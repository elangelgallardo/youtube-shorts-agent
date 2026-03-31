"""
PlanningAgent — uses Gemini to generate 5 VideoPlan objects for today.

Mix: 4 analytics-driven topics + 1 exploratory (new direction).
Deduplication is enforced against published_topics in the state store.
"""
import json
import logging
from datetime import datetime

from google import genai
from google.genai import types

import config
from models.enums import VideoFormat
from models.video_job import AnalyticsContext, VideoPlan
from utils.retry import with_retry
from utils.state_store import is_topic_duplicate, get_recent_published_titles

logger = logging.getLogger(__name__)

_FORMAT_VALUES = [f.value for f in VideoFormat]


class PlanningAgent:
    def __init__(self):
        self._client = genai.Client(api_key=config.GOOGLE_API_KEY)

    def run(
        self,
        analytics: AnalyticsContext,
        existing_titles: list[str] | None = None,
    ) -> list[VideoPlan]:
        existing_titles = list(existing_titles or [])
        existing_titles += get_recent_published_titles(days=30)
        plans: list[VideoPlan] = []
        attempts = 0

        while len(plans) < config.VIDEOS_PER_DAY and attempts < 10:
            attempts += 1
            candidates = self._generate_candidates(analytics, existing_titles, len(plans))
            for c in candidates:
                if len(plans) >= config.VIDEOS_PER_DAY:
                    break
                topic = c.get("topic", "")
                angle = c.get("angle", "")
                if not topic or is_topic_duplicate(topic, angle):
                    logger.debug("Skipping duplicate topic: %s / %s", topic, angle)
                    continue
                try:
                    fmt = VideoFormat(c.get("format", "hook_reveal"))
                except ValueError:
                    fmt = VideoFormat.HOOK_REVEAL

                plan = VideoPlan(
                    title_concept=c.get("title_concept", topic)[:100],
                    topic=topic,
                    angle=angle,
                    format=fmt,
                    target_duration_s=c.get("target_duration_s", 55),
                    is_exploratory=c.get("is_exploratory", False),
                )
                plans.append(plan)
                existing_titles.append(plan.title_concept)

        if len(plans) < config.VIDEOS_PER_DAY:
            logger.warning(
                "Only generated %d/%d plans after %d attempts",
                len(plans), config.VIDEOS_PER_DAY, attempts,
            )

        return plans[: config.VIDEOS_PER_DAY]

    @with_retry(max_attempts=3, exceptions=(json.JSONDecodeError, Exception))
    def _generate_candidates(
        self,
        analytics: AnalyticsContext,
        avoid_titles: list[str],
        already_have: int,
    ) -> list[dict]:
        needed = config.VIDEOS_PER_DAY - already_have
        exploratory_count = max(1, needed // 5)

        # Build performance table from real metrics if available, else top topics
        if analytics.raw_metrics:
            perf_rows = [
                {
                    "title": m.title,
                    "views": m.views,
                    "avg_watch_s": round(m.avg_view_duration_s, 0),
                    "watch_pct": round(m.ctr * 100, 0),
                }
                for m in sorted(analytics.raw_metrics, key=lambda m: m.views, reverse=True)[:10]
            ]
            perf_block = json.dumps(perf_rows, ensure_ascii=False, indent=2)
        else:
            perf_block = None

        avoid_block = json.dumps(avoid_titles, ensure_ascii=False) if avoid_titles else "[]"

        build_on_count = round(needed * 0.6)
        new_topic_count = needed - build_on_count

        prompt = "Here are the recent videos from a YouTube Shorts astrophysics channel (Spanish), sorted by views:\n\n"
        if perf_block:
            prompt += f"{perf_block}\n\n"
        else:
            prompt += "(No performance data yet — use your best judgment for the channel niche below.)\n\n"

        prompt += (
            f"Already published titles to AVOID repeating: {avoid_block}\n\n"
            f"Generate exactly {needed} video ideas in Spanish, split as follows:\n\n"
            f"- {build_on_count} ideas that BUILD ON or GO DEEPER into topics from the best-performing videos above. "
            "These should explore a related angle, a follow-up question, a deeper aspect, or a surprising extension "
            "of a topic that already resonated with the audience. They should feel like natural sequels or companions "
            "to existing content — but with a fresh angle, not a repeat.\n\n"
            f"- {new_topic_count} ideas on COMPLETELY NEW topics not covered in any of the videos above. "
            "These should expand the channel's range while staying within astrophysics/cosmology/physics.\n\n"
            "Mark build-on ideas with `\"is_exploratory\": false` and new-topic ideas with `\"is_exploratory\": true`.\n\n"
            "Return a JSON array where each element has:\n"
            '  "title_concept": string (catchy Spanish title, max 60 chars)\n'
            '  "topic": string (specific subject, 2-5 words, Spanish)\n'
            '  "angle": string (unique angle or surprising fact, 1 sentence, Spanish)\n'
            f'  "format": one of {_FORMAT_VALUES}\n'
            '  "target_duration_s": integer (60-75)\n'
            '  "is_exploratory": boolean\n\n'
            "Return ONLY the JSON array, no comments."
        )

        response = self._client.models.generate_content(
            model=config.GEMINI_FLASH_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.85,
                max_output_tokens=1024,
            ),
        )
        data = json.loads(response.text)
        return data if isinstance(data, list) else []
