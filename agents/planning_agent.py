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

        build_on_count = round(needed * 0.3)
        new_topic_count = needed - build_on_count

        # Extract overused themes from published titles so the LLM knows what to avoid
        overused = _detect_overused_themes(avoid_titles)
        overused_block = ", ".join(overused) if overused else "none"

        prompt = (
            "You are planning content for a YouTube Shorts science channel in Spanish.\n"
            "The channel covers ALL of science — not just astrophysics.\n\n"
        )

        if perf_block:
            prompt += f"Top-performing videos (sorted by views):\n{perf_block}\n\n"

        prompt += (
            f"Already published titles (DO NOT repeat these themes): {avoid_block}\n\n"
            f"⛔ BANNED — do NOT generate ideas about these overused themes: {overused_block}\n"
            f"Any idea touching a banned theme will be rejected. Pick completely different subjects.\n\n"
            f"Generate exactly {needed} video ideas in Spanish.\n\n"
            "DIVERSITY RULES — mandatory:\n"
            f"- The {needed} ideas must span AT LEAST 4 different science domains from this list:\n"
            "  astrophysics, quantum physics, particle physics, biology, neuroscience, chemistry,\n"
            "  geology/earth science, mathematics, technology/engineering, paleontology, medicine, ecology\n"
            "- No more than 2 ideas from any single domain\n"
            "- Prefer unexpected, counterintuitive, or surprising angles over obvious ones\n\n"
            f"Split as follows:\n"
            f"- {build_on_count} ideas that GO DEEPER on a top-performing topic (fresh angle only, not a repeat)\n"
            f"- {new_topic_count} ideas on topics NOT covered before, spread across different domains\n\n"
            "Mark build-on ideas with `\"is_exploratory\": false`, new topics with `\"is_exploratory\": true`.\n\n"
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
                temperature=1.1,
                max_output_tokens=2048,
            ),
        )
        data = json.loads(response.text)
        return data if isinstance(data, list) else []


# ── Helpers ───────────────────────────────────────────────────────────────────

_THEME_KEYWORDS = {
    "agujeros negros": ["agujero negro", "agujeros negros", "black hole"],
    "materia oscura": ["materia oscura", "dark matter"],
    "antimateria": ["antimateria", "antimatter"],
    "luna": ["la luna", "luna desaparec", "luna se acerc"],
    "tierra rotación": ["tierra dejara de girar", "tierra para", "tierra gir"],
    "simulación": ["simulación", "simulacion", "simulation"],
    "viajes en el tiempo": ["viajes en el tiempo", "viaje en el tiempo", "time travel"],
    "vida extraterrestre": ["vida extraterrestre", "extraterrestre", "alien"],
    "big bang": ["big bang", "antes del big bang"],
    "entrelazamiento cuántico": ["entrelazamiento", "quantum entangl"],
}

def _detect_overused_themes(titles: list[str]) -> list[str]:
    """Return theme names that appear 2+ times in the published titles list."""
    from collections import Counter
    counts: Counter = Counter()
    titles_lower = [t.lower() for t in titles]
    for theme, keywords in _THEME_KEYWORDS.items():
        for title in titles_lower:
            if any(kw in title for kw in keywords):
                counts[theme] += 1
    return [theme for theme, count in counts.items() if count >= 1]
