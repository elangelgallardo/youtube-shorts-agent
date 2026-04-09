"""
AnalyticsAgent — fetches channel performance from YouTube Analytics API
and classifies winning topics/formats using Gemini.

Results are cached in SQLite for 23 hours so all 5 daily jobs share one fetch.
"""
import json
import logging
from datetime import datetime, timedelta

from google import genai
from google.genai import types
from googleapiclient.errors import HttpError

import config
from auth import build_analytics_client, build_youtube_client
from models.video_job import AnalyticsContext, VideoMetric, VideoJob
from utils.retry import with_retry
from utils.state_store import get_analytics_cache, save_analytics_cache

logger = logging.getLogger(__name__)


class AnalyticsAgent:
    def __init__(self):
        self._client = genai.Client(api_key=config.GOOGLE_API_KEY)

    def run(self, job: VideoJob) -> VideoJob:
        """Populate job.analytics_context. Uses cache if available."""
        today = datetime.utcnow().date().isoformat()
        cached = get_analytics_cache(today)
        if cached:
            logger.info("[%s] Using cached analytics for %s", job.job_id, today)
            job.analytics_context = cached
            return job

        logger.info("[%s] Fetching YouTube Analytics…", job.job_id)
        try:
            metrics = self._fetch_metrics()
        except HttpError as exc:
            logger.warning("[%s] Analytics API error: %s — using defaults", job.job_id, exc)
            metrics = []

        context = self._build_context(metrics)
        save_analytics_cache(today, context)
        job.analytics_context = context
        logger.info(
            "[%s] Analytics done: %d top topics, avg CTR %.1f%%",
            job.job_id,
            len(context.top_video_topics),
            context.avg_winner_ctr * 100,
        )
        return job

    @with_retry(max_attempts=3, exceptions=(HttpError, Exception))
    def _fetch_metrics(self) -> list[VideoMetric]:
        analytics = build_analytics_client()
        youtube = build_youtube_client()

        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=config.ANALYTICS_LOOKBACK_DAYS)

        # Step 1: get Shorts video IDs from the uploads playlist
        shorts_ids = self._get_shorts_video_ids(youtube, start_date.isoformat())
        if not shorts_ids:
            logger.info("No Shorts found in uploads playlist — no analytics data")
            return []

        logger.info("Found %d Shorts, fetching analytics…", len(shorts_ids))

        # Step 2: fetch Analytics only for those Shorts IDs
        # Analytics API accepts up to 200 IDs in a filter
        metrics: list[VideoMetric] = []
        for batch_start in range(0, len(shorts_ids), 50):
            batch = shorts_ids[batch_start : batch_start + 50]
            filters = "video==" + ",".join(batch)

            response = (
                analytics.reports()
                .query(
                    ids="channel==MINE",
                    startDate=start_date.isoformat(),
                    endDate=end_date.isoformat(),
                    metrics=(
                        "views,estimatedMinutesWatched,likes,comments,"
                        "averageViewDuration,averageViewPercentage,subscribersGained"
                    ),
                    dimensions="video",
                    filters=filters,
                    sort="-views",
                    maxResults=50,
                )
                .execute()
            )

            rows = response.get("rows", [])
            if not rows:
                continue

            # Fetch titles for this batch
            vresp = youtube.videos().list(part="snippet", id=",".join(batch)).execute()
            titles = {item["id"]: item["snippet"]["title"] for item in vresp.get("items", [])}

            for row in rows:
                # averageViewPercentage is 0-100; convert to 0-1 for ctr field
                avg_view_pct = float(row[6] or 0) / 100.0
                metrics.append(VideoMetric(
                    video_id=row[0],
                    title=titles.get(row[0], f"Short {row[0]}"),
                    views=int(row[1]),
                    watch_minutes=float(row[2]),
                    likes=int(row[3] or 0),
                    comments=int(row[4] or 0),
                    avg_view_duration_s=float(row[5] or 0),
                    ctr=avg_view_pct,
                    subs_gained=int(row[7] or 0),
                ))

        # Sort by views descending and cap at TOP_N
        metrics.sort(key=lambda m: m.views, reverse=True)
        return metrics[: config.ANALYTICS_TOP_N_VIDEOS]

    def _get_shorts_video_ids(self, youtube, published_after_date: str) -> list[str]:
        """
        Return video IDs from the channel's uploads that are Shorts (duration ≤ 180s).
        Uses the uploads playlist + contentDetails duration filter.
        """
        # Get the uploads playlist ID for this channel
        ch_resp = youtube.channels().list(part="contentDetails", mine=True).execute()
        items = ch_resp.get("items", [])
        if not items:
            return []
        uploads_playlist = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

        # Page through the playlist to collect recent video IDs
        video_ids: list[str] = []
        page_token = None
        published_after = published_after_date  # "YYYY-MM-DD"

        while True:
            pl_resp = (
                youtube.playlistItems()
                .list(
                    part="contentDetails",
                    playlistId=uploads_playlist,
                    maxResults=50,
                    pageToken=page_token,
                )
                .execute()
            )
            for item in pl_resp.get("items", []):
                published = item["contentDetails"].get("videoPublishedAt", "")[:10]
                if published < published_after:
                    # Playlist is newest-first; stop when we go past the lookback window
                    page_token = None
                    break
                video_ids.append(item["contentDetails"]["videoId"])

            page_token = pl_resp.get("nextPageToken")
            if not page_token:
                break

        if not video_ids:
            return []

        # Filter to Shorts: fetch contentDetails and keep duration ≤ 180s
        shorts_ids: list[str] = []
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i : i + 50]
            vresp = youtube.videos().list(part="contentDetails", id=",".join(batch)).execute()
            for item in vresp.get("items", []):
                duration_s = _parse_iso8601_duration(item["contentDetails"]["duration"])
                if duration_s <= 180:
                    shorts_ids.append(item["id"])

        return shorts_ids

    def _build_context(self, metrics: list[VideoMetric]) -> AnalyticsContext:
        if not metrics:
            return AnalyticsContext(
                channel_niche=config.CHANNEL_NICHE,
                top_video_topics=[config.CHANNEL_NICHE],
                top_formats=["hook_reveal"],
                avg_winner_duration_s=45.0,
                avg_winner_ctr=0.05,
            )

        top5 = metrics[:5]
        avg_duration = sum(m.avg_view_duration_s for m in top5) / len(top5)
        avg_ctr = sum(m.ctr for m in top5) / len(top5)

        # Build per-short performance table for Gemini
        shorts_table = [
            {
                "title": m.title,
                "views": m.views,
                "avg_view_pct": round(m.ctr * 100, 1),
                "avg_view_s": round(m.avg_view_duration_s, 1),
                "likes": m.likes,
            }
            for m in metrics
        ]
        prompt = (
            f"Manejas un canal de YouTube Shorts sobre '{config.CHANNEL_NICHE}'.\n"
            f"Aquí están los Shorts con sus métricas reales de los últimos {config.ANALYTICS_LOOKBACK_DAYS} días:\n"
            f"{json.dumps(shorts_table, ensure_ascii=False)}\n\n"
            "Analiza qué temas y formatos generan mayor retención (avg_view_pct = % del video visto, avg_view_s = segundos vistos).\n"
            "Devuelve JSON con dos claves:\n"
            '  "topics": lista de hasta 10 temas o subtemas que funcionaron mejor (mayor retención y views), en español\n'
            '  "formats": lista de hasta 5 formatos más exitosos (e.g. "hook_reveal", "myth_bust", "countdown", "listicle")\n'
            "Devuelve solo JSON válido, sin comentarios."
        )

        response = self._client.models.generate_content(
            model=config.GEMINI_FLASH_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
                thinking_config=types.ThinkingConfig(thinking_level="high"),
            ),
        )
        try:
            data = json.loads(response.text)
            topics = data.get("topics", [config.CHANNEL_NICHE])
            formats = data.get("formats", ["hook_reveal"])
        except (json.JSONDecodeError, AttributeError):
            topics = [config.CHANNEL_NICHE]
            formats = ["hook_reveal"]

        return AnalyticsContext(
            top_video_topics=topics,
            top_formats=formats,
            avg_winner_duration_s=avg_duration,
            avg_winner_ctr=avg_ctr,
            channel_niche=config.CHANNEL_NICHE,
            raw_metrics=metrics,
        )


def _parse_iso8601_duration(duration: str) -> float:
    """Parse ISO 8601 duration string (e.g. PT1M30S) into total seconds."""
    import re
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
    if not m:
        return 0.0
    h = int(m.group(1) or 0)
    mins = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    return h * 3600 + mins * 60 + s
