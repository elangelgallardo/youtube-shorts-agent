"""
Fetches transcripts from top-performing YouTube Shorts via YouTube Data API (captions).
Uses OAuth (same credentials as uploads) since we're fetching our own channel's captions.
Caches results in SQLite for 72 hours.
"""
import logging
import re
import sqlite3
from datetime import datetime, timedelta

import config

logger = logging.getLogger(__name__)

_TRANSCRIPT_DB = config.BASE_DIR / "transcript_cache.db"
_CACHE_HOURS = 72


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_TRANSCRIPT_DB))
    conn.row_factory = sqlite3.Row
    return conn


def get_top_performer_transcripts(video_ids: list[str], max_videos: int = 3) -> list[dict]:
    """
    Return transcripts for up to `max_videos` of the given IDs.
    Each entry: {"video_id": str, "text": str}
    Caches results in SQLite.
    """
    _ensure_table()
    results = []
    for vid in video_ids:
        if len(results) >= max_videos:
            break
        cached = _get_cached(vid)
        if cached:
            logger.info("Transcript cache hit: %s", vid)
            results.append(cached)
            continue
        transcript = _fetch_via_api(vid)
        if transcript:
            _save_cached(vid, transcript)
            results.append({"video_id": vid, "text": transcript})
    return results


def _fetch_via_api(video_id: str) -> str | None:
    """Fetch caption text via YouTube Data API using OAuth credentials."""
    try:
        from auth import build_youtube_client
        youtube = build_youtube_client()

        # List available captions for this video
        captions_resp = youtube.captions().list(part="snippet", videoId=video_id).execute()
        items = captions_resp.get("items", [])
        if not items:
            logger.debug("No captions found for %s", video_id)
            return None

        # Prefer manually uploaded Spanish, fall back to auto-generated
        caption_id = None
        for item in items:
            lang = item["snippet"].get("language", "")
            track_kind = item["snippet"].get("trackKind", "")
            if lang.startswith("es"):
                caption_id = item["id"]
                if track_kind != "asr":  # prefer manual over auto
                    break

        if not caption_id:
            caption_id = items[0]["id"]

        # Download the caption as plain text (srt format, then strip timestamps)
        caption_bytes = youtube.captions().download(id=caption_id, tfmt="srt").execute()
        text = _srt_to_plain_text(caption_bytes.decode("utf-8") if isinstance(caption_bytes, bytes) else caption_bytes)
        logger.info("Fetched caption for %s via API (%d chars)", video_id, len(text))
        return text

    except Exception as e:
        logger.warning("Caption fetch failed for %s: %s", video_id, e)
        return None


def _srt_to_plain_text(srt: str) -> str:
    """Strip SRT timestamps and indices, return clean narration text."""
    # Remove index numbers and timestamps
    text = re.sub(r"^\d+\s*$", "", srt, flags=re.MULTILINE)
    text = re.sub(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", "", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _ensure_table():
    with _get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transcript_cache (
                video_id   TEXT PRIMARY KEY,
                text       TEXT NOT NULL,
                fetched_at TEXT NOT NULL
            )
        """)


def _get_cached(video_id: str) -> dict | None:
    cutoff = (datetime.utcnow() - timedelta(hours=_CACHE_HOURS)).isoformat()
    with _get_db() as conn:
        row = conn.execute(
            "SELECT text FROM transcript_cache WHERE video_id = ? AND fetched_at > ?",
            (video_id, cutoff),
        ).fetchone()
    return {"video_id": video_id, "text": row[0]} if row else None


def _save_cached(video_id: str, text: str):
    with _get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO transcript_cache (video_id, text, fetched_at) VALUES (?, ?, ?)",
            (video_id, text, datetime.utcnow().isoformat()),
        )
