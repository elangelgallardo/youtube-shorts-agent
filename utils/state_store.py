"""SQLite-backed persistence for VideoJob objects and analytics cache."""
import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

from config import STATE_DB_PATH
from models.enums import JobStatus
from models.video_job import AnalyticsContext, VideoJob

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS video_jobs (
    job_id      TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    scheduled_upload_at TEXT,
    retry_count INTEGER DEFAULT 0,
    data_json   TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analytics_cache (
    date       TEXT PRIMARY KEY,
    data_json  TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS published_topics (
    topic_hash       TEXT PRIMARY KEY,
    title_concept    TEXT,
    published_at     TEXT,
    youtube_video_id TEXT
);

CREATE TABLE IF NOT EXISTS pending_plans (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    plans_json TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(STATE_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL)
    conn.commit()
    return conn


# ── VideoJob persistence ────────────────────────────────────────────────────

def save_job(job: VideoJob) -> None:
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO video_jobs
               (job_id, status, created_at, scheduled_upload_at,
                retry_count, data_json, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(job_id) DO UPDATE SET
                 status=excluded.status,
                 scheduled_upload_at=excluded.scheduled_upload_at,
                 retry_count=excluded.retry_count,
                 data_json=excluded.data_json,
                 updated_at=excluded.updated_at""",
            (
                job.job_id,
                job.status.value,
                job.created_at,
                job.scheduled_upload_at,
                job.retry_count,
                job.to_json(),
                now,
            ),
        )
        conn.commit()


def load_job(job_id: str) -> Optional[VideoJob]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT data_json FROM video_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    if row:
        return VideoJob.from_json(row["data_json"])
    return None


def load_jobs_by_status(*statuses: JobStatus) -> list[VideoJob]:
    placeholders = ",".join("?" * len(statuses))
    values = [s.value for s in statuses]
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT data_json FROM video_jobs WHERE status IN ({placeholders})"
            " ORDER BY created_at ASC",
            values,
        ).fetchall()
    return [VideoJob.from_json(r["data_json"]) for r in rows]


def load_todays_jobs() -> list[VideoJob]:
    today = datetime.utcnow().date().isoformat()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT data_json FROM video_jobs WHERE created_at LIKE ?",
            (f"{today}%",),
        ).fetchall()
    return [VideoJob.from_json(r["data_json"]) for r in rows]


# ── Analytics cache ─────────────────────────────────────────────────────────

def get_analytics_cache(date: str) -> Optional[AnalyticsContext]:
    with _connect() as conn:
        # Try exact date first, then fall back to most recent cached entry
        row = conn.execute(
            "SELECT data_json FROM analytics_cache WHERE date = ?", (date,)
        ).fetchone()
        if not row:
            row = conn.execute(
                "SELECT data_json FROM analytics_cache ORDER BY date DESC LIMIT 1"
            ).fetchone()
    if row:
        data = json.loads(row["data_json"])
        from models.video_job import VideoMetric
        metrics = [VideoMetric(**m) for m in data.get("raw_metrics", [])]
        return AnalyticsContext(
            top_video_topics=data.get("top_video_topics", []),
            top_formats=data.get("top_formats", []),
            avg_winner_duration_s=data.get("avg_winner_duration_s", 45.0),
            avg_winner_ctr=data.get("avg_winner_ctr", 0.05),
            channel_niche=data.get("channel_niche", ""),
            raw_metrics=metrics,
            fetched_at=data.get("fetched_at", ""),
        )
    return None


def save_analytics_cache(date: str, context: AnalyticsContext) -> None:
    import dataclasses
    with _connect() as conn:
        conn.execute(
            """INSERT INTO analytics_cache (date, data_json, created_at)
               VALUES (?, ?, ?)
               ON CONFLICT(date) DO UPDATE SET data_json=excluded.data_json""",
            (date, json.dumps(dataclasses.asdict(context)), datetime.utcnow().isoformat()),
        )
        conn.commit()


# ── Topic deduplication ─────────────────────────────────────────────────────

def _topic_hash(topic: str, angle: str) -> str:
    return hashlib.sha256(f"{topic.lower()}::{angle.lower()}".encode()).hexdigest()[:16]


def get_recent_published_titles(days: int = 30) -> list[str]:
    """Return title_concepts published in the last N days, for planning dedup."""
    cutoff = (datetime.utcnow() - __import__('datetime').timedelta(days=days)).isoformat()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT title_concept FROM published_topics WHERE published_at >= ? ORDER BY published_at DESC",
            (cutoff,),
        ).fetchall()
    return [r["title_concept"] for r in rows]


def is_topic_duplicate(topic: str, angle: str) -> bool:
    h = _topic_hash(topic, angle)
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM published_topics WHERE topic_hash = ?", (h,)
        ).fetchone()
    return row is not None


def register_published_topic(
    topic: str, angle: str, title_concept: str, youtube_video_id: str
) -> None:
    h = _topic_hash(topic, angle)
    with _connect() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO published_topics
               (topic_hash, title_concept, published_at, youtube_video_id)
               VALUES (?, ?, ?, ?)""",
            (h, title_concept, datetime.utcnow().isoformat(), youtube_video_id),
        )
        conn.commit()


# ── Pending plans cache ─────────────────────────────────────────────────────

def save_pending_plans(plans: list) -> None:
    from models.video_job import VideoPlan
    data = [p.model_dump() if hasattr(p, "model_dump") else p.__dict__ for p in plans]
    with _connect() as conn:
        conn.execute("DELETE FROM pending_plans")
        conn.execute(
            "INSERT INTO pending_plans (created_at, plans_json) VALUES (?, ?)",
            (datetime.utcnow().isoformat(), json.dumps(data, ensure_ascii=False)),
        )
        conn.commit()


def load_pending_plans() -> list:
    from models.video_job import VideoPlan
    from models.enums import VideoFormat
    with _connect() as conn:
        row = conn.execute(
            "SELECT plans_json FROM pending_plans ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if not row:
        return []
    data = json.loads(row["plans_json"])
    plans = []
    for d in data:
        try:
            d["format"] = VideoFormat(d.get("format", "hook_reveal"))
            plans.append(VideoPlan(**d))
        except Exception:
            pass
    return plans


def clear_pending_plans() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM pending_plans")
        conn.commit()


# ── Cleanup ─────────────────────────────────────────────────────────────────

def prune_old_jobs(retention_days: int = 90) -> int:
    cutoff = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM video_jobs WHERE created_at < ?", (cutoff,)
        )
        conn.commit()
    return cur.rowcount
