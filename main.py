"""
main.py — 24-hour autonomous YouTube Shorts agent.

Daily schedule (UTC):
  GENERATION_START_HOUR  — fetch analytics, plan 5 videos, generate all media
  UPLOAD_HOURS[0..4]     — staggered uploads (default 08, 10, 12, 14, 16)
  23:00                  — workspace cleanup

Run:
    python main.py

First-time setup:
    python auth.py      ← complete YouTube OAuth in browser
"""
import concurrent.futures
import logging
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

import config
from agents.analytics_agent import AnalyticsAgent
from agents.planning_agent import PlanningAgent
from models.enums import JobStatus
from models.video_job import VideoJob
from pipeline import run_video_pipeline
from utils.state_store import (
    load_jobs_by_status,
    load_todays_jobs,
    prune_old_jobs,
    save_job,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            config.LOGS_DIR / f"agent_{datetime.utcnow().strftime('%Y%m%d')}.log"
        ),
    ],
)
logger = logging.getLogger(__name__)

_analytics_agent = AnalyticsAgent()
_planning_agent = PlanningAgent()


# ── Core daily jobs ──────────────────────────────────────────────────────────

def daily_generation() -> None:
    """
    Runs at GENERATION_START_HOUR UTC.
    1. Fetch analytics (cached per day)
    2. Plan 5 videos
    3. Generate research + script + media for all 5 in parallel
    """
    logger.info("=== Daily generation started ===")

    # Step 1: Analytics (shared across all 5 jobs)
    dummy_job = VideoJob()
    dummy_job = _analytics_agent.run(dummy_job)
    analytics = dummy_job.analytics_context

    # Step 2: Plan topics
    plans = _planning_agent.run(analytics)
    plans = plans[: config.VIDEOS_PER_DAY]  # respect current count (may be 1 in dry-run)
    logger.info("Planned %d videos for today", len(plans))

    # Step 3: Create VideoJob records with scheduled upload times
    jobs: list[VideoJob] = []
    for i, plan in enumerate(plans):
        upload_hour = config.UPLOAD_HOURS[i] if i < len(config.UPLOAD_HOURS) else config.UPLOAD_HOURS[-1]
        scheduled_at = datetime.utcnow().replace(
            hour=upload_hour, minute=0, second=0, microsecond=0
        ).isoformat()

        job = VideoJob()
        job.analytics_context = analytics
        job.plan = plan
        job.scheduled_upload_at = scheduled_at
        save_job(job)
        jobs.append(job)
        logger.info("[%s] Created job: %r → upload at %s", job.job_id, plan.title_concept, scheduled_at)

    # Step 4: Generate research + script + media in parallel (no upload yet)
    logger.info("Generating media for %d jobs in parallel…", len(jobs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_generate_without_upload, job): job.job_id for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job_id = futures[future]
            try:
                updated_job = future.result()
                logger.info(
                    "[%s] Generation complete: status=%s", job_id, updated_job.status.value
                )
            except Exception as exc:
                logger.error("[%s] Generation thread crashed: %s", job_id, exc, exc_info=True)

    logger.info("=== Daily generation finished ===")


def _generate_without_upload(job: VideoJob) -> VideoJob:
    """Run research → script → media → assembly, but NOT upload."""
    job = run_video_pipeline(job, upload=False)
    # Mark as READY so upload jobs can pick it up
    if job.status == JobStatus.DONE:
        job.status = JobStatus.READY
        save_job(job)
    return job


def upload_slot(slot_index: int) -> None:
    """
    Runs at each UPLOAD_HOURS[slot_index] UTC.
    Picks up the next READY job and uploads it.
    """
    logger.info("=== Upload slot %d ===", slot_index)

    # Find ready jobs from today, ordered by creation time
    ready_jobs = [
        j for j in load_todays_jobs()
        if j.status in (JobStatus.READY, JobStatus.FAILED)
        and j.retry_count < 3
    ]

    if not ready_jobs:
        logger.warning("Upload slot %d: no ready jobs found", slot_index)
        return

    job = ready_jobs[0]
    logger.info("[%s] Uploading slot %d: %r", job.job_id, slot_index, job.plan.title_concept)

    # If job previously failed, retry from the failed stage
    if job.status == JobStatus.FAILED:
        logger.info("[%s] Retrying failed job (attempt %d)", job.job_id, job.retry_count + 1)
        job = run_video_pipeline(job, upload=True)
    else:
        # Job is READY — just upload
        from agents.upload_agent import UploadAgent
        try:
            job.status = JobStatus.UPLOADING
            save_job(job)
            job = UploadAgent().run(job)
            job.status = JobStatus.DONE
            save_job(job)
            logger.info("[%s] Upload done: %s", job.job_id, job.upload.youtube_url)
        except Exception as exc:
            job.retry_count += 1
            job.errors.append(f"Upload slot {slot_index}: {exc}")
            job.status = JobStatus.FAILED if job.retry_count < 3 else JobStatus.PERMANENTLY_FAILED
            save_job(job)
            logger.error("[%s] Upload failed: %s", job.job_id, exc, exc_info=True)


def cleanup_workspace() -> None:
    """Delete workspace directories older than WORKSPACE_RETENTION_DAYS."""
    cutoff = datetime.utcnow() - timedelta(days=config.WORKSPACE_RETENTION_DAYS)
    deleted = 0
    for job_dir in config.WORKSPACE_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        mtime = datetime.utcfromtimestamp(job_dir.stat().st_mtime)
        if mtime < cutoff:
            shutil.rmtree(job_dir, ignore_errors=True)
            deleted += 1
    pruned = prune_old_jobs(retention_days=90)
    logger.info("Cleanup: deleted %d workspace dirs, pruned %d DB records", deleted, pruned)


# ── Scheduler setup ──────────────────────────────────────────────────────────

def build_scheduler() -> BlockingScheduler:
    config.LOGS_DIR.mkdir(exist_ok=True)
    config.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    scheduler = BlockingScheduler(
        jobstores={
            "default": SQLAlchemyJobStore(url=f"sqlite:///{config.STATE_DB_PATH}")
        },
        job_defaults={
            "misfire_grace_time": 3600,  # Run up to 1h late if scheduler was down
            "coalesce": True,            # Collapse multiple missed fires into one
        },
        timezone="UTC",
    )

    gen_hour = config.GENERATION_START_HOUR
    scheduler.add_job(
        daily_generation,
        "cron",
        hour=gen_hour,
        minute=0,
        id="daily_generation",
        replace_existing=True,
    )
    logger.info("Scheduled generation at %02d:00 UTC daily", gen_hour)

    for i, upload_hour in enumerate(config.UPLOAD_HOURS):
        scheduler.add_job(
            upload_slot,
            "cron",
            hour=upload_hour,
            minute=0,
            args=[i],
            id=f"upload_slot_{i}",
            replace_existing=True,
        )
        logger.info("Scheduled upload slot %d at %02d:00 UTC daily", i, upload_hour)

    scheduler.add_job(
        cleanup_workspace,
        "cron",
        hour=23,
        minute=0,
        id="cleanup",
        replace_existing=True,
    )

    return scheduler


def run_now(dry_run: bool = False) -> None:
    """Manual trigger: run the full pipeline immediately for testing."""
    logger.info("Running full pipeline NOW (dry_run=%s)…", dry_run)
    import config as cfg
    if dry_run:
        # Override privacy to private and limit to 1 video for quick testing
        cfg.DEFAULT_PRIVACY = "private"
        original_count = cfg.VIDEOS_PER_DAY
        cfg.VIDEOS_PER_DAY = 1
        daily_generation()
        cfg.VIDEOS_PER_DAY = original_count
    else:
        daily_generation()
    logger.info("Generation complete. Triggering upload slot 0 in 5 seconds…")
    import time; time.sleep(5)
    upload_slot(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YouTube Shorts 24h Agent")
    parser.add_argument(
        "--now",
        action="store_true",
        help="Run one full pipeline immediately (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --now: upload as private instead of public",
    )
    args = parser.parse_args()

    if args.now:
        run_now(dry_run=args.dry_run)
    else:
        logger.info("Starting 24h YouTube Shorts Agent scheduler…")
        logger.info("Generation: %02d:00 UTC | Uploads: %s UTC",
                    config.GENERATION_START_HOUR,
                    ", ".join(f"{h:02d}:00" for h in config.UPLOAD_HOURS))
        scheduler = build_scheduler()
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped.")
