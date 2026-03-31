"""
Single-video pipeline: runs all agents in sequence for one VideoJob.
Called by main.py for each of the 5 daily jobs.
"""
import logging
from datetime import datetime

from models.enums import JobStatus
from models.video_job import VideoJob
from agents.research_agent import ResearchAgent
from agents.script_agent import ScriptAgent
from agents.media_agent import MediaAgent
from agents.assembly_agent import AssemblyAgent
from agents.upload_agent import UploadAgent
from utils.state_store import save_job

logger = logging.getLogger(__name__)

_research = ResearchAgent()
_script = ScriptAgent()
_media = MediaAgent()
_assembly = AssemblyAgent()
_upload = UploadAgent()


def run_video_pipeline(job: VideoJob, upload: bool = True) -> VideoJob:
    """
    Execute the full pipeline for a single VideoJob.

    Returns the updated job object regardless of success/failure.
    Caller should check job.status == JobStatus.DONE.
    """
    stages = [
        (JobStatus.RESEARCHING, _research.run),
        (JobStatus.SCRIPTING, _script.run),
        (JobStatus.GENERATING_MEDIA, _media.run),
        (JobStatus.ASSEMBLING, _assembly.run),
    ]
    if upload:
        stages.append((JobStatus.UPLOADING, _upload.run))

    for status, fn in stages:
        job.status = status
        save_job(job)
        try:
            job = fn(job)
        except Exception as exc:
            _handle_failure(job, status, exc)
            return job

    job.status = JobStatus.DONE
    save_job(job)
    logger.info(
        "[%s] Pipeline complete: %s",
        job.job_id,
        job.upload.youtube_url or "(no upload)",
    )
    return job


def _handle_failure(job: VideoJob, failed_at: JobStatus, exc: Exception) -> None:
    job.retry_count += 1
    error_msg = f"{failed_at.value}: {type(exc).__name__}: {exc}"
    job.errors.append(f"[{datetime.utcnow().isoformat()}] {error_msg}")
    logger.error("[%s] Pipeline failed at %s: %s", job.job_id, failed_at.value, exc, exc_info=True)

    if job.retry_count >= 3:
        job.status = JobStatus.PERMANENTLY_FAILED
        logger.error("[%s] Permanently failed after 3 attempts", job.job_id)
    else:
        job.status = JobStatus.FAILED

    save_job(job)
