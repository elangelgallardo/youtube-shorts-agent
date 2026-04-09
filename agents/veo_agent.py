"""
VeoAgent — generates one Veo clip per scene, trimmed to scene duration.
Sequential generation to stay within the 10-videos/day quota.
"""
import logging
import subprocess
import time
from pathlib import Path

from google import genai
from google.genai import types

import config
from models.video_job import VideoJob

logger = logging.getLogger(__name__)

VEO_MODEL = "veo-3.1-generate-preview"
_POLL_INTERVAL_S = 10
_MAX_POLLS = 36  # 6 minutes max per clip


class VeoAgent:
    def __init__(self):
        self._client = genai.Client(api_key=config.GOOGLE_API_KEY)

    def run(self, job: VideoJob) -> VideoJob:
        total = len(job.images)
        logger.info("[%s] Generating %d Veo clips sequentially…", job.job_id, total)

        _EDGE_BUFFER_S = 2.0  # extra footage for first/last clips (intro/outro breathing room)

        for i, asset in enumerate(job.images):
            scene = next((s for s in job.script.scenes if s.scene_id == asset.scene_id), None)
            if not scene:
                logger.warning("[%s] No scene found for asset %d, skipping", job.job_id, asset.scene_id)
                continue

            clip_path = job.workspace_dir / f"veo_scene_{asset.scene_id:03d}.mp4"

            if clip_path.exists():
                logger.info("[%s] Scene %d/%d: clip already exists, reusing", job.job_id, i + 1, total)
                asset.veo_clip_path = str(clip_path)
                continue

            is_edge = (i == 0 or i == total - 1)
            target_duration = min(8.0, asset.duration_s + _EDGE_BUFFER_S) if is_edge else asset.duration_s

            logger.info("[%s] Scene %d/%d: submitting Veo… (%.1fs%s)", job.job_id, i + 1, total,
                        target_duration, " +buffer" if is_edge else "")
            try:
                self._generate_clip(scene.visual_prompt, target_duration, clip_path, job)
                asset.veo_clip_path = str(clip_path)
                logger.info("[%s] Scene %d/%d: done (%.1fs)", job.job_id, i + 1, total, target_duration)
            except Exception as exc:
                logger.error("[%s] Scene %d Veo failed: %s — skipping clip", job.job_id, asset.scene_id, exc)
                job.errors.append(f"Veo scene {asset.scene_id}: {exc}")

        done = sum(1 for a in job.images if a.veo_clip_path)
        logger.info("[%s] VeoAgent done: %d/%d clips generated", job.job_id, done, total)
        return job

    def _generate_clip(self, prompt: str, target_duration: float, out_path: Path, job: VideoJob) -> None:
        operation = self._client.models.generate_videos(
            model=VEO_MODEL,
            prompt=prompt,
            config=types.GenerateVideosConfig(aspect_ratio="9:16"),
        )

        for _ in range(_MAX_POLLS):
            if operation.done:
                break
            time.sleep(_POLL_INTERVAL_S)
            operation = self._client.operations.get(operation)

        if not operation.done or not operation.response.generated_videos:
            raise RuntimeError("Veo operation did not complete in time")

        video_obj = operation.response.generated_videos[0].video
        raw_path = out_path.with_suffix(".raw.mp4")

        if video_obj.video_bytes:
            raw_path.write_bytes(video_obj.video_bytes)
        elif video_obj.uri:
            import requests
            uri = video_obj.uri
            sep = "&" if "?" in uri else "?"
            resp = requests.get(f"{uri}{sep}key={config.GOOGLE_API_KEY}", timeout=120)
            if resp.status_code == 403:
                resp = requests.get(f"{uri}{sep}key={config.GOOGLE_API_KEY}&alt=media", timeout=120)
            resp.raise_for_status()
            raw_path.write_bytes(resp.content)
        else:
            raise RuntimeError("No video data returned from Veo")

        # Trim to scene duration, scale to 1080x1920
        duration = max(target_duration, 1.0)
        subprocess.run([
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", str(raw_path),
            "-t", str(duration),
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
            "-r", "30", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an",
            "-loglevel", "error",
            str(out_path),
        ], check=True)
        raw_path.unlink(missing_ok=True)

        from utils.cost_tracker import record_veo
        record_veo(job, 8.0)
