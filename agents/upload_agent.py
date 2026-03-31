"""
UploadAgent — uploads the final video to YouTube and sets Shorts-optimized metadata.
"""
import logging
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

import config
from auth import build_youtube_client
from models.video_job import UploadMetadata, UploadResult, VideoJob
from utils.retry import with_retry
from utils.state_store import register_published_topic

logger = logging.getLogger(__name__)

# YouTube quota: 1600 units per upload; default daily quota = 10,000 units
_CHUNK_SIZE = 5 * 1024 * 1024  # 5 MB resumable upload chunks


class UploadAgent:
    def __init__(self):
        self._client = genai.Client(api_key=config.GOOGLE_API_KEY)

    def run(self, job: VideoJob) -> VideoJob:
        logger.info("[%s] Uploading: %r", job.job_id, job.plan.title_concept)

        metadata = self._build_metadata(job)
        result = self._upload(Path(job.video.final_path), metadata, job.job_id)
        job.upload = result

        register_published_topic(
            topic=job.plan.topic,
            angle=job.plan.angle,
            title_concept=metadata.title,
            youtube_video_id=result.youtube_video_id,
        )

        from utils.cost_tracker import format_cost_report, save_costs
        logger.info("[%s] Uploaded: %s", job.job_id, result.youtube_url)
        logger.info("[%s] %s", job.job_id, format_cost_report(job))
        save_costs(job)
        return job

    def _build_metadata(self, job: VideoJob) -> UploadMetadata:
        import json

        script_text = " ".join(s.spoken_text for s in job.script.scenes)
        facts_block = "\n".join(f"- {f}" for f in job.research.facts[:5])

        prompt = (
            f"Eres un experto en SEO para YouTube Shorts en español.\n\n"
            f"Genera metadatos optimizados para este video:\n"
            f"Tema: {job.plan.title_concept}\n"
            f"Guión resumido: {script_text[:600]}\n"
            f"Datos clave: {job.research.key_stat}\n"
            f"Hechos:\n{facts_block}\n\n"
            "Devuelve JSON con exactamente estas claves:\n"
            '  "title": título atractivo en español, máx 80 chars, con emoji si ayuda al CTR\n'
            '  "description": descripción de 150-300 chars que explique el video, '
            "termine con los hashtags más relevantes (#Shorts obligatorio)\n"
            '  "tags": array de 10-15 tags en español e inglés, mezcla de términos '
            "específicos y generales, sin # prefix\n\n"
            "Solo JSON, sin markdown."
        )

        try:
            response = self._client.models.generate_content(
                model=config.GEMINI_FLASH_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.7,
                    max_output_tokens=512,
                ),
            )
            from utils.cost_tracker import record_gemini
            if response.usage_metadata:
                record_gemini(job, "upload",
                    response.usage_metadata.prompt_token_count or 0,
                    response.usage_metadata.candidates_token_count or 0)
            data = json.loads(response.text)
            title = data.get("title", job.plan.title_concept)[:100]
            description = data.get("description", "")[:5000]
            tags = data.get("tags", [])[:15]
            logger.info("[%s] LLM metadata: %r", job.job_id, title)
        except Exception as e:
            logger.warning("[%s] LLM metadata failed, using fallback: %s", job.job_id, e)
            title = job.plan.title_concept[:100]
            facts = " ".join(f"• {f}" for f in (job.research.facts[:4] if job.research else []))
            description = f"{job.research.key_stat}\n\n{facts}\n\n#Shorts #Espacio #Ciencia #Astrofísica".strip()[:5000]
            # Build tags from plan topic words + generic science tags
            raw_tags = job.plan.topic.split() + ["espacio", "ciencia", "astrofísica", "universo", "shorts", "nasa", "cosmos"]
            tags = list(dict.fromkeys(t.lower().strip() for t in raw_tags if t.strip()))

        # Enforce YouTube's 500-char total tag limit
        trimmed_tags: list[str] = []
        total = 0
        for tag in tags:
            tag = str(tag).strip()
            if total + len(tag) + 1 > 490:
                break
            trimmed_tags.append(tag)
            total += len(tag) + 1

        return UploadMetadata(
            title=title,
            description=description,
            tags=trimmed_tags,
            category_id=config.YOUTUBE_CATEGORY_ID,
            privacy_status=config.DEFAULT_PRIVACY,
        )

    @with_retry(max_attempts=4, exceptions=(HttpError, Exception))
    def _upload(
        self,
        video_path: Path,
        metadata: UploadMetadata,
        job_id: str,
    ) -> UploadResult:
        youtube = build_youtube_client()

        body = {
            "snippet": {
                "title": metadata.title,
                "description": metadata.description,
                "tags": metadata.tags,
                "categoryId": metadata.category_id,
                "defaultLanguage": "en",
                "defaultAudioLanguage": "en",
            },
            "status": {
                "privacyStatus": metadata.privacy_status,
                "selfDeclaredMadeForKids": False,
                "madeForKids": False,
            },
        }

        media = MediaFileUpload(
            str(video_path),
            mimetype="video/mp4",
            resumable=True,
            chunksize=_CHUNK_SIZE,
        )

        request = youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media,
        )

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                logger.info("[%s] Upload progress: %d%%", job_id, pct)

        video_id = response["id"]
        return UploadResult(
            youtube_video_id=video_id,
            youtube_url=f"https://www.youtube.com/shorts/{video_id}",
            upload_time=datetime.utcnow().isoformat(),
            metadata=metadata,
        )


def _slug(text: str) -> str:
    import re
    return re.sub(r"\W+", "", text.title().replace(" ", ""))


def _build_tags(job: VideoJob) -> list[str]:
    """Build a tag list within YouTube's 500-char total limit."""
    candidates = (
        job.plan.topic.split()
        + job.analytics_context.channel_niche.split()
        + ["shorts", "facts", "didyouknow", "science"]
        + [_slug(t) for t in job.analytics_context.top_video_topics[:5]]
    )

    tags: list[str] = []
    total_chars = 0
    for tag in dict.fromkeys(candidates):  # deduplicate, preserve order
        tag = tag.strip("#").lower()
        if not tag:
            continue
        if total_chars + len(tag) + 1 > 490:
            break
        tags.append(tag)
        total_chars += len(tag) + 1

    return tags
