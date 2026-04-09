"""
ResearchAgent — uses Gemini with Google Search grounding to gather
current, data-backed facts for each video topic.
"""
import json
import logging

from google import genai
from google.genai import types

import config
from models.video_job import ResearchContext, VideoJob
from utils.retry import with_retry

logger = logging.getLogger(__name__)


class ResearchAgent:
    def __init__(self):
        self._client = genai.Client(api_key=config.GOOGLE_API_KEY)

    def run(self, job: VideoJob) -> VideoJob:
        logger.info("[%s] Researching: %s", job.job_id, job.plan.topic)

        try:
            ctx = self._research_with_grounding(job)
        except Exception as exc:
            logger.warning(
                "[%s] Grounded research failed (%s), falling back to ungrounded",
                job.job_id, exc,
            )
            ctx = self._research_fallback(job)
            ctx.grounded = False

        job.research = ctx
        logger.info(
            "[%s] Research done: key_stat=%r, %d facts, grounded=%s",
            job.job_id, ctx.key_stat[:60], len(ctx.facts), ctx.grounded,
        )
        return job

    @with_retry(max_attempts=3, exceptions=(Exception,))
    def _research_with_grounding(self, job: VideoJob) -> ResearchContext:
        response = self._client.models.generate_content(
            model=config.GEMINI_RESEARCH_MODEL,
            contents=self._build_prompt(job),
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2,
                max_output_tokens=1024,
                thinking_config=types.ThinkingConfig(thinking_level="medium"),
                http_options=types.HttpOptions(timeout=180000),
            ),
        )

        # Extract grounding metadata
        sources: list[str] = []
        queries: list[str] = []
        try:
            gm = response.candidates[0].grounding_metadata
            if gm:
                for chunk in gm.grounding_chunks or []:
                    if hasattr(chunk, "web") and chunk.web and chunk.web.uri:
                        sources.append(chunk.web.uri)
                for q in gm.web_search_queries or []:
                    queries.append(q)
        except (AttributeError, IndexError):
            pass

        from utils.cost_tracker import record_gemini
        if response.usage_metadata:
            record_gemini(job, "research",
                response.usage_metadata.prompt_token_count or 0,
                response.usage_metadata.candidates_token_count or 0)
        ctx = self._parse_response(response.text, job)
        ctx.sources = sources[:5]
        ctx.search_queries_used = queries[:3]
        ctx.grounded = True
        return ctx

    def _research_fallback(self, job: VideoJob) -> ResearchContext:
        response = self._client.models.generate_content(
            model=config.GEMINI_RESEARCH_MODEL,
            contents=self._build_prompt(job),
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1024,
                thinking_config=types.ThinkingConfig(thinking_level="medium"),
                http_options=types.HttpOptions(timeout=180000),
            ),
        )
        from utils.cost_tracker import record_gemini
        if response.usage_metadata:
            record_gemini(job, "research",
                response.usage_metadata.prompt_token_count or 0,
                response.usage_metadata.candidates_token_count or 0)
        return self._parse_response(response.text, job)

    def _build_prompt(self, job: VideoJob) -> str:
        return (
            f"Investiga el siguiente tema para un video de YouTube Shorts en español mexicano.\n"
            f"Tema: {job.plan.topic}\n"
            f"Ángulo: {job.plan.angle}\n"
            f"Nicho del canal: {job.analytics_context.channel_niche}\n\n"
            "Devuelve un objeto JSON con estas claves exactas:\n"
            '  "key_stat": una estadística sorprendente o contraintuitiva con un número específico '
            "(ej. 'El 73% de X hace Y'). Este será el gancho. Escríbela en español mexicano.\n"
            '  "facts": lista de 4-6 datos de apoyo con especificidad numérica, '
            "cada uno de 1-2 oraciones, en español mexicano.\n"
            '  "sources": lista de hasta 3 nombres de fuentes o URLs (vacío si se desconocen)\n'
            '  "search_queries_used": []\n'
            '  "grounded": false\n\n'
            "Sé preciso y basado en datos. Prioriza información reciente (post-2022).\n"
            "Devuelve SOLO el objeto JSON."
        )

    def _parse_response(self, text: str, job: VideoJob) -> ResearchContext:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            import re
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            data = json.loads(m.group(1)) if m else {}

        return ResearchContext(
            key_stat=data.get("key_stat", f"Surprising fact about {job.plan.topic}"),
            facts=data.get("facts", [f"Key insight about {job.plan.topic}"]),
            sources=data.get("sources", []),
            search_queries_used=data.get("search_queries_used", []),
            grounded=data.get("grounded", False),
        )
