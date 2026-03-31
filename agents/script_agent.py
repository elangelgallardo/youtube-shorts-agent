"""
ScriptAgent — generates a 5-scene Short script from the research context.

Each scene has spoken_text, a visual_prompt for image generation, and a duration hint.
"""
import json
import logging

from google import genai
from google.genai import types

import config
from models.enums import SceneType
from models.video_job import Scene, Script, VideoJob
from utils.retry import with_retry
from utils.ssml_builder import build_ssml

logger = logging.getLogger(__name__)

_VISUAL_STYLE = (
    "cinematic digital illustration, dark background, vivid colors, "
    "highly detailed, main subject centered, no text no labels no diagrams"
)


class ScriptAgent:
    def __init__(self):
        self._client = genai.Client(api_key=config.GOOGLE_API_KEY)

    def run(self, job: VideoJob) -> VideoJob:
        logger.info("[%s] Writing script: %s", job.job_id, job.plan.title_concept)
        script = self._generate_script(job)
        job.script = script
        logger.info(
            "[%s] Script done: %d scenes, ~%ds, hook=%r",
            job.job_id, len(script.scenes), script.total_duration_estimate_s,
            script.hook_line[:60],
        )
        return job

    @with_retry(max_attempts=3, exceptions=(json.JSONDecodeError, ValueError, Exception))
    def _generate_script(self, job: VideoJob) -> Script:
        response = self._client.models.generate_content(
            model=config.GEMINI_FLASH_MODEL,
            contents=self._build_prompt(job),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.75,
                max_output_tokens=4096,
            ),
        )
        from utils.cost_tracker import record_gemini
        if response.usage_metadata:
            record_gemini(job, "script",
                response.usage_metadata.prompt_token_count or 0,
                response.usage_metadata.candidates_token_count or 0)
        data = _parse_json(response.text)
        return self._parse_script(data, job)

    def _build_prompt(self, job: VideoJob) -> str:
        facts_text = "\n".join(f"- {f}" for f in job.research.facts)
        overshoot = getattr(config, "SCRIPT_WORDS_OVERSHOOT", 1.4)
        target_words = int(job.plan.target_duration_s * config.WORDS_PER_MINUTE / 60 * overshoot)

        scenes_per_video = config.SCENES_PER_VIDEO
        words_per_scene = max(10, target_words // scenes_per_video)

        return (
            f"Escribe un guión informativo para un video de YouTube Shorts.\n"
            f"AUDIENCIA: Explica como si le hablaras a un chico de 12 años. Usa lenguaje simple y cotidiano, "
            f"frases cortas, analogías con cosas del día a día (pelotas, agua, luz del sol, etc.), "
            f"y evita jerga técnica. Si necesitas usar un término científico, explícalo de inmediato con una comparación simple.\n\n"
            f"Tema: {job.plan.title_concept}\n"
            f"Ángulo: {job.plan.angle}\n"
            f"Datos de investigación:\n{facts_text}\n\n"
            f"Objetivo: {target_words} palabras en total, distribuidas en exactamente {scenes_per_video} escenas "
            f"de MÍNIMO {words_per_scene} palabras cada una.\n"
            f"CRÍTICO: cada spoken_text debe tener al menos {words_per_scene} palabras. "
            f"Ejemplo de {words_per_scene} palabras: \""
            + " ".join(["palabra"] * words_per_scene) + f"\". Cuenta las palabras antes de escribirlas.\n\n"
            "Devuelve un objeto JSON con esta estructura exacta:\n"
            "{\n"
            '  "hook_line": "primera oración del narrador",\n'
            '  "total_duration_estimate_s": integer,\n'
            '  "scenes": [\n'
            "    {\n"
            '      "scene_id": 0,\n'
            '      "type": "body",\n'
            f'      "spoken_text": "texto narrado (~{words_per_scene} palabras)",\n'
            f'      "duration_hint_s": {max(4, job.plan.target_duration_s // scenes_per_video)}\n'
            "    },\n"
            f"    ... {scenes_per_video - 1} escenas más ...\n"
            "  ]\n"
            "}\n\n"
            "Reglas:\n"
            f"- Exactamente {scenes_per_video} escenas, todas de tipo 'body'\n"
            "- Contenido 100% informativo: datos, hechos, explicaciones científicas — pero en lenguaje de 12 años\n"
            "- Sin introducción de gancho, sin despedida, sin CTA, sin mencionar redes sociales\n"
            "- El video empieza directamente con información y termina cuando se acaba el tema\n"
            "- spoken_text: frases cortas y simples, con analogías cotidianas cuando sea posible\n"
            "- Devuelve SOLO el objeto JSON, sin markdown ni comentarios"
        )

    def _parse_script(self, data: dict, job: VideoJob) -> Script:
        raw_scenes = data.get("scenes", [])
        if len(raw_scenes) < config.SCENES_PER_VIDEO:
            raise ValueError(
                f"Expected {config.SCENES_PER_VIDEO} scenes, got {len(raw_scenes)}"
            )
        raw_scenes = raw_scenes[: config.SCENES_PER_VIDEO]  # trim any extras

        scenes: list[Scene] = []
        for raw in raw_scenes:
            try:
                scene_type = SceneType(raw.get("type", "body"))
            except ValueError:
                scene_type = SceneType.BODY

            scene = Scene(
                scene_id=int(raw.get("scene_id", len(scenes))),
                type=scene_type,
                spoken_text=raw.get("spoken_text", "").strip(),
                visual_prompt=_enrich_visual_prompt(raw.get("spoken_text", "")),
                duration_hint_s=int(raw.get("duration_hint_s", 10)),
            )
            scenes.append(scene)

        full_ssml = build_ssml(scenes)
        for scene in scenes:
            scene.ssml = f"<speak>{scene.spoken_text}</speak>"

        script = Script(
            hook_line=data.get("hook_line", scenes[0].spoken_text if scenes else ""),
            cta_line="",
            total_duration_estimate_s=int(data.get("total_duration_estimate_s", 55)),
            scenes=scenes,
        )
        script._full_ssml = full_ssml  # type: ignore[attr-defined]
        return script


def _enrich_visual_prompt(spoken_text: str) -> str:
    if not spoken_text:
        return _VISUAL_STYLE
    return f"Illustrate the following scene: {spoken_text}. {_VISUAL_STYLE}"


def _parse_json(text: str) -> dict:
    """Parse JSON from model response, stripping markdown fences if present."""
    import re
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ``` code fences
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fenced:
        text = fenced.group(1).strip()
    return json.loads(text)
