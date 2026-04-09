"""
ScriptAgent — generates a Short script from research context.

Produces a single continuous narration + N independent image prompts.
The narration is split into N equal chunks post-generation so downstream
pipeline (image gen, assembly) works unchanged.
"""
import json
import logging
import re

from google import genai
from google.genai import types

import config
from models.enums import SceneType
from models.video_job import Scene, Script, VideoJob
from utils.retry import with_retry
from utils.ssml_builder import build_ssml

logger = logging.getLogger(__name__)

_VISUAL_STYLE = (
    "ilustración digital, colores vivos, "
    "muy detallado, el sujeto principal llena el encuadre verticalmente, composición retrato 9:16, "
    "sin letterboxing sin barras negras sin bordes"
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
                temperature=0.9,
                max_output_tokens=8192,
                thinking_config=types.ThinkingConfig(thinking_level="medium"),
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
        overshoot = getattr(config, "SCRIPT_WORDS_OVERSHOOT", 1.0)
        target_words = int(job.plan.target_duration_s * config.WORDS_PER_MINUTE / 60 * overshoot)
        n = config.SCENES_PER_VIDEO

        return (
            f"Escribe un guión informativo para un video de YouTube Shorts.\n"
            f"Usa lenguaje claro y accesible, frases directas, y explica los términos técnicos cuando aparezcan.\n\n"
            f"Tema: {job.plan.title_concept}\n"
            f"Ángulo: {job.plan.angle}\n"
            f"Datos de investigación:\n{facts_text}\n\n"
            f"Objetivo: una narración continua y fluida de aproximadamente {target_words} palabras.\n\n"
            "Devuelve un objeto JSON con esta estructura exacta:\n"
            "{\n"
            '  "hook_line": "primera oración del narrador",\n'
            '  "total_duration_estimate_s": integer,\n'
            f'  "narration": "narración completa y continua aquí (~{target_words} palabras)",\n'
            f'  "image_prompts": ["descripción visual 1", "descripción visual 2", ... {n} prompts en total]\n'
            "}\n\n"
            "Reglas para la narración:\n"
            "- Texto continuo y fluido, sin divisiones artificiales en escenas\n"
            "- Empieza presentando el tema desde cero, asumiendo que el espectador NO ha leído el título. "
            "NUNCA empieces con 'Bueno,', 'La respuesta es', 'Sí,' o cualquier cosa que asuma una pregunta previa. "
            "Si el tema es una pregunta, FORMULA esa pregunta explícitamente al inicio antes de responderla.\n"
            "- Sin despedida, sin CTA, sin mencionar redes sociales\n"
            "- Sin emojis en la narración ni en los image_prompts\n"
            "- Varía el ritmo: alterna frases cortas e impactantes con oraciones más elaboradas. "
            "No todas las frases deben tener la misma longitud.\n"
            "- Usa preguntas retóricas para enganchar: '¿Pero qué significa eso realmente?' o '¿Cómo es eso posible?'\n"
            "- Cifras concretas y comparaciones inesperadas en lugar de adjetivos vagos: "
            "no 'muy grande' sino '1.4 millones de Tierras cabrían dentro'. No 'muy rápido' sino 'siete veces alrededor de la Tierra en un segundo'.\n"
            "- Construye tensión: plantea algo sorprendente o contraintuitivo antes de explicarlo\n"
            "- Analogías cotidianas para hacer tangible lo abstracto\n"
            f"- Añade <break time=\"0.4s\"/> entre ideas u oraciones distintas para pausas naturales. "
            "Ejemplo: 'La luz viaja muy rápido. <break time=\"0.4s\"/> Tan rápido que da la vuelta a la Tierra siete veces en un segundo.'\n\n"
            f"Reglas para image_prompts:\n"
            f"- Exactamente {n} descripciones visuales, una por imagen del video\n"
            "- Cada prompt ilustra EXACTAMENTE lo que el narrador está diciendo en ese fragmento de la narración, no lo que viene después\n"
            "- El prompt N debe corresponder al fragmento N de la narración (si el narrador habla de X en ese fragmento, el prompt muestra X)\n"
            "- Los prompts deben cubrir el arco completo del video de inicio a fin, en el mismo orden que la narración\n"
            "- Solo descripción visual, sin texto ni instrucciones de estilo (eso se añade automáticamente)\n"
            "- Devuelve SOLO el objeto JSON, sin markdown ni comentarios"
        )

    def _parse_script(self, data: dict, job: VideoJob) -> Script:
        narration = _strip_emojis(_fix_break_tags(data.get("narration", "").strip()))
        image_prompts = [_strip_emojis(p) for p in data.get("image_prompts", [])]
        n = config.SCENES_PER_VIDEO

        if not narration:
            raise ValueError("LLM returned empty narration")
        if len(image_prompts) < n:
            raise ValueError(f"Expected {n} image_prompts, got {len(image_prompts)}")
        image_prompts = image_prompts[:n]

        chunks = _split_narration(narration, n)

        scenes: list[Scene] = []
        for i, (chunk, prompt) in enumerate(zip(chunks, image_prompts)):
            scene = Scene(
                scene_id=i,
                type=SceneType.BODY,
                spoken_text=chunk,
                visual_prompt=_enrich_visual_prompt(prompt),
                duration_hint_s=max(4, job.plan.target_duration_s // n),
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


def _split_narration(narration: str, n: int) -> list[str]:
    """Split narration into n equal chunks, keeping XML tags (e.g. <break/>) as single tokens."""
    # Match complete XML tags first, then any non-whitespace word
    tokens = re.findall(r"<[^>]+>|\S+", narration)
    total = len(tokens)
    chunk_size = max(1, total // n)
    chunks = []
    for i in range(n):
        start = i * chunk_size
        end = start + chunk_size if i < n - 1 else total
        chunks.append(" ".join(tokens[start:end]))
    return chunks


def _fix_break_tags(text: str) -> str:
    """Ensure <break time="..."> is always self-closing <break time="..."/>."""
    return re.sub(r'<break([^>]*[^/])>', r'<break\1/>', text)


def _strip_emojis(text: str) -> str:
    return re.sub(r"[\U00010000-\U0010ffff\U00002600-\U000027BF\U0001F300-\U0001FAFF]", "", text, flags=re.UNICODE).strip()


def _enrich_visual_prompt(prompt_text: str) -> str:
    clean = re.sub(r"<[^>]+>", "", prompt_text).strip()
    if not clean:
        return _VISUAL_STYLE
    return f"Ilustra la siguiente escena: {clean}. {_VISUAL_STYLE}"


def _parse_json(text: str) -> dict:
    """Parse JSON from model response, stripping markdown fences if present."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fenced:
        text = fenced.group(1).strip()
    return json.loads(text)
