# Original _build_prompt from agents/script_agent.py
# Saved before few-shot transcript injection was added.

def _build_prompt_original(self, job):
    facts_text = "\n".join(f"- {f}" for f in job.research.facts)
    overshoot = getattr(config, "SCRIPT_WORDS_OVERSHOOT", 1.4)
    target_words = int(job.plan.target_duration_s * config.WORDS_PER_MINUTE / 60 * overshoot)

    scenes_per_video = config.SCENES_PER_VIDEO
    words_per_scene = max(10, target_words // scenes_per_video)

    return (
        f"Escribe un guión para un video de YouTube Shorts sobre este tema.\n"
        f"IMPORTANTE: Todo el texto narrado debe estar en español neutro y accesible.\n\n"
        f"Concepto del título: {job.plan.title_concept}\n"
        f"Tema: {job.plan.topic}\n"
        f"Ángulo: {job.plan.angle}\n"
        f"Formato: {job.plan.format.value}\n"
        f"Estadística clave (gancho): {job.research.key_stat}\n"
        f"Datos de apoyo:\n{facts_text}\n\n"
        f"Objetivo: {target_words} palabras narradas en total. "
        f"Exactamente {scenes_per_video} escenas de MÍNIMO {words_per_scene} palabras cada una.\n"
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
        '      "type": "hook",\n'
        f'      "spoken_text": "palabras exactas del narrador (~{words_per_scene} palabras)",\n'
        '      "visual_prompt": "English prompt for the background image",\n'
        f'      "duration_hint_s": {max(4, job.plan.target_duration_s // scenes_per_video)}\n'
        "    },\n"
        f"    ... {scenes_per_video - 1} escenas más (type: body) ...\n"
        "  ]\n"
        "}\n\n"
        "Reglas:\n"
        f"- Exactamente {scenes_per_video} escenas: 1 hook + {scenes_per_video - 1} body\n"
        "- SIN llamada a la acción, SIN pedir que se suscriban, SIN mencionar redes sociales\n"
        "- El video termina cuando se acaba de explicar el tema, sin despedida ni CTA\n"
        "- spoken_text: tono amigable y curioso, lenguaje claro, sin jerga innecesaria, "
        "basado en los datos de investigación\n"
        "- visual_prompt: en inglés, una ilustración que represente visualmente el concepto "
        "narrado en esa escena. Puede ser cualquier cosa relevante: fenómenos cósmicos, partículas, "
        "personas, planetas, reacciones físicas, escenas abstractas, etc. "
        "Sé específico y variado — cada escena debe tener una imagen distinta. "
        "Ejemplo: 'a scientist observing data from a telescope at night', "
        "'glowing subatomic particles colliding in a particle accelerator', "
        "'a lone astronaut floating above Earth', 'two galaxies merging in deep space'.\n"
        "- Devuelve SOLO el objeto JSON, sin markdown ni comentarios"
    )
# NOTE: As of 2026-03-30, script was changed to pure informational content (no hook, no CTA).
