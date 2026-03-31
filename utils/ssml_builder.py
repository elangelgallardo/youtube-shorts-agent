def build_ssml(scenes: list) -> str:
    """Concatenate all scene text into a single string for TTS."""
    return " ".join(scene.spoken_text.strip() for scene in scenes)
