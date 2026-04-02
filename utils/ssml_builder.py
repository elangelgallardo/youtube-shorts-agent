import config


def build_ssml(scenes: list, leading_pause_s: float | None = None) -> str:
    """Build SSML for TTS, with an optional leading pause before narration begins.

    The leading pause creates a natural silence baked into the audio file itself,
    so images + music play for that duration before the voice starts.
    Falls back to config.TTS_LEADING_PAUSE_S if leading_pause_s is not specified.
    """
    pause_s = leading_pause_s if leading_pause_s is not None else getattr(config, "TTS_LEADING_PAUSE_S", 0.0)
    body = " ".join(scene.spoken_text.strip() for scene in scenes)

    if pause_s > 0:
        return f'<speak><break time="{pause_s:.1f}s"/>{body}</speak>'
    return f"<speak>{body}</speak>"
