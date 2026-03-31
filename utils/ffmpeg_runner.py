"""Duration probe helper — MoviePy handles all video processing now."""
from pathlib import Path


def probe_duration(path: Path) -> float:
    """Return duration in seconds of an audio or video file via MoviePy."""
    from moviepy import AudioFileClip, VideoFileClip

    path = Path(path)
    try:
        with AudioFileClip(str(path)) as clip:
            return clip.duration
    except Exception:
        pass
    try:
        with VideoFileClip(str(path)) as clip:
            return clip.duration
    except Exception:
        return 0.0
