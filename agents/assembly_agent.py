"""
AssemblyAgent — MoviePy pipeline producing the final 1080x1920 MP4.

Steps:
  1. Build a Ken Burns zoom VideoClip per scene (alternating zoom-in / zoom-out)
  2. Concatenate all scene clips
  3. Attach audio (trimmed to video length to avoid off-by-one ms crashes)
  4. Overlay subtitles rendered with Pillow
  5. Export final MP4
"""
import logging
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    VideoClip,
    concatenate_videoclips,
)
from moviepy.audio.fx import MultiplyVolume

import config
from models.video_job import VideoJob
from utils.retry import with_retry

logger = logging.getLogger(__name__)


class AssemblyAgent:
    def run(self, job: VideoJob) -> VideoJob:
        ws = job.workspace_dir
        logger.info("[%s] Assembling video with MoviePy in %s", job.job_id, ws)

        scene_clips = [
            self._make_veo_clip(asset) if asset.veo_clip_path else self._make_ken_burns_clip(asset)
            for asset in job.images
        ]
        video = concatenate_videoclips(scene_clips, method="compose")

        audio = AudioFileClip(job.audio.audio_path)
        # Trim audio to video duration to avoid sub-millisecond overflow crashes
        safe_duration = min(audio.duration, video.duration)
        narration = audio.subclipped(0, safe_duration)

        # Mix in background music at low volume if available
        bg_path = Path(config.ASSETS_DIR) / "background_music.mp3"
        if bg_path.exists():
            bg = AudioFileClip(str(bg_path))
            # Loop background music if shorter than video
            if bg.duration < safe_duration:
                from moviepy.audio.fx import AudioLoop
                bg = bg.with_effects([AudioLoop(duration=safe_duration)])
            bg = bg.subclipped(0, safe_duration).with_effects([MultiplyVolume(config.BG_MUSIC_VOLUME)])
            mixed = CompositeAudioClip([narration, bg])
        else:
            mixed = narration

        video = video.subclipped(0, safe_duration).with_audio(mixed)

        video = self._add_subtitles(video, Path(job.audio.subtitles_srt_path), job.job_id)

        final_path = ws / "final_video.mp4"
        video.write_videofile(
            str(final_path),
            fps=config.VIDEO_FPS,
            codec="libx264",
            audio_codec="aac",
            preset="fast",
            threads=4,
            logger=None,
            ffmpeg_params=["-crf", "22", "-pix_fmt", "yuv420p"],
            temp_audiofile=str(ws / "temp_audio.m4a"),
        )

        job.video.final_path = str(final_path)
        job.video.duration_s = video.duration

        audio.close()
        video.close()
        for clip in scene_clips:
            clip.close()

        logger.info("[%s] Assembly done: %.1fs → %s", job.job_id, job.video.duration_s, final_path)
        return job

    # ── Veo clip ──────────────────────────────────────────────────────────

    def _make_veo_clip(self, asset) -> VideoClip:
        from moviepy import VideoFileClip
        clip = VideoFileClip(asset.veo_clip_path)
        # Ensure correct dimensions and duration
        if clip.size != (config.VIDEO_WIDTH, config.VIDEO_HEIGHT):
            clip = clip.resized((config.VIDEO_WIDTH, config.VIDEO_HEIGHT))
        duration = max(3.0, min(float(asset.duration_s), clip.duration))
        return clip.subclipped(0, duration)

    # ── Ken Burns clip ────────────────────────────────────────────────────

    @with_retry(max_attempts=2, exceptions=(Exception,))
    def _make_ken_burns_clip(self, asset) -> VideoClip:
        duration = max(3.0, min(float(asset.duration_s), 20.0))
        zoom_in = asset.scene_id % 2 == 0

        pil_img = Image.open(asset.path).convert("RGB")
        overscan_w = int(config.VIDEO_WIDTH * 1.12)
        overscan_h = int(config.VIDEO_HEIGHT * 1.12)
        pil_img = pil_img.resize((overscan_w, overscan_h), Image.LANCZOS)
        img_array = np.asarray(pil_img)

        src_h, src_w = img_array.shape[:2]
        tgt_w, tgt_h = config.VIDEO_WIDTH, config.VIDEO_HEIGHT

        def make_frame(t: float) -> np.ndarray:
            progress = t / duration
            zoom = (1.0 + 0.10 * progress) if zoom_in else (1.10 - 0.10 * progress)
            crop_w = int(tgt_w / zoom)
            crop_h = int(tgt_h / zoom)
            x1 = max(0, (src_w - crop_w) // 2)
            y1 = max(0, (src_h - crop_h) // 2)
            x2 = min(src_w, x1 + crop_w)
            y2 = min(src_h, y1 + crop_h)
            cropped = img_array[y1:y2, x1:x2]
            return np.asarray(Image.fromarray(cropped).resize((tgt_w, tgt_h), Image.LANCZOS))

        return VideoClip(make_frame, duration=duration).with_fps(config.VIDEO_FPS)

    # ── Subtitle overlay (Pillow-based, no font-path dependencies) ────────

    def _add_subtitles(self, video: VideoClip, srt_path: Path, job_id: str) -> VideoClip:
        if not srt_path.exists():
            logger.warning("[%s] SRT not found — skipping subtitles", job_id)
            return video

        try:
            subtitles = _parse_srt(srt_path)
            if not subtitles:
                return video

            sub_clips = []
            for start, end, text in subtitles:
                if end > video.duration:
                    end = video.duration
                if start >= end:
                    continue
                words = text.split()
                if not words:
                    continue
                word_dur = (end - start) / len(words)
                for i, word in enumerate(words):
                    w_start = start + i * word_dur
                    w_end = w_start + word_dur
                    frame, pos = _render_subtitle_word(word, config.VIDEO_WIDTH, config.VIDEO_HEIGHT)
                    sub_clips.append(
                        ImageClip(frame, duration=w_end - w_start)
                        .with_start(w_start)
                        .with_position(pos)
                    )

            return CompositeVideoClip([video] + sub_clips, size=(config.VIDEO_WIDTH, config.VIDEO_HEIGHT))

        except Exception as exc:
            logger.warning("[%s] Subtitle overlay failed (%s) — skipping", job_id, exc)
            return video


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_srt(path: Path) -> list[tuple[float, float, str]]:
    """Parse an SRT file into (start_s, end_s, text) tuples."""
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n\n+", text.strip())
    results = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            start, end = (_srt_time(t) for t in lines[1].split(" --> "))
            caption = " ".join(lines[2:]).strip()
            results.append((start, end, caption))
        except Exception:
            continue
    return results


def _srt_time(s: str) -> float:
    s = s.strip().replace(",", ".")
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def _render_subtitle_word(word: str, video_w: int, video_h: int) -> tuple[np.ndarray, tuple]:
    """Render a single word as a small RGBA image; return (array, (x, y) position)."""
    scratch = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    max_text_w = video_w - 80  # 40px margin each side

    # Shrink font until the word fits within the video width
    font_size = 110
    while font_size >= 40:
        font = _load_font(font_size)
        bbox = scratch.textbbox((0, 0), word, font=font)
        if (bbox[2] - bbox[0]) <= max_text_w:
            break
        font_size -= 6

    bbox = scratch.textbbox((0, 0), word, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pad_x, pad_y = 6, 6
    outline = 3
    img_w = text_w + 2 * (pad_x + outline)
    img_h = text_h + 2 * (pad_y + outline)

    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Offset by bbox origin so the full glyph (ascenders + descenders) fits inside img
    tx = pad_x + outline - bbox[0]
    ty = pad_y + outline - bbox[1]
    for dx, dy in [(-outline, 0), (outline, 0), (0, -outline), (0, outline),
                   (-2, -2), (2, -2), (-2, 2), (2, 2)]:
        draw.text((tx + dx, ty + dy), word, font=font, fill=(0, 0, 0, 220))
    draw.text((tx, ty), word, font=font, fill=(255, 255, 255, 255))

    # Position: horizontally centered, 500px above bottom (clear of Shorts UI)
    x_pos = (video_w - img_w) // 2
    y_pos = video_h - img_h - 500

    return np.array(img), (x_pos, y_pos)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Try Bebas Neue first, then fall back to bold system fonts."""
    candidates = [
        "/usr/share/fonts/opentype/bebas-neue/BebasNeue-Regular.otf",
        "/usr/share/fonts/opentype/bebas-neue/BebasNeue-Bold.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()
