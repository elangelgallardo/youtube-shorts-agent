"""
AssemblyAgent — ffmpeg-based pipeline producing the final 1080x1920 MP4.

Steps:
  1. Per-scene: Ken Burns via ffmpeg scale+crop+scale with smoothstep expressions (C-level, fast)
  2. Crossfades via ffmpeg xfade filter chain
  3. Audio: narration (with start delay) + background music via ffmpeg amix
  4. Subtitles: SRT → phrase-grouped ASS → ffmpeg ass filter
  5. Black intro/outro + video fades + final encode
  6. Thumbnail generation (Gemini image gen with title text baked in)
"""
import logging
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types

import config
from models.video_job import VideoJob

logger = logging.getLogger(__name__)

# ── Pan presets for Ken Burns ─────────────────────────────────────────────────
# (start_x_frac, start_y_frac, end_x_frac, end_y_frac)
# Fractions of available pan range (OW-W, OH-H); 0=top/left, 1=bottom/right
_MOTION_PRESETS = [
    (0.5, 0.5, 0.0, 0.0),   # center → top-left  (zoom-out feel)
    (0.0, 0.0, 0.5, 0.5),   # top-left → center  (zoom-in feel)
    (1.0, 0.5, 0.0, 0.5),   # right → left
    (0.0, 0.5, 1.0, 0.5),   # left → right
    (0.5, 1.0, 0.5, 0.0),   # bottom → top
    (0.5, 0.0, 0.5, 1.0),   # top → bottom
]

_CROSSFADE_S    = 0.4
_BG_FADE_S      = 1.5
_VIDEO_FADE_S   = 0.5
_START_DELAY_S  = 0.0   # natural pause is baked into TTS via SSML break (TTS_LEADING_PAUSE_S)
_END_DELAY_S    = 3.0
_SUBTITLE_WORDS = 4
_FONT_SIZE      = 77


class AssemblyAgent:
    def __init__(self):
        self._client = genai.Client(api_key=config.GOOGLE_API_KEY)

    def run(self, job: VideoJob) -> VideoJob:
        ws = job.workspace_dir
        logger.info("[%s] Assembling video (ffmpeg pipeline) in %s", job.job_id, ws)

        # ── 1. Ken Burns per scene ────────────────────────────────────────────
        scene_paths: list[Path] = []
        for asset in job.images:
            if asset.veo_clip_path and Path(asset.veo_clip_path).exists():
                scene_paths.append(Path(asset.veo_clip_path))
            else:
                out = ws / f"tmp_scene_{asset.scene_id:03d}.mp4"
                _render_ken_burns(asset, out)
                scene_paths.append(out)

        # ── 2. Concat with crossfades ─────────────────────────────────────────
        content_path = ws / "tmp_content.mp4"
        durations = [_probe_duration(p) for p in scene_paths]
        _concat_xfade(scene_paths, durations, content_path)
        content_dur = _probe_duration(content_path)

        # ── 3. Mix audio ──────────────────────────────────────────────────────
        total_dur = _START_DELAY_S + content_dur + _END_DELAY_S
        audio_path = ws / "tmp_audio.aac"
        _mix_audio(
            Path(job.audio.audio_path),
            Path(config.ASSETS_DIR) / "background_music.mp3",
            audio_path,
            total_dur,
        )

        # ── 4. ASS subtitles ──────────────────────────────────────────────────
        srt_path = Path(job.audio.subtitles_srt_path)
        ass_path: Path | None = None
        if srt_path.exists():
            ass_path = ws / "tmp_subtitles.ass"
            # Whisper transcribes the audio including any SSML silence, so its timestamps
            # already reflect the true timing. Only offset by any external video delay.
            subtitle_offset = _START_DELAY_S
            _srt_to_ass(srt_path, ass_path, subtitle_offset, _SUBTITLE_WORDS)

        # ── 5. Final encode ───────────────────────────────────────────────────
        final_path = ws / "final_video.mp4"
        _final_encode(content_path, audio_path, ass_path, final_path, total_dur)

        job.video.final_path = str(final_path)
        job.video.duration_s = _probe_duration(final_path)

        # Thumbnail generation disabled
        # thumb = self._generate_thumbnail(job)
        # if thumb:
        #     job.video.thumbnail_path = str(thumb)

        # Cleanup temp files
        for p in [content_path, audio_path]:
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        for asset in job.images:
            tmp = ws / f"tmp_scene_{asset.scene_id:03d}.mp4"
            if tmp.exists():
                tmp.unlink()

        logger.info("[%s] Assembly done: %.1fs → %s", job.job_id, job.video.duration_s, final_path)
        return job

    # ── Thumbnail ─────────────────────────────────────────────────────────────

    def _generate_thumbnail(self, job: VideoJob) -> Path | None:
        """Generate a custom thumbnail: Gemini image (no text) + PIL title overlay."""
        import io
        title = job.plan.title_concept
        title_clean = re.sub(r"[^\w\s¿?¡!áéíóúüñÁÉÍÓÚÜÑ,.:'-]", "", title).strip()
        # Use only the first sentence for the thumbnail
        first_sentence = re.split(r"(?<=[?!.])\s+", title_clean)[0].strip()
        if first_sentence:
            title_clean = first_sentence
        topic = job.plan.angle or title_clean

        prompt = (
            f"YouTube Shorts thumbnail background, portrait 9:16, cinematic and dramatic, "
            f"topic: {topic}. Stunning eye-catching illustration, vivid colors, dark cosmic background, "
            f"highly detailed. No text, no letterboxing, no borders, no black bars."
        )

        try:
            response = self._client.models.generate_content(
                model=config.IMAGE_GEN_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio="9:16"),
                ),
            )
            img = None
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    img = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
                    img = img.resize((config.VIDEO_WIDTH, config.VIDEO_HEIGHT), Image.LANCZOS)
                    break

            if img is None:
                logger.warning("[%s] Gemini returned no image for thumbnail", job.job_id)
                return None

            # ── Text overlay ──────────────────────────────────────────────────
            draw = ImageDraw.Draw(img)
            font_size = 120
            font = _load_font(font_size)
            max_w = config.VIDEO_WIDTH - 100

            # Word-wrap (short title usually fits in one line)
            words = title_clean.split()
            lines: list[str] = []
            current: list[str] = []
            for word in words:
                test = " ".join(current + [word])
                bb = draw.textbbox((0, 0), test, font=font)
                if (bb[2] - bb[0]) > max_w and current:
                    lines.append(" ".join(current))
                    current = [word]
                else:
                    current.append(word)
            if current:
                lines.append(" ".join(current))

            line_bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
            line_h = max(bb[3] - bb[1] for bb in line_bboxes) if line_bboxes else font_size
            gap = 16
            total_h = len(lines) * line_h + (len(lines) - 1) * gap

            # Center vertically and horizontally
            y = (config.VIDEO_HEIGHT - total_h) // 2
            stroke = 6
            for line, bb in zip(lines, line_bboxes):
                x = (config.VIDEO_WIDTH - (bb[2] - bb[0])) // 2 - bb[0]
                draw.text((x, y - bb[1]), line, font=font,
                          fill=(255, 255, 255), stroke_width=stroke, stroke_fill=(0, 0, 0))
                y += line_h + gap

            thumb_path = job.workspace_dir / "thumbnail.jpg"
            img.save(str(thumb_path), "JPEG", quality=92)
            logger.info("[%s] Thumbnail saved: %s", job.job_id, thumb_path)
            return thumb_path

        except Exception as e:
            logger.warning("[%s] Thumbnail generation failed: %s", job.job_id, e)
            return None


# ── ffmpeg helpers ─────────────────────────────────────────────────────────────

def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _render_ken_burns(asset, output_path: Path) -> None:
    """Ken Burns pan effect: PIL pre-scales image once; ffmpeg animates crop x,y per-frame.

    ffmpeg crop w/h are evaluated at init only (not per-frame), so true zoom isn't possible
    via filter expressions. Instead we pre-scale to overscan and pan within it — visually
    equivalent at the speeds used in Shorts.
    """
    from PIL import Image as _Image
    duration = max(3.0, min(float(asset.duration_s), 20.0))
    sx_f, sy_f, ex_f, ey_f = _MOTION_PRESETS[asset.scene_id % len(_MOTION_PRESETS)]
    W, H = config.VIDEO_WIDTH, config.VIDEO_HEIGHT
    OW = int(W * 1.04)
    OH = int(H * 1.04)
    fps = config.VIDEO_FPS

    # Pre-scale image to overscan size (done once in Python, not per-frame)
    tmp_img = output_path.with_suffix(".tmp.png")
    img = _Image.open(asset.path).convert("RGB")
    img = img.resize((OW, OH), _Image.LANCZOS)
    img.save(str(tmp_img))

    # Available pan range in pixels
    px = OW - W
    py = OH - H
    sx, sy = sx_f * px, sy_f * py
    ex, ey = ex_f * px, ey_f * py

    # Smoothstep easing for x,y: these ARE evaluated per-frame by ffmpeg crop filter
    sp = f"(3*pow(t/{duration},2)-2*pow(t/{duration},3))"
    cx = f"({sx:.2f}+({ex - sx:.2f})*{sp})"
    cy = f"({sy:.2f}+({ey - sy:.2f})*{sp})"

    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(tmp_img),
            "-vf", f"crop=w={W}:h={H}:x='{cx}':y='{cy}'",
            "-t", str(duration),
            "-r", str(fps),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-loglevel", "error",
            str(output_path),
        ], check=True)
    finally:
        tmp_img.unlink(missing_ok=True)
    logger.debug("Ken Burns scene %d → %.1fs", asset.scene_id, duration)


def _concat_xfade(scene_paths: list[Path], durations: list[float], output_path: Path) -> None:
    """Concatenate scene clips with fade crossfades using ffmpeg xfade filter chain."""
    if len(scene_paths) == 1:
        import shutil
        shutil.copy(scene_paths[0], output_path)
        return

    C = _CROSSFADE_S
    inputs: list[str] = []
    for p in scene_paths:
        inputs += ["-i", str(p)]

    # offset_i = sum(durations[0..i-1]) - i * C
    # (time from start of current composite where next transition begins)
    filters: list[str] = []
    prev = "0:v"
    cumulative = 0.0
    for i in range(1, len(scene_paths)):
        cumulative += durations[i - 1]
        offset = max(0.01, cumulative - i * C)
        label = f"xf{i:02d}"
        filters.append(
            f"[{prev}][{i}:v]xfade=transition=fade:duration={C}:offset={offset:.4f}[{label}]"
        )
        prev = label

    subprocess.run([
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", ";".join(filters),
        "-map", f"[{prev}]",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-loglevel", "error",
        str(output_path),
    ], check=True)


def _mix_audio(narr_path: Path, bg_path: Path, output_path: Path, total_dur: float) -> None:
    """Narration (with start delay) + looped background music, mixed with ffmpeg."""
    delay_ms = int(_START_DELAY_S * 1000)
    fade_out_t = max(0.0, total_dur - _BG_FADE_S)

    leading = getattr(config, "TTS_LEADING_PAUSE_S", 0.0)
    narr_fadein = f",afade=t=in:st={leading:.3f}:d=0.3" if leading > 0 else ""

    if bg_path.exists():
        fc = (
            f"[0:a]adelay={delay_ms}|{delay_ms}{narr_fadein}[narr];"
            f"[1:a]aloop=loop=-1:size=2000000000,"
            f"atrim=end={total_dur:.3f},asetpts=PTS-STARTPTS,"
            f"volume={config.BG_MUSIC_VOLUME},"
            f"afade=t=in:st=0:d={_BG_FADE_S},"
            f"afade=t=out:st={fade_out_t:.3f}:d={_BG_FADE_S}[bg];"
            f"[narr][bg]amix=inputs=2:duration=longest:normalize=0:dropout_transition=0[out]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(narr_path), "-i", str(bg_path),
            "-filter_complex", fc,
            "-map", "[out]", "-t", str(total_dur),
            "-c:a", "aac", "-b:a", "192k", "-loglevel", "error",
            str(output_path),
        ]
    else:
        fc = f"[0:a]adelay={delay_ms}|{delay_ms}{narr_fadein}[out]"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(narr_path),
            "-filter_complex", fc,
            "-map", "[out]", "-t", str(total_dur),
            "-c:a", "aac", "-b:a", "192k", "-loglevel", "error",
            str(output_path),
        ]
    subprocess.run(cmd, check=True)


def _srt_to_ass(srt_path: Path, ass_path: Path, start_delay: float, words_per_phrase: int) -> None:
    """Convert word-level SRT → phrase-grouped ASS with Bebas Neue styling."""
    W, H = config.VIDEO_WIDTH, config.VIDEO_HEIGHT

    # Parse SRT into word cues
    text = srt_path.read_text(encoding="utf-8")
    blocks = re.split(r"\n\n+", text.strip())
    word_cues: list[tuple[float, float, str]] = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            s, e = (_srt_time(t) for t in lines[1].split(" --> "))
            word_cues.append((s, e, " ".join(lines[2:]).strip()))
        except Exception:
            continue

    if not word_cues:
        return

    # Group into phrases
    phrases: list[tuple[float, float, str]] = []
    for i in range(0, len(word_cues), words_per_phrase):
        chunk = word_cues[i:i + words_per_phrase]
        phrases.append((chunk[0][0], chunk[-1][1], " ".join(c[2] for c in chunk)))

    def to_ass_time(t: float) -> str:
        t = max(0.0, t + start_delay)
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        cs = int(round((s % 1) * 100))
        return f"{int(h)}:{int(m):02d}:{int(s):02d}.{cs:02d}"

    # BorderStyle=1 → outline+shadow (4px black stroke); Alignment=2 → center-bottom
    margin_v = 340
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {W}\n"
        f"PlayResY: {H}\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,Bebas Neue,{_FONT_SIZE},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        f"-1,0,0,0,100,100,2,0,1,4,0,2,40,40,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = [
        f"Dialogue: 0,{to_ass_time(s)},{to_ass_time(e)},Default,,0,0,0,,{t}"
        for s, e, t in phrases
    ]
    ass_path.write_text(header + "\n".join(events), encoding="utf-8")


def _final_encode(
    content_path: Path,
    audio_path: Path,
    ass_path: Path | None,
    output_path: Path,
    total_dur: float,
) -> None:
    """Add black padding, fades, subtitles; combine with audio → final MP4."""
    fade_out_t = max(0.0, total_dur - _VIDEO_FADE_S)

    vf = (
        f"tpad=start_duration={_START_DELAY_S}:start_mode=add:stop_duration={_END_DELAY_S}:stop_mode=clone:color=black,"
        f"fade=t=in:st=0:d={_VIDEO_FADE_S},"
        f"fade=t=out:st={fade_out_t:.3f}:d={_VIDEO_FADE_S}"
    )
    if ass_path and ass_path.exists():
        escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")
        vf += f",ass='{escaped}'"

    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(content_path),
        "-i", str(audio_path),
        "-vf", vf,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-t", str(total_dur),
        "-loglevel", "error",
        str(output_path),
    ], check=True)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _load_font(size: int) -> ImageFont.FreeTypeFont:
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


def _srt_time(s: str) -> float:
    s = s.strip().replace(",", ".")
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)
