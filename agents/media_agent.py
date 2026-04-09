"""
MediaAgent — parallel execution of:
  1. Google Cloud TTS: SSML → MP3 + word-level timepoints
  2. gemini-2.5-flash-image (google-genai SDK): visual_prompt → 1080x1920 PNG per scene
  3. SRT generation from timepoints
"""
import concurrent.futures
import logging
from pathlib import Path

from google import genai
from google.genai import types
from google.cloud import texttospeech

import config
from models.video_job import AudioAsset, ImageAsset, Timepoint, VideoJob
from utils.retry import with_retry

logger = logging.getLogger(__name__)


class MediaAgent:
    def __init__(self):
        self._client = genai.Client(api_key=config.GOOGLE_API_KEY)

    def run(self, job: VideoJob, generate_images: bool = True) -> VideoJob:
        job.workspace_dir.mkdir(parents=True, exist_ok=True)

        if generate_images:
            logger.info("[%s] Generating audio + images in parallel…", job.job_id)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                audio_future = pool.submit(self._generate_audio, job)
                images_future = pool.submit(self._generate_images, job)
                audio_asset = audio_future.result()
                image_assets = images_future.result()
        else:
            logger.info("[%s] Generating audio only (images skipped)…", job.job_id)
            audio_asset = self._generate_audio(job)
            # Create placeholder assets — VeoAgent will fill in veo_clip_path
            image_assets = [
                ImageAsset(scene_id=s.scene_id, path="", duration_s=float(s.duration_hint_s))
                for s in job.script.scenes
            ]

        # Generate SRT using Whisper word-level timestamps
        srt_path = job.workspace_dir / "subtitles.srt"
        self._transcribe_srt(Path(audio_asset.audio_path), srt_path, job.script.scenes)
        audio_asset.subtitles_srt_path = str(srt_path)

        # Assign scene durations from timepoints
        _assign_scene_durations(job, audio_asset.timepoints, image_assets)

        # Generate Veo 3 clip for scene 0 (hook) — best effort, skip on failure
        if job.use_veo:
            self._generate_veo_hook(job, image_assets)

        job.audio = audio_asset
        job.images = image_assets

        logger.info(
            "[%s] Media done: audio %.1fs, %d images, SRT written",
            job.job_id,
            audio_asset.total_duration_s,
            len(image_assets),
        )
        return job

    # ── Audio (Google Cloud TTS) ──────────────────────────────────────────

    @with_retry(max_attempts=3, exceptions=(Exception,))
    def _generate_audio(self, job: VideoJob) -> AudioAsset:
        from google.api_core.client_options import ClientOptions
        client = texttospeech.TextToSpeechClient(
            client_options=ClientOptions(api_key=config.GOOGLE_TTS_API_KEY)
        )

        from utils.ssml_builder import build_ssml
        ssml = build_ssml(job.script.scenes)
        synthesis_input = texttospeech.SynthesisInput(ssml=ssml)

        voice = texttospeech.VoiceSelectionParams(
            language_code=config.TTS_LANGUAGE_CODE,
            name=config.TTS_VOICE_NAME,
        )

        is_chirp = "Chirp3" in config.TTS_VOICE_NAME
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            **({} if is_chirp else {"speaking_rate": config.TTS_SPEAKING_RATE}),
            effects_profile_id=["headphone-class-device"],
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )

        audio_path = job.workspace_dir / "audio.mp3"
        audio_path.write_bytes(response.audio_content)

        from utils.cost_tracker import record_tts
        record_tts(job, len(ssml))

        from utils.ffmpeg_runner import probe_duration
        duration = probe_duration(audio_path)

        # Scene-boundary timepoints (still word-count proportional — used for image durations only)
        timepoints = _estimate_timepoints(job.script.scenes, duration)

        logger.info("[%s] TTS: %.1fs audio, %d scenes timed", job.job_id, duration, len(job.script.scenes))
        return AudioAsset(
            audio_path=str(audio_path),
            total_duration_s=duration,
            timepoints=timepoints,
        )

    def _transcribe_srt(self, audio_path: Path, srt_path: Path, scenes) -> None:
        """
        Use Whisper for timestamps, original script for displayed text.
        Aligns script words to Whisper words via sequence matching so spelling
        is always correct while timing comes from the audio.
        """
        from faster_whisper import WhisperModel
        import difflib, re

        logger.debug("Transcribing audio with Whisper for subtitle sync…")
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(
            str(audio_path),
            language="es",
            word_timestamps=True,
        )

        whisper_words = [(w.word.strip(), w.start, w.end) for seg in segments for w in seg.words if w.word.strip()]

        if not whisper_words:
            logger.warning("Whisper returned no words — SRT will be empty")
            srt_path.write_text("")
            return

        script_words = [
            w for scene in scenes
            for w in re.sub(r"<[^>]+>", "", scene.spoken_text).split()
            if not (w.startswith("<") or w.endswith("/>") or w.endswith('">'))
        ]

        def normalize(w):
            return re.sub(r"[^a-záéíóúüñ]", "", w.lower())

        w_norm = [normalize(w) for w, _, _ in whisper_words]
        s_norm = [normalize(w) for w in script_words]

        # Use SequenceMatcher to align script words to Whisper words
        matcher = difflib.SequenceMatcher(None, s_norm, w_norm, autojunk=False)
        # Build map: script_index → whisper_index
        s_to_w = {}
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == "equal":
                for k in range(i2 - i1):
                    s_to_w[i1 + k] = j1 + k
            elif op in ("replace", "insert", "delete"):
                # For non-equal blocks, map by position ratio within the block
                for k in range(i2 - i1):
                    ratio = k / max(i2 - i1 - 1, 1)
                    j = j1 + round(ratio * max(j2 - j1 - 1, 0))
                    s_to_w[i1 + k] = min(j, len(whisper_words) - 1)

        with srt_path.open("w", encoding="utf-8") as f:
            for i, word in enumerate(script_words):
                j = s_to_w.get(i, min(i, len(whisper_words) - 1))
                start = whisper_words[j][1]
                # end = next script word's start, or this word's own end
                if i + 1 < len(script_words):
                    j_next = s_to_w.get(i + 1, min(i + 1, len(whisper_words) - 1))
                    end = whisper_words[j_next][1]
                else:
                    end = whisper_words[j][2]
                f.write(f"{i + 1}\n{_fmt_srt(start)} --> {_fmt_srt(end)}\n{word}\n\n")

    # ── Images (gemini-2.5-flash-image via google-genai SDK) ─────────────

    def _generate_images(self, job: VideoJob) -> list[ImageAsset]:
        assets: list[ImageAsset] = []
        client = genai.Client(api_key=config.GOOGLE_API_KEY)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = {
                pool.submit(self._generate_one_image, client, scene, job): scene.scene_id
                for scene in job.script.scenes
            }
            for future in concurrent.futures.as_completed(futures):
                scene_id = futures[future]
                try:
                    asset = future.result()
                    assets.append(asset)
                except Exception as exc:
                    logger.error(
                        "[%s] Scene %d image failed: %s — using fallback",
                        job.job_id,
                        scene_id,
                        exc,
                    )
                    asset = self._create_fallback_image(job, scene_id)
                    assets.append(asset)

        assets.sort(key=lambda a: a.scene_id)
        return assets

    @with_retry(max_attempts=3, exceptions=(Exception,))
    def _generate_one_image(self, client: genai.Client, scene, job: VideoJob) -> ImageAsset:
        logger.debug("[%s] Generating image for scene %d", job.job_id, scene.scene_id)

        response = client.models.generate_content(
            model=config.IMAGE_GEN_MODEL,
            contents=scene.visual_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio="9:16",
                ),
            ),
        )

        out_path = job.workspace_dir / f"scene_{scene.scene_id:03d}.png"
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                img = _resize_to_fill(part.inline_data.data, config.VIDEO_WIDTH, config.VIDEO_HEIGHT)
                img.save(str(out_path), "PNG")
                break
        else:
            raise RuntimeError(f"No image returned for scene {scene.scene_id}")

        from utils.cost_tracker import record_imagen
        record_imagen(job)
        return ImageAsset(
            scene_id=scene.scene_id,
            path=str(out_path),
            duration_s=float(scene.duration_hint_s),
        )

    def _generate_veo_hook(self, job: VideoJob, image_assets: list) -> None:
        """Generate a Veo 3 video clip for scene 0 (the hook). Best-effort — skips on failure."""
        import time
        import subprocess

        hook_asset = next((a for a in image_assets if a.scene_id == 0), None)
        if not hook_asset:
            return

        hook_scene = next((s for s in job.script.scenes if s.scene_id == 0), None)
        if not hook_scene:
            return

        prompt = (
            hook_scene.visual_prompt.rstrip('.') +
            '. Cinematic, smooth camera motion, photorealistic, vertical 9:16 video, no text, no subtitles.'
        )

        clip_path = job.workspace_dir / "scene_000_veo.mp4"
        if clip_path.exists():
            hook_asset.veo_clip_path = str(clip_path)
            logger.info("[%s] Veo hook clip already exists", job.job_id)
            return

        try:
            logger.info("[%s] Generating Veo 3 hook clip…", job.job_id)
            op = self._client.models.generate_videos(
                model="veo-3.0-generate-001",
                prompt=prompt,
                config=types.GenerateVideosConfig(
                    aspect_ratio="9:16",
                    duration_seconds=8,
                    number_of_videos=1,
                ),
            )
            for _ in range(24):  # wait up to 4 minutes
                if op.done:
                    break
                time.sleep(10)
                op = self._client.operations.get(op)

            if not op.done or not op.response.generated_videos:
                raise RuntimeError("Veo operation did not complete")

            video_obj = op.response.generated_videos[0].video
            raw_path = job.workspace_dir / "scene_000_veo_raw.mp4"
            if video_obj.video_bytes:
                raw_path.write_bytes(video_obj.video_bytes)
            elif video_obj.uri:
                # Download with API key authentication
                import requests
                uri = video_obj.uri
                sep = "&" if "?" in uri else "?"
                resp = requests.get(f"{uri}{sep}key={config.GOOGLE_API_KEY}", timeout=120)
                if resp.status_code == 403:
                    # Try alt=media param
                    resp = requests.get(f"{uri}{sep}key={config.GOOGLE_API_KEY}&alt=media", timeout=120)
                resp.raise_for_status()
                raw_path.write_bytes(resp.content)
            else:
                raise RuntimeError("No video data returned")

            # Resize/crop to 1080x1920 and trim to scene duration
            duration_s = hook_asset.duration_s
            subprocess.run([
                "ffmpeg", "-y",
                "-stream_loop", "-1", "-i", str(raw_path),
                "-t", str(duration_s),
                "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
                "-r", "30", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-an",
                str(clip_path),
            ], check=True, capture_output=True)
            raw_path.unlink(missing_ok=True)

            hook_asset.veo_clip_path = str(clip_path)
            from utils.cost_tracker import record_veo
            record_veo(job, 8.0)  # Veo always generates 8s
            logger.info("[%s] Veo hook clip ready: %.1fs", job.job_id, duration_s)

        except Exception as e:
            logger.warning("[%s] Veo hook clip failed (using image fallback): %s", job.job_id, e)

    def _create_fallback_image(self, job: VideoJob, scene_id: int) -> ImageAsset:
        """Create a plain dark image when Imagen fails."""
        from PIL import Image, ImageDraw, ImageFont

        scene = next((s for s in job.script.scenes if s.scene_id == scene_id), None)
        text = scene.spoken_text[:80] if scene else f"Scene {scene_id}"

        img = Image.new("RGB", (config.VIDEO_WIDTH, config.VIDEO_HEIGHT), color=(15, 20, 40))
        draw = ImageDraw.Draw(img)

        # Draw text centered
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        except Exception:
            font = ImageFont.load_default()

        # Word-wrap text
        words = text.split()
        lines, current = [], []
        for word in words:
            current.append(word)
            if len(" ".join(current)) > 25:
                lines.append(" ".join(current[:-1]))
                current = [word]
        if current:
            lines.append(" ".join(current))

        y = config.VIDEO_HEIGHT // 2 - len(lines) * 30
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            w = bbox[2] - bbox[0]
            draw.text(((config.VIDEO_WIDTH - w) // 2, y), line, fill=(255, 220, 50), font=font)
            y += 70

        out_path = job.workspace_dir / f"scene_{scene_id:03d}.png"
        img.save(str(out_path))
        return ImageAsset(
            scene_id=scene_id,
            path=str(out_path),
            duration_s=float(scene.duration_hint_s if scene else 10),
            is_fallback=True,
        )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _estimate_timepoints(scenes, total_duration: float) -> list[Timepoint]:
    """
    Estimate scene start/end times proportionally by word count.
    Produces scene_N_start and scene_N_end marks used by srt_generator.
    """
    word_counts = [max(1, len(s.spoken_text.split())) for s in scenes]
    total_words = sum(word_counts)
    timepoints: list[Timepoint] = []
    cursor = 0.0
    for scene, wc in zip(scenes, word_counts):
        scene_dur = total_duration * (wc / total_words)
        timepoints.append(Timepoint(mark_name=f"scene_{scene.scene_id}_start", time_seconds=cursor))
        timepoints.append(Timepoint(mark_name=f"scene_{scene.scene_id}_end", time_seconds=cursor + scene_dur))
        cursor += scene_dur
    return timepoints


def _fmt_srt(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _assign_scene_durations(
    job: VideoJob,
    timepoints: list[Timepoint],
    image_assets: list[ImageAsset],
) -> None:
    """
    Use scene_N_start / scene_N_end timepoints to set authoritative durations
    on ImageAssets, overriding the hint from the script.
    """
    import re
    start_times: dict[int, float] = {}
    end_times: dict[int, float] = {}

    for tp in timepoints:
        ms = re.match(r"^scene_(\d+)_start$", tp.mark_name)
        me = re.match(r"^scene_(\d+)_end$", tp.mark_name)
        if ms:
            start_times[int(ms.group(1))] = tp.time_seconds
        if me:
            end_times[int(me.group(1))] = tp.time_seconds

    for asset in image_assets:
        s = start_times.get(asset.scene_id)
        e = end_times.get(asset.scene_id)
        if s is not None and e is not None and e > s:
            asset.duration_s = round(e - s + 0.3, 2)  # small buffer


def _resize_to_fill(image_bytes: bytes, target_w: int, target_h: int) -> "Image.Image":
    """Scale image to fill target dimensions exactly, center-cropping any overflow."""
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))
