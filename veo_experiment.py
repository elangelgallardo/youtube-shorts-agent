"""
Veo 3 experiment: generate video clips per scene, assemble with existing audio + SRT.
Uses the Hubble job (dd4ea1aa-0214-4bac-ac3f-014a4ae5d066).
"""
import concurrent.futures
import logging
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

from google import genai
from google.genai import types
import config
from utils.state_store import load_job

JOB_ID = 'dd4ea1aa-0214-4bac-ac3f-014a4ae5d066'
job = load_job(JOB_ID)
out_dir = job.workspace_dir / 'veo_experiment'
out_dir.mkdir(exist_ok=True)

client = genai.Client(api_key=config.GOOGLE_API_KEY)

def generate_clip(scene, duration_s: float) -> Path:
    clip_path = out_dir / f'clip_{scene.scene_id:03d}.mp4'
    if clip_path.exists():
        logger.info('Scene %d: clip already exists, skipping', scene.scene_id)
        return clip_path

    prompt = (
        scene.visual_prompt.rstrip('.') +
        '. Cinematic, smooth camera motion, photorealistic, vertical 9:16 video, no text, no subtitles.'
    )

    # Retry loop for 429 rate limits
    for attempt in range(6):
        if attempt > 0:
            wait = 60 * attempt
            logger.info('Scene %d: waiting %ds before retry (attempt %d)...', scene.scene_id, wait, attempt + 1)
            time.sleep(wait)
        try:
            logger.info('Scene %d: submitting to Veo 3...', scene.scene_id)
            op = client.models.generate_videos(
                model='veo-3.0-generate-001',
                prompt=prompt,
                config=types.GenerateVideosConfig(
                    aspect_ratio='9:16',
                    duration_seconds=8,
                    number_of_videos=1,
                ),
            )
            break
        except Exception as e:
            if '429' in str(e) and attempt < 5:
                continue
            raise

    while not op.done:
        time.sleep(10)
        op = client.operations.get(op)

    videos = op.response.generated_videos
    if not videos:
        raise RuntimeError(f'No video returned for scene {scene.scene_id}')

    raw_path = out_dir / f'clip_{scene.scene_id:03d}_raw.mp4'
    video_obj = videos[0].video
    if video_obj.video_bytes:
        raw_path.write_bytes(video_obj.video_bytes)
    elif video_obj.uri:
        import urllib.request
        logger.info('Scene %d: downloading from URI...', scene.scene_id)
        urllib.request.urlretrieve(video_obj.uri, str(raw_path))
    else:
        raise RuntimeError(f'No video data for scene {scene.scene_id}')
    logger.info('Scene %d: raw clip saved (%s)', scene.scene_id, raw_path.name)

    # Resize to 1080x1920 and loop/trim to exact duration needed
    subprocess.run([
        'ffmpeg', '-y',
        '-stream_loop', '-1', '-i', str(raw_path),
        '-t', str(duration_s),
        '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920',
        '-r', '30',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
        '-an',
        str(clip_path),
    ], check=True, capture_output=True)
    raw_path.unlink()
    logger.info('Scene %d: final clip %.1fs → %s', scene.scene_id, duration_s, clip_path.name)
    return clip_path

# Build scene→duration map
dur_map = {a.scene_id: a.duration_s for a in job.images}

# Generate clips sequentially to respect rate limits (~1 req/min)
logger.info('Generating %d clips with Veo 3 (sequential)...', len(job.script.scenes))
clip_paths = {}
for i, scene in enumerate(job.script.scenes):
    if i > 0:
        logger.info('Waiting 65s before next clip to respect rate limit...')
        time.sleep(65)
    try:
        clip_paths[scene.scene_id] = generate_clip(scene, dur_map.get(scene.scene_id, 8.0))
    except Exception as e:
        logger.error('Scene %d failed: %s', scene.scene_id, e)

logger.info('All clips ready: %d/%d', len(clip_paths), len(job.script.scenes))

# Concatenate clips in order
concat_list = out_dir / 'concat.txt'
ordered = [clip_paths[s.scene_id] for s in job.script.scenes if s.scene_id in clip_paths]
concat_list.write_text('\n'.join(f"file '{p}'" for p in ordered))

raw_video = out_dir / 'video_raw.mp4'
subprocess.run([
    'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
    '-i', str(concat_list),
    '-c', 'copy', str(raw_video),
], check=True, capture_output=True)
logger.info('Concatenated %d clips → %s', len(ordered), raw_video.name)

# Mix in existing TTS audio + subtitles
final_path = out_dir / 'final_veo_video.mp4'
audio_path = job.audio.audio_path
srt_path = job.audio.subtitles_srt_path
bg_music = str(config.ASSETS_DIR / 'background_music.mp3')

# Load font info from assembly agent approach
font_path = '/usr/share/fonts/opentype/bebas-neue/BebasNeue-Regular.otf'

subtitle_filter = (
    f"subtitles={srt_path}:fontsdir=/usr/share/fonts/opentype/bebas-neue"
    f":force_style='FontName=BebasNeue-Regular,FontSize=22,PrimaryColour=&H00FFFFFF,"
    f"OutlineColour=&H00000000,Outline=3,Shadow=1,Alignment=2,"
    f"MarginV=130,Bold=1'"
)

subprocess.run([
    'ffmpeg', '-y',
    '-i', str(raw_video),
    '-i', audio_path,
    '-i', bg_music,
    '-filter_complex',
    f'[1:a]volume=1.0[voice];'
    f'[2:a]volume={config.BG_MUSIC_VOLUME}[music];'
    f'[voice][music]amix=inputs=2:duration=first[aout];'
    f'[0:v]{subtitle_filter}[vout]',
    '-map', '[vout]', '-map', '[aout]',
    '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
    '-c:a', 'aac', '-b:a', '192k',
    '-shortest',
    str(final_path),
], check=True)

logger.info('VEO EXPERIMENT DONE: %s', final_path)
print(f'OUTPUT:{final_path}')
