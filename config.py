import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
CREDENTIALS_DIR = BASE_DIR / "credentials"
WORKSPACE_DIR = BASE_DIR / "workspace" / "jobs"
LOGS_DIR = BASE_DIR / "logs"
PROMPTS_DIR = BASE_DIR / "prompts"
STATE_DB_PATH = BASE_DIR / "state.db"
ASSETS_DIR = BASE_DIR / "assets"

# ── Audio mix ─────────────────────────────────────────────────────────────────
BG_MUSIC_VOLUME = 0.12      # background music volume relative to narration (12%)
TTS_LEADING_PAUSE_S = 1.0   # natural silence baked into TTS audio before narration starts

# ── Google AI ──────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
GOOGLE_TTS_API_KEY = os.environ.get("GOOGLE_TTS_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))

# Models
GEMINI_FLASH_MODEL = "gemini-3.1-pro-preview"
GEMINI_PRO_MODEL = "gemini-3.1-pro-preview"
GEMINI_RESEARCH_MODEL = "gemini-3.1-pro-preview"
GEMINI_PLANNING_MODEL = "gemini-3.1-pro-preview"
IMAGE_GEN_MODEL = "gemini-3-pro-image-preview"

# ── Google Cloud TTS ───────────────────────────────────────────────────────────
TTS_VOICE_NAME = os.environ.get("TTS_VOICE", "es-US-Chirp3-HD-Fenrir")
TTS_LANGUAGE_CODE = os.environ.get("CHANNEL_LANGUAGE", "es-US")
TTS_SPEAKING_RATE = 1.25  # 1.25x speed (not supported by Chirp3-HD, ignored)

# ── YouTube ────────────────────────────────────────────────────────────────────
YOUTUBE_CLIENT_SECRETS = str(CREDENTIALS_DIR / "client_secrets.json")
YOUTUBE_TOKEN_FILE = str(CREDENTIALS_DIR / "youtube_token.json")
YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]

YOUTUBE_CHANNEL_ID = os.environ.get("YOUTUBE_CHANNEL_ID", "")
YOUTUBE_CATEGORY_ID = os.environ.get("YOUTUBE_CATEGORY_ID", "27")
DEFAULT_PRIVACY = os.environ.get("DEFAULT_PRIVACY", "public")

# ── Channel identity ───────────────────────────────────────────────────────────
CHANNEL_NICHE = os.environ.get("CHANNEL_NICHE", "science facts")
CHANNEL_LANGUAGE = os.environ.get("CHANNEL_LANGUAGE", "en-US")

# Brand color palette used in Imagen prompts to keep visual identity consistent
BRAND_COLORS = "deep navy blue and electric yellow accent, clean white background"

# ── Pipeline settings ──────────────────────────────────────────────────────────
VIDEOS_PER_DAY = 10
ANALYTICS_LOOKBACK_DAYS = 2
ANALYTICS_TOP_N_VIDEOS = 20

# Script / video dimensions
TARGET_DURATION_S = 72       # desired audio output in seconds
WORDS_PER_MINUTE = 180       # measured actual rate for es-US-Chirp3-HD-Fenrir (185 WPM measured, 180 with break-tag buffer)
SCRIPT_WORDS_OVERSHOOT = 1.0
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_FPS = 30
SCENES_PER_VIDEO = 10

# Upload stagger (UTC hours for the 5 daily uploads)
_raw_hours = os.environ.get("UPLOAD_HOURS", "8,11,13,16,19")
UPLOAD_HOURS = [int(h.strip()) for h in _raw_hours.split(",")]

# Generation pipeline starts daily at this UTC hour (before first upload)
GENERATION_START_HOUR = int(os.environ.get("GENERATION_START_HOUR", "10"))  # 6am local (UTC-4), 2h before first upload

# Retry / error handling
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0   # seconds; actual delay = base * 2**attempt + jitter

# Workspace cleanup: delete jobs older than this many days
WORKSPACE_RETENTION_DAYS = 7
