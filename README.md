# YouTube Shorts Autonomous Agent

A fully automated pipeline that creates and uploads YouTube Shorts 24/7 using Google's AI stack. Give it a niche and it handles everything: research, scripting, visuals, voiceover, assembly, and upload.

## How it works

Each video goes through a sequential pipeline of specialized agents:

```
AnalyticsAgent → PlanningAgent → ResearchAgent → ScriptAgent
                                                      ↓
UploadAgent ← AssemblyAgent ← MediaAgent ←──────────┘
```

| Stage | Agent | What it does |
|-------|-------|-------------|
| 1 | **AnalyticsAgent** | Fetches top 20 performing videos from YouTube Analytics (last 30 days) |
| 2 | **PlanningAgent** | Gemini generates 10 video ideas — 60% building on top performers, 40% fresh topics |
| 3 | **ResearchAgent** | Gemini + Google Search grounding gathers data-backed facts per topic |
| 4 | **ScriptAgent** | Gemini writes a 20-scene ~72s script in plain language (ELI12 style) |
| 5 | **MediaAgent** | Google Cloud TTS narration + Imagen image per scene + optional Veo 3 hook clip |
| 6 | **AssemblyAgent** | FFmpeg composes Ken Burns slideshow with burned-in subtitles + background music |
| 7 | **UploadAgent** | Uploads to YouTube with LLM-generated title, description, and tags |

## Tech stack

| Service | Purpose |
|---------|---------|
| Gemini 2.5 Flash Lite | Planning, research, metadata generation |
| Gemini 2.5 Flash Image (Imagen) | 9:16 scene image generation |
| Veo 3 (`veo-3.0-generate-001`) | Optional cinematic hook clip for scene 0 |
| Google Cloud TTS Chirp3-HD | High-quality neural voice narration |
| YouTube Data API v3 | Video upload + metadata patch |
| YouTube Analytics API | Channel performance metrics |
| FFmpeg + MoviePy | Video assembly, Ken Burns zoom, subtitle burn-in |
| SQLite | Job state, analytics cache, topic deduplication |

## Cost per video

| Config | Approx. cost |
|--------|-------------|
| 20 images, no Veo | ~$0.83 |
| 20 images + 8s Veo 3 hook | ~$3.63 |

Breakdown: 20 × Imagen ($0.04) + TTS (~$0.03) + Gemini (~$0.001) + optional Veo 8s ($2.80).

## Setup

### Prerequisites

```bash
python --version   # 3.11+
ffmpeg -version    # required for video assembly

# Install ffmpeg if needed:
apt install ffmpeg        # Ubuntu/Debian
brew install ffmpeg       # macOS
```

### 1. Clone and install

```bash
git clone <repo-url>
cd youtube_shorts_agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
GOOGLE_API_KEY=...          # from Google AI Studio (aistudio.google.com)
GOOGLE_TTS_API_KEY=...      # from Google Cloud Console (Cloud TTS API)
GCP_PROJECT_ID=...          # your Google Cloud project ID
YOUTUBE_CHANNEL_ID=UC...    # your channel ID
CHANNEL_NICHE=science facts
CHANNEL_LANGUAGE=es-US
TTS_VOICE=es-US-Chirp3-HD-Fenrir
DEFAULT_PRIVACY=public
YOUTUBE_CATEGORY_ID=27
UPLOAD_HOURS=8,10,12,14,16
```

### 3. Set up YouTube OAuth

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Enable: **YouTube Data API v3**, **YouTube Analytics API**, **Cloud Text-to-Speech API**
3. Create **OAuth 2.0 credentials** → Desktop App
4. Download → save as `credentials/client_secrets.json`
5. Run the auth flow:

```bash
python auth.py
```

This opens a browser once and saves `credentials/youtube_token.json`. Required scopes:
- `https://www.googleapis.com/auth/youtube` (upload + metadata update)
- `https://www.googleapis.com/auth/yt-analytics.readonly`

### 4. Run a single video (test)

```bash
python -c "
import uuid
from datetime import date
from utils.state_store import get_analytics_cache
from agents import ResearchAgent, ScriptAgent, MediaAgent, AssemblyAgent, UploadAgent
from models.video_job import VideoJob, VideoPlan

plan = VideoPlan(
    topic='Black holes',
    title_concept='¿Qué hay dentro de un agujero negro?',
    angle='Exploramos qué pasaría si cayeras en uno.',
)
job = VideoJob(job_id=str(uuid.uuid4())[:8], plan=plan, use_veo=False)
for Agent in [ResearchAgent, ScriptAgent, MediaAgent, AssemblyAgent, UploadAgent]:
    job = Agent().run(job)
"
```

### 5. Start the scheduler

```bash
python main.py
```

## Interactive workflow (with Claude Code)

The recommended way to use this agent is interactively via Claude Code:

```
# Get ideas
"give me 10 ideas"

# Pick one, optionally with Veo
"3"                     # uses Veo by default
"3; no Veo"             # skips Veo hook, cheaper
"my own idea: '...'"    # custom topic

# Check status
"status?"
```

Claude runs the full pipeline in the background and notifies you when done.

## Configuration reference

Key settings in `config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEOS_PER_DAY` | `10` | Ideas generated per planning batch |
| `SCENES_PER_VIDEO` | `20` | Images (and script scenes) per video |
| `TARGET_DURATION_S` | `72` | Target audio length in seconds |
| `TTS_SPEAKING_RATE` | `1.25` | Speed multiplier (ignored for Chirp3-HD) |
| `BG_MUSIC_VOLUME` | `0.02` | Background music level (2% of narration) |
| `WORKSPACE_RETENTION_DAYS` | `7` | Days before auto-deleting job files |

### Veo 3

Veo generates a cinematic 8-second clip for the first scene (hook). It's optional — pass `use_veo=False` to the `VideoJob` to skip it. Adds ~$2.80 per video.

### TTS voices

Any [Google Cloud TTS voice](https://cloud.google.com/text-to-speech/docs/voices) works. Chirp3-HD voices (`es-US-Chirp3-HD-Fenrir`, etc.) are highest quality but don't support `speaking_rate`. Studio voices (`es-US-Studio-B`, etc.) support speed control.

### 60/40 planning mix

The planning agent generates:
- **60%** build-on ideas — deeper angles on your best-performing topics
- **40%** new ideas — fresh topics to expand reach

### Script style

Scripts are pure informational content — no hook framing, no CTA, no filler. Written in plain accessible language (explain-like-you're-12 style) with everyday analogies.

## File structure

```
youtube_shorts_agent/
├── main.py                  # Scheduler + daily pipeline orchestrator
├── pipeline.py              # Single-video pipeline runner
├── auth.py                  # YouTube OAuth flow
├── config.py                # All configuration
├── setup.py                 # One-time setup & API verification
│
├── agents/
│   ├── analytics_agent.py   # YouTube Analytics fetcher
│   ├── planning_agent.py    # Gemini topic planner (10 ideas, 60/40 mix)
│   ├── research_agent.py    # Gemini + Search grounding
│   ├── script_agent.py      # Gemini script writer (20 scenes, ELI12)
│   ├── media_agent.py       # TTS + Imagen + optional Veo 3 hook
│   ├── assembly_agent.py    # FFmpeg video assembly
│   └── upload_agent.py      # YouTube upload + LLM metadata
│
├── models/
│   ├── video_job.py         # Central VideoJob dataclass + all sub-models
│   └── enums.py             # JobStatus, VideoFormat, SceneType
│
├── utils/
│   ├── state_store.py       # SQLite persistence (jobs, cache, dedup)
│   ├── cost_tracker.py      # Per-video cost tracking + DB persistence
│   ├── ssml_builder.py      # Scene text → TTS input
│   ├── srt_generator.py     # TTS timepoints → SRT subtitles
│   ├── ffmpeg_runner.py     # FFmpeg subprocess wrapper
│   └── retry.py             # Exponential backoff decorator
│
├── credentials/             # OAuth tokens (gitignored)
│   └── client_secrets.json  # ← place your OAuth credentials here
│
├── workspace/jobs/          # Per-job working files (auto-cleaned after 7 days)
├── logs/                    # Daily log files
├── assets/
│   └── background_music.mp3
├── .env.example             # Environment variable template
└── requirements.txt
```

## Default schedule (UTC)

| Time | Action |
|------|--------|
| 06:00 | Fetch analytics, generate 10 ideas, create job queue |
| 08:00 | Upload video 1 |
| 10:00 | Upload video 2 |
| 12:00 | Upload video 3 |
| 14:00 | Upload video 4 |
| 16:00 | Upload video 5 |
| 23:00 | Cleanup old workspace files |

Change upload times via `UPLOAD_HOURS=8,10,12,14,16` in `.env`.

## Cost tracking

Every uploaded video records actual costs to `state.db` (`video_costs` table):

```sql
SELECT title, total_usd, has_veo FROM video_costs ORDER BY created_at;
```

Previous videos without tracked costs are estimated from scene count and Veo usage.

## YouTube API quota

With 5 uploads/day, daily quota usage:
- Upload: 5 × 1,600 = 8,000 units
- Analytics + reads: ~200 units
- **Total: ~8,200 / 10,000 units/day**

Request a quota increase in Google Cloud Console if needed.

## Error handling

- Each pipeline stage retries up to 3× with exponential backoff
- One failed video doesn't block others
- Jobs survive restarts (persisted in `state.db`)
- Veo failures fall back gracefully to static image for scene 0
- LLM metadata failures fall back to rule-based title/description/tags
