"""
Central data model. Every agent receives and mutates a VideoJob instance.
The full object is serialized to SQLite between pipeline stages.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from models.enums import JobStatus, VideoFormat, SceneType


# ── Sub-models ──────────────────────────────────────────────────────────────


@dataclass
class VideoMetric:
    video_id: str
    title: str
    views: int
    watch_minutes: float
    likes: int
    comments: int
    avg_view_duration_s: float
    ctr: float
    subs_gained: int


@dataclass
class AnalyticsContext:
    top_video_topics: list[str] = field(default_factory=list)
    top_formats: list[str] = field(default_factory=list)
    avg_winner_duration_s: float = 45.0
    avg_winner_ctr: float = 0.05
    channel_niche: str = ""
    raw_metrics: list[VideoMetric] = field(default_factory=list)
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class VideoPlan:
    title_concept: str = ""
    topic: str = ""
    angle: str = ""
    format: VideoFormat = VideoFormat.HOOK_REVEAL
    target_duration_s: int = 55
    rationale: str = ""
    is_exploratory: bool = False   # True for the 20% new-topic videos


@dataclass
class ResearchContext:
    key_stat: str = ""
    facts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    search_queries_used: list[str] = field(default_factory=list)
    grounded: bool = True   # False if Search grounding was unavailable


@dataclass
class Scene:
    scene_id: int = 0
    type: SceneType = SceneType.BODY
    spoken_text: str = ""
    ssml: str = ""
    visual_prompt: str = ""
    duration_hint_s: int = 10


@dataclass
class Script:
    hook_line: str = ""
    cta_line: str = ""
    total_duration_estimate_s: int = 55
    scenes: list[Scene] = field(default_factory=list)

    @property
    def full_narration(self) -> str:
        return " ".join(s.spoken_text for s in self.scenes)


@dataclass
class Timepoint:
    mark_name: str = ""
    time_seconds: float = 0.0


@dataclass
class AudioAsset:
    audio_path: str = ""          # str for JSON serialization; use Path(x) when needed
    total_duration_s: float = 0.0
    timepoints: list[Timepoint] = field(default_factory=list)
    subtitles_srt_path: str = ""


@dataclass
class ImageAsset:
    scene_id: int = 0
    path: str = ""
    duration_s: float = 10.0
    veo_clip_path: str = ""  # optional Veo 3 video clip for this scene


@dataclass
class VideoAsset:
    raw_path: str = ""
    final_path: str = ""
    duration_s: float = 0.0


@dataclass
class UploadMetadata:
    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    category_id: str = "27"
    privacy_status: str = "public"


@dataclass
class UploadResult:
    youtube_video_id: str = ""
    youtube_url: str = ""
    upload_time: str = ""
    metadata: UploadMetadata = field(default_factory=UploadMetadata)


# ── Root model ──────────────────────────────────────────────────────────────


@dataclass
class VideoJob:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.QUEUED
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    scheduled_upload_at: str = ""
    retry_count: int = 0
    errors: list[str] = field(default_factory=list)

    analytics_context: AnalyticsContext = field(default_factory=AnalyticsContext)
    plan: VideoPlan = field(default_factory=VideoPlan)
    research: ResearchContext = field(default_factory=ResearchContext)
    script: Script = field(default_factory=Script)
    audio: AudioAsset = field(default_factory=AudioAsset)
    images: list[ImageAsset] = field(default_factory=list)
    video: VideoAsset = field(default_factory=VideoAsset)
    upload: UploadResult = field(default_factory=UploadResult)
    costs: dict = field(default_factory=dict)  # usage tracking per agent
    use_veo: bool = True  # whether to generate a Veo 3 clip for scene 0

    # ── Serialization ──────────────────────────────────────────────────────

    def to_json(self) -> str:
        def _convert(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _convert(v) for k, v in asdict(obj).items()}
            if isinstance(obj, list):
                return [_convert(i) for i in obj]
            return obj

        return json.dumps(_convert(self), indent=2)

    @classmethod
    def from_json(cls, data: str | dict) -> "VideoJob":
        if isinstance(data, str):
            data = json.loads(data)

        job = cls()
        job.job_id = data.get("job_id", job.job_id)
        job.status = JobStatus(data.get("status", JobStatus.QUEUED))
        job.created_at = data.get("created_at", job.created_at)
        job.scheduled_upload_at = data.get("scheduled_upload_at", "")
        job.retry_count = data.get("retry_count", 0)
        job.errors = data.get("errors", [])

        if ac := data.get("analytics_context"):
            metrics = [VideoMetric(**m) for m in ac.get("raw_metrics", [])]
            job.analytics_context = AnalyticsContext(
                top_video_topics=ac.get("top_video_topics", []),
                top_formats=ac.get("top_formats", []),
                avg_winner_duration_s=ac.get("avg_winner_duration_s", 45.0),
                avg_winner_ctr=ac.get("avg_winner_ctr", 0.05),
                channel_niche=ac.get("channel_niche", ""),
                raw_metrics=metrics,
                fetched_at=ac.get("fetched_at", ""),
            )

        if p := data.get("plan"):
            job.plan = VideoPlan(
                title_concept=p.get("title_concept", ""),
                topic=p.get("topic", ""),
                angle=p.get("angle", ""),
                format=VideoFormat(p.get("format", VideoFormat.HOOK_REVEAL)),
                target_duration_s=p.get("target_duration_s", 55),
                rationale=p.get("rationale", ""),
                is_exploratory=p.get("is_exploratory", False),
            )

        if r := data.get("research"):
            job.research = ResearchContext(**r)

        if s := data.get("script"):
            scenes = [
                Scene(
                    scene_id=sc.get("scene_id", i),
                    type=SceneType(sc.get("type", SceneType.BODY)),
                    spoken_text=sc.get("spoken_text", ""),
                    ssml=sc.get("ssml", ""),
                    visual_prompt=sc.get("visual_prompt", ""),
                    duration_hint_s=sc.get("duration_hint_s", 10),
                )
                for i, sc in enumerate(s.get("scenes", []))
            ]
            job.script = Script(
                hook_line=s.get("hook_line", ""),
                cta_line=s.get("cta_line", ""),
                total_duration_estimate_s=s.get("total_duration_estimate_s", 55),
                scenes=scenes,
            )

        if a := data.get("audio"):
            tps = [Timepoint(**t) for t in a.get("timepoints", [])]
            job.audio = AudioAsset(
                audio_path=a.get("audio_path", ""),
                total_duration_s=a.get("total_duration_s", 0.0),
                timepoints=tps,
                subtitles_srt_path=a.get("subtitles_srt_path", ""),
            )

        job.images = [ImageAsset(**img) for img in data.get("images", [])]

        if v := data.get("video"):
            job.video = VideoAsset(**v)

        if u := data.get("upload"):
            meta = UploadMetadata(**u.get("metadata", {})) if u.get("metadata") else UploadMetadata()
            job.upload = UploadResult(
                youtube_video_id=u.get("youtube_video_id", ""),
                youtube_url=u.get("youtube_url", ""),
                upload_time=u.get("upload_time", ""),
                metadata=meta,
            )

        return job

    @property
    def workspace_dir(self) -> Path:
        from config import WORKSPACE_DIR
        return WORKSPACE_DIR / self.job_id
