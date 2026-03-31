from enum import Enum


class JobStatus(str, Enum):
    QUEUED = "queued"
    RESEARCHING = "researching"
    SCRIPTING = "scripting"
    GENERATING_MEDIA = "generating_media"
    ASSEMBLING = "assembling"
    READY = "ready"
    UPLOADING = "uploading"
    DONE = "done"
    FAILED = "failed"
    PERMANENTLY_FAILED = "permanently_failed"


class VideoFormat(str, Enum):
    HOOK_REVEAL = "hook_reveal"      # "You won't believe X... here's why"
    COUNTDOWN = "countdown"          # "5 facts about X"
    MYTH_BUST = "myth_bust"          # "X is actually wrong because..."
    TUTORIAL = "tutorial"            # "How to do X in 60 seconds"
    LISTICLE = "listicle"            # "Top 5 X"


class SceneType(str, Enum):
    HOOK = "hook"
    BODY = "body"
    CTA = "cta"
