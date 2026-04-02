# YouTube Shorts Agent — Claude Code Instructions

## Long-running operations
Assembly and full pipeline runs take 5–15+ minutes. Always run them as background tasks using `run_in_background: true` on the Bash tool so they never hit the 10-minute timeout. Examples:

- AssemblyAgent.run()
- MediaAgent.run() (Whisper transcription + image generation)
- Full pipeline (planning → scripting → media → assembly)

Use `.venv/bin/python` for all Python commands.

## Workflow
The user requests video ideas interactively. Use `state_store.save_pending_plans()` to cache the list when showing ideas, and `load_pending_plans()` to retrieve the correct plan when the user picks a number.

**ALWAYS use `create_job(plan, use_veo=True/False)` from `utils.state_store` when creating a job interactively** — never `VideoJob()` directly. This saves to DB immediately with a full UUID, preventing orphaned workspace folders with truncated IDs.

## Key paths
- Workspace jobs: `workspace/jobs/<job_id>/`
- Final video: `workspace/jobs/<job_id>/final_video.mp4`
- State DB: `state.db`
- Background music: `assets/background_music.mp3`
