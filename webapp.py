"""
Web UI for the YouTube Shorts pipeline.

Endpoints:
  GET  /           → serve the single-page UI
  POST /ideas      → generate 10 video ideas via PlanningAgent
  POST /generate   → start a pipeline job (background thread)
  GET  /job/{id}   → poll job status + result
"""
import sys
import os
import threading
import logging

# Ensure agents can import config, models, utils
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from agents.planning_agent import PlanningAgent
from models.video_job import AnalyticsContext, VideoPlan
from models.enums import VideoFormat
from utils.state_store import create_job, load_job, save_job, save_pending_plans, load_pending_plans, save_proposed_ideas
from pipeline import run_video_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
_planning_agent = PlanningAgent()

# ── In-memory job registry (status updates come from state_store too) ──────────
_running: dict[str, bool] = {}


# ── Models ─────────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    # Either pick from the cached plan list by index (0-based) or supply custom fields
    plan_index: int | None = None
    title_concept: str | None = None
    angle: str | None = None
    use_veo: bool = False
    upload: bool = True


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(os.path.dirname(__file__), "webapp_ui.html")) as f:
        return f.read()


@app.post("/ideas")
async def get_ideas():
    try:
        plans = _planning_agent.run(AnalyticsContext())
        save_pending_plans(plans)
        save_proposed_ideas(plans)
        return {"ideas": [
            {"index": i, "title": p.title_concept, "angle": p.angle}
            for i, p in enumerate(plans)
        ]}
    except Exception as e:
        logger.exception("Failed to generate ideas")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/generate")
async def generate(req: GenerateRequest):
    try:
        if req.plan_index is not None:
            plans = load_pending_plans()
            if req.plan_index < 0 or req.plan_index >= len(plans):
                return JSONResponse(status_code=400, content={"error": "Invalid plan index"})
            plan = plans[req.plan_index]
        elif req.title_concept:
            plan = VideoPlan(
                title_concept=req.title_concept,
                topic=req.title_concept,
                angle=req.angle or "",
                format=VideoFormat.HOOK_REVEAL,
                target_duration_s=70,
            )
        else:
            return JSONResponse(status_code=400, content={"error": "Provide plan_index or title_concept"})

        job = create_job(plan, use_veo=req.use_veo)
        _running[job.job_id] = True

        def _run():
            try:
                run_video_pipeline(job, upload=req.upload)
            except Exception:
                logger.exception("[%s] Pipeline error", job.job_id)
            finally:
                _running.pop(job.job_id, None)

        threading.Thread(target=_run, daemon=True).start()
        return {"job_id": job.job_id}

    except Exception as e:
        logger.exception("Failed to start pipeline")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/job/{job_id}")
async def job_status(job_id: str):
    try:
        job = load_job(job_id)
        result = {
            "job_id": job_id,
            "status": job.status.value if hasattr(job.status, "value") else str(job.status),
            "title": job.plan.title_concept,
            "running": _running.get(job_id, False),
        }
        if job.upload and job.upload.youtube_url:
            result["youtube_url"] = job.upload.youtube_url
        if job.costs:
            total = sum(
                v.get("cost_usd", 0) if isinstance(v, dict) else 0
                for v in job.costs.values()
            )
            result["cost_usd"] = round(total, 4)
        if job.errors:
            result["errors"] = job.errors
        return result
    except Exception as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
