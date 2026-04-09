"""
Manual Veo pipeline runner.

Usage:
    python veo_runner.py <plan_index>

Runs: Research → Script → Audio → Veo clips → Assembly
Does NOT upload — use the web app or upload_agent manually for that.
"""
import logging
import sys

sys.path.insert(0, ".")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run(plan_index: int) -> str:
    from utils.state_store import load_pending_plans, create_job, save_job
    from agents.research_agent import ResearchAgent
    from agents.script_agent import ScriptAgent
    from agents.media_agent import MediaAgent
    from agents.veo_agent import VeoAgent
    from agents.assembly_agent import AssemblyAgent

    plans = load_pending_plans()
    if plan_index < 0 or plan_index >= len(plans):
        raise ValueError(f"Invalid plan index {plan_index} — {len(plans)} plans available")

    plan = plans[plan_index]
    logger.info("Starting Veo pipeline for: %s", plan.title_concept)

    job = create_job(plan, use_veo=False)
    logger.info("Job ID: %s", job.job_id)

    job = ResearchAgent().run(job);  save_job(job); logger.info("Research done")
    job = ScriptAgent().run(job);    save_job(job); logger.info("Script done")
    job = MediaAgent().run(job, generate_images=False); save_job(job); logger.info("Audio done")
    job = VeoAgent().run(job);       save_job(job); logger.info("Veo clips done")
    job = AssemblyAgent().run(job);  save_job(job); logger.info("Assembly done: %s", job.video.final_path)

    return job.video.final_path


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    final = run(idx)
    print(f"\nDone: {final}")
