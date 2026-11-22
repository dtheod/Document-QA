from fastapi import APIRouter
from pydantic import BaseModel
import asyncio
import logging
from app.services.evaluation.ragas import run_ragas_evaluation

logger = logging.getLogger(__name__)

router = APIRouter()

# The latest evaluation run. It takes many minutes, so it runs as a background job that the
# frontend polls. One run at a time, kept in memory (the app runs a single worker).
job = {"status": "idle", "results": None, "error": None}
tasks = set()  # keeps a reference so the running task is not garbage collected


class EvaluationRequest(BaseModel):
    context: str = ""  # Optional, might rely on indexed docs


async def run_job():
    try:
        # Runs the global verification suite defined in app/services/evaluation/ragas.py;
        # 'request.context' is ignored for now
        job["results"] = await run_ragas_evaluation()
        job["status"] = "done"
    except Exception as e:
        logger.error(f"Error in evaluation: {e}", exc_info=True)
        job["error"] = str(e)
        job["status"] = "failed"


@router.post("/evaluate", status_code=202)
async def start_evaluation(request: EvaluationRequest):
    """Start an evaluation run, unless one is already running. Poll GET /evaluate for results."""
    if job["status"] != "running":
        job.update(status="running", results=None, error=None)
        task = asyncio.create_task(run_job())
        tasks.add(task)
        task.add_done_callback(tasks.discard)
    return job


@router.get("/evaluate")
async def evaluation_status():
    """Status of the latest run: idle, running, done (with results) or failed (with error)."""
    return job
