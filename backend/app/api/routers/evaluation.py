from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
from app.services.evaluation.ragas import run_ragas_evaluation

logger = logging.getLogger(__name__)

router = APIRouter()


class EvaluationRequest(BaseModel):
    context: str = ""  # Optional, might rely on indexed docs


@router.post("/evaluate")
async def evaluate_endpoint(request: EvaluationRequest):
    try:
        # Trigger the full RAGAS evaluation suite
        # Note: This ignores the 'request.context' or specific document context for now
        # and runs the global verification suite defined in app/services/evaluation/ragas.py

        results = await run_ragas_evaluation()
        return results

    except Exception as e:
        logger.error(f"Error in evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
