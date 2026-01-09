from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
from app.services.workflow.graph import process_workflow_query

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []


@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Convert history to format expected by workflow if needed
        # Assuming frontend sends [{role: 'user', content: '...'}, ...] matches

        result = process_workflow_query(request.query, request.history)
        answer = result.get("response", "No answer generated.")

        return {
            "answer": answer,
            "judge_result": result.get("judge_result"),
            "judge_feedback": result.get("judge_feedback"),
        }
    except Exception as e:
        logger.error(f"Error in chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
