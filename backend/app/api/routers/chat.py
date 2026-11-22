from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
from app.services.workflow.chat import process_query

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

        result = await process_query(request.query, request.history)

        return {
            "answer": result.response,
            "sources": [source.model_dump() for source in result.sources],
            "judge_result": result.judge_result,
            "judge_feedback": result.judge_feedback,
        }
    except Exception as e:
        logger.error(f"Error in chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
