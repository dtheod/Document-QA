import time
import logging
import asyncio

from app.models.schemas import ChatResponseModel, SourceReference
from app.services.workflow.graph import process_workflow_query

log = logging.getLogger(__name__)


async def process_query(query: str, chat_history: list | None) -> ChatResponseModel:
    """
    Process a user query and return an LLM-generated response with
    source references.

    Args:
        query (str): The user's natural language query string.
        chat_history (list): List of previous chat messages for context.

    Returns:
        ChatResponseModel: A model containing the LLM-generated response
            and a list of source document references.

    Raises:
        Exception: If there's an error during query processing.
    """
    log.debug(f"Processing query: {query}.... with chat history: {chat_history}")
    start_time = time.time()

    try:
        # Run the synchronous workflow in a separate thread to avoid blocking the event loop
        result = await asyncio.to_thread(process_workflow_query, query, chat_history)

        message_content = result.get("response", "No response generated.")
        retrieved_docs = result.get("retriever_results", [])
        judge_result = result.get("judge_result", True)
        judge_feedback = result.get("judge_feedback", "Not evaluated.")

        sources = []
        for doc in retrieved_docs:
            source = SourceReference(
                document_id=doc.metadata.get("id", "unknown"),
                document_name=doc.metadata.get("source", "unknown"),
                # Use the relevance score attached by the retriever agent, simpler logic if missing
                relevance_score=doc.metadata.get("relevance_score", 0.0),
                content_excerpt=doc.page_content[:5000],  # Limit content length
            )
            sources.append(source)

        processing_time = time.time() - start_time
        log.info(
            f"Query processed successfully in {processing_time:.2f}s. "
            f"Sources: {len(sources)}"
        )

        return ChatResponseModel(
            response=message_content,
            sources=sources,
            judge_result=judge_result,
            judge_feedback=judge_feedback,
        )
    except Exception as e:
        log.exception(f"Error processing query: {e}")
        raise
