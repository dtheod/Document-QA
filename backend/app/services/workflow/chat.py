import time
import logging
import asyncio

from app.models.schemas import ChatResponseModel, SourceReference
from app.services.workflow.graph import process_workflow_query
from app.utils import page_number

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
        judge_result = result.get("judge_result")
        judge_feedback = result.get("judge_feedback")

        sources = []
        for doc in retrieved_docs:
            source = SourceReference(
                document_id=doc.metadata.get("id", "unknown"),
                document_name=doc.metadata.get("source", "unknown"),
                page=page_number(doc.metadata),
                # Relevance score attached by the retriever agent (absent for summaries)
                relevance_score=doc.metadata.get("relevance_score"),
                content_excerpt=doc.page_content[:500],  # Limit content length
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
