import logging
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.services.workflow.builder import GraphBuilder

logger = logging.getLogger(__name__)

# Initialize LLM
# This should ideally be passed from a config or dependency container
llm = ChatOpenAI(model=settings.openai_model_name, api_key=settings.openai_api_key)

# Initialize Builder
builder = GraphBuilder(llm)

# Compile the graph
app = builder.build()


def process_workflow_query(query: str, chat_history: list[dict] = []):
    """Process a query through the workflow."""
    result = app.invoke({"query": query, "chat_history": chat_history})
    return result
