import os
import logging
from langchain_openai import ChatOpenAI
from app.services.workflow.builder import GraphBuilder

logger = logging.getLogger(__name__)

# Initialize LLM
# This should ideally be passed from a config or dependency container
llm = ChatOpenAI(temperature=0, model=os.getenv("OPENAI_MODEL_NAME"))

# Initialize Builder
builder = GraphBuilder(llm)

# Compile the graph
app = builder.build()

# Write graph to an image file
try:
    png_graph = app.get_graph().draw_mermaid_png()
    # Navigate from app/services/workflow/ to backend/data/outputs
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "data",
        "outputs",
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "workflow_graph.png"), "wb") as f:
        f.write(png_graph)
except Exception as e:
    print(f"Could not generate workflow graph image: {e}")


def process_workflow_query(query: str, chat_history: list[dict] = []):
    """Process a query through the workflow."""
    result = app.invoke({"query": query, "chat_history": chat_history})
    return result
