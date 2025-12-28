import os
import logging
from langgraph.graph import StateGraph, END

from backend.agents import (
    State, 
    injection_agent, 
    classification_agent, 
    qa_agent, 
    general_agent,
    summarisation_agent,
    retriever_agent,
    judge_agent,
    contextualize_agent,
)

logger = logging.getLogger(__name__)

def route_query(state):
    """Route to appropriate agent based on category."""
    company = state.get("company", "general")
    query_type = state.get("query_type", "unknown")
    logger.info(f"--- Route Query --- Company: {company}")
    
    if query_type in ["summarize"]:
        return "summarisation_agent"
    elif query_type in ["rag"]:
        if company in ["general"]:
            return "general_agent"
        else:
            return "retriever_agent"
    else:
        return "general_agent"

def retrieve_router(state):
    """Route to appropriate agent based on category."""
    query_type = state.get("query_type", "unknown")
    logger.info(f"--- Retrieve Router --- Query Type: {query_type}")
    
    if query_type in ["rag"]:
        return "qa_agent"
    else:
        return "general_agent"    


def injection_router(state):
    """Route to appropriate agent based on query type."""
    query_safety_type = state.get("query_safety_type", "general")
    logger.info(f"--- Injection Router --- Safety Type: {query_safety_type}")
    
    if query_safety_type in ["INJECTION"]:
        return END
    else:
        return "contextualizer"

# Build the graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("injection_agent", injection_agent)
workflow.add_node("contextualizer", contextualize_agent)
workflow.add_node("classifier", classification_agent)
workflow.add_node("judge_agent", judge_agent)
workflow.add_node("qa_agent", qa_agent)
workflow.add_node("general_agent", general_agent)
workflow.add_node("summarisation_agent", summarisation_agent)
workflow.add_node("retriever_agent", retriever_agent)

# Set entry point
workflow.set_entry_point("injection_agent")

workflow.add_conditional_edges(
    "injection_agent",
    injection_router,
    {
        END: END,
        "contextualizer": "contextualizer"
    }
)

workflow.add_edge("contextualizer", "classifier")

# Add conditional routing
workflow.add_conditional_edges(
    "classifier",
    route_query,
    {
        "retriever_agent": "retriever_agent",
        "general_agent": "general_agent",
        "summarisation_agent": "summarisation_agent"
    }
)

# All agents end the workflow
workflow.add_conditional_edges(
    "retriever_agent",
    retrieve_router,
    {
        "qa_agent": "qa_agent",
        "general_agent": "general_agent"
    }
)

workflow.add_edge("qa_agent", "judge_agent")
workflow.add_edge("summarisation_agent", "judge_agent")
workflow.add_edge("general_agent", END)

workflow.add_edge("judge_agent", END)

# Compile the graph

app = workflow.compile()

# Write graph to an image file
try:
    png_graph = app.get_graph().draw_mermaid_png()
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "workflow_graph.png"), "wb") as f:
        f.write(png_graph)
except Exception as e:
    print(f"Could not generate workflow graph image: {e}")


def process_workflow_query(query: str, chat_history: list[dict] = []):
    """Process a query through the workflow."""
    result = app.invoke({"query": query, "chat_history": chat_history})
    return result