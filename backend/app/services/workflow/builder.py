from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END

from app.models.schemas import State
from app.services.workflow.agents import WorkflowAgents
import logging

logger = logging.getLogger(__name__)

# --- Router Functions ---
# Moved here to avoid circular imports between graph.py and builder.py


def route_query(state: State):
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


def retrieve_router(state: State):
    """Route to appropriate agent based on category."""
    query_type = state.get("query_type", "unknown")
    logger.info(f"--- Retrieve Router --- Query Type: {query_type}")

    if query_type in ["rag"]:
        return "qa_agent"
    else:
        return "general_agent"


def injection_router(state: State):
    """Route to appropriate agent based on query type."""
    query_safety_type = state.get("query_safety_type", "general")
    logger.info(f"--- Injection Router --- Safety Type: {query_safety_type}")

    if query_safety_type in ["INJECTION"]:
        return END
    else:
        return "contextualizer"


class GraphBuilder:
    def __init__(self, llm: BaseChatModel):
        # Instantiate agents container
        self.agents = WorkflowAgents(llm)

    def build(self):
        """Builds and compiles the workflow graph with injected dependencies."""

        # Build the graph
        workflow = StateGraph(State)

        # Add nodes using bound methods
        workflow.add_node("injection_agent", self.agents.injection_agent)
        workflow.add_node("contextualizer", self.agents.contextualize_agent)
        workflow.add_node("classifier", self.agents.classification_agent)
        workflow.add_node("judge_agent", self.agents.judge_agent)
        workflow.add_node("qa_agent", self.agents.qa_agent)
        workflow.add_node("general_agent", self.agents.general_agent)
        workflow.add_node("summarisation_agent", self.agents.summarisation_agent)
        workflow.add_node("retriever_agent", self.agents.retriever_agent)

        # Set entry point
        workflow.set_entry_point("injection_agent")

        workflow.add_conditional_edges(
            "injection_agent",
            injection_router,
            {END: END, "contextualizer": "contextualizer"},
        )

        workflow.add_edge("contextualizer", "classifier")

        workflow.add_conditional_edges(
            "classifier",
            route_query,
            {
                "retriever_agent": "retriever_agent",
                "general_agent": "general_agent",
                "summarisation_agent": "summarisation_agent",
            },
        )

        workflow.add_conditional_edges(
            "retriever_agent",
            retrieve_router,
            {"qa_agent": "qa_agent", "general_agent": "general_agent"},
        )

        workflow.add_edge("qa_agent", "judge_agent")
        workflow.add_edge("summarisation_agent", "judge_agent")
        workflow.add_edge("general_agent", END)
        workflow.add_edge("judge_agent", END)

        return workflow.compile()
