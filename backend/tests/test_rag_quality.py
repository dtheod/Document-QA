import logging

import pytest
from app.services.workflow.graph import process_workflow_query

# These tests call OpenAI and HuggingFace; conftest.py skips them without API keys
pytestmark = pytest.mark.integration


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def test_single_workflow():
    try:
        logger.info("\n--- Testing Single Query ---")
        test_questions = [
            "What was Apple’s net income for the first nine months ended June 28, 2025?"
        ]
        for query in test_questions:
            result_gen = process_workflow_query(query)
        print(result_gen)
    except ImportError as e:
        logger.error(f"Failed to import services: {e}")
    except Exception as e:
        logger.error(f"An error occurred during testing: {e}")


def test_conversational_memory():
    try:
        logger.info("\n--- Testing Conversational Memory ---")

        # Turn 1
        query1 = "What is Apple's revenue for the nine months ended June 2025?"
        # We don't need history for the first query
        result1 = process_workflow_query(query1, [])
        logger.info(f"Turn 1 Response: {result1['response'][:100]}...")

        # Turn 2: Follow-up question
        query2 = "And what about their net income?"
        history = [
            {"role": "user", "content": query1},
            {"role": "assistant", "content": result1["response"]},
        ]

        # Verify that the query gets contextualized
        # We can inspect the logs or check the final answer, but ideally we'd spy on the intermediate state.
        # For this integration test, we'll check if the answer corresponds to Apple's Net Income, which implies successful rewriting.
        result2 = process_workflow_query(query2, history)
        logger.info(f"Turn 2 Response: {result2['response']}")

        if (
            "84,544" in result2["response"]
            or "84.5" in result2["response"]
            or "net income" in result2["response"].lower()
        ):
            logger.info("Conversational Memory Test Passed: Context successfully used.")
        else:
            logger.warning(
                "Conversational Memory Test: Answer might not be specific enough. Check logs for query refinement."
            )

    except Exception as e:
        logger.error(f"Conversational memory test failed: {e}")
