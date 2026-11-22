"""
Workflow testing with detailed logging.
Run with: uv run python test_workflow_debug.py
"""

import logging
from app.services.workflow.graph import process_workflow_query

if __name__ == "__main__":
    # Set up detailed logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Test a single query with full logging
    query = "What was Apple's revenue in Q1 2024?"

    print(f"\n{'=' * 60}")
    print(f"Testing Query: {query}")
    print(f"{'=' * 60}\n")

    result = process_workflow_query(query)

    print(f"\n{'=' * 60}")
    print("FINAL RESULT")
    print(f"{'=' * 60}")
    print(f"Company: {result.get('company')}")
    print(f"Query Type: {result.get('query_type')}")
    print(f"Safety Type: {result.get('query_safety_type')}")
    print(f"Documents Retrieved: {len(result.get('retriever_results', []))}")
    print(f"Judge Result: {result.get('judge_result')}")
    print(f"Judge Feedback: {result.get('judge_feedback')}")
    print(f"\nResponse:\n{result.get('response')}")
    print(f"{'=' * 60}\n")
