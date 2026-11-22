"""
Quick workflow testing script.
Run with: uv run python test_workflow.py
"""

from app.services.workflow.graph import process_workflow_query
import dotenv

dotenv.load_dotenv()

if __name__ == "__main__":
    # Test queries
    test_queries = [
        "What was Apple's revenue in 2024?",
    ]

    print("=" * 60)
    print("WORKFLOW TESTING")
    print("=" * 60)

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 60)

        result = process_workflow_query(query)

        # Print key state information
        print(f"🏢 Company: {result.get('company', 'N/A')}")
        print(f"🎯 Query Type: {result.get('query_type', 'N/A')}")
        print(f"🔒 Safety: {result.get('query_safety_type', 'N/A')}")
        print(f"📚 Docs Retrieved: {len(result.get('retriever_results', []))}")
        print(f"✅ Judge Valid: {result.get('judge_result', 'N/A')}")
        print(f"\n💬 Response: {result.get('response', 'N/A')[:200]}...")
        print("=" * 60)
