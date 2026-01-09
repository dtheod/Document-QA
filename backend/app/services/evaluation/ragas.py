import logging
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import asyncio
import concurrent.futures
from app.services.workflow.graph import process_workflow_query

logger = logging.getLogger(__name__)

# Reusing the test data as ground truth for now
TEST_QUESTIONS = [
    "What was Apple's net income for the first nine months ended June 28, 2025?",
    "What was Apple's net income for the first nine months ended on June 29, 2024?",
    # "Wie vielle Modelle umfasst das Hybridangebot der Cayenne-Reihe von Porsche?",
    # "What was the basic net loss per share attributable to Uber Technologies, Inc. in the first quarter of 2024?",
    # "What is Ubers free Cash flow for 2024?",
    # "What is Ubers revenue for delivery in Q1 2023?",
    # "How many new markets in Video advertising did Uber enter?",
    # "How is the weather today?",
    # "What is the net income of Tesla for 2024?",
    # "Give me a financial summary for BMW?"
]

TEST_ANSWERS = [
    "Apple reported net income of $84,544 million for the first nine months ended June 28, 2025.",
    "Apple's net income for the first nine months ended on June 29, 2024, totaled $79,000 million",
    # "Das Hybridangebot der Cayenne-Reihe von Porsche umfasst insgesamt drei Modelle",
    # "The basic net loss per share attributable to Uber Technologies, Inc. in the first quarter of 2024 was $0.31",
    # "Ubers free cash flow for 2024 was $1,359 millions",
    # "The revenue for Uber's delivery segment in Q1 2023 was $3,093 million (or $3.093 billion)",
    # "Uber expanded video Journey Ads to new markets including Australia, Brazil, and Chile, which indicates entry into 3 new markets",
    # "I'm sorry, but I don't have enough information to answer that",
    # "I don't know. The provided context does not include any information about Tesla's net income for 2024.",
    # "No documents found for this company."
]


async def run_ragas_evaluation():
    try:
        logger.info("Initializing RAGAS evaluation...")

        llm = ChatOpenAI(temperature=0, model=os.getenv("OPENAI_MODEL_NAME"))
        embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL"), api_key=os.getenv("OPENAI_API_KEY")
        )

        results = []

        # Run workflow for each question
        # This part is async (process_workflow_query uses LangGraph which is sync/async?)
        # process_workflow_query is currently SYNC (app.invoke).
        # So we can run this loop here without issues if it doesn't block too much,
        # or we might want to offload the whole thing.
        # However, the error specifically came from ragas.evaluate.

        for query in TEST_QUESTIONS:
            logger.info(f"Processing evaluation query: {query}")
            # process_workflow_query is a sync function wrapping app.invoke
            result_gen = process_workflow_query(query)
            print("RESULT GENERATOR")
            print(result_gen)

            # Extract contexts
            contexts = []
            if "retriever_results" in result_gen and isinstance(
                result_gen["retriever_results"], list
            ):
                contexts = [
                    doc.page_content if hasattr(doc, "page_content") else str(doc)
                    for doc in result_gen["retriever_results"]
                ]

            results.append(
                {
                    "question": query,
                    "answer": result_gen.get("response", "No response"),
                    "contexts": contexts,
                }
            )

        data = {
            "question": [r["question"] for r in results],
            "contexts": [r["contexts"] for r in results],
            "answer": [r["answer"] for r in results],
            "ground_truth": TEST_ANSWERS,
        }

        dataset = Dataset.from_dict(data)

        logger.info("Running RAGAS evaluate...")

        # Define a wrapper for the sync execution of ragas.evaluate
        def evaluate_wrapper():
            return evaluate(
                dataset=dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    answer_correctness,
                    context_precision,
                    context_recall,
                ],
                llm=llm,
                embeddings=embeddings,
            )

        # Run in a separate thread
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            scores = await loop.run_in_executor(pool, evaluate_wrapper)

        # Structure the output for the frontend
        df = scores.to_pandas()

        output_results = []
        for index, row in df.iterrows():
            output_results.append(
                {
                    "question": row["user_input"],
                    "response": row["response"],
                    "faithfulness": row["faithfulness"],
                    "answer_relevancy": row["answer_relevancy"],
                    "answer_correctness": row["answer_correctness"],
                    "context_precision": row["context_precision"],
                    "context_recall": row["context_recall"],
                }
            )

        return output_results

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}", exc_info=True)
        raise e


if __name__ == "__main__":
    asyncio.run(run_ragas_evaluation())
