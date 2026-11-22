import json
import logging
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
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
import pandas as pd
from app.core.config import settings
from app.services.workflow.graph import process_workflow_query

logger = logging.getLogger(__name__)

# Eval set over apple.pdf, uber.pdf and volkswagen.pdf; ground truths cite their source page
EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
EVAL_SET = json.loads(EVAL_SET_PATH.read_text())

METRICS = [
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall,
]
METRIC_NAMES = [metric.name for metric in METRICS]


def answer_eval_item(item: dict) -> dict:
    """Run one eval-set question through the workflow and return the item with its answer."""
    logger.info(f"Processing evaluation query: {item['question']}")
    result = process_workflow_query(item["question"])

    contexts = []
    if isinstance(result.get("retriever_results"), list):
        contexts = [
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in result["retriever_results"]
        ]

    return {
        **item,
        "answer": result.get("response", "No response"),
        "contexts": contexts,
        "judge_valid": result.get("judge_result"),
    }


def score_with_ragas(rows: list[dict]) -> pd.DataFrame:
    """Score answered rows with RAGAS. Returns one row per input row, in the same order."""
    logger.info("Running RAGAS evaluate...")
    # The configured model only accepts its default temperature and may reject n > 1,
    # so stop RAGAS from overriding either
    llm = LangchainLLMWrapper(
        ChatOpenAI(model=settings.openai_model_name, api_key=settings.openai_api_key),
        bypass_temperature=True,
        bypass_n=True,
    )
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )
    dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in rows],
            "contexts": [r["contexts"] for r in rows],
            "answer": [r["answer"] for r in rows],
            "ground_truth": [r["ground_truth"] for r in rows],
        }
    )
    scores = evaluate(dataset=dataset, metrics=METRICS, llm=llm, embeddings=embeddings)
    return scores.to_pandas()


async def run_ragas_evaluation():
    try:
        logger.info("Initializing RAGAS evaluation...")

        # process_workflow_query is sync (app.invoke), so run it off the event loop
        rows = await asyncio.to_thread(
            lambda: [answer_eval_item(item) for item in EVAL_SET]
        )

        # ragas.evaluate runs its own event loop, so it has to run in a separate thread
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            df = await loop.run_in_executor(pool, score_with_ragas, rows)

        # Structure the output for the frontend
        output_results = []
        for index, row in df.iterrows():
            output_results.append(
                {
                    "question": row["user_input"],
                    "response": row["response"],
                    # RAGAS gives NaN when it cannot score an answer; NaN is not valid JSON
                    **{
                        name: None if pd.isna(row[name]) else row[name]
                        for name in METRIC_NAMES
                    },
                }
            )

        return output_results

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}", exc_info=True)
        raise e


if __name__ == "__main__":
    asyncio.run(run_ragas_evaluation())
