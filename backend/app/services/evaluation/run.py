"""Run the eval set through the workflow, score it with RAGAS and log the run to MLflow.

From backend/:
    uv run python -m app.services.evaluation.run                                  # run an evaluation
    uv run python -m app.services.evaluation.dataset                              # only sync the dataset
    scripts/mlflow-server.sh                                                      # browse runs at http://localhost:5001
"""

import hashlib
import logging
import re
from pathlib import Path

import mlflow
import pandas as pd

from app.core.config import settings
from app.services.evaluation.dataset import sync_eval_dataset
from app.services.evaluation.ragas import (
    EVAL_SET,
    EVAL_SET_PATH,
    METRIC_NAMES,
    answer_eval_item,
    score_with_ragas,
)
from app.services.rag.indexing import chunking_params, get_retriever
from app.services.workflow.agents import RETRIEVER_K

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[3]
MLFLOW_DIR = BACKEND_DIR / "data" / "mlflow"
TRACKING_URI = f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}"
EXPERIMENT_NAME = "document-qa-eval"

# Questions the system should decline to answer
REFUSAL_CATEGORIES = {"no_answer", "off_topic"}
# Phrases the QA, general and summarisation agents use when they decline to answer
REFUSAL_PATTERN = re.compile(
    r"don't know|do not know|(don't|do not) have enough|not enough (context|information)"
    r"|no documents found"
    r"|(does|do) not (contain|include|mention|provide|report|disclose|specify)"
    r"|not (available|provided|mentioned|reported|disclosed|specified) in",
    re.IGNORECASE,
)


def is_refusal(answer: str) -> bool:
    """Heuristic: does the answer decline to answer rather than state a fact?"""
    return bool(REFUSAL_PATTERN.search(answer.replace("’", "'")))


def _sha(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def index_chunker(vector_store) -> str:
    """Chunker the index was built with; chunks from before it was recorded are recursive."""
    metadatas = vector_store._collection.get(include=["metadatas"])["metadatas"]
    chunkers = {m.get("chunker", "recursive") for m in metadatas if m}
    if not chunkers:
        return "empty"
    return chunkers.pop() if len(chunkers) == 1 else "mixed"


def use_experiment(name: str):
    """Point MLflow at the local store and select (or create) an experiment."""
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(
            name, artifact_location=(MLFLOW_DIR / "artifacts").as_uri()
        )
    return mlflow.set_experiment(name)


def run_params() -> dict:
    """Settings that explain why two runs differ."""
    _, vector_store = get_retriever()
    return {
        "llm_model": settings.openai_model_name,  # also the RAGAS judge
        "embedding_model": settings.embedding_model,
        "retriever_k": RETRIEVER_K,
        # Settings of the chunker the index was built with, assuming they have not changed since
        **chunking_params(index_chunker(vector_store)),
        "prompts_sha": _sha(*sorted((BACKEND_DIR / "prompts").glob("*.yml"))),
        "eval_set_sha": _sha(EVAL_SET_PATH),
        "eval_set_size": len(EVAL_SET),
    }


def summarise(results: pd.DataFrame) -> dict:
    """Mean of each RAGAS metric overall and per category, plus refusal accuracy."""
    metrics = {}
    for name in METRIC_NAMES:
        metrics[name] = results[name].mean()
        for category, mean in results.groupby("category")[name].mean().items():
            metrics[f"{name}/{category}"] = mean

    should_refuse = results[results["category"].isin(REFUSAL_CATEGORIES)]
    if not should_refuse.empty:
        metrics["refusal_accuracy"] = should_refuse["refused"].mean()

    # A metric RAGAS could not compute for any row comes back as NaN
    return {key: float(value) for key, value in metrics.items() if pd.notna(value)}


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    experiment = use_experiment(EXPERIMENT_NAME)
    dataset = sync_eval_dataset(experiment.experiment_id)

    with mlflow.start_run() as run:
        mlflow.log_params(run_params())
        mlflow.log_artifact(str(EVAL_SET_PATH))
        mlflow.log_input(dataset, context="evaluation")

        # Trace only the workflow; RAGAS's own LLM calls would bury it
        mlflow.langchain.autolog()
        rows = []
        for item in EVAL_SET:
            row = answer_eval_item(item)
            row["trace_id"] = mlflow.get_last_active_trace_id()
            rows.append(row)
        mlflow.langchain.autolog(disable=True)

        scores = score_with_ragas(rows)

        results = pd.DataFrame(rows)
        results[METRIC_NAMES] = scores[METRIC_NAMES].to_numpy()
        results["refused"] = results["answer"].map(is_refusal)
        results["contexts"] = results["contexts"].map("\n---\n".join)

        metrics = summarise(results)
        mlflow.log_metrics(metrics)
        mlflow.log_text(results.to_csv(index=False), "results.csv")

    print(f"\nMLflow run {run.info.run_id} in experiment '{EXPERIMENT_NAME}'")
    for name in [*METRIC_NAMES, "refusal_accuracy"]:
        value = (
            f"{metrics[name]:.3f}"
            if name in metrics
            else "not computed (see errors above)"
        )
        print(f"  {name:<20} {value}")
    print("\nBrowse runs: scripts/mlflow-server.sh, then open http://localhost:5001")


if __name__ == "__main__":
    main()
