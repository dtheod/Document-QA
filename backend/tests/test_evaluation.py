"""Offline tests for the evaluation run's refusal check and metric summary."""

import math
from types import SimpleNamespace

import pandas as pd
import pytest

import mlflow

from app.services.evaluation import dataset as dataset_module
from app.services.evaluation.ragas import EVAL_SET, METRIC_NAMES
from app.services.evaluation.run import index_chunker, is_refusal, summarise


@pytest.mark.parametrize(
    "answer",
    [
        "I don't know. The provided context does not include any information about Tesla.",
        "I don’t have enough context to answer that question.",
        "No documents found for this company.",
        "The document does not disclose the number of iPhone units sold.",
        "This information is not reported in the provided documents.",
    ],
)
def test_detects_refusals(answer):
    assert is_refusal(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "Apple reported net income of $84,544 million for the first nine months of 2025.",
        "Uber's free cash flow for Q1 2024 was $1,359 million.",
    ],
)
def test_factual_answers_are_not_refusals(answer):
    assert not is_refusal(answer)


def test_summarise_averages_per_category_and_scores_refusals():
    results = pd.DataFrame(
        {
            "category": ["table_number", "table_number", "no_answer", "off_topic"],
            "refused": [False, False, True, False],
            **{name: [1.0, 0.5, 0.0, float("nan")] for name in METRIC_NAMES},
        }
    )

    metrics = summarise(results)

    assert metrics["answer_correctness/table_number"] == 0.75
    assert math.isclose(metrics["answer_correctness"], 0.5)  # NaN rows are skipped
    assert "answer_correctness/off_topic" not in metrics  # all-NaN metrics are dropped
    assert metrics["refusal_accuracy"] == 0.5  # only no_answer and off_topic count


def fake_vector_store(metadatas):
    collection = SimpleNamespace(get=lambda include: {"metadatas": metadatas})
    return SimpleNamespace(_collection=collection)


def test_index_chunker_reports_what_the_index_was_built_with():
    assert index_chunker(fake_vector_store([{"chunker": "docling"}])) == "docling"
    # Chunks indexed before the chunker was recorded came from the recursive splitter
    assert index_chunker(fake_vector_store([{"company": "apple"}])) == "recursive"
    assert (
        index_chunker(fake_vector_store([{"chunker": "docling"}, {"company": "uber"}]))
        == "mixed"
    )
    assert index_chunker(fake_vector_store([])) == "empty"


def test_eval_set_entries_are_complete():
    ids = [item["id"] for item in EVAL_SET]
    assert len(ids) == len(set(ids))
    for item in EVAL_SET:
        assert item["question"] and item["ground_truth"] and item["category"]


def test_eval_dataset_mirrors_the_eval_set(tmp_path, monkeypatch):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment_id = mlflow.create_experiment(
        "test", artifact_location=tmp_path.as_uri()
    )
    items = [
        {
            "id": f"q-{i}",
            "category": "fact_lookup",
            "question": f"Question {i}?",
            "ground_truth": f"Answer {i}.",
            "source": "x.pdf",
        }
        for i in range(3)
    ]
    monkeypatch.setattr(dataset_module, "EVAL_SET", items)
    dataset_module.sync_eval_dataset(experiment_id)

    # Reword one question, change one answer and drop one item
    items = [
        {**items[0], "question": "Question 0, reworded?"},
        {**items[1], "ground_truth": "New."},
    ]
    monkeypatch.setattr(dataset_module, "EVAL_SET", items)
    records = dataset_module.sync_eval_dataset(experiment_id).to_df()

    assert sorted(r["question"] for r in records["inputs"]) == [
        "Question 0, reworded?",
        "Question 1?",
    ]
    by_id = {t["id"]: e for t, e in zip(records["tags"], records["expectations"])}
    assert by_id == {
        "q-0": {"expected_response": "Answer 0."},
        "q-1": {"expected_response": "New."},
    }
