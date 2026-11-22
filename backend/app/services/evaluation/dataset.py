"""Mirror eval_set.json into an MLflow evaluation dataset.

eval_set.json stays the source of truth (it is versioned in git); the MLflow dataset is a copy that
the MLflow UI can browse and that each evaluation run is linked to.

From backend/:
    uv run python -m app.services.evaluation.dataset
"""

import json

from mlflow.genai.datasets import EvaluationDataset, create_dataset, search_datasets

from app.services.evaluation.ragas import EVAL_SET

DATASET_NAME = "document-qa-eval-set"


def to_record(item: dict) -> dict:
    return {
        "inputs": {"question": item["question"]},
        "expectations": {"expected_response": item["ground_truth"]},
        "tags": {
            "id": item["id"],
            "category": item["category"],
            "source": item["source"],
        },
    }


def sync_eval_dataset(experiment_id: str) -> EvaluationDataset:
    """Create the dataset if needed and make its records match eval_set.json."""
    found = search_datasets(filter_string=f"name = '{DATASET_NAME}'")
    dataset = (
        found[0]
        if found
        else create_dataset(name=DATASET_NAME, experiment_id=experiment_id)
    )

    records = [to_record(item) for item in EVAL_SET]

    # MLflow matches records by their inputs, so a reworded question would otherwise
    # leave its old version behind
    wanted = {json.dumps(r["inputs"], sort_keys=True) for r in records}
    existing = dataset.to_df()
    stale = [
        row["dataset_record_id"]
        for _, row in existing.iterrows()
        if json.dumps(row["inputs"], sort_keys=True) not in wanted
    ]
    if stale:
        dataset.delete_records(stale)

    return dataset.merge_records(records)


if __name__ == "__main__":
    from app.services.evaluation.run import EXPERIMENT_NAME, use_experiment

    experiment = use_experiment(EXPERIMENT_NAME)
    dataset = sync_eval_dataset(experiment.experiment_id)
    print(
        f"Dataset '{DATASET_NAME}' ({dataset.dataset_id}): {len(dataset.to_df())} records"
    )
    print(
        f"Browse: scripts/mlflow-server.sh, then http://localhost:5001 (experiment '{EXPERIMENT_NAME}', Datasets tab)"
    )
