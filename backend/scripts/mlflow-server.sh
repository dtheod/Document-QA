#!/bin/sh
# MLflow UI and tracking server for the eval runs and the app's request traces.
# Open http://localhost:5001
#   --allowed-hosts lets the backend container send traces via host.docker.internal
#   port 5001 because macOS uses 5000 for the AirPlay Receiver
cd "$(dirname "$0")/.." || exit 1
exec uv run mlflow server \
  --backend-store-uri sqlite:///data/mlflow/mlflow.db \
  --artifacts-destination data/mlflow/artifacts \
  --host 127.0.0.1 --port 5001 \
  --allowed-hosts "localhost:*,127.0.0.1:*,host.docker.internal:*"
