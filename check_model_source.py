from pathlib import Path

import mlflow
from mlflow import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parent
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
)

mlflow.set_registry_uri(
    f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
)

MODEL_NAME = "telco_churn_baseline"

client = MlflowClient()

champion = client.get_model_version_by_alias(
    name=MODEL_NAME,
    alias="champion",
)

print(f"Model name: {MODEL_NAME}")
print(f"Champion version: {champion.version}")
print(f"Run ID: {champion.run_id}")
print(f"Source URI: {champion.source}")
print(f"Status: {champion.status}")