from pathlib import Path
import shutil

import mlflow
import mlflow.sklearn


PROJECT_ROOT = Path(__file__).resolve().parent

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
)

mlflow.set_registry_uri(
    f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
)

MODEL_URI = "models:/telco_churn_baseline@champion"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "models"
    / "champion_pipeline"
)

if OUTPUT_PATH.exists():
    shutil.rmtree(OUTPUT_PATH)

model = mlflow.sklearn.load_model(MODEL_URI)

mlflow.sklearn.save_model(
    sk_model=model,
    path=str(OUTPUT_PATH),
    serialization_format=(
        mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
    ),
)

print("Champion model exported successfully.")
print(f"Source URI: {MODEL_URI}")
print(f"Saved path: {OUTPUT_PATH}")