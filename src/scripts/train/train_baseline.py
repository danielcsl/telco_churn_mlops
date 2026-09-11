from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Find the root directory of the project.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Allow train.py to import files from src/.
sys.path.append(str(PROJECT_ROOT / "src"))

# Configure MLflow paths inside this project.
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACTS_PATH = PROJECT_ROOT / "mlartifacts"

MLFLOW_ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")

experiment_name = "telco-churn-baseline"

try:
    mlflow.create_experiment(
        experiment_name,
        artifact_location=f"file:///{MLFLOW_ARTIFACTS_PATH.as_posix()}",
    )
except mlflow.exceptions.MlflowException:
    pass

mlflow.set_experiment(experiment_name)

# Store temporary files here before logging them as MLflow artifacts.
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from scripts.data.deterministic_cleaning import load_and_clean


def create_preprocessor(X_train):
    """Create transformations for numeric and categorical input columns."""

    numerical_columns = X_train.select_dtypes(
        include=["int64", "float64"],
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["object", "string"],
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    return preprocessor


def create_model_pipeline(X_train):
    """Combine learned preprocessing and Logistic Regression."""

    preprocessor = create_preprocessor(X_train)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    return pipeline


def train_baseline_model():
    """Load, clean, split features/target, then fit the model pipeline."""

    train_df, test_df = load_and_clean()

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]

    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    pipeline = create_model_pipeline(X_train)

    pipeline.fit(X_train, y_train)

    return pipeline, X_train, X_test, y_test


def evaluate_model(pipeline, X_test, y_test):
    """Calculate test-set metrics and create evaluation outputs."""

    y_pred = pipeline.predict(X_test)

    class_names = list(pipeline.named_steps["model"].classes_)

    churn_class_index = class_names.index("Yes")

    y_churn_probability = pipeline.predict_proba(X_test)[:, churn_class_index]

    report = classification_report(
        y_test,
        y_pred,
        labels=class_names,
        output_dict=True,
    )

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_churn_probability),
        "pr_auc": average_precision_score(
            y_test == "Yes",
            y_churn_probability,
        ),
        "churn_precision": report["Yes"]["precision"],
        "churn_recall": report["Yes"]["recall"],
        "churn_f1": report["Yes"]["f1-score"],
    }

    return metrics, y_pred, report, class_names


def log_evaluation_artifacts(y_test, y_pred, report, class_names):
    """Save and log a JSON report and confusion-matrix image."""

    report_path = RESULTS_DIR / "classification_report.json"

    with report_path.open("w") as file:
        json.dump(report, file, indent=4)

    mlflow.log_artifact(str(report_path), artifact_path="evaluation")

    figure, axis = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=class_names,
        ax=axis,
        cmap="Blues",
    )

    axis.set_title("Telco Churn Confusion Matrix")

    matrix_path = RESULTS_DIR / "confusion_matrix.png"

    figure.savefig(matrix_path, bbox_inches="tight")

    plt.close(figure)

    mlflow.log_artifact(str(matrix_path), artifact_path="evaluation")


if __name__ == "__main__":
    
    with mlflow.start_run(run_name="logistic-regression-balanced"):
        pipeline, X_train, X_test, y_test = train_baseline_model()

        metrics, y_pred, report, class_names = evaluate_model(
            pipeline,
            X_test,
            y_test,
        )

        model = pipeline.named_steps["model"]

        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "max_iter": model.max_iter,
                "class_weight": str(model.class_weight),
                "random_state": model.random_state,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            }
        )

        mlflow.log_metrics(metrics)

        mlflow.set_tags(
            {
                "project": "telco-churn-mlops",
                "dataset": "blastchar/telco-customer-churn",
                "data_stage": "processed",
                "model_status": "baseline",
                "model_version": "baseline_v1",
            }
        )

        log_evaluation_artifacts(
            y_test,
            y_pred,
            report,
            class_names,
        )

        input_example = X_train.head(3)

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="telco_churn_pipeline",
            input_example = input_example,
            serialization_format="cloudpickle",
        )

        print("\n===== Test Metrics =====")

        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

        print("\nMLflow model logged successfully.")
        print(f"Model URI: {model_info.model_uri}")