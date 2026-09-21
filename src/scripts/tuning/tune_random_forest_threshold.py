from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src"))

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")

EXPERIMENT_NAME = "telco-churn-baseline"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from scripts.data.deterministic_cleaning import load_and_clean


CHAMPION_RECALL = 0.7834224598930482
THRESHOLDS = [round(value / 100, 2) for value in range(10, 91, 5)]


def load_tuned_run_params(model_version):
    """Load parameters from the latest matching MLflow tuning run."""

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError(
            f"MLflow experiment '{EXPERIMENT_NAME}' was not found."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.model_version = '{model_version}'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        raise ValueError(
            f"No MLflow run found for model_version='{model_version}'."
        )

    source_run_id = runs.iloc[0]["run_id"]
    source_run = mlflow.get_run(source_run_id)

    return {
        "run_id": source_run_id,
        "params": source_run.data.params,
    }


def parse_max_features(value):
    """Convert MLflow's string parameter to sklearn's expected type."""

    if value == "None":
        return None

    try:
        numeric_value = float(value)

        if numeric_value.is_integer():
            return int(numeric_value)

        return numeric_value

    except ValueError:
        return value


def create_preprocessor(X_train):
    """Create preprocessing for numeric and categorical features."""

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

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def create_model_pipeline(X_train, params):
    """Build a Random Forest using MLflow-loaded tuned parameters."""

    preprocessor = create_preprocessor(X_train)

    max_depth = params["best_max_depth"]

    if max_depth == "None":
        max_depth = None
    else:
        max_depth = int(max_depth)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=int(params["best_n_estimators"]),
                    max_depth=max_depth,
                    min_samples_split=int(
                        params["best_min_samples_split"]
                    ),
                    min_samples_leaf=int(
                        params["best_min_samples_leaf"]
                    ),
                    max_features=parse_max_features(
                        params["best_max_features"]
                    ),
                    class_weight=params["class_weight"],
                    random_state=int(params["random_state"]),
                    n_jobs=int(params["n_jobs"]),
                ),
            ),
        ]
    )


def select_threshold(pipeline, X_train, y_train):
    """Select the threshold with best precision above champion recall."""

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    cv_probabilities = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )

    churn_probabilities = cv_probabilities[:, 1]

    candidates = []

    for threshold in THRESHOLDS:
        predictions = [
            "Yes" if probability >= threshold else "No"
            for probability in churn_probabilities
        ]

        recall = recall_score(
            y_train,
            predictions,
            pos_label="Yes",
        )

        precision = precision_score(
            y_train,
            predictions,
            pos_label="Yes",
            zero_division=0,
        )

        if recall > CHAMPION_RECALL:
            candidates.append(
                {
                    "threshold": threshold,
                    "cv_recall": recall,
                    "cv_precision": precision,
                }
            )

    if not candidates:
        raise ValueError(
            "No threshold exceeded champion recall during CV."
        )

    return max(
        candidates,
        key=lambda candidate: candidate["cv_precision"],
    )


def evaluate_model(pipeline, X_test, y_test, threshold):
    """Evaluate the final fitted pipeline at the selected threshold."""

    class_names = list(pipeline.named_steps["model"].classes_)
    churn_class_index = class_names.index("Yes")

    churn_probabilities = pipeline.predict_proba(X_test)[
        :, churn_class_index
    ]

    y_pred = [
        "Yes" if probability >= threshold else "No"
        for probability in churn_probabilities
    ]

    report = classification_report(
        y_test,
        y_pred,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, churn_probabilities),
        "pr_auc": average_precision_score(
            y_test == "Yes",
            churn_probabilities,
        ),
        "churn_precision": report["Yes"]["precision"],
        "churn_recall": report["Yes"]["recall"],
        "churn_f1": report["Yes"]["f1-score"],
    }

    return metrics, y_pred, report, class_names


def log_evaluation_artifacts(y_test, y_pred, report, class_names):
    """Save and log final evaluation artifacts."""

    report_path = (
        RESULTS_DIR
        / "random_forest_hyperparameter_threshold_report.json"
    )

    with report_path.open("w") as file:
        json.dump(report, file, indent=4)

    mlflow.log_artifact(
        str(report_path),
        artifact_path="evaluation",
    )

    figure, axis = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=class_names,
        ax=axis,
        cmap="Blues",
    )

    axis.set_title(
        "Random Forest: Hyperparameter + Threshold-Tuned Matrix"
    )

    matrix_path = (
        RESULTS_DIR
        / "random_forest_hyperparameter_threshold_matrix.png"
    )

    figure.savefig(matrix_path, bbox_inches="tight")
    plt.close(figure)

    mlflow.log_artifact(
        str(matrix_path),
        artifact_path="evaluation",
    )


if __name__ == "__main__":
    mlflow.set_experiment(EXPERIMENT_NAME)

    train_df, test_df = load_and_clean()

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]

    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    source_run = load_tuned_run_params(
        "random_forest_hyperparameter_tuned_v1"
    )

    print("\n===== Loaded Hyperparameter Run =====")
    print(f"Source run ID: {source_run['run_id']}")

    pipeline = create_model_pipeline(
        X_train,
        source_run["params"],
    )

    threshold_result = select_threshold(
        pipeline,
        X_train,
        y_train,
    )

    selected_threshold = threshold_result["threshold"]

    pipeline.fit(X_train, y_train)

    metrics, y_pred, report, class_names = evaluate_model(
        pipeline,
        X_test,
        y_test,
        selected_threshold,
    )

    model = pipeline.named_steps["model"]

    model_params = {
        "model_type": "RandomForestClassifier",
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "min_samples_split": model.min_samples_split,
        "min_samples_leaf": model.min_samples_leaf,
        "max_features": str(model.max_features),
        "class_weight": str(model.class_weight),
        "random_state": model.random_state,
        "n_jobs": model.n_jobs,
        "decision_threshold": selected_threshold,
        "threshold_selection": (
            "5-fold CV; recall > champion; max precision"
        ),
        "source_hyperparameter_run_id": source_run["run_id"],
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }

    with mlflow.start_run(
        run_name="random-forest-hyperparameter-threshold-tuned"
    ):
        mlflow.log_params(model_params)

        mlflow.log_metrics(
            {
                **metrics,
                "cv_churn_recall": threshold_result["cv_recall"],
                "cv_churn_precision": threshold_result["cv_precision"],
            }
        )

        mlflow.set_tags(
            {
                "project": "telco-churn-mlops",
                "model_status": "challenger",
                "model_version": (
                    "random_forest_hyperparameter_threshold_tuned_v1"
                ),
                "optimization_goal": (
                    "recall_above_current_champion"
                ),
            }
        )

        log_evaluation_artifacts(
            y_test,
            y_pred,
            report,
            class_names,
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name=(
                "random_forest_hyperparameter_"
                "threshold_tuned_pipeline"
            ),
            input_example=X_train.head(3),
            serialization_format="cloudpickle",
        )

    print("\n===== Final Model Parameters =====")
    for parameter_name, parameter_value in model_params.items():
        print(f"{parameter_name}: {parameter_value}")

    print("\n===== Threshold Selection =====")
    print(f"Selected threshold: {selected_threshold:.2f}")
    print(f"CV churn recall: {threshold_result['cv_recall']:.4f}")
    print(f"CV churn precision: {threshold_result['cv_precision']:.4f}")

    print("\n===== Test Metrics =====")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nMLflow model logged successfully.")
    print(f"Model URI: {model_info.model_uri}")