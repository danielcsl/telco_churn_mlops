from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src"))

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")

EXPERIMENT_NAME = "telco-churn-baseline"
mlflow.set_experiment(EXPERIMENT_NAME)

from scripts.data.deterministic_cleaning import load_and_clean


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


def create_model_pipeline(X_train):
    """Create an untuned Gradient Boosting pipeline for the search."""

    preprocessor = create_preprocessor(X_train)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                GradientBoostingClassifier(
                    random_state=42,
                ),
            ),
        ]
    )


def create_search(pipeline):
    """Search Gradient Boosting settings using churn recall."""

    parameter_distributions = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__max_depth": [1, 2, 3, 4, 5],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__max_features": [None, "sqrt", "log2"],
    }

    scoring = {
        "churn_recall": make_scorer(
            recall_score,
            pos_label="Yes",
        ),
        "churn_precision": make_scorer(
            precision_score,
            pos_label="Yes",
            zero_division=0,
        ),
        "churn_f1": make_scorer(
            f1_score,
            pos_label="Yes",
            zero_division=0,
        ),
        "roc_auc": "roc_auc",
        "pr_auc": make_scorer(
            average_precision_score,
            needs_proba=True,
            pos_label="Yes",
        ),
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_distributions,
        n_iter=20,
        scoring=scoring,
        refit="churn_recall",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        return_train_score=False,
        verbose=1,
    )


def evaluate_model(pipeline, X_test, y_test):
    """Evaluate the selected model once on the test set."""

    y_pred = pipeline.predict(X_test)

    class_names = list(pipeline.named_steps["model"].classes_)
    churn_class_index = class_names.index("Yes")

    churn_probabilities = pipeline.predict_proba(X_test)[:, churn_class_index]

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
            y_test,
            churn_probabilities,
            pos_label="Yes",
        ),
        "churn_precision": report["Yes"]["precision"],
        "churn_recall": report["Yes"]["recall"],
        "churn_f1": report["Yes"]["f1-score"],
    }

    return metrics, y_pred, report, class_names


def log_evaluation_artifacts(y_test, y_pred, report, class_names):
    """Save and log final evaluation outputs."""

    report_path = RESULTS_DIR / "gb_hyperparameter_report.json"

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

    axis.set_title("Gradient Boosting: Hyperparameter-Tuned Matrix")

    matrix_path = RESULTS_DIR / "gb_hyperparameter_matrix.png"

    figure.savefig(matrix_path, bbox_inches="tight")
    plt.close(figure)

    mlflow.log_artifact(str(matrix_path), artifact_path="evaluation")


if __name__ == "__main__":
    train_df, test_df = load_and_clean()

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]

    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    pipeline = create_model_pipeline(X_train)
    search = create_search(pipeline)

    with mlflow.start_run(
        run_name="gradient-boosting-hyperparameter-tuned",
    ):
        search.fit(X_train, y_train)

        best_pipeline = search.best_estimator_

        metrics, y_pred, report, class_names = evaluate_model(
            best_pipeline,
            X_test,
            y_test,
        )

        best_model = best_pipeline.named_steps["model"]

        mlflow.log_params(
            {
                "model_type": "GradientBoostingClassifier",
                "search_method": "RandomizedSearchCV",
                "search_iterations": 20,
                "cross_validation_folds": 5,
                "selection_metric": "churn_recall",
                "random_state": best_model.random_state,
                "best_n_estimators": best_model.n_estimators,
                "best_learning_rate": best_model.learning_rate,
                "best_max_depth": best_model.max_depth,
                "best_min_samples_split": best_model.min_samples_split,
                "best_min_samples_leaf": best_model.min_samples_leaf,
                "best_subsample": best_model.subsample,
                "best_max_features": best_model.max_features,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            }
        )

        mlflow.log_metrics(
            {
                **metrics,
                "cv_best_churn_recall": search.best_score_,
            }
        )

        mlflow.set_tags(
            {
                "project": "telco-churn-mlops",
                "model_status": "challenger",
                "model_version": "gradient_boosting_hyperparameter_tuned_v1",
                "optimization_goal": "maximize_churn_recall",
            }
        )

        log_evaluation_artifacts(
            y_test,
            y_pred,
            report,
            class_names,
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            name="gradient_boosting_hyperparameter_tuned_pipeline",
            input_example=X_train.head(3),
            serialization_format="cloudpickle",
        )

    print("\n===== Best Cross-Validation Result =====")
    print(f"Best CV churn recall: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")

    print("\n===== Final Test Metrics =====")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nMLflow model logged successfully.")
    print(f"Model URI: {model_info.model_uri}")