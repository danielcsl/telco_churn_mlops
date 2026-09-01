from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    roc_auc_score,
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# locating the deterministic_cleaning.py for funnction import
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / 'src'))

from scripts.data.deterministic_cleaning import load_and_clean

# create preprocess pipeline based on training feature types:
def create_preprocessor(X_train):

    numerical_columns = X_train.select_dtypes(
        include = ['int64', 'float64']
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include = ['object']
    ).columns.tolist()

    # setup the data cleaning pipeline
    numeric_pipeline = Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
        ]
    )

    # assemble the pipelines
    preprocessor = ColumnTransformer(
        transformers = [
            ('numeric', numeric_pipeline, numerical_columns),
            ('categorical', categorical_pipeline, categorical_columns),
        ]
    )

    return preprocessor

# assembling the preprocessor and logistic regression training

def create_model_pipeline(X_train):

    preprocessor = create_preprocessor(X_train)

    model_pipeline = Pipeline(
        steps = [
            ('preprocessor', preprocessor),
            ('model', 
             LogisticRegression(
                max_iter = 1000,
                class_weight='balanced',
                random_state = 42,
                ),
            ),
        ]
    )

    return model_pipeline

# function to train the base model
def train_baseline_model():
    
    train_df, test_df = load_and_clean()

    X_train = train_df.drop(columns = ['Churn'])
    y_train = train_df['Churn']

    X_test = test_df.drop(columns = ['Churn'])
    y_test = test_df['Churn']

    pipeline = create_model_pipeline(X_train)

    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test, len(X_train)

# model evaluation
def evaluate_model(pipeline, X_test, y_test):
    """Evaluate the fitted pipeline on the held-out test set."""

    y_pred = pipeline.predict(X_test)

    class_names = list(pipeline.named_steps["model"].classes_)

    churn_class_index = class_names.index("Yes")

    y_churn_probability = pipeline.predict_proba(X_test)[:, churn_class_index]

    report = classification_report(
        y_test,
        y_pred,
        labels = class_names,
        output_dict= True,
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

    print("\n===== Test Metrics =====")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return metrics, y_pred, report, class_names

if __name__ == "__main__":
    mlflow.sklearn.autolog(
        log_models = True,
        exclusive = False,
    )

    with mlflow.start_run(run_name="logistic-regression-balanced"):
        pipeline, X_test, y_test, train_rows = train_baseline_model()

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
                "train_rows": train_rows,
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
            }
        )

        with open("classification_report.json", "w") as file:
            json.dump(report, file, indent=4)

        mlflow.log_artifact("classification_report.json")

        figure, axis = plt.subplots(figsize=(6, 5))

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            labels=class_names,
            ax=axis,
            cmap="Blues",
        )

        figure.savefig("confusion_matrix.png", bbox_inches="tight")

        plt.close(figure)

        mlflow.log_artifact("confusion_matrix.png")

    print("starting MLFLOW model Logging..")
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
    )
    print("MLFLOW model logged")