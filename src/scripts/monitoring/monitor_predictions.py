from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

LOG_PATH = (
    PROJECT_ROOT
    / "results"
    / "prediction_logs.jsonl"
)


def load_logs():
    if not LOG_PATH.exists():
        raise FileNotFoundError(
            "No prediction logs found. "
            "Use the /predict endpoint first."
        )

    records = []

    with LOG_PATH.open() as file:
        for line in file:
            records.append(json.loads(line))

    return pd.DataFrame(records)


def calculate_metrics(logs):
    total_predictions = len(logs)

    mean_probability = logs[
        "churn_probability"
    ].mean()

    predicted_churn_rate = (
        logs["churn_prediction"]
        .eq("Yes")
        .mean()
    )

    return {
        "total_predictions": total_predictions,
        "mean_churn_probability": mean_probability,
        "predicted_churn_rate": predicted_churn_rate,
        "model_uri": logs["model_uri"].iloc[-1],
    }


if __name__ == "__main__":
    prediction_logs = load_logs()

    metrics = calculate_metrics(prediction_logs)

    print("\n===== Prediction Monitoring =====")
    print(
        "Total predictions: "
        f"{metrics['total_predictions']}"
    )
    print(
        "Mean churn probability: "
        f"{metrics['mean_churn_probability']:.4f}"
    )
    print(
        "Predicted churn rate: "
        f"{metrics['predicted_churn_rate']:.4f}"
    )
    print(
        "Model URI: "
        f"{metrics['model_uri']}"
    )