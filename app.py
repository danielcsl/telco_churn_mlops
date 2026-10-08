from datetime import datetime, timezone
from pathlib import Path
import json

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_NAME = "telco_churn_baseline"

MODEL_URI = f"models:/{MODEL_NAME}@champion"

LOCAL_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "champion_pipeline"
)

DECISION_THRESHOLD = 0.25

PREDICTION_LOG_PATH = (
    PROJECT_ROOT
    / "results"
    / "prediction_logs.jsonl"
)

PREDICTION_LOG_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

if not LOCAL_MODEL_PATH.exists():
    raise FileNotFoundError(
        "Exported champion model was not found at: "
        f"{LOCAL_MODEL_PATH}. "
        "Run: python export_champion_model.py"
    )

model = mlflow.sklearn.load_model(
    str(LOCAL_MODEL_PATH)
)

app = FastAPI(
    title="Telco Churn Prediction API",
    version="1.0.0",
    description=(
        "Predicts customer churn using the exported "
        "MLflow champion Gradient Boosting model."
    ),
)


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    return {
        "message": "Telco churn prediction API is running.",
        "docs_url": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_uri": MODEL_URI,
        "local_model_path": str(LOCAL_MODEL_PATH),
        "decision_threshold": DECISION_THRESHOLD,
    }


@app.post("/predict")
def predict(customer: Customer):
    try:
        customer_data = pd.DataFrame(
            [customer.model_dump()]
        )

        class_names = list(
            model.named_steps["model"].classes_
        )

        churn_class_index = class_names.index("Yes")

        churn_probability = float(
            model.predict_proba(customer_data)[
                0,
                churn_class_index,
            ]
        )

        churn_prediction = (
            "Yes"
            if churn_probability >= DECISION_THRESHOLD
            else "No"
        )

        prediction_log = {
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "model_uri": MODEL_URI,
            "local_model_path": str(LOCAL_MODEL_PATH),
            "decision_threshold": DECISION_THRESHOLD,
            "churn_probability": churn_probability,
            "churn_prediction": churn_prediction,
        }

        with PREDICTION_LOG_PATH.open("a") as file:
            file.write(
                json.dumps(prediction_log)
                + "\n"
            )

        return {
            "churn_prediction": churn_prediction,
            "churn_probability": round(
                churn_probability,
                4,
            ),
            "decision_threshold": DECISION_THRESHOLD,
            "model_uri": MODEL_URI,
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )