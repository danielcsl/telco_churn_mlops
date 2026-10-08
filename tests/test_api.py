from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 1026.0,
}


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert "running" in response.json()["message"].lower()


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["decision_threshold"] == 0.25


def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json=VALID_CUSTOMER,
    )

    body = response.json()

    assert response.status_code == 200
    assert body["churn_prediction"] in ["Yes", "No"]
    assert 0 <= body["churn_probability"] <= 1
    assert body["decision_threshold"] == 0.25
    assert "@champion" in body["model_uri"]


def test_missing_field_is_rejected():
    incomplete_customer = VALID_CUSTOMER.copy()
    incomplete_customer.pop("MonthlyCharges")

    response = client.post(
        "/predict",
        json=incomplete_customer,
    )

    assert response.status_code == 422