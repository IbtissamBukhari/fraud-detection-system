import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

BASE = {
    "TransactionAmt": 150.0,
    "ProductCD": "W",
    "card4": "visa",
    "card6": "debit",
    "P_emaildomain": "gmail.com",
    "transaction_hour": 14,
    "transaction_day": 3,
    "transaction_week": 10,
}


def predict(**overrides):
    response = client.post("/predict", json={**BASE, **overrides})
    assert response.status_code == 200, response.text
    return response.json()


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_frontend_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "Fraud Detection AI" in response.text


def test_predict_response_shape():
    body = predict()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["risk_level"] in {"Low", "Medium", "High"}
    assert len(body["top_features"]) == 10


def test_categorical_inputs_change_the_prediction():
    # Regression test: these inputs used to be ignored
    low = predict(card4="visa", card6="debit", P_emaildomain="gmail.com", ProductCD="W")
    high = predict(card4="discover", card6="credit", P_emaildomain="mail.com", ProductCD="C")
    assert low["fraud_probability"] != high["fraud_probability"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"transaction_day": 7},
        {"transaction_hour": 24},
        {"TransactionAmt": 0},
        {"card4": "rupay"},
        {"ProductCD": "Z"},
    ],
)
def test_invalid_input_is_rejected(overrides):
    assert client.post("/predict", json={**BASE, **overrides}).status_code == 422


def test_history_and_export():
    predict()
    history = client.get("/history").json()
    assert history and history[-1]["transaction_amount"] == 150.0

    export = client.get("/history/export")
    assert export.status_code == 200
    assert export.headers["content-type"].startswith("text/csv")
    assert "fraud_probability" in export.text.splitlines()[0]


def test_categories_and_model_info():
    categories = client.get("/categories").json()
    assert "gmail.com" in categories["P_emaildomain"]
    info = client.get("/model-info").json()
    assert info["auc_roc"] == 0.9107
    assert info["features_used"] == 106
