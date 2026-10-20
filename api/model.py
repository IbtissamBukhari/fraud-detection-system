import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from api.encoding import encode, high_risk_card, high_risk_email

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
HISTORY_FILE = Path(os.getenv("HISTORY_PATH", BASE_DIR / "predictions_history.json"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "500"))

model = joblib.load(MODELS_DIR / "fraud_model.pkl")
encoders = joblib.load(MODELS_DIR / "encoders.pkl")
feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
# Training medians are used for every feature the user does not supply, so the
# model sees a realistic transaction rather than an all-zero vector.
feature_medians = joblib.load(MODELS_DIR / "feature_medians.pkl")

explainer = shap.TreeExplainer(model)

CATEGORICAL_INPUTS = ["ProductCD", "card4", "card6", "P_emaildomain"]

_history_lock = threading.Lock()


def load_history() -> list:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_to_history(record: dict) -> None:
    with _history_lock:
        history = load_history()
        history.append(record)
        history = history[-MAX_HISTORY:]
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)


def preprocess_input(data: dict) -> pd.DataFrame:
    row = {col: feature_medians.get(col, 0.0) for col in feature_columns}

    row["TransactionAmt"] = data["TransactionAmt"]
    row["log_transaction_amt"] = float(np.log1p(data["TransactionAmt"]))
    row["transaction_hour"] = data["transaction_hour"]
    row["transaction_day"] = data["transaction_day"]
    row["transaction_week"] = data["transaction_week"]

    for field in CATEGORICAL_INPUTS:
        row[field] = encode(field, data[field], encoders)

    # Engineered flags from notebook 03, derived from the user's input
    row["high_risk_email"] = high_risk_email(data["P_emaildomain"])
    row["high_risk_card"] = high_risk_card(data["card4"], data["card6"])

    return pd.DataFrame([row], columns=feature_columns)


def risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "High"
    if probability >= 0.4:
        return "Medium"
    return "Low"


def confidence_label(probability: float) -> str:
    # Distance from the 0.5 decision boundary
    distance = abs(probability - 0.5)
    if distance >= 0.3:
        return "High Confidence"
    if distance >= 0.15:
        return "Medium Confidence"
    return "Low Confidence - Review Manually"


def predict(data: dict) -> dict:
    row = preprocess_input(data)
    probability = float(model.predict_proba(row)[0][1])

    shap_values = explainer.shap_values(row)
    ranked = sorted(zip(feature_columns, shap_values[0]), key=lambda x: abs(x[1]), reverse=True)
    top_features = [{"feature": f, "shap_value": round(float(v), 4)} for f, v in ranked[:10]]

    result = {
        "fraud_probability": round(probability, 4),
        "risk_level": risk_level(probability),
        "confidence": confidence_label(probability),
        "top_features": top_features,
    }

    save_to_history({
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "transaction_amount": round(float(data["TransactionAmt"]), 2),
        "fraud_probability": result["fraud_probability"],
        "risk_level": result["risk_level"],
        "confidence": result["confidence"],
    })
    return result
