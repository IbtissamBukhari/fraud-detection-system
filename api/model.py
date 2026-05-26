import joblib
import numpy as np
import pandas as pd
import json
import os
import shap
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

model = joblib.load(MODELS_DIR / "fraud_model.pkl")
encoders = joblib.load(MODELS_DIR / "encoders.pkl")
feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")


explainer = shap.TreeExplainer(model)

HISTORY_FILE = BASE_DIR / "predictions_history.json"

def save_to_history(record: dict):
    # load existing history if file exists, otherwise start with empty list
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
    else:
        history = []

    # append new record and save
    history.append(record)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def preprocess_input(data: dict) -> pd.DataFrame:
    # start with a row of zeros for all 106 features
    row = pd.DataFrame([np.zeros(len(feature_columns))], columns=feature_columns)

    # fill in the values the user provided
    row['TransactionAmt'] = data['TransactionAmt']
    row['log_transaction_amt'] = np.log1p(data['TransactionAmt'])
    row['transaction_hour'] = data['transaction_hour']
    row['transaction_day'] = data['transaction_day']
    row['transaction_week'] = data['transaction_week']

    # encode categorical fields using the same encoders from training
    cat_fields = ['ProductCD', 'card4', 'card6', 'P_emaildomain']

    for field in cat_fields:
        value = str(data[field])
        known_categories = set(encoders[field].classes_)
        if value not in known_categories:
            value = 'unknown'
        row[field] = encoders[field].transform([value])[0]

    return row


def predict(data: dict) -> dict:
    row = preprocess_input(data)

    # get fraud probability from model
    fraud_probability = model.predict_proba(row)[0][1]

    # determine risk level from probability
    if fraud_probability >= 0.7:
        risk_level = "High"
    elif fraud_probability >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # calculate SHAP values for this transaction
    shap_values = explainer.shap_values(row)

    # sort features by absolute SHAP value
    feature_importance = list(zip(feature_columns, shap_values[0]))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

    # take top 10 and format for frontend
    top_features = []
    for f, v in feature_importance[:10]:
        top_features.append({
            "feature": f,
            "shap_value": round(float(v), 4)
        })

   # determine confidence based on how far probability is from the 0.5 boundary
    distance = abs(fraud_probability - 0.5)
    if distance >= 0.3:
        confidence = "High Confidence"
    elif distance >= 0.15:
        confidence = "Medium Confidence"
    else:
        confidence = "Low Confidence - Review Manually"

    # build history record and save it
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "transaction_amount": round(float(data['TransactionAmt']), 2),
        "fraud_probability": round(float(fraud_probability), 4),
        "risk_level": risk_level,
        "confidence": confidence
    }
    save_to_history(record)

    return {
        "fraud_probability": round(float(fraud_probability), 4),
        "risk_level": risk_level,
        "confidence": confidence,
        "top_features": top_features
    }