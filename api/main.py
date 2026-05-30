from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api.schemas import TransactionInput, PredictionResponse
from api.model import predict, HISTORY_FILE
import json
import os

app = FastAPI(
    title="Fraud Detection API",
    description="Predicts whether a transaction is fraudulent using XGBoost",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,          # must be False when allow_origins=["*"]
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,                     # cache preflight 1 hr — stops repeated OPTIONS requests
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/history")
def get_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)


@app.get("/model-info")
def model_info():
    return {
        "model": "XGBoost",
        "auc_roc": 0.9107,
        "training_data": "IEEE-CIS Fraud Detection",
        "training_rows": 590540,
        "features_used": 106,
        "fraud_rate_in_training": "3.50%",
        "top_features": ["TransactionAmt", "M4", "V308", "log_transaction_amt", "V258"],
    }


@app.get("/history/export")
def export_history():
    from fastapi.responses import StreamingResponse
    import csv
    import io

    if not os.path.exists(HISTORY_FILE):
        return JSONResponse({"error": "No history found"}, status_code=404)

    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)

    if not history:
        return JSONResponse({"error": "History is empty"}, status_code=404)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions_history.csv"},
    )


@app.post("/predict", response_model=PredictionResponse)
def make_prediction(transaction: TransactionInput):
    result = predict(transaction.model_dump())
    return result