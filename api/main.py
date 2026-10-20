import csv
import io

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api.encoding import RAW_CATEGORIES
from api.model import BASE_DIR, load_history, predict
from api.schemas import PredictionResponse, TransactionInput

FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="Fraud Detection API",
    description="Scores card transactions for fraud risk with XGBoost and explains each prediction with SHAP.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def make_prediction(transaction: TransactionInput):
    return predict(transaction.model_dump())


@app.get("/history")
def get_history():
    return load_history()


@app.get("/history/export")
def export_history():
    history = load_history()
    if not history:
        raise HTTPException(status_code=404, detail="No prediction history yet")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(history[0].keys()))
    writer.writeheader()
    writer.writerows(history)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions_history.csv"},
    )


@app.get("/categories")
def categories():
    """Accepted values for the categorical inputs, used to fill the dashboard form."""
    return {
        "ProductCD": RAW_CATEGORIES["ProductCD"],
        "card4": [v for v in RAW_CATEGORIES["card4"] if v != "unknown"],
        "card6": [v for v in RAW_CATEGORIES["card6"] if v != "unknown"],
        "P_emaildomain": [v for v in RAW_CATEGORIES["P_emaildomain"] if v != "unknown"],
    }


@app.get("/model-info")
def model_info():
    return {
        "model": "XGBoost",
        "auc_roc": 0.9107,
        "training_data": "IEEE-CIS Fraud Detection",
        "dataset_rows": 590540,
        "training_rows": 472432,
        "test_rows": 118108,
        "features_used": 106,
        "fraud_rate": "3.50%",
        "top_features": ["TransactionAmt", "M4", "V308", "log_transaction_amt", "V258"],
    }
