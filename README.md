# Fraud Detection AI System

[![CI](https://github.com/IbtissamBukhari/fraud-detection-system/actions/workflows/ci.yml/badge.svg)](https://github.com/IbtissamBukhari/fraud-detection-system/actions/workflows/ci.yml)

An end-to-end machine learning system that scores card transactions for fraud risk in real time and explains every prediction. An XGBoost model trained on 590,540 real transactions from the IEEE-CIS dataset is served through a FastAPI backend, with SHAP explanations and a browser dashboard for analysts.

**Live demo:** https://fraud-detection-system-wplk.onrender.com
**API docs:** https://fraud-detection-system-wplk.onrender.com/docs

Hosted on Render's free plan: if the app has been idle, the first request can take up to a minute while it wakes up.

---

## What it does

- Takes a transaction (amount, product code, card network and type, email domain, time) and returns a fraud probability, a risk level and a confidence label
- Shows the ten features that pushed the score up or down, using SHAP TreeExplainer
- Logs every prediction for review and CSV export
- Runs as a single Docker container that serves both the API and the dashboard

## Results

| Metric | Value |
|:--|:--|
| Model | XGBoost (LightGBM compared, AUC 0.9056) |
| AUC-ROC on held-out test set | **0.9107** |
| Dataset | IEEE-CIS Fraud Detection, 590,540 transactions |
| Train / test split | 472,432 / 118,108 (stratified 80/20) |
| Fraud rate | 3.5% |
| Features | 106 (from 434 raw) |
| Class imbalance | SMOTE on the training set only |

## Machine learning pipeline

The four notebooks in `notebooks/` cover the full workflow.

**1. Data cleaning.** Merged the transaction and identity tables, cut memory use from 2.5 GB to 1.68 GB by downcasting types, dropped 12 columns with more than 90% missing values, and filled the rest (median for numbers, `unknown` for categories).

**2. Exploratory analysis.** Confirmed the 96.5 / 3.5 class split and found strong risk signals: `mail.com` purchasers have an 18.9% fraud rate, Discover credit cards 7.9%, and product code C 11.7%, against a 3.5% baseline.

**3. Feature engineering.** Added hour, day and week features from the transaction clock, a log-transformed amount, and binary flags for high-risk email domains and card combinations. Kept the 20 most informative of 339 anonymised V features, for 106 features in total.

**4. Model training.** Stratified split, label encoding fitted on training data only, SMOTE applied after the split to avoid leakage, then XGBoost and LightGBM trained with early stopping. XGBoost was selected and explained with SHAP.

## API

| Method | Endpoint | Description |
|:--:|:--|:--|
| GET | `/health` | Health check |
| POST | `/predict` | Score a transaction |
| GET | `/history` | Recent predictions |
| GET | `/history/export` | Prediction history as CSV |
| GET | `/categories` | Accepted values for categorical inputs |
| GET | `/model-info` | Model metadata and performance |

Example request:

```json
POST /predict
{
  "TransactionAmt": 150.00,
  "ProductCD": "W",
  "card4": "visa",
  "card6": "debit",
  "P_emaildomain": "gmail.com",
  "transaction_hour": 14,
  "transaction_day": 3,
  "transaction_week": 10
}
```

Example response:

```json
{
  "fraud_probability": 0.0878,
  "risk_level": "Low",
  "confidence": "High Confidence",
  "top_features": [
    {"feature": "TransactionAmt", "shap_value": -0.3126},
    {"feature": "C9", "shap_value": 0.1011}
  ]
}
```

Input rules: product codes are `W, H, C, S, R` (anonymised in the dataset); day index is 0-6 and week index 0-26, following the dataset's relative clock; email domains not seen in training are treated as unknown. Invalid input returns HTTP 422.

## Run locally

Requires Python 3.10.

```bash
git clone https://github.com/IbtissamBukhari/fraud-detection-system.git
cd fraud-detection-system
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r api/requirements.txt
uvicorn api.main:app --reload
```

Open http://127.0.0.1:8000 for the dashboard and http://127.0.0.1:8000/docs for the interactive API docs.

### With Docker

```bash
docker compose up --build
```

The app runs on http://localhost:8000 and prediction history is kept in a Docker volume.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

GitHub Actions runs the test suite and a Docker build with a health-check smoke test on every push.

## Deployment

The repository includes a `render.yaml` blueprint. On [Render](https://render.com), choose **New > Blueprint**, connect this repository, and Render builds the Dockerfile and checks `/health`. The container reads the port from `$PORT`, so it also runs on other container hosts.

On the free plan the service sleeps when idle and the first request can take up to a minute. Prediction history is stored on the container's disk and resets on redeploy.

## Project structure

```
fraud-detection-system/
├── api/
│   ├── main.py            FastAPI app and endpoints
│   ├── model.py           Model loading, preprocessing, prediction, history
│   ├── encoding.py        Maps raw categorical values to model codes
│   ├── schemas.py         Request and response validation
│   └── requirements.txt
├── frontend/              Dashboard (HTML, CSS, JavaScript), served by FastAPI
├── models/                Trained model, encoders, feature list, feature medians
├── notebooks/             Cleaning, EDA, feature engineering, training
├── tests/                 Pytest suite for encoding and API
├── .github/workflows/     CI pipeline
├── Dockerfile
├── docker-compose.yml
└── render.yaml
```

## Design decisions

**SMOTE after the split.** Oversampling before splitting would leak synthetic copies of test rows into training and inflate the score.

**Training medians as defaults.** The dashboard collects 8 inputs, but the model uses 106 features. Unsupplied features are filled with their training medians so the model sees a realistic transaction rather than an all-zero vector.

**No feature scaling.** Tree-based models are invariant to monotonic scaling.

**Explicit category mapping.** The encoding cell in the training notebook was run twice, so the saved encoders expect integer codes rather than raw strings like `visa`. `api/encoding.py` rebuilds the exact mapping and is covered by tests, and the notebook now has a guard that stops the cell from running twice. Retraining from the notebook would remove the need for the mapping.

## Limitations

- Most of the 106 features are filled with medians, so predictions reflect a typical transaction with the given inputs, not a full customer profile
- Product codes and V features are anonymised in the source data, so their real-world meaning is unknown
- History is a JSON file; a production system would use a database such as PostgreSQL

## Tech stack

Python, Pandas, NumPy, scikit-learn, XGBoost, LightGBM, imbalanced-learn (SMOTE), SHAP, FastAPI, Pydantic, Uvicorn, HTML/CSS/JavaScript, Docker, GitHub Actions, Pytest

## Author

**Syed Ali Ibtissam Bukhari**
[GitHub](https://github.com/IbtissamBukhari) · [LinkedIn](https://www.linkedin.com/in/syed-ali-ibtissam-bukhari-50438028a)

## License

MIT
