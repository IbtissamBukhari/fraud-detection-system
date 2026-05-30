# Fraud Detection AI System

An end-to-end machine learning system that detects fraudulent financial transactions in real time. Built with XGBoost, FastAPI, and a clean editorial frontend. Trained on the IEEE-CIS Fraud Detection dataset from Kaggle.

---

## Project Overview

This project demonstrates a complete machine learning pipeline from raw data to a deployed API with a browser-based dashboard. It covers data cleaning, exploratory data analysis, feature engineering, model training with class imbalance handling, model explainability using SHAP, and a production-ready REST API.

The system allows a fraud analyst to enter transaction details, receive an instant fraud probability score, understand which features drove the prediction, and review past predictions — all from a single-page dashboard.

---

## Live Demo

Frontend: Open `frontend/index.html` in your browser after starting the API.

API Documentation: `http://127.0.0.1:8000/docs`

---

## Dataset

Source: IEEE-CIS Fraud Detection (Kaggle Competition)

- 590,540 transactions
- 434 raw features across transaction and identity tables
- 3.5% fraud rate (severe class imbalance)
- 180-day time period

The dataset is not included in this repository due to size. Download it from Kaggle and place the CSV files in the `data/` folder.

---

## Machine Learning Pipeline

### Notebook 1 — Data Cleaning
- Merged transaction and identity tables on TransactionID
- Reduced memory usage from 2.5GB to 1.68GB using dtype optimization
- Dropped 12 columns with more than 90% missing values
- Filled numerical NaN with median, categorical NaN with unknown

### Notebook 2 — Exploratory Data Analysis
- Confirmed severe class imbalance: 96.5% legitimate, 3.5% fraud
- Found high-risk email domains: mail.com at 18.9% fraud rate
- Found high-risk card combinations: Discover credit at 7.93% fraud rate
- Found ProductCD C has 11.7% fraud rate versus 3.5% average
- Identified top V features by correlation with fraud label

### Notebook 3 — Feature Engineering
- Created time features: transaction_hour, transaction_day, transaction_week
- Created log_transaction_amt to handle skewed amount distribution
- Created high_risk_email and high_risk_card binary flags
- Dropped 319 low-signal V features, kept top 20 by correlation
- Final feature set: 106 features

### Notebook 4 — Model Training
- Train/test split: 80/20 stratified to preserve fraud rate
- Applied SMOTE on training set only to balance classes
- Trained XGBoost and LightGBM with identical settings
- XGBoost AUC-ROC: 0.9107 (selected)
- LightGBM AUC-ROC: 0.9056
- Generated SHAP explanations using TreeExplainer
- Saved model, encoders, feature columns, and feature medians

---

## Model Performance

| Metric | Value |
|:-------------------------|:-------------------------:|
| Algorithm | XGBoost |
| AUC-ROC Score | 0.9107 |
| Training Rows | 590,540 |
| Features Used | 106 |
| Fraud Rate in Training | 3.50% |
| Class Balancing | SMOTE |

---

## API Endpoints

| Method | Endpoint | Description |
|:-----:|:------------------------------|:--------------------------------|
| GET | /health | API health check |
| POST | /predict | Predict fraud for a transaction |
| GET | /history | Retrieve prediction history |
| GET | /model-info | Model metadata and performance |
| GET | /history/export | Download history as CSV |

### Predict Request Body

```json
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

### Predict Response

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

---

## Project Structure
fraud-detection-system/
├── api/
│   ├── init.py
│   ├── main.py                  FastAPI application and endpoints
│   ├── model.py                 Model loading, preprocessing, prediction
│   ├── schemas.py               Pydantic request and response schemas
│   └── requirements.txt         Python dependencies
├── frontend/
│   ├── index.html               Single-page dashboard
│   ├── style.css                Editorial cream and orange theme
│   └── app.js                   API integration and UI logic
├── models/
│   ├── fraud_model.pkl          Trained XGBoost model
│   ├── encoders.pkl             Label encoders for categorical features
│   ├── feature_columns.pkl      Ordered list of 106 feature names
│   └── feature_medians.pkl      Training medians for inference defaults
├── notebooks/
│   ├── 01-data-cleaning.ipynb
│   ├── 02-eda.ipynb
│   ├── 03-feature-engineering.ipynb
│   └── 04-model-training.ipynb
├── data/
│   └── .gitkeep
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
---

## Running Locally

### Prerequisites

- Python 3.10
- Conda or virtualenv
- Git

### Setup

Clone the repository:

```bash
git clone https://github.com/IbtissamBukhari/fraud-detection-system.git
cd fraud-detection-system
```

Create and activate conda environment:

```bash
conda create --prefix ./env python=3.10
conda activate ./env
```

Install dependencies:

```bash
pip install -r api/requirements.txt
```

Start the API:

```bash
uvicorn api.main:app --reload
```

Open the frontend by opening `frontend/index.html` in your browser.

---

## Running with Docker

Build and start the container:

```bash
docker-compose up --build
```

API will be available at `http://localhost:8000`.

Open `frontend/index.html` in your browser to use the dashboard.

---

## Running in GitHub Codespaces

Open this repository on GitHub and click Code, then Codespaces, then Create codespace on main.

Once the Codespace loads, run:

```bash
docker-compose up --build
```

Codespaces will detect port 8000 and offer to open it in the browser.

---

## Tech Stack

| Layer | Technology |
|:------|:-----------------------------------------|
| Model | XGBoost, LightGBM, Scikit-learn |
| Explainability | SHAP TreeExplainer |
| Class Balancing | SMOTE (imbalanced-learn) |
| API | FastAPI, Uvicorn |
| Validation | Pydantic |
| Frontend | HTML, CSS, Vanilla JavaScript |
| Containerization | Docker, Docker Compose |
| Data Processing | Pandas, NumPy |
| Notebooks | Jupyter (Kaggle) |
| Version Control | Git, GitHub |

---

## Key Design Decisions

**SMOTE on training set only** — applying SMOTE to the test set would cause data leakage and inflate performance metrics. SMOTE is applied only after the train/test split.

**Feature medians as inference defaults** — the model was trained on real feature distributions. Using training medians for unknown features at inference time ensures the model sees a realistic input rather than an anomalous all-zero vector.

**No feature scaling** — XGBoost and LightGBM are tree-based models that are scale invariant. Standardization is not needed and was intentionally skipped.

**Label encoding after train/test split** — encoders are fitted on training data only, then applied to both train and test. This prevents test data from influencing the encoding.

**JSON file for prediction history** — a flat JSON file is used for simplicity. A production system would use a proper database such as PostgreSQL.

---

## Author

Ibtissam Bukhari

GitHub: https://github.com/IbtissamBukhari

---

## License

MIT License. Free to use for learning and portfolio purposes.