# Fraud Detection System

An end-to-end machine learning system for detecting fraudulent transactions
using the IEEE-CIS Fraud Detection dataset (590,000 transactions, 400+ features).

## Project Structure

| Folder | Description |
|---|---|
| notebooks/ | Data cleaning, EDA, feature engineering, model training |
| models/ | Trained XGBoost model and preprocessor files |
| api/ | FastAPI backend for serving predictions |
| frontend/ | HTML/CSS/JS dashboard |
| data/ | Raw data (not committed, too large for GitHub) |

## Results

| Model | AUC-ROC |
|---|---|
| XGBoost | 0.9107 |
| LightGBM | 0.9056 |

## Tech Stack

- Python, Pandas, NumPy
- XGBoost, LightGBM, Scikit-learn
- SMOTE for class imbalance
- SHAP for model explainability
- FastAPI for model serving
- Docker for containerization

## Workflow

Training was done on Kaggle due to dataset size (590K rows, 2.5GB in memory).
The trained model was downloaded and integrated into a FastAPI backend served via Docker.

## Setup

```bash
conda create --prefix ./env python=3.10
conda activate ./env
pip install -r api/requirements.txt
uvicorn api.main:app --reload
```