from typing import List, Literal

from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    TransactionAmt: float = Field(gt=0, le=1_000_000, description="Transaction amount in USD")
    ProductCD: Literal["W", "H", "C", "S", "R"] = Field(
        default="W", description="Product code (anonymised in the IEEE-CIS dataset)"
    )
    card4: Literal["visa", "mastercard", "discover", "american express"] = Field(
        default="visa", description="Card network"
    )
    card6: Literal["debit", "credit", "debit or credit", "charge card"] = Field(
        default="debit", description="Card type"
    )
    P_emaildomain: str = Field(
        default="gmail.com",
        max_length=100,
        description="Purchaser email domain. Domains not seen in training are treated as unknown.",
    )
    transaction_hour: int = Field(default=12, ge=0, le=23, description="Hour of day (0-23)")
    transaction_day: int = Field(default=0, ge=0, le=6, description="Day index within the week (0-6)")
    transaction_week: int = Field(
        default=0, ge=0, le=26, description="Week index (training data covers weeks 0-26)"
    )


class FeatureImpact(BaseModel):
    feature: str
    shap_value: float


class PredictionResponse(BaseModel):
    fraud_probability: float
    risk_level: Literal["Low", "Medium", "High"]
    confidence: str
    top_features: List[FeatureImpact]
