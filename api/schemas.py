from pydantic import BaseModel, Field
from typing import Optional
class TransactionInput(BaseModel):
    TransactionAmt: float = Field(gt=0, description="Transaction amount in USD")
    ProductCD: str = Field(default="W", description="Product code")
    card4: str = Field(default="visa", description="Card network")
    card6: str = Field(default="debit", description="Card type")
    P_emaildomain: str = Field(default="gmail.com", description="Purchaser email domain")
    transaction_hour: int = Field(default=12, ge=0, le=23, description="Hour of transaction")
    transaction_day: int = Field(default=0, ge=0, le=6, description="Day of week")
    transaction_week: int = Field(default=0, ge=0, description="Week number")

class PredictionResponse(BaseModel):
    fraud_probability: float
    risk_level: str
    confidence: str
    top_features: list

