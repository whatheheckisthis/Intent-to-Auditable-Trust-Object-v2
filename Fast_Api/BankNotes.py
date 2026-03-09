"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field


class BankNoteFeatures(BaseModel):
    """Input payload used to classify a bank note."""

    variance: float = Field(..., description="Variance of wavelet transformed image")
    skewness: float = Field(..., description="Skewness of wavelet transformed image")
    curtosis: float = Field(..., description="Curtosis of wavelet transformed image")
    entropy: float = Field(..., description="Entropy of image")


class PredictionResponse(BaseModel):
    """Prediction payload returned by the API."""

    prediction: str
    label: int
