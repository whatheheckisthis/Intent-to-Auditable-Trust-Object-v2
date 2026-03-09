"""FastAPI application for bank note authenticity prediction."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException

from Fast_Api.BankNotes import BankNoteFeatures, PredictionResponse

APP_NAME = "Banknote Authentication API"
APP_VERSION = "2.0.0"
MODEL_PATH = Path(__file__).resolve().parent / "classifier.pkl"

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Predict whether a bank note is genuine or fake using a trained classifier.",
)

classifier: Any | None = None
model_load_error: str | None = None


def load_model(path: Path) -> Any:
    """Load and return a pickle-serialized model."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")

    with path.open("rb") as model_file:
        return pickle.load(model_file)


try:
    classifier = load_model(MODEL_PATH)
except Exception as exc:  # pragma: no cover - startup guard
    model_load_error = str(exc)


@app.get("/", tags=["meta"])
def index() -> dict[str, str]:
    return {
        "message": "Welcome to the Banknote Authentication API",
        "docs": "/docs",
    }


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model_loaded": "true" if classifier is not None else "false",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict_banknote(data: BankNoteFeatures) -> PredictionResponse:
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not available. "
                f"Load error: {model_load_error or 'unknown error'}"
            ),
        )

    features = [[data.variance, data.skewness, data.curtosis, data.entropy]]

    try:
        raw_prediction = int(classifier.predict(features)[0])
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    label = 1 if raw_prediction > 0 else 0
    prediction = "Fake note" if label == 1 else "Genuine note"
    return PredictionResponse(prediction=prediction, label=label)


if __name__ == "__main__":
    uvicorn.run("Fast_Api.app:app", host="0.0.0.0", port=8000, reload=True)
