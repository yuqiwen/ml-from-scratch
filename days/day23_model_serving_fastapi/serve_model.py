from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from train_and_save_model import MODEL_PATH, TinyRegressionModel, load_model


app = FastAPI(title="Tiny Regression Model API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: TinyRegressionModel | None = None


class PredictRequest(BaseModel):
    """
    Input request schema.

    features must contain exactly two numbers.
    """

    features: list[float] = Field(..., min_length=2, max_length=2)


class PredictResponse(BaseModel):
    prediction: float


def load_serving_model(model_path: Path = MODEL_PATH) -> TinyRegressionModel:
    """
    Load model for serving.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run `python train_and_save_model.py` first."
        )

    model = load_model(model_path)
    model = model.to(device)
    model.eval()

    return model


def get_model() -> TinyRegressionModel:
    """
    Lazily load model so imports do not fail before the model file exists.
    """
    global model

    if model is None:
        model = load_serving_model()

    return model


def predict_features(features: list[float]) -> float:
    """
    Core inference logic.

    This function is separated so it can be unit-tested without running FastAPI.
    """
    if len(features) != 2:
        raise ValueError("features must contain exactly two numbers.")

    x = torch.tensor([features], dtype=torch.float32, device=device)

    serving_model = get_model()

    with torch.no_grad():
        y_hat = serving_model(x)

    prediction = float(y_hat.item())

    return prediction


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        prediction = predict_features(request.features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictResponse(prediction=prediction)
