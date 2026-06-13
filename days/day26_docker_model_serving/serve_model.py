from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from model import TinyRegressionModel


MODEL_PATH = Path(__file__).with_name("regression_model.pt")

app = FastAPI(
    title="Day 26 Dockerized Regression API",
    version="1.0.0",
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class PredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=2,
        max_length=2,
    )


class PredictResponse(BaseModel):
    prediction: float


def load_model(
    path: Path = MODEL_PATH,
) -> TinyRegressionModel:
    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {path}. "
            "Run `python train_model.py` before starting the service."
        )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model = TinyRegressionModel(
        hidden_dim=checkpoint["hidden_dim"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model


model = load_model()


def predict_features(
    features: list[float],
) -> float:
    if len(features) != 2:
        raise ValueError(
            "features must contain exactly two values."
        )

    X = torch.tensor(
        [features],
        dtype=torch.float32,
        device=device,
    )

    with torch.inference_mode():
        prediction = model(X)

    return float(prediction.item())


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "device": device.type,
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
)
def predict(
    request: PredictRequest,
) -> PredictResponse:
    prediction = predict_features(
        request.features
    )

    return PredictResponse(
        prediction=prediction
    )
