from pathlib import Path
import time

import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from metrics import MetricsStore
from model import TinyRegressionModel


MODEL_PATH = Path(__file__).with_name("regression_model.pt")

app = FastAPI(
    title="Day 27 Monitored Model Service",
    version="1.0.0",
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

metrics_store = MetricsStore()
model: TinyRegressionModel | None = None


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
            "Run `python train_model.py` first."
        )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    loaded_model = TinyRegressionModel(
        hidden_dim=checkpoint["hidden_dim"]
    )

    loaded_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    loaded_model = loaded_model.to(device)
    loaded_model.eval()

    return loaded_model


@app.on_event("startup")
def startup_event() -> None:
    """
    Load model when the FastAPI service starts.
    """
    global model

    model = load_model()


@app.middleware("http")
async def monitoring_middleware(
    request: Request,
    call_next,
):
    """
    Record total request latency and success/failure.
    """
    start_time = time.perf_counter()

    try:
        response = await call_next(request)

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        if response.status_code < 400:
            metrics_store.record_success(
                latency_ms
            )
        else:
            metrics_store.record_failure(
                latency_ms
            )

        return response

    except Exception:
        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        metrics_store.record_failure(
            latency_ms
        )

        raise


def predict_features(
    features: list[float],
) -> float:
    if model is None:
        raise RuntimeError(
            "Model is not loaded."
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
    """
    Liveness check:
    confirms the HTTP service is alive.
    """
    return {
        "status": "alive",
    }


@app.get("/ready")
def ready() -> dict[str, bool | str]:
    """
    Readiness check:
    confirms the model is loaded.
    """
    is_ready = model is not None

    return {
        "ready": is_ready,
        "device": device.type,
    }


@app.get("/metrics")
def metrics() -> dict[str, float | int]:
    return metrics_store.snapshot()


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
