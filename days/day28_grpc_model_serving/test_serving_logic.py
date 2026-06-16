from pathlib import Path

import grpc
import torch

import model_pb2
from grpc_server import ModelService
from model import TinyRegressionModel
from train_model import MODEL_PATH, save_model


class FakeContext:
    """
    Minimal fake gRPC context for unit tests.
    """

    def abort(
        self,
        code: grpc.StatusCode,
        details: str,
    ) -> None:
        raise ValueError(
            f"{code.name}: {details}"
        )


def ensure_model_exists() -> None:
    if MODEL_PATH.exists():
        return

    model = TinyRegressionModel(hidden_dim=32)
    save_model(model)


def load_test_service() -> ModelService:
    ensure_model_exists()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = TinyRegressionModel(
        hidden_dim=checkpoint["hidden_dim"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return ModelService(
        model=model,
        device=torch.device("cpu"),
    )


def test_predict_returns_response() -> None:
    service = load_test_service()

    request = model_pb2.PredictRequest(
        features=[1.0, 2.0]
    )

    response = service.Predict(
        request,
        FakeContext(),
    )

    print("Test 1: Predict returns response")
    print(f"prediction = {response.prediction}")

    assert isinstance(
        response,
        model_pb2.PredictResponse,
    )
    assert isinstance(
        response.prediction,
        float,
    )

    print("Passed.\n")


def test_predict_rejects_wrong_length() -> None:
    service = load_test_service()

    request = model_pb2.PredictRequest(
        features=[1.0]
    )

    print("Test 2: invalid request rejected")

    try:
        service.Predict(
            request,
            FakeContext(),
        )
        raised = False
    except ValueError:
        raised = True

    assert raised

    print("Passed.\n")


def test_health_returns_device() -> None:
    service = load_test_service()

    request = model_pb2.HealthRequest()

    response = service.Health(
        request,
        FakeContext(),
    )

    print("Test 3: Health response")
    print(
        f"status={response.status}, "
        f"device={response.device}"
    )

    assert response.status == "ok"
    assert response.device == "cpu"

    print("Passed.\n")


def test_model_is_eval_mode() -> None:
    service = load_test_service()

    print("Test 4: model is in eval mode")

    assert service.model.training is False

    print("Passed.\n")


def test_inference_does_not_track_grad() -> None:
    service = load_test_service()

    X = torch.tensor(
        [[1.0, 2.0]],
        requires_grad=True,
    )

    with torch.inference_mode():
        output = service.model(X)

    print("Test 5: inference mode")

    assert output.requires_grad is False

    print("Passed.\n")


def main() -> None:
    test_predict_returns_response()
    test_predict_rejects_wrong_length()
    test_health_returns_device()
    test_model_is_eval_mode()
    test_inference_does_not_track_grad()

    print("All Day 28 tests passed.")


if __name__ == "__main__":
    main()