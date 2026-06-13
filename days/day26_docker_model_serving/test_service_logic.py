from pathlib import Path

import torch

from model import TinyRegressionModel
from train_model import MODEL_PATH, save_model


def ensure_model_exists() -> None:
    """
    Create a valid random checkpoint for import tests
    when the trained checkpoint does not yet exist.
    """
    if MODEL_PATH.exists():
        return

    model = TinyRegressionModel(hidden_dim=32)
    save_model(model, MODEL_PATH)


ensure_model_exists()

import serve_model  # noqa: E402


def test_model_checkpoint_exists() -> None:
    print("Test 1: model checkpoint exists")

    assert Path(MODEL_PATH).exists()

    print("Passed.\n")


def test_model_is_in_eval_mode() -> None:
    print("Test 2: model is in eval mode")

    assert serve_model.model.training is False

    print("Passed.\n")


def test_prediction_returns_float() -> None:
    prediction = serve_model.predict_features(
        [1.0, 2.0]
    )

    print("Test 3: prediction returns float")
    print(f"prediction = {prediction}")

    assert isinstance(prediction, float)

    print("Passed.\n")


def test_wrong_feature_count_rejected() -> None:
    print("Test 4: wrong feature count rejected")

    try:
        serve_model.predict_features([1.0])
        raised = False
    except ValueError:
        raised = True

    assert raised

    print("Passed.\n")


def test_inference_mode_disables_grad() -> None:
    X = torch.tensor(
        [[1.0, 2.0]],
        dtype=torch.float32,
        device=serve_model.device,
        requires_grad=True,
    )

    with torch.inference_mode():
        output = serve_model.model(X)

    print("Test 5: inference mode disables grad")

    assert output.requires_grad is False

    print("Passed.\n")


def test_health_response() -> None:
    response = serve_model.health()

    print("Test 6: health response")
    print(response)

    assert response["status"] == "ok"
    assert response["device"] in {"cpu", "cuda"}

    print("Passed.\n")


def main() -> None:
    test_model_checkpoint_exists()
    test_model_is_in_eval_mode()
    test_prediction_returns_float()
    test_wrong_feature_count_rejected()
    test_inference_mode_disables_grad()
    test_health_response()

    print("All Day 26 tests passed.")


if __name__ == "__main__":
    main()