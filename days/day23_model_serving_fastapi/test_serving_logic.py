from pathlib import Path

import torch

from train_and_save_model import MODEL_PATH, TinyRegressionModel, save_model
import serve_model


def ensure_dummy_model_exists() -> None:
    """
    Create a dummy model file if missing.

    This keeps tests simple and independent of full training.
    """
    if MODEL_PATH.exists():
        return

    model = TinyRegressionModel(hidden_dim=32)
    save_model(model, MODEL_PATH)


def test_model_file_exists_or_created():
    ensure_dummy_model_exists()

    print("Test 1: model file exists")
    print(f"MODEL_PATH = {MODEL_PATH}")

    assert MODEL_PATH.exists()

    print("Passed.\n")


def test_predict_features_returns_float():
    ensure_dummy_model_exists()

    prediction = serve_model.predict_features([1.0, 2.0])

    print("Test 2: predict_features returns float")
    print(f"prediction = {prediction}")

    assert isinstance(prediction, float)

    print("Passed.\n")


def test_predict_features_rejects_wrong_length():
    print("Test 3: predict_features rejects wrong length")

    try:
        serve_model.predict_features([1.0])
        raised = False
    except ValueError:
        raised = True

    assert raised

    print("Passed.\n")


def test_model_eval_mode():
    ensure_dummy_model_exists()

    print("Test 4: model is in eval mode")

    assert serve_model.get_model().training is False

    print("Passed.\n")


def test_no_grad_prediction_does_not_create_input_grad():
    ensure_dummy_model_exists()

    print("Test 5: no_grad inference behavior")

    x = torch.tensor([[1.0, 2.0]], requires_grad=True, device=serve_model.device)

    with torch.no_grad():
        y = serve_model.get_model()(x)

    assert y.requires_grad is False

    print("Passed.\n")


def main():
    test_model_file_exists_or_created()
    test_predict_features_returns_float()
    test_predict_features_rejects_wrong_length()
    test_model_eval_mode()
    test_no_grad_prediction_does_not_create_input_grad()

    print("All Day 23 tests passed.")


if __name__ == "__main__":
    main()
