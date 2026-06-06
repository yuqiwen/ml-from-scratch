from pathlib import Path
import shutil

import torch
from torch import nn

from implementation import (
    TinyRegressionModel,
    evaluate,
    generate_data,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
    train_with_checkpoints,
)


def test_model_state_dict_contains_parameters():
    model = TinyRegressionModel()

    state = model.state_dict()

    print("Test 1: model.state_dict contains parameters")
    print(state.keys())

    assert "linear.weight" in state
    assert "linear.bias" in state

    print("Passed.\n")


def test_save_and_load_checkpoint_restores_model(tmp_path: Path):
    X, y = generate_data(n_samples=50, seed=1)

    model = TinyRegressionModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    train_one_epoch(model, X, y, optimizer, loss_fn)

    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        best_val_loss=0.123,
    )

    original_output = model(X)

    new_model = TinyRegressionModel()
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=0.01)

    start_epoch, best_val_loss = load_checkpoint(
        path=checkpoint_path,
        model=new_model,
        optimizer=new_optimizer,
    )

    loaded_output = new_model(X)

    print("Test 2: checkpoint restores model")
    print(f"start_epoch = {start_epoch}")
    print(f"best_val_loss = {best_val_loss}")

    assert start_epoch == 4
    assert best_val_loss == 0.123
    assert torch.allclose(original_output, loaded_output)

    print("Passed.\n")


def test_optimizer_state_is_saved_after_adamw_step(tmp_path: Path):
    X, y = generate_data(n_samples=50, seed=2)

    model = TinyRegressionModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    train_one_epoch(model, X, y, optimizer, loss_fn)

    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=0,
        best_val_loss=1.0,
    )

    checkpoint = torch.load(checkpoint_path, weights_only=False)

    print("Test 3: optimizer state saved")
    print(checkpoint["optimizer_state_dict"].keys())

    assert "state" in checkpoint["optimizer_state_dict"]
    assert len(checkpoint["optimizer_state_dict"]["state"]) > 0

    print("Passed.\n")


def test_train_with_checkpoints_creates_files(tmp_path: Path):
    train_losses, val_losses = train_with_checkpoints(
        output_dir=tmp_path,
        num_epochs=5,
    )

    print("Test 4: train_with_checkpoints creates files")
    print(f"train_losses = {train_losses}")
    print(f"val_losses = {val_losses}")

    assert (tmp_path / "last_checkpoint.pt").exists()
    assert (tmp_path / "best_checkpoint.pt").exists()
    assert len(train_losses) == 5
    assert len(val_losses) == 5

    print("Passed.\n")


def test_loaded_best_checkpoint_can_evaluate(tmp_path: Path):
    X, y = generate_data(n_samples=100, seed=3)

    val_X = X[80:]
    val_y = y[80:]

    train_with_checkpoints(
        output_dir=tmp_path,
        num_epochs=5,
    )

    model = TinyRegressionModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    load_checkpoint(
        path=tmp_path / "best_checkpoint.pt",
        model=model,
        optimizer=optimizer,
    )

    val_loss = evaluate(model, val_X, val_y, loss_fn)

    print("Test 5: loaded best checkpoint can evaluate")
    print(f"val_loss = {val_loss:.6f}")

    assert val_loss > 0

    print("Passed.\n")


def make_test_dir(name: str) -> Path:
    test_dir = Path(__file__).parent / "_test_runs" / name
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


def main():
    test_model_state_dict_contains_parameters()

    test_save_and_load_checkpoint_restores_model(make_test_dir("save_load"))

    test_optimizer_state_is_saved_after_adamw_step(make_test_dir("optimizer_state"))

    test_train_with_checkpoints_creates_files(make_test_dir("train_outputs"))

    test_loaded_best_checkpoint_can_evaluate(make_test_dir("best_eval"))

    print("All Day 19 tests passed.")


if __name__ == "__main__":
    main()
