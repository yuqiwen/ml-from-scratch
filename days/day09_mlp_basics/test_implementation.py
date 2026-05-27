import torch
from torch import nn

from implementation import (
    MLPBinaryClassifier,
    compute_accuracy_from_logits,
    create_dataloaders,
    evaluate,
    generate_nonlinear_classification_data,
    train_model,
    train_one_epoch,
)


def test_mlp_output_shape():
    model = MLPBinaryClassifier(input_dim=2, hidden_dim=16)

    X = torch.randn(8, 2)
    logits = model(X)

    print("Test 1: MLP output shape")
    print(f"logits shape = {logits.shape}")

    assert logits.shape == (8, 1)

    print("Passed.\n")


def test_accuracy_from_logits():
    logits = torch.tensor(
        [
            [10.0],
            [-10.0],
            [5.0],
            [-5.0],
        ]
    )

    y = torch.tensor(
        [
            [1.0],
            [0.0],
            [1.0],
            [0.0],
        ]
    )

    accuracy = compute_accuracy_from_logits(logits, y)

    print("Test 2: accuracy from logits")
    print(f"accuracy = {accuracy:.4f}")

    assert accuracy == 1.0

    print("Passed.\n")


def test_train_one_epoch_runs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = generate_nonlinear_classification_data(
        n_samples=200,
        noise_std=0.1,
        seed=1,
    )

    train_loader, _ = create_dataloaders(
        X=X,
        y=y,
        train_ratio=0.8,
        batch_size=32,
        seed=1,
    )

    model = MLPBinaryClassifier(input_dim=2, hidden_dim=16).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)

    loss = train_one_epoch(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )

    print("Test 3: train_one_epoch runs")
    print(f"loss = {loss:.6f}")

    assert loss > 0

    print("Passed.\n")


def test_evaluate_returns_loss_and_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = generate_nonlinear_classification_data(
        n_samples=200,
        noise_std=0.1,
        seed=2,
    )

    _, val_loader = create_dataloaders(
        X=X,
        y=y,
        train_ratio=0.8,
        batch_size=32,
        seed=2,
    )

    model = MLPBinaryClassifier(input_dim=2, hidden_dim=16).to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    val_loss, val_accuracy = evaluate(
        model=model,
        data_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
    )

    print("Test 4: evaluate returns loss and accuracy")
    print(f"val_loss = {val_loss:.6f}")
    print(f"val_accuracy = {val_accuracy:.4f}")

    assert val_loss > 0
    assert 0.0 <= val_accuracy <= 1.0

    print("Passed.\n")


def test_mlp_learns_nonlinear_pattern():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = generate_nonlinear_classification_data(
        n_samples=800,
        noise_std=0.05,
        seed=3,
    )

    train_loader, val_loader = create_dataloaders(
        X=X,
        y=y,
        train_ratio=0.8,
        batch_size=32,
        seed=3,
    )

    model = MLPBinaryClassifier(input_dim=2, hidden_dim=32).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)

    train_losses, val_losses, val_accuracies, best_model_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=30,
    )

    model.load_state_dict(best_model_state)

    final_val_loss, final_val_accuracy = evaluate(
        model=model,
        data_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
    )

    print("Test 5: MLP learns nonlinear pattern")
    print(f"initial val loss = {val_losses[0]:.6f}")
    print(f"final val loss = {final_val_loss:.6f}")
    print(f"final val accuracy = {final_val_accuracy:.4f}")

    assert final_val_loss < val_losses[0]
    assert final_val_accuracy > 0.85

    print("Passed.\n")


def main():
    test_mlp_output_shape()
    test_accuracy_from_logits()
    test_train_one_epoch_runs()
    test_evaluate_returns_loss_and_accuracy()
    test_mlp_learns_nonlinear_pattern()

    print("All Day 09 tests passed.")


if __name__ == "__main__":
    main()