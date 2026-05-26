import torch
from torch import nn

from implementation import (
    LinearRegressionModel,
    create_dataloaders,
    evaluate,
    generate_linear_regression_data,
    train_model,
    train_one_epoch,
)


def test_dataloaders_create_batches():
    X, y = generate_linear_regression_data(
        n_samples=100,
        noise_std=0.1,
        seed=1,
    )

    train_loader, val_loader = create_dataloaders(
        X=X,
        y=y,
        train_ratio=0.8,
        batch_size=16,
        seed=1,
    )

    print("Test 1: DataLoaders create batches")
    print(f"train batches = {len(train_loader)}")
    print(f"val batches = {len(val_loader)}")

    assert len(train_loader.dataset) == 80
    assert len(val_loader.dataset) == 20

    x_batch, y_batch = next(iter(train_loader))

    assert x_batch.shape[1] == 2
    assert y_batch.shape[1] == 1

    print("Passed.\n")


def test_train_one_epoch_reduces_or_runs_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = generate_linear_regression_data(
        n_samples=200,
        noise_std=0.1,
        seed=2,
    )

    train_loader, _ = create_dataloaders(
        X=X,
        y=y,
        train_ratio=0.8,
        batch_size=32,
        seed=2,
    )

    model = LinearRegressionModel(input_dim=2).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    loss_epoch_1 = train_one_epoch(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )

    loss_epoch_2 = train_one_epoch(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )

    print("Test 2: train_one_epoch runs and loss usually decreases")
    print(f"loss_epoch_1 = {loss_epoch_1:.6f}")
    print(f"loss_epoch_2 = {loss_epoch_2:.6f}")

    assert loss_epoch_1 > 0
    assert loss_epoch_2 > 0
    assert loss_epoch_2 < loss_epoch_1

    print("Passed.\n")


def test_evaluate_does_not_create_gradients():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = generate_linear_regression_data(
        n_samples=100,
        noise_std=0.1,
        seed=3,
    )

    train_loader, val_loader = create_dataloaders(
        X=X,
        y=y,
        train_ratio=0.8,
        batch_size=16,
        seed=3,
    )

    model = LinearRegressionModel(input_dim=2).to(device)
    loss_fn = nn.MSELoss()

    # Gradients should initially be None.
    for param in model.parameters():
        assert param.grad is None

    val_loss = evaluate(
        model=model,
        data_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
    )

    print("Test 3: evaluate uses no_grad")
    print(f"val_loss = {val_loss:.6f}")

    for param in model.parameters():
        assert param.grad is None

    print("Passed.\n")


def test_train_model_learns_linear_relationship():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = generate_linear_regression_data(
        n_samples=500,
        noise_std=0.1,
        seed=4,
    )

    train_loader, val_loader = create_dataloaders(
        X=X,
        y=y,
        train_ratio=0.8,
        batch_size=32,
        seed=4,
    )

    model = LinearRegressionModel(input_dim=2).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    train_losses, val_losses, best_model_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=20,
    )

    model.load_state_dict(best_model_state)

    learned_w = model.linear.weight.detach().cpu().flatten()
    learned_b = model.linear.bias.detach().cpu().item()

    print("Test 4: model learns linear relationship")
    print(f"learned_w = {learned_w}")
    print(f"learned_b = {learned_b:.6f}")

    assert torch.allclose(learned_w, torch.tensor([2.0, 3.0]), atol=0.25)
    assert abs(learned_b - 1.0) < 0.25
    assert val_losses[-1] < val_losses[0]

    print("Passed.\n")


def main():
    test_dataloaders_create_batches()
    test_train_one_epoch_reduces_or_runs_loss()
    test_evaluate_does_not_create_gradients()
    test_train_model_learns_linear_relationship()

    print("All Day 08 tests passed.")


if __name__ == "__main__":
    main()