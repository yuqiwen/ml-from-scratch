import torch
from torch import nn

from implementation import (
    TinyMLP,
    compute_grad_norm,
    generate_data,
    get_current_lr,
    gradient_clipping_demo,
    train_one_epoch,
    train_with_cosine_lr_demo,
    train_with_step_lr_demo,
)


def test_step_lr_changes_learning_rate():
    lrs = train_with_step_lr_demo()

    print("Test 1: StepLR changes learning rate")
    print(f"lrs = {lrs}")

    assert lrs[0] == 0.01
    assert lrs[5] < lrs[0]
    assert lrs[10] < lrs[5]

    print("Passed.\n")


def test_cosine_lr_decreases():
    lrs = train_with_cosine_lr_demo()

    print("Test 2: CosineAnnealingLR changes learning rate")
    print(f"lrs = {lrs}")

    assert lrs[0] > lrs[-1]

    print("Passed.\n")


def test_gradient_clipping_reduces_norm():
    grad_before, grad_after = gradient_clipping_demo()

    print("Test 3: gradient clipping reduces norm")
    print(f"grad_before = {grad_before:.6f}")
    print(f"grad_after = {grad_after:.6f}")

    assert grad_after <= grad_before
    assert grad_after <= 1.0001

    print("Passed.\n")


def test_train_one_epoch_returns_positive_loss():
    X, y = generate_data(n_samples=128, seed=10)

    model = TinyMLP(hidden_dim=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    loss, grad_before, grad_after = train_one_epoch(
        model=model,
        X=X,
        y=y,
        optimizer=optimizer,
        loss_fn=loss_fn,
        max_grad_norm=1.0,
    )

    print("Test 4: train_one_epoch returns values")
    print(f"loss = {loss:.6f}")
    print(f"grad_before = {grad_before:.6f}")
    print(f"grad_after = {grad_after:.6f}")

    assert loss > 0
    assert grad_before >= 0
    assert grad_after >= 0

    print("Passed.\n")


def test_get_current_lr():
    model = TinyMLP(hidden_dim=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.123)

    lr = get_current_lr(optimizer)

    print("Test 5: get current learning rate")
    print(f"lr = {lr}")

    assert lr == 0.123

    print("Passed.\n")


def test_compute_grad_norm_after_backward():
    X, y = generate_data(n_samples=64, seed=11)

    model = TinyMLP(hidden_dim=16)
    loss_fn = nn.MSELoss()

    y_hat = model(X)
    loss = loss_fn(y_hat, y)

    loss.backward()

    grad_norm = compute_grad_norm(model)

    print("Test 6: compute grad norm")
    print(f"grad_norm = {grad_norm:.6f}")

    assert grad_norm > 0

    print("Passed.\n")


def main():
    test_step_lr_changes_learning_rate()
    test_cosine_lr_decreases()
    test_gradient_clipping_reduces_norm()
    test_train_one_epoch_returns_positive_loss()
    test_get_current_lr()
    test_compute_grad_norm_after_backward()

    print("All Day 20 tests passed.")


if __name__ == "__main__":
    main()