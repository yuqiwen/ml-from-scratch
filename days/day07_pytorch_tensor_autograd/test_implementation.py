import torch


def test_scalar_autograd():
    """
    loss = w^2
    dloss/dw = 2w

    If w = 2, gradient should be 4.
    """
    w = torch.tensor(2.0, requires_grad=True)

    loss = w ** 2
    loss.backward()

    print("Test 1: scalar autograd")
    print(f"w.grad = {w.grad.item()}")

    assert torch.isclose(w.grad, torch.tensor(4.0))

    print("Passed.\n")


def test_linear_regression_autograd_matches_manual_gradient():
    """
    Check PyTorch autograd against manual linear regression gradients.
    """
    X = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ]
    )

    y = torch.tensor([6.0, 8.0, 9.0])

    w = torch.zeros(2, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    y_hat = X @ w + b
    loss = torch.mean((y_hat - y) ** 2)
    loss.backward()

    n_samples = X.shape[0]
    error = y_hat.detach() - y

    manual_dw = (2 / n_samples) * (X.T @ error)
    manual_db = (2 / n_samples) * torch.sum(error)

    print("Test 2: autograd matches manual gradient")
    print(f"w.grad = {w.grad}")
    print(f"manual_dw = {manual_dw}")
    print(f"b.grad = {b.grad}")
    print(f"manual_db = {manual_db}")

    assert torch.allclose(w.grad, manual_dw)
    assert torch.allclose(b.grad, manual_db)

    print("Passed.\n")


def test_optimizer_step_updates_parameters():
    """
    Check that optimizer.step() changes parameters.
    """
    X = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ]
    )

    y = torch.tensor([6.0, 8.0, 9.0])

    w = torch.zeros(2, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    optimizer = torch.optim.SGD([w, b], lr=0.01)

    old_w = w.detach().clone()
    old_b = b.detach().clone()

    y_hat = X @ w + b
    loss = torch.mean((y_hat - y) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Test 3: optimizer step updates parameters")
    print(f"old_w = {old_w}")
    print(f"new_w = {w.detach()}")
    print(f"old_b = {old_b}")
    print(f"new_b = {b.detach()}")

    assert not torch.allclose(w.detach(), old_w)
    assert not torch.allclose(b.detach(), old_b)

    print("Passed.\n")


def test_gradients_accumulate_without_zero_grad():
    """
    PyTorch gradients accumulate by default.
    """
    w = torch.tensor(2.0, requires_grad=True)

    loss1 = w ** 2
    loss1.backward()

    first_grad = w.grad.detach().clone()

    loss2 = w ** 2
    loss2.backward()

    second_grad = w.grad.detach().clone()

    print("Test 4: gradients accumulate without zero_grad")
    print(f"first_grad = {first_grad}")
    print(f"second_grad = {second_grad}")

    assert torch.isclose(first_grad, torch.tensor(4.0))
    assert torch.isclose(second_grad, torch.tensor(8.0))

    print("Passed.\n")


def test_zero_grad_clears_gradient():
    """
    Check that optimizer.zero_grad() clears parameter gradients.
    """
    w = torch.tensor(2.0, requires_grad=True)
    optimizer = torch.optim.SGD([w], lr=0.1)

    loss = w ** 2
    loss.backward()

    assert w.grad is not None

    optimizer.zero_grad()

    print("Test 5: zero_grad clears gradient")
    print(f"w.grad after zero_grad = {w.grad}")

    # Depending on PyTorch version, zero_grad may set grad to None by default.
    assert w.grad is None or torch.isclose(w.grad, torch.tensor(0.0))

    print("Passed.\n")


def main():
    test_scalar_autograd()
    test_linear_regression_autograd_matches_manual_gradient()
    test_optimizer_step_updates_parameters()
    test_gradients_accumulate_without_zero_grad()
    test_zero_grad_clears_gradient()

    print("All Day 07 tests passed.")


if __name__ == "__main__":
    main()