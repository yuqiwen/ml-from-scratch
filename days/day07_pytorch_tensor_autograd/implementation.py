import torch


def scalar_autograd_demo() -> None:
    """
    Demonstrate autograd with a scalar variable.

    loss = w^2
    dloss/dw = 2w

    If w = 2, gradient should be 4.
    """
    w = torch.tensor(2.0, requires_grad=True)

    loss = w ** 2

    loss.backward()

    print("Scalar autograd demo")
    print(f"w = {w.item()}")
    print(f"loss = {loss.item()}")
    print(f"w.grad = {w.grad.item()}")
    print()


def linear_regression_autograd_demo() -> None:
    """
    Demonstrate autograd with a simple linear regression example.

    Model:
        y_hat = X @ w + b

    Loss:
        mean((y_hat - y)^2)

    PyTorch automatically computes gradients:
        dloss/dw
        dloss/db
    """
    X = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ]
    )

    y = torch.tensor([6.0, 8.0, 9.0])

    # Initialize parameters.
    w = torch.zeros(2, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    # Forward pass.
    y_hat = X @ w + b

    # Compute MSE loss.
    loss = torch.mean((y_hat - y) ** 2)

    # Backward pass.
    loss.backward()

    print("Linear regression autograd demo")
    print(f"y_hat = {y_hat.detach().numpy()}")
    print(f"loss = {loss.item():.6f}")
    print(f"w.grad = {w.grad}")
    print(f"b.grad = {b.grad}")
    print()


def manual_gradient_check() -> None:
    """
    Compare PyTorch autograd gradients with manual gradients.

    This verifies that autograd gives the same result as:

        error = y_hat - y
        dw = (2/n) * X.T @ error
        db = (2/n) * sum(error)
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

    # Manual gradient computation.
    n_samples = X.shape[0]
    error = y_hat.detach() - y

    manual_dw = (2 / n_samples) * (X.T @ error)
    manual_db = (2 / n_samples) * torch.sum(error)

    print("Manual gradient check")
    print(f"autograd w.grad = {w.grad}")
    print(f"manual dw       = {manual_dw}")
    print(f"autograd b.grad = {b.grad}")
    print(f"manual db       = {manual_db}")
    print()


def optimizer_step_demo() -> None:
    """
    Demonstrate optimizer.zero_grad(), loss.backward(), and optimizer.step().
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

    print("Optimizer step demo")
    print(f"Before update: w = {w.detach()}, b = {b.detach()}")

    y_hat = X @ w + b
    loss = torch.mean((y_hat - y) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"loss = {loss.item():.6f}")
    print(f"After update:  w = {w.detach()}, b = {b.detach()}")
    print()


def gradient_accumulation_demo() -> None:
    """
    Demonstrate that PyTorch accumulates gradients by default.

    Calling backward twice without zero_grad accumulates gradients.
    """
    w = torch.tensor(2.0, requires_grad=True)

    loss1 = w ** 2
    loss1.backward()

    print("Gradient accumulation demo")
    print(f"After first backward, w.grad = {w.grad.item()}")

    loss2 = w ** 2
    loss2.backward()

    print(f"After second backward without zero_grad, w.grad = {w.grad.item()}")

    w.grad.zero_()

    loss3 = w ** 2
    loss3.backward()

    print(f"After manually zeroing grad and backward again, w.grad = {w.grad.item()}")
    print()


def main() -> None:
    scalar_autograd_demo()
    linear_regression_autograd_demo()
    manual_gradient_check()
    optimizer_step_demo()
    gradient_accumulation_demo()


if __name__ == "__main__":
    main()