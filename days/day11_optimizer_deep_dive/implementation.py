import torch
from torch import nn


def manual_sgd_update_demo() -> None:
    """
    Show that SGD update is:

        w = w - lr * grad
    """
    w = torch.tensor(2.0, requires_grad=True)

    loss = w ** 2
    loss.backward()

    lr = 0.1

    print("Manual SGD update demo")
    print(f"Before update: w = {w.item():.6f}")
    print(f"w.grad = {w.grad.item():.6f}")

    with torch.no_grad():
        w -= lr * w.grad

    print(f"After update:  w = {w.item():.6f}")
    print()


def torch_sgd_update_demo() -> None:
    """
    Show PyTorch SGD updates a parameter using parameter.grad.
    """
    w = torch.tensor(2.0, requires_grad=True)

    optimizer = torch.optim.SGD([w], lr=0.1)

    loss = w ** 2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("PyTorch SGD update demo")
    print("loss = w^2, w starts at 2")
    print(f"After optimizer.step(), w = {w.item():.6f}")
    print("Expected: 1.6")
    print()


def momentum_optimizer_state_demo() -> None:
    """
    Show that SGD with momentum stores optimizer state.
    """
    w = torch.tensor(2.0, requires_grad=True)

    optimizer = torch.optim.SGD(
        [w],
        lr=0.1,
        momentum=0.9,
    )

    for step in range(3):
        loss = w ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Momentum step {step + 1}: w = {w.item():.6f}")

    print("\nOptimizer state for momentum SGD:")
    print(optimizer.state)
    print()


def adam_optimizer_state_demo() -> None:
    """
    Show that Adam stores first and second moment estimates.
    """
    w = torch.tensor(2.0, requires_grad=True)

    optimizer = torch.optim.Adam(
        [w],
        lr=0.1,
    )

    for step in range(3):
        loss = w ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Adam step {step + 1}: w = {w.item():.6f}")

    print("\nOptimizer state for Adam:")
    print(optimizer.state)
    print()


def train_small_model_with_optimizer(
    optimizer_name: str = "sgd",
    num_epochs: int = 50,
) -> tuple[list[float], nn.Module]:
    """
    Train a tiny linear regression model using a selected optimizer.

    True relationship:
        y = 2*x1 + 3*x2 + 1
    """
    torch.manual_seed(42)

    X = torch.randn(200, 2)

    true_w = torch.tensor([[2.0], [3.0]])
    true_b = 1.0

    y = X @ true_w + true_b + 0.1 * torch.randn(200, 1)

    model = nn.Linear(2, 1)
    loss_fn = nn.MSELoss()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    elif optimizer_name == "momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.01)
    else:
        raise ValueError(f"Unknown optimizer_name: {optimizer_name}")

    losses = []

    for epoch in range(num_epochs):
        y_hat = model(X)
        loss = loss_fn(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    return losses, model


def compare_optimizers_demo() -> None:
    """
    Compare SGD, momentum, Adam, and AdamW on a tiny regression task.
    """
    optimizer_names = ["sgd", "momentum", "adam", "adamw"]

    print("Compare optimizers demo")

    for name in optimizer_names:
        losses, model = train_small_model_with_optimizer(
            optimizer_name=name,
            num_epochs=50,
        )

        print(
            f"{name:8s}: "
            f"initial_loss={losses[0]:.6f}, "
            f"final_loss={losses[-1]:.6f}"
        )

    print()


def main() -> None:
    manual_sgd_update_demo()
    torch_sgd_update_demo()
    momentum_optimizer_state_demo()
    adam_optimizer_state_demo()
    compare_optimizers_demo()


if __name__ == "__main__":
    main()