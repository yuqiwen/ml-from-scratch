import torch
from torch import nn


class TinyMLP(nn.Module):
    """
    A small MLP for regression.

    Input:
        X: (B, 2)

    Output:
        y_hat: (B, 1)
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X)


def generate_data(
    n_samples: int = 256,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate nonlinear regression data.

    y = x1^2 + 0.5*x2 + noise
    """
    torch.manual_seed(seed)

    X = torch.randn(n_samples, 2)
    noise = 0.1 * torch.randn(n_samples, 1)

    y = X[:, [0]] ** 2 + 0.5 * X[:, [1]] + noise

    return X, y


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Return current learning rate from first parameter group.
    """
    return optimizer.param_groups[0]["lr"]


def compute_grad_norm(model: nn.Module) -> float:
    """
    Compute total gradient norm across all parameters.

    This is useful for monitoring whether gradients are exploding.
    """
    total_norm_squared = 0.0

    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.detach().norm(2)
            total_norm_squared += param_norm.item() ** 2

    return total_norm_squared ** 0.5


def train_one_epoch(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    max_grad_norm: float | None = None,
) -> tuple[float, float, float]:
    """
    Train for one epoch on full batch.

    Returns:
        loss value
        grad norm before clipping
        grad norm after clipping
    """
    model.train()

    y_hat = model(X)
    loss = loss_fn(y_hat, y)

    optimizer.zero_grad()
    loss.backward()

    grad_norm_before = compute_grad_norm(model)

    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_grad_norm,
        )

    grad_norm_after = compute_grad_norm(model)

    optimizer.step()

    return float(loss.item()), grad_norm_before, grad_norm_after


def train_with_step_lr_demo() -> list[float]:
    """
    Train with StepLR and print learning rate changes.
    """
    X, y = generate_data(n_samples=256, seed=1)

    model = TinyMLP(hidden_dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5,
    )

    loss_fn = nn.MSELoss()

    lrs = []

    print("Training with StepLR")

    for epoch in range(12):
        current_lr = get_current_lr(optimizer)
        lrs.append(current_lr)

        loss, grad_before, grad_after = train_one_epoch(
            model=model,
            X=X,
            y=y,
            optimizer=optimizer,
            loss_fn=loss_fn,
            max_grad_norm=1.0,
        )

        scheduler.step()

        print(
            f"epoch={epoch:02d}, "
            f"lr={current_lr:.6f}, "
            f"loss={loss:.6f}, "
            f"grad_before={grad_before:.6f}, "
            f"grad_after={grad_after:.6f}"
        )

    print()

    return lrs


def train_with_cosine_lr_demo() -> list[float]:
    """
    Train with CosineAnnealingLR and return learning rates.
    """
    X, y = generate_data(n_samples=256, seed=2)

    model = TinyMLP(hidden_dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10,
    )

    loss_fn = nn.MSELoss()

    lrs = []

    print("Training with CosineAnnealingLR")

    for epoch in range(10):
        current_lr = get_current_lr(optimizer)
        lrs.append(current_lr)

        loss, grad_before, grad_after = train_one_epoch(
            model=model,
            X=X,
            y=y,
            optimizer=optimizer,
            loss_fn=loss_fn,
            max_grad_norm=1.0,
        )

        scheduler.step()

        print(
            f"epoch={epoch:02d}, "
            f"lr={current_lr:.6f}, "
            f"loss={loss:.6f}, "
            f"grad_before={grad_before:.6f}, "
            f"grad_after={grad_after:.6f}"
        )

    print()

    return lrs


def gradient_clipping_demo() -> tuple[float, float]:
    """
    Create a situation with large gradients and show clipping effect.
    """
    torch.manual_seed(3)

    model = TinyMLP(hidden_dim=32)

    # Scale inputs and targets to create larger gradients.
    X = 20.0 * torch.randn(128, 2)
    y = 50.0 * torch.randn(128, 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    loss, grad_before, grad_after = train_one_epoch(
        model=model,
        X=X,
        y=y,
        optimizer=optimizer,
        loss_fn=loss_fn,
        max_grad_norm=1.0,
    )

    print("Gradient clipping demo")
    print(f"loss = {loss:.6f}")
    print(f"grad_norm_before = {grad_before:.6f}")
    print(f"grad_norm_after  = {grad_after:.6f}")
    print()

    return grad_before, grad_after


def main() -> None:
    train_with_step_lr_demo()
    train_with_cosine_lr_demo()
    gradient_clipping_demo()


if __name__ == "__main__":
    main()