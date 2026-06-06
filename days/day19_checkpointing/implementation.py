from pathlib import Path

import torch
from torch import nn


class TinyRegressionModel(nn.Module):
    """
    A tiny linear regression model.

    Input:
        X: (B, 2)

    Output:
        y_hat: (B, 1)
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X)


def generate_data(
    n_samples: int = 200,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate simple regression data.

    True function:
        y = 2*x1 + 3*x2 + 1 + noise
    """
    torch.manual_seed(seed)

    X = torch.randn(n_samples, 2)

    true_w = torch.tensor([[2.0], [3.0]])
    true_b = 1.0

    y = X @ true_w + true_b + 0.1 * torch.randn(n_samples, 1)

    return X, y


def train_one_epoch(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    """
    Train for one epoch on the full dataset.
    """
    model.train()

    y_hat = model(X)
    loss = loss_fn(y_hat, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
) -> float:
    """
    Evaluate without gradient tracking.
    """
    model.eval()

    with torch.no_grad():
        y_hat = model(X)
        loss = loss_fn(y_hat, y)

    return float(loss.item())


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
) -> None:
    """
    Save full training checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float]:
    """
    Load full training checkpoint.

    Returns:
        start_epoch
        best_val_loss
    """
    checkpoint = torch.load(path, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["best_val_loss"]

    return start_epoch, best_val_loss


def train_with_checkpoints(
    output_dir: str | Path,
    num_epochs: int = 20,
) -> tuple[list[float], list[float]]:
    """
    Train model and save both best and last checkpoints.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = generate_data(n_samples=200, seed=1)

    train_X = X[:160]
    train_y = y[:160]
    val_X = X[160:]
    val_y = y[160:]

    model = TinyRegressionModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.01)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model=model,
            X=train_X,
            y=train_y,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )

        val_loss = evaluate(
            model=model,
            X=val_X,
            y=val_y,
            loss_fn=loss_fn,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        save_checkpoint(
            path=output_dir / "last_checkpoint.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                path=output_dir / "best_checkpoint.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
            )

        print(
            f"epoch={epoch:03d}, "
            f"train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}, "
            f"best_val_loss={best_val_loss:.6f}"
        )

    return train_losses, val_losses


def resume_training_demo(checkpoint_path: str | Path, extra_epochs: int = 5) -> float:
    """
    Resume training from checkpoint.
    """
    X, y = generate_data(n_samples=200, seed=1)

    train_X = X[:160]
    train_y = y[:160]

    model = TinyRegressionModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.01)
    loss_fn = nn.MSELoss()

    start_epoch, best_val_loss = load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
    )

    print(f"\nResuming from epoch {start_epoch}")
    print(f"loaded best_val_loss = {best_val_loss:.6f}")

    last_loss = 0.0

    for epoch in range(start_epoch, start_epoch + extra_epochs):
        last_loss = train_one_epoch(
            model=model,
            X=train_X,
            y=train_y,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )

        print(f"resumed epoch={epoch:03d}, train_loss={last_loss:.6f}")

    return last_loss


def main() -> None:
    output_dir = Path("checkpoints")

    train_with_checkpoints(
        output_dir=output_dir,
        num_epochs=20,
    )

    resume_training_demo(
        checkpoint_path=output_dir / "last_checkpoint.pt",
        extra_epochs=5,
    )


if __name__ == "__main__":
    main()