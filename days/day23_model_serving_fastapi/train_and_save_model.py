from pathlib import Path

import torch
from torch import nn


MODEL_PATH = Path(__file__).with_name("regression_model.pt")


class TinyRegressionModel(nn.Module):
    """
    Tiny regression model.

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
    n_samples: int = 512,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate regression data.

    True function:
        y = x1^2 + 0.5*x2 + noise
    """
    torch.manual_seed(seed)

    X = torch.randn(n_samples, 2)
    noise = 0.1 * torch.randn(n_samples, 1)

    y = X[:, [0]] ** 2 + 0.5 * X[:, [1]] + noise

    return X, y


def train_model(
    num_epochs: int = 200,
    lr: float = 0.01,
) -> TinyRegressionModel:
    """
    Train and return model.
    """
    X, y = generate_data()

    model = TinyRegressionModel(hidden_dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()

        y_hat = model(X)
        loss = loss_fn(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"epoch={epoch:03d}, loss={loss.item():.6f}")

    return model


def save_model(model: nn.Module, path: Path = MODEL_PATH) -> None:
    """
    Save model state_dict.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dim": 32,
        },
        path,
    )


def load_model(path: Path = MODEL_PATH) -> TinyRegressionModel:
    """
    Load model from checkpoint.
    """
    checkpoint = torch.load(path, weights_only=False)

    model = TinyRegressionModel(hidden_dim=checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def main() -> None:
    model = train_model()
    save_model(model, MODEL_PATH)

    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
