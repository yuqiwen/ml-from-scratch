from pathlib import Path

import torch
from torch import nn

from model import TinyRegressionModel


MODEL_PATH = Path(__file__).with_name("regression_model.pt")
HIDDEN_DIM = 32


def generate_data(
    n_samples: int = 512,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic regression data.

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
    learning_rate: float = 0.01,
) -> TinyRegressionModel:
    X, y = generate_data()

    model = TinyRegressionModel(hidden_dim=HIDDEN_DIM)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()

        prediction = model(X)
        loss = loss_fn(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(
                f"epoch={epoch:03d}, "
                f"loss={loss.item():.6f}"
            )

    return model


def save_model(
    model: TinyRegressionModel,
    path: Path = MODEL_PATH,
) -> None:
    torch.save(
        {
            "hidden_dim": HIDDEN_DIM,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def main() -> None:
    model = train_model()
    save_model(model)

    print(f"Saved model to: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()
