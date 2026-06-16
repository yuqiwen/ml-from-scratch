import torch
from torch import nn


class TinyRegressionModel(nn.Module):
    """
    Tiny regression model.

    Input:
        X: (B, 2)

    Output:
        prediction: (B, 1)
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