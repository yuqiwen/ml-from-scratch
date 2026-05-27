import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


def generate_nonlinear_classification_data(
    n_samples: int = 1000,
    noise_std: float = 0.1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a simple nonlinear binary classification dataset.

    Rule:
        y = 1 if x1^2 + x2^2 > 1
        y = 0 otherwise

    This creates a circular decision boundary.

    A linear model struggles with this.
    An MLP can learn it better.
    """
    torch.manual_seed(seed)

    X = torch.empty(n_samples, 2).uniform_(-2.0, 2.0)

    radius_squared = X[:, 0] ** 2 + X[:, 1] ** 2

    y = (radius_squared > 1.0).float()

    # Add a little label noise by perturbing boundary-related score.
    noisy_score = radius_squared + noise_std * torch.randn(n_samples)
    y = (noisy_score > 1.0).float()

    # Shape should be (n_samples, 1) for BCEWithLogitsLoss.
    y = y.unsqueeze(1)

    return X, y


def create_dataloaders(
    X: torch.Tensor,
    y: torch.Tensor,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    """
    dataset = TensorDataset(X, y)

    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


class MLPBinaryClassifier(nn.Module):
    """
    A small MLP for binary classification.

    Architecture:
        input -> Linear -> ReLU -> Linear -> logit

    Input:
        X shape: (batch_size, input_dim)

    Output:
        logits shape: (batch_size, 1)
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 16):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Returns raw logits, not probabilities.
        """
        return self.network(X)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train model for one epoch.
    """
    model.train()

    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = x_batch.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def compute_accuracy_from_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """
    Compute binary classification accuracy from raw logits.

    logits -> sigmoid -> probability -> threshold 0.5
    """
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).float()

    accuracy = (predictions == y).float().mean()

    return float(accuracy.item())


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate model.

    Returns:
        average loss
        average accuracy
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()
            correct = (predictions == y_batch).float().sum().item()

            batch_size = x_batch.shape[0]

            total_loss += loss.item() * batch_size
            total_correct += correct
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples

    return avg_loss, avg_accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 30,
) -> tuple[list[float], list[float], list[float], dict[str, torch.Tensor]]:
    """
    Train model and keep best validation checkpoint.

    Returns:
        train_losses
        val_losses
        val_accuracies
        best_model_state
    """
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy = evaluate(
            model=model,
            data_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        print(
            f"epoch={epoch + 1:03d}, "
            f"train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}, "
            f"val_accuracy={val_accuracy:.4f}, "
            f"best_val_loss={best_val_loss:.6f}"
        )

    return train_losses, val_losses, val_accuracies, best_model_state


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    X, y = generate_nonlinear_classification_data(
        n_samples=1000,
        noise_std=0.1,
        seed=42,
    )

    train_loader, val_loader = create_dataloaders(
        X=X,
        y=y,
        train_ratio=0.8,
        batch_size=32,
        seed=42,
    )

    model = MLPBinaryClassifier(
        input_dim=2,
        hidden_dim=16,
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
        weight_decay=0.001,
    )

    train_losses, val_losses, val_accuracies, best_model_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=30,
    )

    model.load_state_dict(best_model_state)

    final_val_loss, final_val_accuracy = evaluate(
        model=model,
        data_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
    )

    print("\nFinal result:")
    print(f"best validation loss = {final_val_loss:.6f}")
    print(f"validation accuracy = {final_val_accuracy:.4f}")

    test_points = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.5, 0.5],
            [-2.0, -2.0],
        ],
        dtype=torch.float32,
    ).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(test_points)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()

    print("\nExample predictions:")
    for point, prob, pred in zip(
        test_points.cpu(),
        probabilities.cpu(),
        predictions.cpu(),
    ):
        print(
            f"x={point.numpy()}, "
            f"probability={prob.item():.4f}, "
            f"prediction={int(pred.item())}"
        )


if __name__ == "__main__":
    main()