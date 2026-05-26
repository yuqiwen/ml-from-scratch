import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


def generate_linear_regression_data(
    n_samples: int = 500,
    noise_std: float = 0.2,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic linear regression data.

    True relationship:
        y = 2*x1 + 3*x2 + 1 + noise
    """
    torch.manual_seed(seed)

    X = torch.randn(n_samples, 2)

    true_w = torch.tensor([2.0, 3.0])
    true_b = 1.0

    noise = noise_std * torch.randn(n_samples)
    y = X @ true_w + true_b + noise

    # Make y shape (n_samples, 1) to match model output.
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
    Create train and validation DataLoaders.

    TensorDataset stores (X, y) pairs.

    DataLoader handles batching and shuffling.
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


class LinearRegressionModel(nn.Module):
    """
    A simple PyTorch linear regression model.

    Model:
        y_hat = Linear(X)

    For input shape:
        X: (batch_size, 2)

    Output shape:
        y_hat: (batch_size, 1)
    """

    def __init__(self, input_dim: int = 2):
        super().__init__()

        self.linear = nn.Linear(input_dim, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        return self.linear(X)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train the model for one epoch.

    Returns:
        Average training loss over all samples.
    """
    model.train()

    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass.
        y_hat = model(x_batch)

        # Compute batch loss.
        loss = loss_fn(y_hat, y_batch)

        # Backward pass and parameter update.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = x_batch.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    return avg_loss


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate the model without updating parameters.

    Returns:
        Average loss over all samples.
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_hat = model(x_batch)
            loss = loss_fn(y_hat, y_batch)

            batch_size = x_batch.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    return avg_loss


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 20,
) -> tuple[list[float], list[float], dict[str, torch.Tensor]]:
    """
    Train model for multiple epochs and keep the best validation checkpoint.

    Returns:
        train_losses
        val_losses
        best_model_state
    """
    train_losses = []
    val_losses = []

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

        val_loss = evaluate(
            model=model,
            data_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        print(
            f"epoch={epoch + 1:03d}, "
            f"train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}, "
            f"best_val_loss={best_val_loss:.6f}"
        )

    return train_losses, val_losses, best_model_state


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    X, y = generate_linear_regression_data(
        n_samples=500,
        noise_std=0.2,
        seed=42,
    )

    train_loader, val_loader = create_dataloaders(
        X=X,
        y=y,
        train_ratio=0.8,
        batch_size=32,
        seed=42,
    )

    model = LinearRegressionModel(input_dim=2).to(device)

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.05,
    )

    train_losses, val_losses, best_model_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=20,
    )

    # Load the best model based on validation loss.
    model.load_state_dict(best_model_state)

    final_val_loss = evaluate(
        model=model,
        data_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
    )

    print("\nFinal result:")
    print(f"best validation loss = {final_val_loss:.6f}")

    learned_w = model.linear.weight.detach().cpu().numpy()
    learned_b = model.linear.bias.detach().cpu().numpy()

    print(f"learned weight = {learned_w}")
    print(f"learned bias = {learned_b}")


if __name__ == "__main__":
    main()