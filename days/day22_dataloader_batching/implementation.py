import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


class SyntheticRegressionDataset(Dataset):
    """
    A simple synthetic regression dataset.

    Each sample:
        x: (2,)
        y: (1,)

    True function:
        y = x1^2 + 0.5*x2 + noise
    """

    def __init__(
        self,
        n_samples: int = 1024,
        seed: int = 42,
        simulate_delay: bool = False,
    ):
        super().__init__()

        torch.manual_seed(seed)

        self.X = torch.randn(n_samples, 2)
        noise = 0.1 * torch.randn(n_samples, 1)
        self.y = self.X[:, [0]] ** 2 + 0.5 * self.X[:, [1]] + noise

        self.simulate_delay = simulate_delay

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return one sample.

        simulate_delay is used to mimic expensive preprocessing.
        """
        if self.simulate_delay:
            time.sleep(0.001)

        return self.X[idx], self.y[idx]


class TinyMLP(nn.Module):
    """
    Tiny regression model.
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


@dataclass
class LoaderConfig:
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False


def create_train_val_loaders(
    dataset: Dataset,
    train_ratio: float = 0.8,
    config: LoaderConfig | None = None,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Split dataset into train/val and create DataLoaders.
    """
    if config is None:
        config = LoaderConfig()

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
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, val_loader


def move_batch_to_device(
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Move batch tensors to device.

    non_blocking=True can help when using pinned memory and CUDA.
    """
    x_batch, y_batch = batch

    x_batch = x_batch.to(device, non_blocking=True)
    y_batch = y_batch.to(device, non_blocking=True)

    return x_batch, y_batch


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """
    Train for one epoch.
    """
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in train_loader:
        x_batch, y_batch = move_batch_to_device(batch, device)

        y_hat = model(x_batch)
        loss = loss_fn(y_hat, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = x_batch.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate model.
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in val_loader:
        x_batch, y_batch = move_batch_to_device(batch, device)

        y_hat = model(x_batch)
        loss = loss_fn(y_hat, y_batch)

        batch_size = x_batch.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def inspect_loader_demo() -> None:
    """
    Show DataLoader batch shapes.
    """
    dataset = SyntheticRegressionDataset(n_samples=100, seed=1)

    config = LoaderConfig(
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    train_loader, val_loader = create_train_val_loaders(
        dataset=dataset,
        train_ratio=0.8,
        config=config,
        seed=1,
    )

    x_batch, y_batch = next(iter(train_loader))

    print("DataLoader inspect demo")
    print(f"dataset size       = {len(dataset)}")
    print(f"train size         = {len(train_loader.dataset)}")
    print(f"val size           = {len(val_loader.dataset)}")
    print(f"x_batch shape      = {x_batch.shape}")
    print(f"y_batch shape      = {y_batch.shape}")
    print(f"train num batches  = {len(train_loader)}")
    print(f"val num batches    = {len(val_loader)}")
    print()


def train_with_loader_demo() -> None:
    """
    Train a model using DataLoader.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SyntheticRegressionDataset(n_samples=1024, seed=2)

    config = LoaderConfig(
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    train_loader, val_loader = create_train_val_loaders(
        dataset=dataset,
        train_ratio=0.8,
        config=config,
        seed=2,
    )

    model = TinyMLP(hidden_dim=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    print("Training with DataLoader")
    print(f"device = {device}")
    print(f"pin_memory = {config.pin_memory}")

    for epoch in range(5):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        val_loss = evaluate(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        print(
            f"epoch={epoch:02d}, "
            f"train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}"
        )

    print()


def benchmark_loader_iteration(
    num_workers: int,
    simulate_delay: bool,
    batch_size: int = 64,
) -> float:
    """
    Benchmark iterating over a DataLoader.

    This is a simple demo. Results depend heavily on machine, OS, and environment.
    """
    dataset = SyntheticRegressionDataset(
        n_samples=512,
        seed=3,
        simulate_delay=simulate_delay,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    start = time.perf_counter()

    for _ in loader:
        pass

    elapsed = time.perf_counter() - start

    return elapsed


def dataloader_benchmark_demo() -> None:
    """
    Compare DataLoader iteration with different num_workers.

    On some small/local environments, num_workers=0 can be faster due to worker startup overhead.
    The important idea is that num_workers helps when per-sample loading is expensive.
    """
    print("DataLoader benchmark demo")

    for simulate_delay in [False, True]:
        print(f"simulate_delay = {simulate_delay}")

        for num_workers in [0, 2]:
            elapsed = benchmark_loader_iteration(
                num_workers=num_workers,
                simulate_delay=simulate_delay,
                batch_size=64,
            )

            print(
                f"num_workers={num_workers}, "
                f"elapsed={elapsed:.4f}s"
            )

        print()


def main() -> None:
    inspect_loader_demo()
    train_with_loader_demo()
    dataloader_benchmark_demo()


if __name__ == "__main__":
    main()