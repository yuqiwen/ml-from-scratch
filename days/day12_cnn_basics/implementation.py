import torch
from torch import nn


def conv_output_size(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> int:
    """
    Compute convolution output size for one spatial dimension.

    Formula:
        output = floor((input + 2*padding - kernel_size) / stride) + 1
    """
    return (input_size + 2 * padding - kernel_size) // stride + 1


def conv_shape_demo() -> None:
    """
    Demonstrate Conv2d input/output shapes.
    """
    batch_size = 4
    in_channels = 1
    height = 28
    width = 28

    X = torch.randn(batch_size, in_channels, height, width)

    conv = nn.Conv2d(
        in_channels=1,
        out_channels=8,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    Y = conv(X)

    print("Conv2d shape demo")
    print(f"input shape  = {X.shape}")
    print(f"output shape = {Y.shape}")
    print()


def pooling_shape_demo() -> None:
    """
    Demonstrate MaxPool2d shape change.
    """
    X = torch.randn(4, 8, 28, 28)

    pool = nn.MaxPool2d(
        kernel_size=2,
        stride=2,
    )

    Y = pool(X)

    print("MaxPool2d shape demo")
    print(f"input shape  = {X.shape}")
    print(f"output shape = {Y.shape}")
    print()


class SmallCNNBinaryClassifier(nn.Module):
    """
    A small CNN for binary classification.

    Input:
        X shape: (batch_size, 1, 28, 28)

    Architecture:
        Conv2d(1, 8, kernel_size=3, padding=1)
        ReLU
        MaxPool2d(2)

        Conv2d(8, 16, kernel_size=3, padding=1)
        ReLU
        MaxPool2d(2)

        Flatten
        Linear(16 * 7 * 7, 1)

    Output:
        logits shape: (batch_size, 1)
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Linear(16 * 7 * 7, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Returns raw logits.
        """
        features = self.features(X)
        flattened = torch.flatten(features, start_dim=1)
        logits = self.classifier(flattened)
        return logits


def generate_synthetic_image_data(
    n_samples: int = 256,
    image_size: int = 28,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate simple synthetic image data.

    Class 1:
        image has a bright square in the center.

    Class 0:
        image is mostly noise.

    Shape:
        X: (n_samples, 1, image_size, image_size)
        y: (n_samples, 1)
    """
    torch.manual_seed(seed)

    X = 0.1 * torch.randn(n_samples, 1, image_size, image_size)
    y = torch.zeros(n_samples, 1)

    half = n_samples // 2

    # Class 1: add a bright center square.
    center_start = image_size // 2 - 3
    center_end = image_size // 2 + 3

    X[:half, :, center_start:center_end, center_start:center_end] += 1.0
    y[:half] = 1.0

    # Shuffle samples.
    indices = torch.randperm(n_samples)
    X = X[indices]
    y = y[indices]

    return X, y


def train_cnn_demo() -> None:
    """
    Train a small CNN on synthetic image data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = generate_synthetic_image_data(n_samples=256, seed=1)

    model = SmallCNNBinaryClassifier().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.001,
    )

    X = X.to(device)
    y = y.to(device)

    print("Training SmallCNNBinaryClassifier")

    for epoch in range(10):
        model.train()

        logits = model(X)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()
            accuracy = (predictions == y).float().mean().item()

        print(
            f"epoch={epoch + 1:02d}, "
            f"loss={loss.item():.6f}, "
            f"accuracy={accuracy:.4f}"
        )

    print()


def main() -> None:
    conv_shape_demo()
    pooling_shape_demo()

    model = SmallCNNBinaryClassifier()
    X = torch.randn(4, 1, 28, 28)
    logits = model(X)

    print("SmallCNN forward demo")
    print(f"input shape  = {X.shape}")
    print(f"logits shape = {logits.shape}")
    print()

    train_cnn_demo()


if __name__ == "__main__":
    main()