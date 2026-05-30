import torch
from torch import nn

from implementation import (
    SmallCNNBinaryClassifier,
    conv_output_size,
    generate_synthetic_image_data,
)


def test_conv_output_size_without_padding():
    output_size = conv_output_size(
        input_size=28,
        kernel_size=3,
        stride=1,
        padding=0,
    )

    print("Test 1: conv output size without padding")
    print(f"output_size = {output_size}")

    assert output_size == 26

    print("Passed.\n")


def test_conv_output_size_with_padding():
    output_size = conv_output_size(
        input_size=28,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    print("Test 2: conv output size with padding")
    print(f"output_size = {output_size}")

    assert output_size == 28

    print("Passed.\n")


def test_conv2d_shape():
    X = torch.randn(4, 1, 28, 28)

    conv = nn.Conv2d(
        in_channels=1,
        out_channels=8,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    Y = conv(X)

    print("Test 3: Conv2d output shape")
    print(f"Y shape = {Y.shape}")

    assert Y.shape == (4, 8, 28, 28)

    print("Passed.\n")


def test_maxpool_shape():
    X = torch.randn(4, 8, 28, 28)

    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    Y = pool(X)

    print("Test 4: MaxPool2d output shape")
    print(f"Y shape = {Y.shape}")

    assert Y.shape == (4, 8, 14, 14)

    print("Passed.\n")


def test_small_cnn_output_shape():
    model = SmallCNNBinaryClassifier()

    X = torch.randn(8, 1, 28, 28)
    logits = model(X)

    print("Test 5: SmallCNN output shape")
    print(f"logits shape = {logits.shape}")

    assert logits.shape == (8, 1)

    print("Passed.\n")


def test_synthetic_image_data_shape():
    X, y = generate_synthetic_image_data(
        n_samples=100,
        image_size=28,
        seed=1,
    )

    print("Test 6: synthetic image data shape")
    print(f"X shape = {X.shape}")
    print(f"y shape = {y.shape}")

    assert X.shape == (100, 1, 28, 28)
    assert y.shape == (100, 1)

    print("Passed.\n")


def test_cnn_training_reduces_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = generate_synthetic_image_data(
        n_samples=128,
        image_size=28,
        seed=2,
    )

    X = X.to(device)
    y = y.to(device)

    model = SmallCNNBinaryClassifier().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)

    model.train()

    logits = model(X)
    initial_loss = loss_fn(logits, y).item()

    for _ in range(20):
        logits = model(X)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logits = model(X)
    final_loss = loss_fn(logits, y).item()

    print("Test 7: CNN training reduces loss")
    print(f"initial_loss = {initial_loss:.6f}")
    print(f"final_loss = {final_loss:.6f}")

    assert final_loss < initial_loss

    print("Passed.\n")


def main():
    test_conv_output_size_without_padding()
    test_conv_output_size_with_padding()
    test_conv2d_shape()
    test_maxpool_shape()
    test_small_cnn_output_shape()
    test_synthetic_image_data_shape()
    test_cnn_training_reduces_loss()

    print("All Day 12 tests passed.")


if __name__ == "__main__":
    main()