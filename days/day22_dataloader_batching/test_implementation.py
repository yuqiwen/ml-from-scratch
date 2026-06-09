import torch
from torch import nn

from implementation import (
    LoaderConfig,
    SyntheticRegressionDataset,
    TinyMLP,
    create_train_val_loaders,
    evaluate,
    move_batch_to_device,
    train_one_epoch,
)


def test_dataset_length_and_sample_shape():
    dataset = SyntheticRegressionDataset(n_samples=100, seed=1)

    x, y = dataset[0]

    print("Test 1: dataset length and sample shape")
    print(f"len(dataset) = {len(dataset)}")
    print(f"x shape = {x.shape}")
    print(f"y shape = {y.shape}")

    assert len(dataset) == 100
    assert x.shape == (2,)
    assert y.shape == (1,)

    print("Passed.\n")


def test_dataloader_batch_shapes():
    dataset = SyntheticRegressionDataset(n_samples=100, seed=2)

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
        seed=2,
    )

    x_batch, y_batch = next(iter(train_loader))

    print("Test 2: dataloader batch shapes")
    print(f"x_batch shape = {x_batch.shape}")
    print(f"y_batch shape = {y_batch.shape}")

    assert len(train_loader.dataset) == 80
    assert len(val_loader.dataset) == 20
    assert x_batch.shape[1] == 2
    assert y_batch.shape[1] == 1
    assert x_batch.shape[0] <= 16

    print("Passed.\n")


def test_val_loader_shuffle_false_by_behavior():
    dataset = SyntheticRegressionDataset(n_samples=50, seed=3)

    config = LoaderConfig(
        batch_size=10,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    _, val_loader = create_train_val_loaders(
        dataset=dataset,
        train_ratio=0.8,
        config=config,
        seed=3,
    )

    first_pass = [batch[0] for batch in val_loader]
    second_pass = [batch[0] for batch in val_loader]

    print("Test 3: val loader deterministic order")
    print(f"num val batches = {len(first_pass)}")

    for a, b in zip(first_pass, second_pass):
        assert torch.allclose(a, b)

    print("Passed.\n")


def test_move_batch_to_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(4, 2)
    y = torch.randn(4, 1)

    x_device, y_device = move_batch_to_device((x, y), device)

    print("Test 4: move batch to device")
    print(f"device = {device}")
    print(f"x_device.device = {x_device.device}")

    assert x_device.device == device
    assert y_device.device == device

    print("Passed.\n")


def test_training_with_dataloader_reduces_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SyntheticRegressionDataset(n_samples=512, seed=4)

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
        seed=4,
    )

    model = TinyMLP(hidden_dim=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    initial_val_loss = evaluate(
        model=model,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
    )

    for _ in range(5):
        train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

    final_val_loss = evaluate(
        model=model,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
    )

    print("Test 5: training with dataloader reduces val loss")
    print(f"initial_val_loss = {initial_val_loss:.6f}")
    print(f"final_val_loss = {final_val_loss:.6f}")

    assert final_val_loss < initial_val_loss

    print("Passed.\n")


def main():
    test_dataset_length_and_sample_shape()
    test_dataloader_batch_shapes()
    test_val_loader_shuffle_false_by_behavior()
    test_move_batch_to_device()
    test_training_with_dataloader_reduces_loss()

    print("All Day 22 tests passed.")


if __name__ == "__main__":
    main()