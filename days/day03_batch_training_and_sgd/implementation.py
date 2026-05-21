import math
import numpy as np


def compute_steps_per_epoch(
    dataset_size: int,
    batch_size: int,
    drop_last: bool = False,
) -> int:
    """
    Compute how many training steps are needed for one epoch.

    If drop_last is False:
        Keep the last incomplete batch.
        Use ceil(dataset_size / batch_size).

    If drop_last is True:
        Drop the last incomplete batch.
        Use floor(dataset_size / batch_size).
    """
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if drop_last:
        return dataset_size // batch_size

    return math.ceil(dataset_size / batch_size)


def create_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Split dataset into mini-batches.

    Args:
        X: Feature matrix with shape (n_samples, n_features).
        y: Target vector with shape (n_samples,).
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle data before batching.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        A list of (X_batch, y_batch).
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")

    dataset_size = len(X)
    indices = np.arange(dataset_size)

    if shuffle:
        np.random.shuffle(indices)

    batches = []

    for start in range(0, dataset_size, batch_size):
        end = start + batch_size
        batch_indices = indices[start:end]

        if drop_last and len(batch_indices) < batch_size:
            continue

        X_batch = X[batch_indices]
        y_batch = y[batch_indices]

        batches.append((X_batch, y_batch))

    return batches


def demo_epoch_batch_relationship() -> None:
    """
    Demonstrate the relationship between dataset size, batch size, epoch, and steps.
    """
    dataset_size = 1030
    batch_size = 100
    epochs = 5

    steps_keep_last = compute_steps_per_epoch(
        dataset_size=dataset_size,
        batch_size=batch_size,
        drop_last=False,
    )

    steps_drop_last = compute_steps_per_epoch(
        dataset_size=dataset_size,
        batch_size=batch_size,
        drop_last=True,
    )

    print("Dataset / batch / epoch relationship")
    print(f"dataset_size = {dataset_size}")
    print(f"batch_size = {batch_size}")
    print(f"epochs = {epochs}")

    print("\nWithout drop_last:")
    print(f"steps_per_epoch = {steps_keep_last}")
    print(f"total_steps = {epochs * steps_keep_last}")

    print("\nWith drop_last:")
    print(f"steps_per_epoch = {steps_drop_last}")
    print(f"total_steps = {epochs * steps_drop_last}")


def demo_create_batches() -> None:
    """
    Demonstrate how a dataset is split into batches.
    """
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    batch_size = 4

    print("\nCreate batches demo")
    print("X:")
    print(X)
    print("y:")
    print(y)

    batches = create_batches(
        X=X,
        y=y,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    print(f"\nbatch_size = {batch_size}")
    print(f"number of batches = {len(batches)}")

    for i, (X_batch, y_batch) in enumerate(batches):
        print(f"\nBatch {i + 1}")
        print("X_batch:")
        print(X_batch)
        print("y_batch:")
        print(y_batch)


def main() -> None:
    demo_epoch_batch_relationship()
    demo_create_batches()


if __name__ == "__main__":
    main()