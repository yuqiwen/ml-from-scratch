import numpy as np

from implementation import (
    compute_steps_per_epoch,
    create_batches,
)


def test_steps_per_epoch_without_drop_last():
    dataset_size = 1030
    batch_size = 100

    steps = compute_steps_per_epoch(
        dataset_size=dataset_size,
        batch_size=batch_size,
        drop_last=False,
    )

    print("Test 1: steps_per_epoch without drop_last")
    print(f"steps = {steps}")

    assert steps == 11

    print("Passed.\n")


def test_steps_per_epoch_with_drop_last():
    dataset_size = 1030
    batch_size = 100

    steps = compute_steps_per_epoch(
        dataset_size=dataset_size,
        batch_size=batch_size,
        drop_last=True,
    )

    print("Test 2: steps_per_epoch with drop_last")
    print(f"steps = {steps}")

    assert steps == 10

    print("Passed.\n")


def test_create_batches_without_shuffle():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    batches = create_batches(
        X=X,
        y=y,
        batch_size=4,
        shuffle=False,
        drop_last=False,
    )

    print("Test 3: create batches without shuffle")
    print(f"number of batches = {len(batches)}")

    assert len(batches) == 3

    first_X_batch, first_y_batch = batches[0]
    last_X_batch, last_y_batch = batches[-1]

    assert first_X_batch.shape == (4, 2)
    assert first_y_batch.shape == (4,)

    assert last_X_batch.shape == (2, 2)
    assert last_y_batch.shape == (2,)

    print("Passed.\n")


def test_create_batches_with_drop_last():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    batches = create_batches(
        X=X,
        y=y,
        batch_size=4,
        shuffle=False,
        drop_last=True,
    )

    print("Test 4: create batches with drop_last")
    print(f"number of batches = {len(batches)}")

    assert len(batches) == 2

    for X_batch, y_batch in batches:
        assert X_batch.shape == (4, 2)
        assert y_batch.shape == (4,)

    print("Passed.\n")


def test_total_steps():
    dataset_size = 1000
    batch_size = 100
    epochs = 5

    steps_per_epoch = compute_steps_per_epoch(
        dataset_size=dataset_size,
        batch_size=batch_size,
        drop_last=False,
    )

    total_steps = epochs * steps_per_epoch

    print("Test 5: total steps")
    print(f"steps_per_epoch = {steps_per_epoch}")
    print(f"total_steps = {total_steps}")

    assert steps_per_epoch == 10
    assert total_steps == 50

    print("Passed.\n")


def main():
    test_steps_per_epoch_without_drop_last()
    test_steps_per_epoch_with_drop_last()
    test_create_batches_without_shuffle()
    test_create_batches_with_drop_last()
    test_total_steps()

    print("All Day 03 tests passed.")


if __name__ == "__main__":
    main()