import numpy as np

from implementation import (
    PolynomialRegressionFromScratch,
    generate_noisy_quadratic_data,
    train_val_test_split,
)


def test_split_sizes():
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X,
        y,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        shuffle=False,
    )

    print("Test 1: split sizes")
    print(f"train size = {len(X_train)}")
    print(f"val size   = {len(X_val)}")
    print(f"test size  = {len(X_test)}")

    assert len(X_train) == 35
    assert len(X_val) == 7
    assert len(X_test) == 8

    assert len(y_train) == 35
    assert len(y_val) == 7
    assert len(y_test) == 8

    print("Passed.\n")


def test_split_without_shuffle_preserves_order():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X,
        y,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        shuffle=False,
    )

    print("Test 2: split without shuffle preserves order")

    assert np.array_equal(y_train, np.array([0, 1, 2, 3, 4, 5]))
    assert np.array_equal(y_val, np.array([6, 7]))
    assert np.array_equal(y_test, np.array([8, 9]))

    print("Passed.\n")


def test_polynomial_degree_2_better_than_degree_1():
    x, y = generate_noisy_quadratic_data(
        n_samples=120,
        noise_std=0.2,
        seed=1,
    )

    x_train, y_train, x_val, y_val, _, _ = train_val_test_split(
        x,
        y,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        shuffle=True,
        seed=1,
    )

    degree_1_model = PolynomialRegressionFromScratch(
        degree=1,
        learning_rate=0.01,
        epochs=3000,
    )
    degree_1_model.fit(x_train, y_train)

    degree_2_model = PolynomialRegressionFromScratch(
        degree=2,
        learning_rate=0.01,
        epochs=3000,
    )
    degree_2_model.fit(x_train, y_train)

    val_loss_d1 = degree_1_model.compute_loss(degree_1_model.predict(x_val), y_val)
    val_loss_d2 = degree_2_model.compute_loss(degree_2_model.predict(x_val), y_val)

    print("Test 3: degree 2 should fit quadratic data better than degree 1")
    print(f"degree 1 val loss = {val_loss_d1:.6f}")
    print(f"degree 2 val loss = {val_loss_d2:.6f}")

    assert val_loss_d2 < val_loss_d1

    print("Passed.\n")


def main():
    test_split_sizes()
    test_split_without_shuffle_preserves_order()
    test_polynomial_degree_2_better_than_degree_1()

    print("All Day 04 tests passed.")


if __name__ == "__main__":
    main()