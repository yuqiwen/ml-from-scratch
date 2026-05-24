import numpy as np

from implementation import (
    RegularizedLinearRegression,
    generate_correlated_data,
)


def test_l2_penalty_is_zero_when_lambda_zero():
    X, y = generate_correlated_data(n_samples=50, seed=1)

    model = RegularizedLinearRegression(
        learning_rate=0.01,
        epochs=100,
        l2_lambda=0.0,
    )

    model.fit(X, y, verbose=False)

    penalty = model.compute_l2_penalty()

    print("Test 1: L2 penalty is zero when lambda is zero")
    print(f"penalty = {penalty:.6f}")

    assert np.isclose(penalty, 0.0)

    print("Passed.\n")


def test_l2_penalty_is_positive_when_lambda_positive():
    X, y = generate_correlated_data(n_samples=50, seed=2)

    model = RegularizedLinearRegression(
        learning_rate=0.01,
        epochs=100,
        l2_lambda=0.1,
    )

    model.fit(X, y, verbose=False)

    penalty = model.compute_l2_penalty()

    print("Test 2: L2 penalty is positive when lambda is positive")
    print(f"penalty = {penalty:.6f}")

    assert penalty > 0.0

    print("Passed.\n")


def test_l2_reduces_weight_norm():
    X, y = generate_correlated_data(n_samples=100, seed=3)

    model_no_reg = RegularizedLinearRegression(
        learning_rate=0.01,
        epochs=1000,
        l2_lambda=0.0,
    )

    model_l2 = RegularizedLinearRegression(
        learning_rate=0.01,
        epochs=1000,
        l2_lambda=0.1,
    )

    model_no_reg.fit(X, y, verbose=False)
    model_l2.fit(X, y, verbose=False)

    norm_no_reg = np.linalg.norm(model_no_reg.w)
    norm_l2 = np.linalg.norm(model_l2.w)

    print("Test 3: L2 regularization reduces weight norm")
    print(f"norm_no_reg = {norm_no_reg:.6f}")
    print(f"norm_l2 = {norm_l2:.6f}")

    assert norm_l2 < norm_no_reg

    print("Passed.\n")


def test_training_loss_decreases():
    X, y = generate_correlated_data(n_samples=100, seed=4)

    model = RegularizedLinearRegression(
        learning_rate=0.01,
        epochs=500,
        l2_lambda=0.01,
    )

    losses = model.fit(X, y, verbose=False)

    print("Test 4: training loss decreases")
    print(f"first loss = {losses[0]:.6f}")
    print(f"last loss = {losses[-1]:.6f}")

    assert losses[-1] < losses[0]

    print("Passed.\n")


def main():
    test_l2_penalty_is_zero_when_lambda_zero()
    test_l2_penalty_is_positive_when_lambda_positive()
    test_l2_reduces_weight_norm()
    test_training_loss_decreases()

    print("All Day 06 tests passed.")


if __name__ == "__main__":
    main()
