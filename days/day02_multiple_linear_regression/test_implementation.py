import numpy as np

from implementation import (
    MultipleLinearRegressionFromScratch,
)


def test_two_feature_linear_regression():
    """
    Test whether the model can learn:

        y = 2*x1 + 3*x2 + 1
    """
    X = np.array(
        [
            [1, 1],
            [2, 1],
            [1, 2],
            [3, 2],
            [2, 3],
            [4, 1],
            [1, 4],
        ],
        dtype=float,
    )

    y = np.array([6, 8, 9, 13, 14, 12, 15], dtype=float)

    model = MultipleLinearRegressionFromScratch(
        learning_rate=0.01,
        epochs=5000,
    )

    model.fit(X, y, verbose=False)

    print("Test: y = 2*x1 + 3*x2 + 1")
    print(f"Learned w = {model.w}")
    print(f"Learned b = {model.b:.6f}")

    assert abs(model.w[0] - 2.0) < 0.05
    assert abs(model.w[1] - 3.0) < 0.05
    assert abs(model.b - 1.0) < 0.10

    print("Passed.\n")


def test_prediction_shape():
    """
    Test whether prediction shape is correct.
    """
    X = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
        ],
        dtype=float,
    )

    y = np.array([9, 19, 29], dtype=float)

    model = MultipleLinearRegressionFromScratch(
        learning_rate=0.01,
        epochs=1000,
    )

    model.fit(X, y, verbose=False)

    predictions = model.predict(X)

    print("Test: prediction shape")
    print(f"Prediction shape = {predictions.shape}")

    assert predictions.shape == (3,)

    print("Passed.\n")


def main():
    test_two_feature_linear_regression()
    test_prediction_shape()
    print("All Day 02 tests passed.")


if __name__ == "__main__":
    main()
