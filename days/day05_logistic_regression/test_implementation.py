import numpy as np

from implementation import (
    LogisticRegressionFromScratch,
    generate_binary_classification_data,
)


def test_sigmoid_output_range():
    model = LogisticRegressionFromScratch()

    z = np.array([-1000, -1, 0, 1, 1000], dtype=float)
    probabilities = model.sigmoid(z)

    print("Test 1: sigmoid output range")
    print(f"probabilities = {probabilities}")

    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.isclose(probabilities[2], 0.5)

    print("Passed.\n")


def test_binary_cross_entropy_loss_small_when_correct():
    model = LogisticRegressionFromScratch()

    y = np.array([1, 0], dtype=float)

    good_predictions = np.array([0.99, 0.01], dtype=float)
    bad_predictions = np.array([0.01, 0.99], dtype=float)

    good_loss = model.compute_loss(good_predictions, y)
    bad_loss = model.compute_loss(bad_predictions, y)

    print("Test 2: BCE loss is smaller for correct confident predictions")
    print(f"good_loss = {good_loss:.6f}")
    print(f"bad_loss = {bad_loss:.6f}")

    assert good_loss < bad_loss

    print("Passed.\n")


def test_logistic_regression_learns_linearly_separable_data():
    X, y = generate_binary_classification_data(
        n_samples=300,
        seed=1,
    )

    model = LogisticRegressionFromScratch(
        learning_rate=0.1,
        epochs=1000,
    )

    model.fit(X, y, verbose=False)

    accuracy = model.accuracy(X, y)

    print("Test 3: logistic regression learns simple linearly separable data")
    print(f"accuracy = {accuracy:.4f}")

    assert accuracy > 0.95

    print("Passed.\n")


def test_predict_shape():
    X, y = generate_binary_classification_data(
        n_samples=50,
        seed=2,
    )

    model = LogisticRegressionFromScratch(
        learning_rate=0.1,
        epochs=500,
    )

    model.fit(X, y, verbose=False)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    print("Test 4: prediction shapes")
    print(f"predictions shape = {predictions.shape}")
    print(f"probabilities shape = {probabilities.shape}")

    assert predictions.shape == (50,)
    assert probabilities.shape == (50,)

    print("Passed.\n")


def main():
    test_sigmoid_output_range()
    test_binary_cross_entropy_loss_small_when_correct()
    test_logistic_regression_learns_linearly_separable_data()
    test_predict_shape()

    print("All Day 05 tests passed.")


if __name__ == "__main__":
    main()