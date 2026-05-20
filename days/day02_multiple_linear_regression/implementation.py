import numpy as np


class MultipleLinearRegressionFromScratch:
    """
    Multiple Linear Regression trained with gradient descent.

    Model:
        y_hat = X @ w + b

    Loss:
        MSE = mean((y_hat - y)^2)
    """

    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.w = None
        self.b = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predictions.

        Args:
            X: Feature matrix with shape (n_samples, n_features).

        Returns:
            Prediction vector with shape (n_samples,).
        """
        return X @ self.w + self.b

    def compute_loss(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Mean Squared Error.
        """
        return float(np.mean((y_hat - y) ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> None:
        """
        Train the model using gradient descent.

        Args:
            X: Feature matrix with shape (n_samples, n_features).
            y: Target vector with shape (n_samples,).
            verbose: Whether to print training progress.
        """
        n_samples, n_features = X.shape

        # Initialize one weight for each feature.
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.epochs):
            # 1. Forward pass
            y_hat = self.predict(X)

            # 2. Compute loss
            loss = self.compute_loss(y_hat, y)

            # 3. Compute error
            error = y_hat - y

            # 4. Compute gradients
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            # 5. Update parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if verbose and epoch % 100 == 0:
                print(
                    f"epoch={epoch:4d}, "
                    f"loss={loss:.6f}, "
                    f"w={self.w}, "
                    f"b={self.b:.6f}"
                )


def main():
    # Dataset:
    # True relationship:
    # y = 2*x1 + 3*x2 + 1
    X = np.array(
        [
            [1, 1],
            [2, 1],
            [1, 2],
            [3, 2],
            [2, 3],
        ],
        dtype=float,
    )

    y = np.array([6, 8, 9, 13, 14], dtype=float)

    model = MultipleLinearRegressionFromScratch(
        learning_rate=0.01,
        epochs=2000,
    )

    model.fit(X, y)

    print("\nFinal result:")
    print(f"w = {model.w}")
    print(f"b = {model.b:.6f}")

    test_X = np.array(
        [
            [4, 1],
            [1, 4],
        ],
        dtype=float,
    )

    predictions = model.predict(test_X)

    print("\nPredictions:")
    for xi, pred in zip(test_X, predictions):
        print(f"x={xi}, y_hat={pred:.6f}")


if __name__ == "__main__":
    main()