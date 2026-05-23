import numpy as np


class LogisticRegressionFromScratch:
    """
    Logistic Regression for binary classification.

    Model:
        z = X @ w + b
        p = sigmoid(z)

    Loss:
        Binary Cross Entropy

    Gradients:
        error = p - y
        dw = (1/n) * X.T @ error
        db = (1/n) * sum(error)
    """

    def __init__(self, learning_rate: float = 0.1, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w: np.ndarray | None = None
        self.b = 0.0

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid function.

        sigmoid(z) = 1 / (1 + exp(-z))

        np.clip is used to avoid overflow when z is very large or very small.
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability P(y = 1 | x).
        """
        if self.w is None:
            raise ValueError("Model has not been trained yet.")

        z = X @ self.w + self.b
        return self.sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels using a probability threshold.
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def compute_loss(self, probabilities: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Binary Cross Entropy loss.

        Add epsilon to avoid log(0).
        """
        epsilon = 1e-12
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)

        loss = -np.mean(
            y * np.log(probabilities)
            + (1 - y) * np.log(1 - probabilities)
        )

        return float(loss)

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> list[float]:
        """
        Train logistic regression using gradient descent.

        Args:
            X: Feature matrix with shape (n_samples, n_features).
            y: Binary labels with shape (n_samples,).
            verbose: Whether to print training progress.

        Returns:
            A list of loss values.
        """
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        losses = []

        for epoch in range(self.epochs):
            # 1. Forward pass
            probabilities = self.predict_proba(X)

            # 2. Compute loss
            loss = self.compute_loss(probabilities, y)
            losses.append(loss)

            # 3. Compute gradients
            error = probabilities - y

            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            # 4. Update parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if verbose and epoch % 100 == 0:
                accuracy = self.accuracy(X, y)
                print(
                    f"epoch={epoch:4d}, "
                    f"loss={loss:.6f}, "
                    f"accuracy={accuracy:.4f}, "
                    f"w={self.w}, "
                    f"b={self.b:.6f}"
                )

        return losses

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.
        """
        predictions = self.predict(X)
        return float(np.mean(predictions == y))


def generate_binary_classification_data(
    n_samples: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a simple linearly separable binary classification dataset.

    Rule:
        y = 1 if x1 + x2 > 0
        y = 0 otherwise
    """
    rng = np.random.default_rng(seed)

    X = rng.normal(0, 1, size=(n_samples, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    return X, y


def main() -> None:
    X, y = generate_binary_classification_data(n_samples=200)

    model = LogisticRegressionFromScratch(
        learning_rate=0.1,
        epochs=1000,
    )

    model.fit(X, y, verbose=True)

    final_accuracy = model.accuracy(X, y)

    print("\nFinal result:")
    print(f"w = {model.w}")
    print(f"b = {model.b:.6f}")
    print(f"accuracy = {final_accuracy:.4f}")

    test_X = np.array(
        [
            [2.0, 1.0],
            [-2.0, -1.0],
            [1.0, -0.2],
            [-1.0, 0.2],
        ]
    )

    probabilities = model.predict_proba(test_X)
    predictions = model.predict(test_X)

    print("\nPredictions:")
    for x, p, pred in zip(test_X, probabilities, predictions):
        print(f"x={x}, probability={p:.4f}, prediction={pred}")


if __name__ == "__main__":
    main()