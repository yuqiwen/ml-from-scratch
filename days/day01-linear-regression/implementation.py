import numpy as np


class LinearRegressionFromScratch:
    """
    A minimal linear regression model trained with gradient descent.

    Model:
        y_hat = w * x + b

    Loss:
        MSE = mean((y_hat - y)^2)
    """

    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = 0.0
        self.b = 0.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.w * x + self.b

    def compute_loss(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((y_hat - y) ** 2))

    def fit(self, x: np.ndarray, y: np.ndarray, verbose: bool = True) -> None:
        n = len(x)

        for epoch in range(self.epochs):
            y_hat = self.predict(x)
            loss = self.compute_loss(y_hat, y)
            error = y_hat - y

            dw = (2 / n) * np.sum(error * x)
            db = (2 / n) * np.sum(error)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if verbose and epoch % 100 == 0:
                print(
                    f"epoch={epoch:4d}, "
                    f"loss={loss:.6f}, "
                    f"w={self.w:.6f}, "
                    f"b={self.b:.6f}"
                )


def main():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2, 4, 6, 8, 10], dtype=float)

    model = LinearRegressionFromScratch(learning_rate=0.01, epochs=1000)
    model.fit(x, y)

    print("\nFinal result:")
    print(f"w = {model.w:.6f}")
    print(f"b = {model.b:.6f}")

    test_x = np.array([6, 7, 8], dtype=float)
    predictions = model.predict(test_x)

    print("\nPredictions:")
    for xi, pred in zip(test_x, predictions):
        print(f"x={xi}, y_hat={pred:.6f}")


if __name__ == "__main__":
    main()
