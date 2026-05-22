import numpy as np


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = True,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into train, validation, and test sets.

    Args:
        X: Feature matrix with shape (n_samples, n_features).
        y: Target vector with shape (n_samples,).
        train_ratio: Ratio of training data.
        val_ratio: Ratio of validation data.
        test_ratio: Ratio of test data.
        shuffle: Whether to shuffle before splitting.
        seed: Random seed for reproducibility.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")

    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0.")

    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    return (
        X[train_indices],
        y[train_indices],
        X[val_indices],
        y[val_indices],
        X[test_indices],
        y[test_indices],
    )


class PolynomialRegressionFromScratch:
    """
    Simple polynomial regression using manually constructed polynomial features.

    This is still linear regression after feature expansion.

    Example:
        degree = 3
        x -> [x, x^2, x^3]
    """

    def __init__(self, degree: int, learning_rate: float = 0.01, epochs: int = 1000):
        self.degree = degree
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w: np.ndarray | None = None
        self.b = 0.0

    def make_features(self, x: np.ndarray) -> np.ndarray:
        """
        Convert one-dimensional input x into polynomial features.

        Args:
            x: Input array with shape (n_samples,).

        Returns:
            Feature matrix with shape (n_samples, degree).
        """
        return np.column_stack([x ** power for power in range(1, self.degree + 1)])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict target values.
        """
        if self.w is None:
            raise ValueError("Model has not been trained yet.")

        X_poly = self.make_features(x)
        return X_poly @ self.w + self.b

    def compute_loss(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Mean Squared Error.
        """
        return float(np.mean((y_hat - y) ** 2))

    def fit(self, x: np.ndarray, y: np.ndarray, verbose: bool = False) -> list[float]:
        """
        Train polynomial regression with gradient descent.

        Returns:
            A list of training losses.
        """
        X_poly = self.make_features(x)
        n_samples, n_features = X_poly.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        losses = []

        for epoch in range(self.epochs):
            y_hat = X_poly @ self.w + self.b
            loss = self.compute_loss(y_hat, y)
            losses.append(loss)

            error = y_hat - y

            dw = (2 / n_samples) * (X_poly.T @ error)
            db = (2 / n_samples) * np.sum(error)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if verbose and epoch % 500 == 0:
                print(f"epoch={epoch}, loss={loss:.6f}")

        return losses


def generate_noisy_quadratic_data(
    n_samples: int = 100,
    noise_std: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a small synthetic regression dataset.

    True relationship:
        y = 2x^2 + 1

    Noise is added to make the problem realistic.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, size=n_samples)
    noise = rng.normal(0, noise_std, size=n_samples)
    y = 2 * x ** 2 + 1 + noise
    return x, y


def main() -> None:
    x, y = generate_noisy_quadratic_data(n_samples=120)

    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    ) = train_val_test_split(x, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    print("Dataset split:")
    print(f"train size = {len(x_train)}")
    print(f"val size   = {len(x_val)}")
    print(f"test size  = {len(x_test)}")

    # Degree 1: likely underfitting for quadratic data.
    model_degree_1 = PolynomialRegressionFromScratch(
        degree=1,
        learning_rate=0.01,
        epochs=3000,
    )
    model_degree_1.fit(x_train, y_train)

    train_loss_d1 = model_degree_1.compute_loss(model_degree_1.predict(x_train), y_train)
    val_loss_d1 = model_degree_1.compute_loss(model_degree_1.predict(x_val), y_val)

    # Degree 2: matches the true data pattern better.
    model_degree_2 = PolynomialRegressionFromScratch(
        degree=2,
        learning_rate=0.01,
        epochs=3000,
    )
    model_degree_2.fit(x_train, y_train)

    train_loss_d2 = model_degree_2.compute_loss(model_degree_2.predict(x_train), y_train)
    val_loss_d2 = model_degree_2.compute_loss(model_degree_2.predict(x_val), y_val)

    print("\nModel comparison:")
    print("Degree 1 model:")
    print(f"train loss = {train_loss_d1:.6f}")
    print(f"val loss   = {val_loss_d1:.6f}")

    print("\nDegree 2 model:")
    print(f"train loss = {train_loss_d2:.6f}")
    print(f"val loss   = {val_loss_d2:.6f}")

    # Use validation loss to pick the better model.
    if val_loss_d2 < val_loss_d1:
        best_model = model_degree_2
        best_name = "degree 2"
    else:
        best_model = model_degree_1
        best_name = "degree 1"

    test_loss = best_model.compute_loss(best_model.predict(x_test), y_test)

    print(f"\nBest model selected by validation loss: {best_name}")
    print(f"Final test loss = {test_loss:.6f}")


if __name__ == "__main__":
    main()