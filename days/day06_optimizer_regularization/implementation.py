import numpy as np


class RegularizedLinearRegression:
    """
    Linear regression with optional L2 regularization.

    Model:
        y_hat = X @ w + b

    Data loss:
        mean((y_hat - y)^2)

    L2 regularized loss:
        data_loss + l2_lambda * sum(w^2)

    L2 gradient:
        dw = data_gradient + 2 * l2_lambda * w
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        l2_lambda: float = 0.0,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_lambda = l2_lambda

        self.w: np.ndarray | None = None
        self.b = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict values using the linear model.
        """
        if self.w is None:
            raise ValueError("Model has not been trained yet.")

        return X @ self.w + self.b

    def compute_data_loss(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """
        Compute MSE data loss.
        """
        return float(np.mean((y_hat - y) ** 2))

    def compute_l2_penalty(self) -> float:
        """
        Compute L2 penalty.

        Bias is not regularized.
        """
        if self.w is None:
            raise ValueError("Model has not been trained yet.")

        return float(self.l2_lambda * np.sum(self.w ** 2))

    def compute_total_loss(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """
        Compute total loss = data loss + L2 penalty.
        """
        data_loss = self.compute_data_loss(y_hat, y)
        l2_penalty = self.compute_l2_penalty()
        return data_loss + l2_penalty

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> list[float]:
        """
        Train model using gradient descent.

        Args:
            X: Feature matrix with shape (n_samples, n_features).
            y: Target values with shape (n_samples,).
            verbose: Whether to print progress.

        Returns:
            List of total losses.
        """
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        losses = []

        for epoch in range(self.epochs):
            # 1. Forward pass
            y_hat = self.predict(X)

            # 2. Compute losses
            data_loss = self.compute_data_loss(y_hat, y)
            total_loss = self.compute_total_loss(y_hat, y)
            losses.append(total_loss)

            # 3. Compute data gradient
            error = y_hat - y
            data_dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            # 4. Add L2 gradient
            l2_dw = 2 * self.l2_lambda * self.w
            dw = data_dw + l2_dw

            # 5. Update parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if verbose and epoch % 100 == 0:
                print(
                    f"epoch={epoch:4d}, "
                    f"data_loss={data_loss:.6f}, "
                    f"total_loss={total_loss:.6f}, "
                    f"w_norm={np.linalg.norm(self.w):.6f}, "
                    f"b={self.b:.6f}"
                )

        return losses


def generate_correlated_data(
    n_samples: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data with correlated features.

    True relationship:
        y = 3*x1 + noise

    x2 is highly correlated with x1, so without regularization,
    weights may become less stable.
    """
    rng = np.random.default_rng(seed)

    x1 = rng.normal(0, 1, size=n_samples)
    x2 = x1 + rng.normal(0, 0.1, size=n_samples)

    X = np.column_stack([x1, x2])

    noise = rng.normal(0, 0.2, size=n_samples)
    y = 3 * x1 + noise

    return X, y


def main() -> None:
    X, y = generate_correlated_data(n_samples=100)

    print("Training without L2 regularization")
    model_no_reg = RegularizedLinearRegression(
        learning_rate=0.01,
        epochs=1000,
        l2_lambda=0.0,
    )
    model_no_reg.fit(X, y, verbose=False)

    y_hat_no_reg = model_no_reg.predict(X)
    data_loss_no_reg = model_no_reg.compute_data_loss(y_hat_no_reg, y)

    print(f"w = {model_no_reg.w}")
    print(f"b = {model_no_reg.b:.6f}")
    print(f"data_loss = {data_loss_no_reg:.6f}")
    print(f"w_norm = {np.linalg.norm(model_no_reg.w):.6f}")

    print("\nTraining with L2 regularization")
    model_l2 = RegularizedLinearRegression(
        learning_rate=0.01,
        epochs=1000,
        l2_lambda=0.1,
    )
    model_l2.fit(X, y, verbose=False)

    y_hat_l2 = model_l2.predict(X)
    data_loss_l2 = model_l2.compute_data_loss(y_hat_l2, y)

    print(f"w = {model_l2.w}")
    print(f"b = {model_l2.b:.6f}")
    print(f"data_loss = {data_loss_l2:.6f}")
    print(f"w_norm = {np.linalg.norm(model_l2.w):.6f}")

    print("\nObservation:")
    print("The L2-regularized model usually has a smaller weight norm.")
    print("Its training data loss may be slightly higher, but the weights are more constrained.")


if __name__ == "__main__":
    main()