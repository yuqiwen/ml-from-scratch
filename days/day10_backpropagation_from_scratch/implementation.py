import numpy as np


class TwoLayerMLPFromScratch:
    """
    A small two-layer MLP implemented with NumPy.

    Architecture:
        X -> Linear1 -> ReLU -> Linear2 -> output

    Forward:
        Z1 = X @ W1 + b1
        A1 = ReLU(Z1)
        Y_hat = A1 @ W2 + b2

    Loss:
        mean((Y_hat - y)^2)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        learning_rate: float = 0.01,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)

        # Small random initialization.
        self.W1 = 0.1 * rng.normal(size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        self.W2 = 0.1 * rng.normal(size=(hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)

        self.learning_rate = learning_rate

        # Cache values from forward pass for backward pass.
        self.cache: dict[str, np.ndarray] = {}

    def relu(self, X: np.ndarray) -> np.ndarray:
        """
        ReLU activation.
        """
        return np.maximum(0, X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Returns:
            Y_hat with shape (batch_size, output_dim)
        """
        Z1 = X @ self.W1 + self.b1
        A1 = self.relu(Z1)
        Y_hat = A1 @ self.W2 + self.b2

        self.cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "Y_hat": Y_hat,
        }

        return Y_hat

    def compute_loss(self, Y_hat: np.ndarray, y: np.ndarray) -> float:
        """
        Mean Squared Error loss.
        """
        return float(np.mean((Y_hat - y) ** 2))

    def backward(self, y: np.ndarray) -> dict[str, np.ndarray]:
        """
        Manual backpropagation.

        Uses cached values from forward pass.

        Returns:
            Gradients for W1, b1, W2, b2.
        """
        X = self.cache["X"]
        Z1 = self.cache["Z1"]
        A1 = self.cache["A1"]
        Y_hat = self.cache["Y_hat"]

        batch_size = X.shape[0]

        # Loss = mean((Y_hat - y)^2)
        # Since output_dim = 1 in this example:
        # dY_hat = 2 * (Y_hat - y) / batch_size
        dY_hat = 2 * (Y_hat - y) / batch_size

        # Linear2:
        # Y_hat = A1 @ W2 + b2
        dW2 = A1.T @ dY_hat
        db2 = np.sum(dY_hat, axis=0)

        # Gradient passed back into hidden activation.
        dA1 = dY_hat @ self.W2.T

        # ReLU:
        # A1 = ReLU(Z1)
        dZ1 = dA1 * (Z1 > 0)

        # Linear1:
        # Z1 = X @ W1 + b1
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        grads = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
        }

        return grads

    def step(self, grads: dict[str, np.ndarray]) -> None:
        """
        Gradient descent parameter update.
        """
        self.W1 -= self.learning_rate * grads["W1"]
        self.b1 -= self.learning_rate * grads["b1"]
        self.W2 -= self.learning_rate * grads["W2"]
        self.b2 -= self.learning_rate * grads["b2"]

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        One training step:
            forward
            loss
            backward
            update
        """
        Y_hat = self.forward(X)
        loss = self.compute_loss(Y_hat, y)
        grads = self.backward(y)
        self.step(grads)

        return loss


def generate_regression_data(
    n_samples: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate simple nonlinear regression data.

    True function:
        y = x1^2 + 0.5*x2 + noise
    """
    rng = np.random.default_rng(seed)

    X = rng.normal(size=(n_samples, 2))
    noise = 0.1 * rng.normal(size=(n_samples, 1))

    y = (X[:, [0]] ** 2) + 0.5 * X[:, [1]] + noise

    return X, y


def main() -> None:
    X, y = generate_regression_data(n_samples=200, seed=1)

    model = TwoLayerMLPFromScratch(
        input_dim=2,
        hidden_dim=16,
        output_dim=1,
        learning_rate=0.01,
        seed=1,
    )

    print("Training TwoLayerMLPFromScratch")

    for epoch in range(500):
        loss = model.train_step(X, y)

        if epoch % 50 == 0:
            print(f"epoch={epoch:03d}, loss={loss:.6f}")

    final_predictions = model.forward(X)
    final_loss = model.compute_loss(final_predictions, y)

    print("\nFinal result:")
    print(f"final loss = {final_loss:.6f}")


if __name__ == "__main__":
    main()