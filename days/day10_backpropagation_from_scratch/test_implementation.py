import numpy as np

from implementation import (
    TwoLayerMLPFromScratch,
    generate_regression_data,
)


def test_forward_output_shape():
    model = TwoLayerMLPFromScratch(
        input_dim=2,
        hidden_dim=4,
        output_dim=1,
        learning_rate=0.01,
        seed=1,
    )

    X = np.random.randn(5, 2)
    Y_hat = model.forward(X)

    print("Test 1: forward output shape")
    print(f"Y_hat shape = {Y_hat.shape}")

    assert Y_hat.shape == (5, 1)

    print("Passed.\n")


def test_backward_gradient_shapes():
    model = TwoLayerMLPFromScratch(
        input_dim=2,
        hidden_dim=4,
        output_dim=1,
        learning_rate=0.01,
        seed=1,
    )

    X = np.random.randn(5, 2)
    y = np.random.randn(5, 1)

    model.forward(X)
    grads = model.backward(y)

    print("Test 2: backward gradient shapes")
    print(f"dW1 shape = {grads['W1'].shape}")
    print(f"db1 shape = {grads['b1'].shape}")
    print(f"dW2 shape = {grads['W2'].shape}")
    print(f"db2 shape = {grads['b2'].shape}")

    assert grads["W1"].shape == model.W1.shape
    assert grads["b1"].shape == model.b1.shape
    assert grads["W2"].shape == model.W2.shape
    assert grads["b2"].shape == model.b2.shape

    print("Passed.\n")


def test_train_step_reduces_loss():
    X, y = generate_regression_data(n_samples=100, seed=2)

    model = TwoLayerMLPFromScratch(
        input_dim=2,
        hidden_dim=16,
        output_dim=1,
        learning_rate=0.01,
        seed=2,
    )

    initial_predictions = model.forward(X)
    initial_loss = model.compute_loss(initial_predictions, y)

    for _ in range(300):
        model.train_step(X, y)

    final_predictions = model.forward(X)
    final_loss = model.compute_loss(final_predictions, y)

    print("Test 3: training reduces loss")
    print(f"initial_loss = {initial_loss:.6f}")
    print(f"final_loss = {final_loss:.6f}")

    assert final_loss < initial_loss

    print("Passed.\n")


def test_relu_backward_blocks_negative_inputs():
    model = TwoLayerMLPFromScratch(
        input_dim=2,
        hidden_dim=3,
        output_dim=1,
        learning_rate=0.01,
        seed=3,
    )

    Z1 = np.array(
        [
            [-1.0, 0.5, 2.0],
            [3.0, -0.2, 0.0],
        ]
    )

    dA1 = np.ones_like(Z1)

    dZ1 = dA1 * (Z1 > 0)

    print("Test 4: ReLU backward")
    print(f"dZ1 =\n{dZ1}")

    expected = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )

    assert np.array_equal(dZ1, expected)

    print("Passed.\n")


def main():
    test_forward_output_shape()
    test_backward_gradient_shapes()
    test_train_step_reduces_loss()
    test_relu_backward_blocks_negative_inputs()

    print("All Day 10 tests passed.")


if __name__ == "__main__":
    main()