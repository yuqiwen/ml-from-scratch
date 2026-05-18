# Day 01: Linear Regression from Scratch

## 1. Goal

Today's goal is to understand the most basic machine learning training loop:

```text
Input -> Model -> Prediction -> Loss -> Gradient -> Parameter Update
```

Linear regression is the simplest example of this process.

## 2. Model

For one input feature `x`, linear regression uses:

```text
y_hat = w * x + b
```

Where:

- `x` is the input feature
- `y` is the true label
- `y_hat` is the predicted value
- `w` is the weight
- `b` is the bias

The model learns `w` and `b` from data.

## 3. Loss Function

We use Mean Squared Error:

```text
Loss = (1 / n) * sum((y_hat_i - y_i)^2)
```

The loss measures how far the prediction is from the true value.

Smaller loss means the model is better.

## 4. Gradients

Since:

```text
y_hat_i = w * x_i + b
```

The loss is:

```text
Loss(w, b) = (1 / n) * sum((w * x_i + b - y_i)^2)
```

Let:

```text
error_i = y_hat_i - y_i
```

Then the gradients are:

```text
dw = (2 / n) * sum(error_i * x_i)
db = (2 / n) * sum(error_i)
```

These tell us how the loss changes with respect to `w` and `b`.

## 5. Gradient Descent Update

Gradient points in the direction where the loss increases fastest.

To reduce the loss, we move in the opposite direction:

```text
w = w - learning_rate * dw
b = b - learning_rate * db
```

The learning rate controls the step size.

If it is too large, training may become unstable.

If it is too small, training may be very slow.

## 6. Key Intuition

Machine learning is about learning parameters that minimize a loss function.

For linear regression:

```text
Learn w and b so that y_hat = w * x + b fits the data.
```

For neural networks and LLMs, the idea is the same, but the model has many more parameters and a more complex function.

## 7. Connection to Deep Learning and LLMs

Linear Regression:

```text
y_hat = w * x + b
```

Neural Network:

```text
y_hat = neural_network(x; weights)
```

LLM:

```text
next_token_prediction = Transformer(previous_tokens; weights)
```

The training loop is still:

```text
Prediction -> Loss -> Gradient -> Parameter Update
```

So linear regression is the smallest prototype of modern machine learning.

## 8. Checklist

- [ ] Understand linear regression model: `y_hat = w * x + b`
- [ ] Understand MSE loss
- [ ] Understand gradients `dw` and `db`
- [ ] Understand gradient descent update
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass
