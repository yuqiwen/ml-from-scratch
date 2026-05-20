# Day 02: Multiple Linear Regression from Scratch

## 1. Goal

Today's goal is to extend linear regression from one input feature to multiple input features.

Day 01:

```text
y_hat = w * x + b
```

Day 02:

```text
y_hat = w1*x1 + w2*x2 + ... + wd*xd + b
```

In vector form:

```text
y_hat = X @ w + b
```

## 2. Model

For a single sample with multiple features:

```text
x = [x1, x2, ..., xd]
w = [w1, w2, ..., wd]
```

The prediction is:

```text
y_hat = dot(x, w) + b
```

For multiple samples:

```text
y_hat = X @ w + b
```

Where:

- `X` is the feature matrix with shape `(n_samples, n_features)`
- `w` is the weight vector with shape `(n_features,)`
- `b` is the scalar bias
- `y_hat` is the prediction vector with shape `(n_samples,)`

## 3. Loss Function

We still use Mean Squared Error:

```text
Loss = mean((y_hat - y)^2)
```

Since:

```text
y_hat = X @ w + b
```

The loss is:

```text
Loss = mean((X @ w + b - y)^2)
```

## 4. Gradients

Let:

```text
error = y_hat - y
```

Then:

```text
dw = (2 / n) * X.T @ error
db = (2 / n) * sum(error)
```

Where:

- `dw` has the same shape as `w`
- `db` is a scalar

Each element in `dw` tells us how the corresponding feature weight should change.

## 5. Gradient Descent Update

```text
w = w - learning_rate * dw
b = b - learning_rate * db
```

The learning rate controls the step size of each update.

## 6. Key Intuition

Multiple linear regression is still a linear model.

The only difference from Day 01 is that the model now uses multiple input features.

Instead of learning one weight, the model learns one weight per feature.

## 7. Connection to Neural Networks

Multiple linear regression:

```text
y_hat = X @ w + b
```

A neural network layer:

```text
h = X @ W + b
```

So understanding multiple linear regression helps us understand the basic computation inside neural networks.

## 8. Checklist

- [ ] Understand multiple input features
- [ ] Understand vector form: `y_hat = X @ w + b`
- [ ] Understand shapes of `X`, `w`, `b`, and `y_hat`
- [ ] Understand MSE loss for multiple features
- [ ] Understand gradients: `dw = (2 / n) * X.T @ error`
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass
