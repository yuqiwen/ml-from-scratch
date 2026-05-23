# Day 05: Logistic Regression from Scratch

## 1. Goal

Today's goal is to understand logistic regression for binary classification.

Linear regression predicts continuous values.

Logistic regression predicts probabilities for binary labels:

```text
y = 0 or y = 1
```

## 2. Why Not Use Linear Regression for Classification?

Linear regression outputs any real number:

```text
y_hat = X @ w + b
```

But classification needs a probability:

```text
0 <= p <= 1
```

So we use the sigmoid function to convert a linear score into a probability.

## 3. Sigmoid Function

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

Where:

```text
z = X @ w + b
```

The logistic regression model is:

```text
p = sigmoid(X @ w + b)
```

Here, `p` means:

```text
P(y = 1 | x)
```

## 4. Prediction Rule

To convert probability into class label:

```text
if p >= 0.5:
    predict 1
else:
    predict 0
```

The threshold can be adjusted depending on the application.

## 5. Binary Cross Entropy Loss

For binary classification, we use Binary Cross Entropy:

```text
loss = -mean(y * log(p) + (1 - y) * log(1 - p))
```

Where:

- `y` is the true label, either 0 or 1
- `p` is the predicted probability of class 1

## 6. Loss Intuition

If the true label is `y = 1`:

```text
loss = -log(p)
```

So:

```text
p close to 1 -> small loss
p close to 0 -> large loss
```

If the true label is `y = 0`:

```text
loss = -log(1 - p)
```

So:

```text
p close to 0 -> small loss
p close to 1 -> large loss
```

Cross entropy heavily penalizes confident wrong predictions.

## 7. Gradients

Let:

```text
p = sigmoid(X @ w + b)
error = p - y
```

Then:

```text
dw = (1 / n) * X.T @ error
db = (1 / n) * sum(error)
```

This is similar to linear regression, but the prediction is now a probability.

## 8. Optimizer Update

Using gradient descent:

```text
w = w - learning_rate * dw
b = b - learning_rate * db
```

## 9. Connection to Neural Networks

Logistic regression can be viewed as a neural network with no hidden layer:

```text
input -> linear layer -> sigmoid -> probability
```

A simple neural network adds hidden layers:

```text
input -> linear layer -> activation -> linear layer -> sigmoid
```

## 10. Connection to AI Infra / LLMs

Logistic regression introduces classification and cross entropy.

This matters because modern language models are also trained with cross entropy.

Binary classification:

```text
choose between class 0 and class 1
```

LLM next-token prediction:

```text
choose the next token from a vocabulary
```

The idea of predicting probabilities and penalizing wrong confident predictions is shared.

## 11. Checklist

- [ ] Understand binary classification
- [ ] Understand sigmoid
- [ ] Understand probability output
- [ ] Understand binary cross entropy
- [ ] Understand logistic regression gradients
- [ ] Understand prediction threshold
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass
