# Day 06: Optimizer and Regularization

## 1. Goal

Today's goal is to understand optimizers and L2 regularization.

Key concepts:

```text
optimizer
gradient descent
SGD
mini-batch SGD
L2 regularization
weight decay
overfitting
```

## 2. Optimizer

An optimizer is a rule for updating model parameters using gradients.

For basic gradient descent:

```text
w = w - learning_rate * dw
b = b - learning_rate * db
```

In PyTorch, this corresponds to:

```python
loss.backward()
optimizer.step()
```

`loss.backward()` computes gradients.

`optimizer.step()` updates parameters.

## 3. Full-Batch GD, SGD, and Mini-Batch SGD

The update formula can be the same:

```text
parameter = parameter - learning_rate * gradient
```

The difference is how much data is used to compute the gradient.

### Full-Batch Gradient Descent

```text
batch_size = dataset_size
```

Uses the full training set for one update.

### Stochastic Gradient Descent

```text
batch_size = 1
```

Uses one sample for one update.

### Mini-Batch SGD

```text
batch_size = 32, 64, 128, ...
```

Uses a small batch of samples for one update.

Mini-batch SGD is the most common approach in modern deep learning.

## 4. Regularization

Regularization adds a penalty term to the loss.

Original loss:

```text
loss = data_loss
```

Regularized loss:

```text
loss = data_loss + regularization_penalty
```

The goal is to reduce overfitting by discouraging overly complex models.

## 5. L2 Regularization

L2 regularization penalizes large weights:

```text
loss = data_loss + lambda * sum(w^2)
```

Where:

```text
lambda = regularization strength
```

Usually, we regularize weights `w`, but not bias `b`.

## 6. L2 Gradient

For the regularization term:

```text
lambda * sum(w^2)
```

The gradient is:

```text
2 * lambda * w
```

So the total gradient becomes:

```text
dw = data_gradient + 2 * lambda * w
```

The update becomes:

```text
w = w - learning_rate * (data_gradient + 2 * lambda * w)
```

This pushes weights toward smaller values.

## 7. Weight Decay

Weight decay is closely related to L2 regularization.

A simple weight decay update can be written as:

```text
w = w * (1 - learning_rate * weight_decay) - learning_rate * data_gradient
```

This means every update slightly shrinks the weight values.

## 8. Why Regularization Helps

Without regularization, a model may use very large weights to fit noise in the training set.

With L2 regularization, large weights are penalized.

This often leads to:

```text
slightly higher training loss
lower validation loss
better generalization
```

## 9. ML Systems Connection

In real training systems, optimizer settings are part of the training configuration:

```yaml
optimizer: AdamW
learning_rate: 0.0001
weight_decay: 0.01
batch_size: 64
epochs: 10
```

These values strongly affect training stability, convergence speed, memory usage, and generalization.

## 10. Checklist

- [ ] Understand what an optimizer does
- [ ] Understand full-batch GD
- [ ] Understand SGD
- [ ] Understand mini-batch SGD
- [ ] Understand L2 regularization
- [ ] Understand L2 gradient
- [ ] Understand weight decay intuition
- [ ] Understand why regularization can reduce overfitting
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass
