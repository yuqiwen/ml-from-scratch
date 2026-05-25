# Day 07: PyTorch Tensor and Autograd

## 1. Goal

Today's goal is to understand PyTorch tensors and automatic differentiation.

Key concepts:

```text
Tensor
requires_grad
computation graph
loss.backward()
parameter.grad
optimizer.step()
optimizer.zero_grad()
```

This is the bridge from NumPy from-scratch ML to real deep learning training loops.

## 2. Tensor

A PyTorch tensor is similar to a NumPy array, but it also supports:

```text
GPU acceleration
automatic differentiation
deep learning computation graphs
```

Example:

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
```

## 3. requires_grad

If a tensor is a trainable parameter, set:

```python
w = torch.tensor(0.0, requires_grad=True)
```

This tells PyTorch to track operations involving `w` so gradients can be computed later.

## 4. Computation Graph

When tensors with `requires_grad=True` participate in computation, PyTorch builds a computation graph.

Example:

```python
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)

y_hat = w * x
loss = y_hat ** 2
```

The graph is:

```text
w -> y_hat -> loss
```

Calling:

```python
loss.backward()
```

computes:

```text
d loss / d w
```

using the chain rule.

## 5. loss.backward()

`loss.backward()` computes gradients for all tensors in the computation graph that have:

```text
requires_grad=True
```

The computed gradient is stored in:

```python
parameter.grad
```

Example:

```python
w.grad
b.grad
```

Important:

```text
loss.backward() computes gradients.
It does not update parameters.
```

## 6. optimizer.step()

`optimizer.step()` updates parameters using their gradients.

For basic SGD:

```text
w = w - learning_rate * w.grad
```

This corresponds to the manual update from previous days:

```python
w -= learning_rate * dw
```

## 7. optimizer.zero_grad()

PyTorch accumulates gradients by default.

If we call `loss.backward()` multiple times without clearing gradients, the gradients will be added together.

Therefore, before each training step, we usually call:

```python
optimizer.zero_grad()
```

This clears old gradients stored in `parameter.grad`.

Important:

```text
zero_grad() clears gradients.
It does not clear parameter values.
```

## 8. Standard Training Step

A standard PyTorch training step is:

```python
y_hat = model(x_batch)
loss = loss_fn(y_hat, y_batch)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Meaning:

```text
forward pass
compute loss
clear old gradients
compute new gradients
update parameters
```

## 9. Connection to Previous From-Scratch Code

NumPy from scratch:

```python
dw = (2 / n) * X.T @ error
db = (2 / n) * sum(error)

w -= lr * dw
b -= lr * db
```

PyTorch autograd:

```python
loss.backward()
optimizer.step()
```

PyTorch automatically computes `dw` and `db`.

The math is the same.

## 10. ML Systems Connection

Autograd is the foundation of modern deep learning frameworks.

In real ML systems, training loops depend on:

```text
automatic differentiation
optimizer state
gradient accumulation
checkpointing
mixed precision
distributed training
```

Understanding autograd is the first step toward understanding training infrastructure.

## 11. Checklist

- [ ] Understand PyTorch Tensor
- [ ] Understand `requires_grad`
- [ ] Understand computation graph
- [ ] Understand `loss.backward()`
- [ ] Understand `.grad`
- [ ] Understand `optimizer.step()`
- [ ] Understand `optimizer.zero_grad()`
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass
