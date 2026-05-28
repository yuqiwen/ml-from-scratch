# Day 10: Backpropagation from Scratch

## 1. Goal

Today's goal is to understand backpropagation.

Backpropagation is the algorithm used to compute gradients in neural networks.

Key concepts:

```text
computation graph
chain rule
local gradient
upstream gradient
linear layer backward
ReLU backward
manual backpropagation
```

## 2. Backpropagation

Backpropagation means:

```text
starting from the loss,
move backward through the computation graph,
use chain rule,
compute gradients for parameters.
```

In PyTorch, this is done by:

```python
loss.backward()
```

Today, we implement the same idea manually.

## 3. Chain Rule

If:

```text
z = f(y)
y = g(x)
```

Then:

```text
dz/dx = dz/dy * dy/dx
```

In a computation graph:

```text
upstream gradient * local gradient = downstream gradient
```

## 4. Local Gradient and Upstream Gradient

For:

```text
y = f(x)
loss = L(y)
```

To compute:

```text
dLoss/dx
```

we use:

```text
dLoss/dx = dLoss/dy * dy/dx
```

Where:

```text
dLoss/dy = upstream gradient
dy/dx = local gradient
dLoss/dx = downstream gradient
```

## 5. Linear Layer Forward

A linear layer computes:

```text
Y = X @ W + b
```

Shapes:

```text
X: (batch_size, input_dim)
W: (input_dim, output_dim)
b: (output_dim,)
Y: (batch_size, output_dim)
```

## 6. Linear Layer Backward

Given upstream gradient:

```text
dY = dLoss/dY
```

The gradients are:

```text
dW = X.T @ dY
db = sum(dY over batch)
dX = dY @ W.T
```

## 7. ReLU Forward

```text
Y = max(0, X)
```

## 8. ReLU Backward

```text
dX = dY * (X > 0)
```

If the original input was positive, gradient passes through.

If the original input was zero or negative, gradient becomes zero.

## 9. Two-Layer MLP

Forward:

```text
Z1 = X @ W1 + b1
A1 = ReLU(Z1)
Y_hat = A1 @ W2 + b2
Loss = mean((Y_hat - y)^2)
```

Backward:

```text
dY_hat = 2 * (Y_hat - y) / n

dW2 = A1.T @ dY_hat
db2 = sum(dY_hat)

dA1 = dY_hat @ W2.T

dZ1 = dA1 * (Z1 > 0)

dW1 = X.T @ dZ1
db1 = sum(dZ1)
```

## 10. Shape Analysis for Backward Pass

Shape analysis is the key to understanding backpropagation.

Define:

```text
B = batch_size
D = input_dim
H = hidden_dim
O = output_dim
```

The two-layer MLP is:

```text
X -> Linear1 -> ReLU -> Linear2 -> Y_hat
```

### 10.1 Forward Shapes

Input:

```text
X: (B, D)
```

First linear layer:

```text
Z1 = X @ W1 + b1
```

Shapes:

```text
X:  (B, D)
W1: (D, H)
b1: (H,)
Z1: (B, H)
```

ReLU:

```text
A1 = ReLU(Z1)
```

Shapes:

```text
Z1: (B, H)
A1: (B, H)
```

Second linear layer:

```text
Y_hat = A1 @ W2 + b2
```

Shapes:

```text
A1:    (B, H)
W2:    (H, O)
b2:    (O,)
Y_hat: (B, O)
```

Label:

```text
y: (B, O)
```

Loss:

```text
Loss = mean((Y_hat - y)^2)
```

Shape:

```text
Loss: scalar
```

### 10.2 Gradient of Loss with Respect to Y_hat

Since:

```text
Y_hat: (B, O)
y:     (B, O)
```

The error has shape:

```text
Y_hat - y: (B, O)
```

For MSE:

```text
Loss = mean((Y_hat - y)^2)
```

In the simplified case where the mean is taken over the batch dimension:

```text
dY_hat = 2 * (Y_hat - y) / B
```

Shape:

```text
dY_hat: (B, O)
```

The gradient of a tensor has the same shape as the tensor itself.

So:

```text
dLoss/dY_hat has the same shape as Y_hat
```

### 10.3 Linear2 Backward Shapes

Forward:

```text
Y_hat = A1 @ W2 + b2
```

Shapes:

```text
A1:     (B, H)
W2:     (H, O)
b2:     (O,)
Y_hat:  (B, O)
```

Upstream gradient:

```text
dY_hat: (B, O)
```

Gradient for `W2`:

```text
dW2 = A1.T @ dY_hat
```

Shapes:

```text
A1.T:   (H, B)
dY_hat: (B, O)
dW2:    (H, O)
```

This matches:

```text
W2:     (H, O)
```

Gradient for `b2`:

```text
db2 = sum(dY_hat over batch)
```

Shapes:

```text
dY_hat: (B, O)
db2:    (O,)
```

This matches:

```text
b2:     (O,)
```

Gradient passed back to `A1`:

```text
dA1 = dY_hat @ W2.T
```

Shapes:

```text
dY_hat: (B, O)
W2.T:   (O, H)
dA1:    (B, H)
```

This matches:

```text
A1:     (B, H)
```

### 10.4 ReLU Backward Shapes

Forward:

```text
A1 = ReLU(Z1)
```

Shapes:

```text
Z1: (B, H)
A1: (B, H)
```

Backward:

```text
dZ1 = dA1 * (Z1 > 0)
```

Shapes:

```text
dA1:     (B, H)
Z1 > 0:  (B, H)
dZ1:     (B, H)
```

This matches:

```text
Z1:      (B, H)
```

ReLU does not change the shape. It only gates the gradient:

```text
if Z1 > 0:
    gradient passes through

if Z1 <= 0:
    gradient becomes 0
```

### 10.5 Linear1 Backward Shapes

Forward:

```text
Z1 = X @ W1 + b1
```

Shapes:

```text
X:  (B, D)
W1: (D, H)
b1: (H,)
Z1: (B, H)
```

Upstream gradient:

```text
dZ1: (B, H)
```

Gradient for `W1`:

```text
dW1 = X.T @ dZ1
```

Shapes:

```text
X.T: (D, B)
dZ1: (B, H)
dW1: (D, H)
```

This matches:

```text
W1:  (D, H)
```

Gradient for `b1`:

```text
db1 = sum(dZ1 over batch)
```

Shapes:

```text
dZ1: (B, H)
db1: (H,)
```

This matches:

```text
b1:  (H,)
```

### 10.6 Full Shape Table

Forward:

```text
X:      (B, D)

W1:     (D, H)
b1:     (H,)
Z1:     (B, H)

A1:     (B, H)

W2:     (H, O)
b2:     (O,)
Y_hat:  (B, O)

y:      (B, O)
Loss:   scalar
```

Backward:

```text
dY_hat: (B, O)

dW2:    (H, O)
db2:    (O,)

dA1:    (B, H)

dZ1:    (B, H)

dW1:    (D, H)
db1:    (H,)
```

### 10.7 Most Important Shape Rules

Rule 1:

```text
The gradient of an activation has the same shape as that activation.
```

Examples:

```text
Y_hat:  (B, O)  -> dY_hat: (B, O)
A1:     (B, H)  -> dA1:    (B, H)
Z1:     (B, H)  -> dZ1:    (B, H)
```

Rule 2:

```text
The gradient of a parameter has the same shape as that parameter.
```

Examples:

```text
W2: (H, O) -> dW2: (H, O)
b2: (O,)   -> db2: (O,)

W1: (D, H) -> dW1: (D, H)
b1: (H,)   -> db1: (H,)
```

Rule 3:

```text
For a linear layer Y = X @ W + b:
```

```text
dW = X.T @ dY
db = sum(dY over batch)
dX = dY @ W.T
```

This rule appears again and again in neural networks.

### 10.8 Concrete Example

Suppose:

```text
B = 4
D = 2
H = 3
O = 1
```

Forward:

```text
X:      (4, 2)

W1:     (2, 3)
b1:     (3,)
Z1:     (4, 3)
A1:     (4, 3)

W2:     (3, 1)
b2:     (1,)
Y_hat:  (4, 1)

y:      (4, 1)
Loss:   scalar
```

Backward:

```text
dY_hat: (4, 1)

dW2 = A1.T @ dY_hat
A1.T:   (3, 4)
dY_hat: (4, 1)
dW2:    (3, 1)

db2:    (1,)

dA1 = dY_hat @ W2.T
dY_hat: (4, 1)
W2.T:   (1, 3)
dA1:    (4, 3)

dZ1:    (4, 3)

dW1 = X.T @ dZ1
X.T:    (2, 4)
dZ1:    (4, 3)
dW1:    (2, 3)

db1:    (3,)
```

### 10.9 Key Intuition

Backpropagation becomes much easier if the shapes are clear.

The most useful debugging rule is:

```text
Every parameter gradient must have the same shape as the parameter.
```

So:

```text
dW1 must match W1
db1 must match b1
dW2 must match W2
db2 must match b2
```

This is also how we debug shape errors in larger models such as CNNs, Transformers, and LLMs.

## 11. Connection to PyTorch

Manual backprop:

```python
grads = model.backward(...)
```

PyTorch autograd:

```python
loss.backward()
```

The math is the same.

PyTorch just builds the computation graph and applies chain rule automatically.

## 12. ML Systems Connection

Backpropagation is the core of training infrastructure.

Modern systems optimize around:

```text
forward pass
activation storage
backward pass
gradient memory
optimizer state
checkpointing
mixed precision
distributed gradient communication
```

Understanding backprop helps explain why training uses much more memory than inference.

## 13. Checklist

- [ ] Understand backpropagation
- [ ] Understand chain rule
- [ ] Understand local gradient
- [ ] Understand upstream gradient
- [ ] Understand linear layer backward
- [ ] Understand ReLU backward
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass
