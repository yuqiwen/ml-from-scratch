# Day 11: PyTorch Optimizer Deep Dive

## 1. Goal

Today's goal is to understand how PyTorch optimizers update model parameters.

Key concepts:

```text
parameter
parameter.grad
optimizer.step()
SGD
momentum
Adam
AdamW
optimizer state
```

## 2. What Optimizer Manages

An optimizer is created from model parameters:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

The optimizer receives references to trainable parameters such as:

```text
linear.weight
linear.bias
```

The optimizer does not compute gradients by itself.

It reads gradients that were computed by:

```python
loss.backward()
```

## 3. backward() vs optimizer.step()

The division of work is:

```text
loss.backward():
  compute gradients
  store gradients in parameter.grad

optimizer.step():
  read parameter.grad
  update parameter values
```

Example:

```text
w = 2.0
w.grad = 4.0
lr = 0.1
```

SGD update:

```text
w = w - lr * w.grad
w = 2.0 - 0.1 * 4.0
w = 1.6
```

## 4. SGD

SGD update rule:

```text
w = w - learning_rate * gradient
```

PyTorch:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

SGD is simple and useful for understanding optimization.

## 5. Momentum

Momentum adds a velocity term:

```text
v = momentum * v + gradient
w = w - learning_rate * v
```

Intuition:

```text
consistent gradient direction -> accelerate
noisy gradient direction -> smooth
```

PyTorch:

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
)
```

## 6. Adam

Adam uses moving averages of gradients and squared gradients.

Simplified:

```text
m = moving average of gradients
v = moving average of squared gradients
```

The update uses something like:

```text
m / sqrt(v)
```

Intuition:

```text
m tracks gradient direction
v tracks gradient scale
```

Adam is adaptive and often easier to use than plain SGD.

PyTorch:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

## 7. AdamW

AdamW is Adam with decoupled weight decay.

L2 regularization modifies the gradient.

Weight decay modifies the parameter update.

For SGD-style optimizers, L2 and weight decay are often equivalent.

For Adam, they are not exactly equivalent because Adam processes gradients.

AdamW applies weight decay separately from the Adam gradient update.

PyTorch:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,
)
```

## 8. Optimizer State

Some optimizers store extra state.

SGD without momentum has little state.

SGD with momentum stores:

```text
velocity
```

Adam / AdamW store:

```text
first moment
second moment
step count
```

This is called optimizer state.

For resuming training, a checkpoint should save both:

```text
model state
optimizer state
```

Example:

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
}, "checkpoint.pt")
```

## 9. ML Systems Connection

Optimizers are a core part of training infrastructure.

They affect:

```text
convergence speed
training stability
memory usage
checkpoint size
resume training correctness
```

Adam and AdamW use extra memory because they store optimizer state for each parameter.

For large models, optimizer state can be a major part of training memory.

## 10. Checklist

- [ ] Understand `parameter.grad`
- [ ] Understand `optimizer.step()`
- [ ] Understand SGD
- [ ] Understand momentum
- [ ] Understand Adam intuition
- [ ] Understand AdamW intuition
- [ ] Understand optimizer state
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass
