# Day 08: PyTorch Training Loop

## 1. Goal

Today's goal is to build a complete PyTorch training loop.

Key concepts:

```text
Dataset
DataLoader
batch
epoch
train_one_epoch
evaluate
model.train()
model.eval()
torch.no_grad()
best validation loss
```

This is the first step from basic PyTorch autograd to real ML training infrastructure.

## 2. Training Loop Overview

A common training loop looks like:

```python
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
    val_loss = evaluate(model, val_loader, loss_fn)

    if val_loss < best_val_loss:
        save_best_model()
```

Each epoch has two phases:

```text
training phase:
  update model parameters

validation phase:
  evaluate model without updating parameters
```

## 3. One Training Step

For one batch:

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

## 4. Dataset and DataLoader

`TensorDataset` wraps tensors into a dataset:

```python
dataset = TensorDataset(X, y)
```

`DataLoader` creates mini-batches:

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
)
```

Then we can iterate:

```python
for x_batch, y_batch in loader:
    ...
```

## 5. model.train()

`model.train()` switches the model to training mode.

This matters for layers like:

```text
Dropout
BatchNorm
```

Even if a simple linear model does not behave differently, it is still standard practice to call it during training.

## 6. model.eval()

`model.eval()` switches the model to evaluation mode.

Use it during:

```text
validation
testing
inference
```

## 7. torch.no_grad()

During validation, we do not need gradients.

So we use:

```python
with torch.no_grad():
    ...
```

This prevents PyTorch from building a computation graph.

Benefits:

```text
lower memory usage
faster evaluation
no accidental gradient accumulation
```

## 8. Best Validation Loss

Training loss can keep decreasing while validation loss starts increasing.

So we often track:

```text
best_val_loss
best_model_state
```

A common rule is:

```text
save the model with the lowest validation loss
```

not necessarily the final epoch.

## 9. ML Systems Connection

A real ML training system usually separates:

```text
data loading
training step
evaluation step
logging
checkpointing
configuration
metrics
```

Today's training loop is the foundation for later topics:

```text
checkpointing
mixed precision
batching
model serving
inference benchmark
distributed training
```

## 10. Checklist

- [ ] Understand Dataset
- [ ] Understand DataLoader
- [ ] Understand one training step
- [ ] Understand `train_one_epoch`
- [ ] Understand `evaluate`
- [ ] Understand `model.train()`
- [ ] Understand `model.eval()`
- [ ] Understand `torch.no_grad()`
- [ ] Understand best validation loss
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass
