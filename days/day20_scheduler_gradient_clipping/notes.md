# Day 20: Learning Rate Scheduler and Gradient Clipping

## 1. Goal

Today's goal is to understand two common training stability tools:

```text
learning rate scheduler
gradient clipping
```

These are important parts of ML training infrastructure.

---

## 2. Learning Rate Scheduler

A learning rate scheduler changes the optimizer's learning rate during training.

The optimizer updates parameters.

The scheduler updates the learning rate used by the optimizer.

Example:

```python
optimizer.step()
scheduler.step()
```

---

## 3. Why Change Learning Rate?

A fixed learning rate is not always ideal.

Common intuition:

```text
early training:
  larger learning rate helps fast progress

late training:
  smaller learning rate helps stable convergence
```

So the learning rate often decays during training.

---

## 4. StepLR

StepLR reduces the learning rate every fixed number of epochs.

Example:

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5,
)
```

Meaning:

```text
every 5 epochs, multiply lr by 0.5
```

If initial lr is:

```text
0.1
```

Then:

```text
epoch 0-4:   0.1
epoch 5-9:   0.05
epoch 10-14: 0.025
```

---

## 5. CosineAnnealingLR

CosineAnnealingLR changes the learning rate smoothly following a cosine curve.

Example:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
)
```

It usually starts high and gradually decreases.

---

## 6. Gradient Clipping

Gradient clipping prevents gradients from becoming too large.

Large gradients can cause:

```text
unstable updates
loss explosion
NaN values
training divergence
```

PyTorch example:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)
```

---

## 7. Where Gradient Clipping Happens

The order is:

```python
optimizer.zero_grad()
loss.backward()

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
```

Gradient clipping happens after gradients are computed and before parameters are updated.

---

## 8. Training Loop With Scheduler and Clipping

```python
for epoch in range(num_epochs):
    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        loss_fn,
        max_grad_norm=1.0,
    )

    val_loss = evaluate(model, val_loader, loss_fn)

    scheduler.step()
```

---

## 9. ML Systems Connection

Schedulers and gradient clipping are common in real training jobs.

They affect:

```text
training stability
convergence speed
loss spikes
large model training reliability
experiment reproducibility
```

In large-scale training configs, you often see:

```yaml
optimizer: AdamW
learning_rate: 0.0003
scheduler: cosine
warmup_steps: 1000
max_grad_norm: 1.0
```

---

## 10. Checklist

- [ ] Understand why learning rate may change
- [ ] Understand scheduler vs optimizer
- [ ] Understand StepLR
- [ ] Understand CosineAnnealingLR
- [ ] Understand gradient clipping
- [ ] Understand where clipping happens in the loop
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass
