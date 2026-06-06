# Day 19: Checkpointing / Save and Resume Training

## 1. Goal

Today's goal is to understand checkpointing in PyTorch training systems.

Key concepts:

```text
checkpoint
model.state_dict()
optimizer.state_dict()
save checkpoint
load checkpoint
resume training
best checkpoint
last checkpoint
```

---

## 2. What Is a Checkpoint?

A checkpoint is a saved training state.

A full training checkpoint usually contains:

```text
model parameters
optimizer state
current epoch
best validation loss
training config
```

Checkpointing allows us to:

```text
resume interrupted training
save best model
debug training
deploy selected model
compare experiments
```

---

## 3. model.state_dict()

`model.state_dict()` returns a dictionary containing model parameters.

Examples:

```text
linear.weight
linear.bias
layer_norm.weight
layer_norm.bias
attention.q_proj.weight
```

It is the model's parameter snapshot.

---

## 4. optimizer.state_dict()

`optimizer.state_dict()` stores optimizer state.

For AdamW, this includes:

```text
first moment
second moment
step count
```

If we resume training without optimizer state, AdamW loses its history.

So for proper resume training, save both:

```text
model_state_dict
optimizer_state_dict
```

---

## 5. Best Checkpoint vs Last Checkpoint

Last checkpoint:

```text
latest training state
used for resuming training
```

Best checkpoint:

```text
model with best validation loss
used for final evaluation or deployment
```

The last checkpoint is not always the best checkpoint.

---

## 6. Saving a Checkpoint

```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_val_loss": best_val_loss,
}, "checkpoint.pt")
```

---

## 7. Loading a Checkpoint

```python
checkpoint = torch.load("checkpoint.pt")

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

start_epoch = checkpoint["epoch"] + 1
best_val_loss = checkpoint["best_val_loss"]
```

---

## 8. ML Systems Connection

Checkpointing is a core ML infrastructure feature.

It matters for:

```text
long training jobs
preemptible GPU instances
fault tolerance
experiment tracking
distributed training
model deployment
resume training correctness
```

For large models, checkpoints can be huge because they may include:

```text
model weights
optimizer states
scheduler states
mixed precision scaler states
distributed shards
```

---

## 9. Checklist

- [ ] Understand checkpoint
- [ ] Understand `model.state_dict()`
- [ ] Understand `optimizer.state_dict()`
- [ ] Understand save checkpoint
- [ ] Understand load checkpoint
- [ ] Understand resume training
- [ ] Understand best vs last checkpoint
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass
