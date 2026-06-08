# Day 21: Mixed Precision Training and AMP

## 1. Goal

Today's goal is to understand mixed precision training.

Key concepts:

```text
FP32
FP16
BF16
mixed precision
AMP
autocast
GradScaler
loss scaling
```

---

## 2. Precision

Common floating point types:

```text
FP32 = float32
FP16 = float16
BF16 = bfloat16
```

Memory per number:

```text
FP32: 4 bytes
FP16: 2 bytes
BF16: 2 bytes
```

Lower precision can reduce memory usage and improve GPU throughput.

---

## 3. Mixed Precision

Mixed precision means not everything is forced into low precision.

Some operations use FP16/BF16 for speed.

Some operations stay in FP32 for stability.

This gives a balance between:

```text
speed
memory efficiency
training stability
```

---

## 4. AMP

AMP means:

```text
Automatic Mixed Precision
```

In PyTorch, we use:

```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    output = model(input)
    loss = loss_fn(output, target)
```

`autocast` automatically chooses appropriate precision for supported operations.

---

## 5. Why Not Use FP16 Everywhere?

FP16 has smaller numerical range.

It can cause:

```text
overflow
underflow
gradient becoming zero
loss becoming inf or nan
```

So FP16 training often needs loss scaling.

---

## 6. GradScaler

GradScaler helps prevent FP16 gradient underflow.

Idea:

```text
scale loss up before backward
compute scaled gradients
unscale gradients before optimizer step
adjust scale dynamically
```

PyTorch:

```python
scaler = torch.amp.GradScaler("cuda")

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 7. BF16

BF16 uses 2 bytes like FP16, but has a larger exponent range.

BF16 is often more stable than FP16.

Many modern LLM training systems prefer BF16 when hardware supports it.

BF16 often does not require GradScaler.

---

## 8. Standard FP16 AMP Training Step

```python
optimizer.zero_grad()

with torch.autocast(device_type="cuda", dtype=torch.float16):
    y_hat = model(x_batch)
    loss = loss_fn(y_hat, y_batch)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 9. Standard BF16 AMP Training Step

```python
optimizer.zero_grad()

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    y_hat = model(x_batch)
    loss = loss_fn(y_hat, y_batch)

loss.backward()
optimizer.step()
```

---

## 10. ML Systems Connection

Mixed precision is a core AI infrastructure technique.

It affects:

```text
GPU memory usage
training throughput
activation memory
gradient memory
tensor core utilization
numerical stability
large model feasibility
```

In real configs, you often see:

```yaml
precision: bf16
use_amp: true
gradient_clipping: 1.0
optimizer: AdamW
```

---

## 11. Checklist

- [ ] Understand FP32 / FP16 / BF16
- [ ] Understand why mixed precision saves memory
- [ ] Understand `torch.autocast`
- [ ] Understand GradScaler
- [ ] Understand loss scaling intuition
- [ ] Understand FP16 vs BF16
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass
