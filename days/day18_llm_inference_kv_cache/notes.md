# Day 18: LLM Inference, Prefill, Decode, and KV Cache

## 1. Goal

Today's goal is to understand how GPT-style LLM inference works.

Key concepts:

```text
training vs inference
prefill
decode
KV cache
cached K/V
autoregressive generation
memory bandwidth
```

---

## 2. Training vs Inference

During training, we have both inputs and targets.

Example:

```text
tokens = [10, 20, 30, 40, 50]
```

Training uses:

```text
input_ids  = [10, 20, 30, 40]
target_ids = [20, 30, 40, 50]
```

The model outputs:

```text
logits: (B, T, vocab_size)
```

Then we compute cross entropy loss and update parameters:

```python
loss.backward()
optimizer.step()
```

During inference, there are no target tokens and no parameter updates.

Inference is forward-only:

```text
prompt -> logits -> sample next token -> append -> repeat
```

---

## 3. Prefill

Prefill is the first stage of inference.

Given a full prompt:

```text
prompt = [10, 20, 30, 40]
```

The model processes the entire prompt in parallel.

Prefill does two things:

```text
1. computes logits for the prompt
2. builds KV cache for all prompt tokens
```

After prefill:

```text
KV cache length = prompt length
```

Usually, we use the last position logits to generate the first new token:

```python
next_token_logits = logits[:, -1, :]
```

---

## 4. Decode

Decode generates tokens one by one.

After prefill, each decode step usually receives only the newest token:

```text
input_ids: (B, 1)
```

For the new token, the model computes:

```text
Q_new, K_new, V_new
```

Then:

```text
Q_new attends to cached K/V
```

After that, the new K/V are appended to the KV cache.

So:

```text
cache length grows by 1 each decode step
```

---

## 5. KV Cache Shape

For one layer:

```text
K_cache: (B, H, past_len, head_dim)
V_cache: (B, H, past_len, head_dim)
```

Where:

```text
B = batch size
H = num_heads
past_len = number of cached tokens
head_dim = D / H
```

Across the full model:

```text
num_layers x 2 x B x H x past_len x head_dim
```

The `2` means K and V.

---

## 6. Why KV Cache Helps

Without KV cache:

```text
each new token recomputes K/V for all previous tokens
```

With KV cache:

```text
only compute K/V for the new token
reuse cached K/V for previous tokens
```

This greatly speeds up autoregressive decoding.

---

## 7. Why Decode Is Still Expensive

At decode step, the new query still attends to all previous keys.

```text
Q_new:   (B, H, 1, head_dim)
K_cache: (B, H, past_len, head_dim)

scores:  (B, H, 1, past_len)
```

As `past_len` grows, each decode step reads more KV cache.

So:

```text
prefill is compute-heavy
decode is memory-bandwidth-heavy
```

---

## 8. Prefill vs Decode Summary

Prefill:

```text
input: full prompt
shape: (B, T_prompt)
computes K/V for all prompt tokens
initializes KV cache
usually uses last logits for first generation
```

Decode:

```text
input: one new token
shape: (B, 1)
computes K/V only for new token
appends K/V to cache
uses current logits for next token
```

---

## 9. ML Systems Connection

KV cache is a central systems problem in LLM serving.

It affects:

```text
GPU memory usage
memory bandwidth
batch scheduling
long context cost
continuous batching
PagedAttention
latency
throughput
```

Understanding KV cache is essential for AI infrastructure roles.

---

## 10. Checklist

- [ ] Understand training vs inference
- [ ] Understand prefill
- [ ] Understand decode
- [ ] Understand KV cache shape
- [ ] Understand why cache grows with generated tokens
- [ ] Understand why decode is memory-bandwidth-heavy
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass
