# Day 14: Multi-Head Attention from Scratch

## 1. Goal

Today's goal is to implement multi-head self-attention from scratch.

Key concepts:

```text
d_model
num_heads
head_dim
split heads
scaled dot-product attention
attention weights
concat heads
output projection
```

## 2. From Single-Head to Multi-Head

Single-head attention uses one attention operation:

```text
Q = X @ Wq
K = X @ Wk
V = X @ Wv

Attention(Q, K, V) = softmax(QK.T / sqrt(D)) V
```

Multi-head attention runs multiple attention heads in parallel.

Each head works on a smaller subspace.

## 3. Important Variables

```text
B = batch_size
T = sequence length
D = d_model
H = num_heads
Hd = head_dim = D / H
```

Example:

```text
D = 64
H = 4
Hd = 16
```

## 4. Input Shape

The input hidden states have shape:

```text
X: (B, T, D)
```

Meaning:

```text
B sequences
T tokens per sequence
D-dimensional representation per token
```

## 5. Q / K / V Projection

We first compute:

```text
Q = Wq(X)
K = Wk(X)
V = Wv(X)
```

Shapes:

```text
X: (B, T, D)

Q: (B, T, D)
K: (B, T, D)
V: (B, T, D)
```

Implementation usually uses:

```python
nn.Linear(D, D)
```

for each projection.

## 6. Split Heads

We reshape:

```text
(B, T, D)
```

into:

```text
(B, T, H, Hd)
```

because:

```text
D = H * Hd
```

Then we transpose to:

```text
(B, H, T, Hd)
```

This makes it easy to compute attention independently for each head.

## 7. Attention Scores

For each head:

```text
scores = Q @ K.transpose(-2, -1) / sqrt(Hd)
```

Shapes:

```text
Q:      (B, H, T, Hd)
K.T:    (B, H, Hd, T)

scores: (B, H, T, T)
```

So each head has its own attention matrix.

## 8. Attention Weights

Apply softmax over the last dimension:

```text
attention_weights = softmax(scores, dim=-1)
```

Shape:

```text
attention_weights: (B, H, T, T)
```

Each row sums to 1.

For each token, the row tells how much it attends to every token.

## 9. Weighted Sum of Values

```text
head_output = attention_weights @ V
```

Shapes:

```text
attention_weights: (B, H, T, T)
V:                 (B, H, T, Hd)

head_output:       (B, H, T, Hd)
```

## 10. Concat Heads

We convert:

```text
(B, H, T, Hd)
```

back to:

```text
(B, T, H, Hd)
```

then reshape to:

```text
(B, T, D)
```

This concatenates all heads.

## 11. Output Projection

After concatenating heads, we apply:

```text
output = Wo(concat_output)
```

Shape:

```text
concat_output: (B, T, D)
output:        (B, T, D)
```

The output projection mixes information across heads.

Without it, heads are just placed side by side.

With it, each output dimension can use information from all heads.

## 12. Full Shape Summary

```text
X:       (B, T, D)

Q:       (B, T, D)
K:       (B, T, D)
V:       (B, T, D)

Q split: (B, H, T, Hd)
K split: (B, H, T, Hd)
V split: (B, H, T, Hd)

scores:  (B, H, T, T)
weights: (B, H, T, T)

heads:   (B, H, T, Hd)

concat:  (B, T, D)

output:  (B, T, D)
```

## 13. Multi-Head Attention Formula

```text
head_i = Attention(XWq_i, XWk_i, XWv_i)

concat = [head_1, head_2, ..., head_H]

output = concat @ Wo
```

Where:

```text
Attention(Q, K, V) = softmax(QK.T / sqrt(Hd)) V
```

## 14. ML Systems Connection

Multi-head attention is central to LLM inference.

Important systems implications:

```text
attention score shape is (B, H, T, T)
memory grows with T^2
KV cache stores K/V per layer and per head
head_dim affects attention kernel performance
output projection is a GEMM
```

This connects directly to:

```text
FlashAttention
KV cache
prefill vs decode
tensor parallelism
inference benchmarking
```

## 15. Checklist

- [ ] Understand `D = num_heads * head_dim`
- [ ] Understand Q/K/V projection
- [ ] Understand split heads
- [ ] Understand attention score shape `(B, H, T, T)`
- [ ] Understand concat heads
- [ ] Understand output projection
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass
