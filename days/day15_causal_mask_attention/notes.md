# Day 15: Causal Mask and Autoregressive Attention

## 1. Goal

Today's goal is to understand causal masking in decoder-style Transformer models.

Key concepts:

```text
causal mask
decoder self-attention
autoregressive generation
masked attention scores
prefill
decode
KV cache
```

## 2. Why Causal Mask?

In normal self-attention, every token can attend to every token.

For GPT-style language models, this is not allowed.

When predicting the next token, the model must not see future tokens.

So we enforce:

```text
position i can only attend to positions <= i
```

This is called causal masking.

## 3. Causal Mask Matrix

For sequence length:

```text
T = 4
```

The causal mask is:

```text
[
  [1, 0, 0, 0],
  [1, 1, 0, 0],
  [1, 1, 1, 0],
  [1, 1, 1, 1]
]
```

Where:

```text
1 = allowed
0 = masked
```

## 4. Apply Mask to Attention Scores

Attention scores:

```text
scores = Q @ K.T / sqrt(head_dim)
```

Before softmax, masked positions are set to a very negative value:

```text
scores[masked_positions] = -inf
```

Then:

```text
attention_weights = softmax(scores)
```

Because:

```text
softmax(-inf) = 0
```

future positions get zero attention weight.

## 5. Causal Self-Attention Formula

```text
Q = X @ Wq
K = X @ Wk
V = X @ Wv

scores = Q @ K.T / sqrt(head_dim)

scores = apply_causal_mask(scores)

weights = softmax(scores)

output = weights @ V
```

## 6. Autoregressive Generation

GPT-style models generate one token at a time:

```text
tokens_so_far -> predict next token -> append next token -> repeat
```

The model predicts:

```text
P(next_token | previous_tokens)
```

So future tokens must not be visible during training.

## 7. Training vs Inference

During training:

```text
full sequence is processed in parallel
causal mask prevents future-token leakage
```

During inference:

```text
tokens are generated one by one
KV cache stores previous K/V
new token attends to cached history
```

## 8. Prefill and Decode

Prefill:

```text
process the full prompt
compute K/V for all prompt tokens
initialize KV cache
requires causal mask
```

Decode:

```text
process one new token at a time
append new K/V to cache
new query attends to cached K/V
usually no full T x T causal mask needed
```

## 9. KV Cache Shape

Conceptually:

```text
K_cache: (B, H, past_len, head_dim)
V_cache: (B, H, past_len, head_dim)
```

For each layer, we store K/V for previous tokens.

As generation continues:

```text
past_len grows
KV cache grows
```

## 10. ML Systems Connection

Causal attention is central to LLM inference.

Important systems implications:

```text
prefill is compute-heavy and parallel
decode is memory-bandwidth-heavy
KV cache grows with generated length
attention reads all previous K/V
long context increases memory pressure
```

This connects to:

```text
FlashAttention
PagedAttention
KV cache management
continuous batching
speculative decoding
```

## 11. Checklist

- [ ] Understand why GPT needs causal mask
- [ ] Understand causal mask matrix
- [ ] Understand masking scores before softmax
- [ ] Understand autoregressive generation
- [ ] Understand prefill vs decode
- [ ] Understand KV cache growth
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass
