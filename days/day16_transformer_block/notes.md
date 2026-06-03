# Day 16: Transformer Block Basics

## 1. Goal

Today's goal is to understand and implement a basic GPT-style Transformer block.

Key concepts:

```text
Transformer block
residual connection
LayerNorm
causal multi-head self-attention
feed-forward network
FFN / MLP
Pre-LN Transformer
```

## 2. Transformer Block Structure

A GPT-style Transformer block usually contains:

```text
causal self-attention
feed-forward network
residual connections
layer normalization
```

A common Pre-LN structure is:

```text
x = x + attention(layer_norm(x))
x = x + ffn(layer_norm(x))
```

## 3. Residual Connection

Residual connection means:

```text
output = x + sublayer(x)
```

Example:

```text
x = x + attention_output
```

Benefits:

```text
preserves original information
helps gradient flow
makes deep networks easier to train
```

The layer only needs to learn a useful correction instead of rewriting the entire representation.

## 4. LayerNorm

LayerNorm normalizes each token representation across the hidden dimension.

If:

```text
x: (B, T, D)
```

Then:

```python
nn.LayerNorm(D)
```

keeps the same shape:

```text
(B, T, D) -> (B, T, D)
```

LayerNorm helps stabilize training.

## 5. Pre-LN Transformer Block

Pre-LN means LayerNorm is applied before each sublayer.

```text
attn_input = ln1(x)
attn_output = attention(attn_input)
x = x + attn_output

ffn_input = ln2(x)
ffn_output = ffn(ffn_input)
x = x + ffn_output
```

This structure is common in modern decoder-only Transformer models.

## 6. Feed-Forward Network

The feed-forward network is usually:

```text
Linear(D, ffn_dim)
GELU
Linear(ffn_dim, D)
```

Often:

```text
ffn_dim = 4 * D
```

Example:

```text
D = 64
ffn_dim = 256
```

The FFN is applied independently to each token.

It does not mix information across tokens.

Token mixing is done by self-attention.

## 7. GELU

GELU is a smooth activation function commonly used in Transformers.

PyTorch:

```python
nn.GELU()
```

For now, it is enough to know that GELU is often used inside Transformer FFNs.

## 8. Shape Summary

Define:

```text
B = batch_size
T = sequence length
D = d_model
H = num_heads
Hd = head_dim
```

Input:

```text
x: (B, T, D)
```

Causal attention:

```text
attention output: (B, T, D)
attention weights: (B, H, T, T)
```

Residual add:

```text
x + attention_output: (B, T, D)
```

FFN:

```text
Linear(D, ffn_dim): (B, T, D) -> (B, T, ffn_dim)
GELU:               (B, T, ffn_dim)
Linear(ffn_dim, D): (B, T, ffn_dim) -> (B, T, D)
```

Final output:

```text
output: (B, T, D)
```

## 9. Why Attention + FFN?

Self-attention:

```text
mixes information across tokens
```

FFN:

```text
transforms each token representation independently
```

Together:

```text
attention handles context
FFN increases nonlinear representation capacity
```

## 10. ML Systems Connection

A Transformer block is the core unit of LLMs.

Large models stack many blocks:

```text
token embeddings
+ positional embeddings
-> Transformer block 1
-> Transformer block 2
-> ...
-> Transformer block N
-> lm_head
-> vocabulary logits
```

Systems topics connected to Transformer blocks:

```text
activation memory
attention memory
KV cache per layer
MLP/FFN GEMMs
residual stream
LayerNorm kernels
operator fusion
mixed precision
tensor parallelism
```

## 11. Checklist

- [ ] Understand Transformer block structure
- [ ] Understand residual connection
- [ ] Understand LayerNorm
- [ ] Understand Pre-LN
- [ ] Understand FFN / MLP
- [ ] Understand attention vs FFN roles
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass
