# Day 13: Transformer Basics

## 1. Goal

Today's goal is to understand the foundation of Transformer models.

Key concepts:

```text
token
embedding
sequence
self-attention
Q / K / V
attention scores
softmax
scaled dot-product attention
multi-head attention intuition
```

## 2. Why Transformer?

MLP works on vector inputs.

CNN works well on images and local spatial patterns.

Transformer works well on sequences because each token can dynamically attend to other tokens.

This is the core idea behind modern language models.

## 3. Token

A token is a discrete unit of text.

Example:

```text
"I love machine learning"
```

can be represented as token IDs:

```text
[2, 5, 9, 4]
```

The model works with token IDs, not raw strings.

## 4. Embedding

Token IDs are converted into vectors using an embedding table.

PyTorch:

```python
embedding = nn.Embedding(vocab_size, d_model)
```

If:

```text
tokens: (batch_size, seq_len)
```

Then:

```text
X = embedding(tokens)
X: (batch_size, seq_len, d_model)
```

## 5. Self-Attention Intuition

Self-attention answers:

```text
For each token, which other tokens should it pay attention to?
```

Each token produces a new representation by mixing information from other tokens.

The mixing weights are computed dynamically from the input itself.

## 6. Q / K / V

Each token representation is projected into:

```text
Q = Query
K = Key
V = Value
```

Intuition:

```text
Query: what information am I looking for?
Key: what information do I contain for matching?
Value: what content do I provide if attended to?
```

Formula:

```text
Q = X @ Wq
K = X @ Wk
V = X @ Wv
```

Shapes:

```text
X:  (B, T, D)
Wq: (D, D)
Wk: (D, D)
Wv: (D, D)

Q:  (B, T, D)
K:  (B, T, D)
V:  (B, T, D)
```

Where:

```text
B = batch_size
T = sequence length
D = d_model
```

## 7. Attention Scores

Attention scores are computed with dot products:

```text
scores = Q @ K.T
```

In PyTorch batch form:

```python
scores = Q @ K.transpose(-2, -1)
```

Shapes:

```text
Q:      (B, T, D)
K.T:    (B, D, T)
scores: (B, T, T)
```

`scores[i, j]` means how much token `i` attends to token `j`.

## 8. Scaled Dot-Product Attention

The scores are scaled by:

```text
sqrt(D)
```

Formula:

```text
scores = (Q @ K.T) / sqrt(D)
```

This prevents dot products from becoming too large when `D` is large.

## 9. Softmax

Softmax converts raw scores into attention weights:

```text
attention_weights = softmax(scores)
```

Each row sums to 1.

Shape:

```text
attention_weights: (B, T, T)
```

## 10. Weighted Sum of Values

The output is:

```text
output = attention_weights @ V
```

Shapes:

```text
attention_weights: (B, T, T)
V:                 (B, T, D)
output:            (B, T, D)
```

Each token's output representation is a weighted sum of all value vectors.

## 11. Self-Attention Formula

```text
Q = X @ Wq
K = X @ Wk
V = X @ Wv

scores = (Q @ K.T) / sqrt(D)

attention_weights = softmax(scores)

output = attention_weights @ V
```

## 12. Multi-Head Attention Intuition

Multi-head attention runs several attention heads in parallel.

If:

```text
d_model = 64
num_heads = 4
```

Then:

```text
head_dim = 16
```

Each head can learn a different attention pattern.

The outputs of all heads are concatenated and projected back to `d_model`.

## 13. ML Systems Connection

Attention is central to LLM systems.

Important systems topics later include:

```text
attention complexity O(T^2)
KV cache
prefill vs decode
FlashAttention
PagedAttention
tensor parallelism
memory bandwidth bottleneck
```

The self-attention formula is the foundation for understanding LLM inference optimization.

## 14. Checklist

- [ ] Understand token IDs
- [ ] Understand embeddings
- [ ] Understand self-attention intuition
- [ ] Understand Q / K / V
- [ ] Understand attention score shape `(B, T, T)`
- [ ] Understand scaled dot-product attention
- [ ] Understand softmax attention weights
- [ ] Understand weighted sum of values
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass
