# Day 17: Tiny GPT / Decoder-only Language Model Skeleton

## 1. Goal

Today's goal is to build a tiny GPT-style decoder-only language model skeleton.

Key concepts:

```text
token embedding
positional embedding
Transformer block stack
lm_head
vocabulary logits
next-token prediction
input / target shift
CrossEntropyLoss
```

---

## 2. GPT-style Model Structure

A decoder-only GPT-style model looks like:

```text
token_ids
-> token embedding
-> positional embedding
-> Transformer blocks
-> final LayerNorm
-> lm_head
-> logits over vocabulary
```

Shape flow:

```text
token_ids: (B, T)

hidden states: (B, T, D)

logits: (B, T, vocab_size)
```

---

## 3. Token Embedding

Token IDs are integers:

```text
[5, 12, 9, 3]
```

They are converted into vectors by:

```python
nn.Embedding(vocab_size, d_model)
```

If:

```text
input_ids: (B, T)
```

Then:

```text
token_emb: (B, T, D)
```

---

## 4. Positional Embedding

Self-attention does not naturally know token order.

So we add positional embeddings:

```python
nn.Embedding(max_seq_len, d_model)
```

Position IDs:

```text
[0, 1, 2, 3]
```

Final input representation:

```text
x = token_embedding(input_ids) + position_embedding(position_ids)
```

Shape:

```text
x: (B, T, D)
```

---

## 5. Transformer Blocks

Each Transformer block keeps shape:

```text
(B, T, D) -> (B, T, D)
```

A Pre-LN block:

```text
x = x + attention(layer_norm(x))
x = x + ffn(layer_norm(x))
```

Stacking many blocks increases model capacity.

---

## 6. lm_head

The language modeling head maps hidden states to vocabulary logits:

```python
lm_head = nn.Linear(d_model, vocab_size)
```

Shape:

```text
hidden: (B, T, D)

logits: (B, T, vocab_size)
```

For each position, logits contain one score for every token in the vocabulary.

---

## 7. Logits vs Probabilities

Logits are raw scores.

Probabilities are obtained by:

```python
probabilities = softmax(logits, dim=-1)
```

During training, we usually do not manually apply softmax.

`nn.CrossEntropyLoss()` expects raw logits.

---

## 8. Next-Token Prediction

Given a sequence:

```text
[10, 20, 30, 40, 50]
```

We use:

```text
input_ids  = [10, 20, 30, 40]
target_ids = [20, 30, 40, 50]
```

So each position predicts the next token.

This is the training objective for GPT-style language models.

---

## 9. CrossEntropyLoss Shape

Language model logits:

```text
logits: (B, T, vocab_size)
```

Targets:

```text
targets: (B, T)
```

PyTorch CrossEntropyLoss expects:

```text
logits:  (N, C)
targets: (N,)
```

So we flatten:

```python
logits = logits.view(B * T, vocab_size)
targets = targets.view(B * T)
loss = loss_fn(logits, targets)
```

---

## 10. ML Systems Connection

TinyGPT introduces the full LLM training skeleton.

This connects to:

```text
causal attention
KV cache
cross entropy over vocabulary
token-level batching
activation memory
checkpointing
mixed precision
model parallelism
inference logits
sampling
```

This is the bridge from learning Transformer components to building ML / AI infrastructure.

---

## 11. Checklist

- [ ] Understand token embedding
- [ ] Understand positional embedding
- [ ] Understand Transformer block stack
- [ ] Understand lm_head
- [ ] Understand logits over vocabulary
- [ ] Understand next-token prediction shift
- [ ] Understand CrossEntropyLoss flattening
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass
