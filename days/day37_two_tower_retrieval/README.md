# Day 37: Two-Tower Retrieval Model

## 1. Goal

Today's goal is to understand how user and item embeddings are trained for retrieval, and how a two-tower model supports large-scale candidate generation.

Key concepts:

```text
Two-Tower model
user tower
item tower
positive pair
negative sample
in-batch negatives
similarity matrix
contrastive learning
temperature
embedding normalization
```

## 2. Two-Tower Architecture

```text
user features
-> user tower
-> user embedding

item features
-> item tower
-> item embedding
```

The user and item embeddings must have the same dimension.

A similarity function produces a relevance score:

```text
score(user, item)
=
dot(user_embedding, item_embedding)
```

## 3. Positive and Negative Pairs

Positive pairs come from real interactions:

```text
click
purchase
watch
save
```

Example:

```text
(user_1, camera_item)
```

Negative pairs represent unrelated or non-selected items.

Training pushes:

```text
positive score up
negative score down
```

## 4. In-Batch Negatives

For a batch of aligned positive pairs:

```text
user_i <-> item_i
```

Calculate:

```python
logits = user_embeddings @ item_embeddings.T
```

Shape:

```text
(B, D) @ (D, B)
=
(B, B)
```

The diagonal contains positive pairs.

Off-diagonal entries are treated as negatives.

## 5. Cross-Entropy Objective

For batch size `B`:

```python
labels = torch.arange(B)
loss = cross_entropy(logits, labels)
```

For each user row, the correct item column is the matching diagonal position.

This day's implementation also uses the symmetric version:

```text
user -> item loss
+
item -> user loss
```

## 6. Temperature

```python
logits = similarity / temperature
```

Smaller temperature creates a sharper softmax distribution and increases the pressure to separate positives from negatives.

## 7. L2 Normalization

Normalize embeddings:

```python
embedding = normalize(embedding)
```

Then:

```text
dot product = cosine similarity
```

Normalization also prevents the model from increasing scores only by increasing vector magnitude.

## 8. Serving Architecture

Offline:

```text
item features
-> item tower
-> item embeddings
-> ANN index
```

Online:

```text
user features
-> user tower
-> user embedding
-> ANN search
-> candidates
```

## 9. Retrieval and Ranking

Two-tower models are commonly used for retrieval because they are fast and allow precomputed item embeddings.

A separate ranking model can use richer user-item interactions:

```text
retrieval:
  millions -> hundreds

ranking:
  hundreds -> top-k
```

## 10. Code Mapping

This day's mini-project maps the retrieval pipeline to concrete code:

- `Tower`: user tower and item tower MLPs that produce normalized embeddings
- `TwoTowerModel.similarity_matrix(...)`: builds the full in-batch user-item score matrix
- `TwoTowerModel.forward(...)`: computes the symmetric contrastive-style retrieval loss
- `offline_encode_items(...)`: simulates offline item embedding generation
- `retrieve_top_k(...)`: uses a live user embedding against precomputed item embeddings

The key systems idea is:

```text
train joint embeddings online
serve retrieval with precomputed item vectors
```

## 11. Commands

Run the demo:

```bash
python days/day37_two_tower_retrieval/two_tower.py
```

Run the tests:

```bash
python days/day37_two_tower_retrieval/test_two_tower.py
```

## 12. Checklist

- [x] Understand user tower
- [x] Understand item tower
- [x] Understand positive pairs
- [x] Understand negative samples
- [x] Understand in-batch negatives
- [x] Understand similarity matrix shapes
- [x] Understand cross-entropy labels
- [x] Understand temperature
- [x] Understand offline item embedding generation
- [x] Run demo
- [x] Run tests
