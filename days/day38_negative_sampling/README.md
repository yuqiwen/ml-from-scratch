# Day 38: Hard Negative Mining and Negative Sampling

## 1. Goal

Today's goal is to understand how negative sample quality affects retrieval model training, especially for recommendation and embedding-based retrieval systems.

Key concepts:

```text
random negative
easy negative
hard negative
semi-hard negative
false negative
in-batch negative
negative sampling
sampled softmax
```

## 2. Random Negatives

Random negatives are sampled from the entire item catalog.

They are cheap to obtain but are often too easy.

Example:

```text
positive:
  camera lens

random negative:
  kitchen sponge
```

The model may already separate them easily, so the gradient signal can be weak.

## 3. Hard Negatives

Hard negatives are items that receive high model scores but are not labeled as positive.

Example:

```text
positive:
  Sony 85mm lens

hard negative:
  Sigma 85mm lens
```

Hard negatives provide stronger training signals because they are close to the decision boundary.

## 4. False Negatives

A false negative is an item labeled as negative even though the user may actually like it.

```text
not observed as positive
does not always mean
true negative
```

False negatives can push relevant user and item embeddings apart and hurt retrieval quality.

## 5. In-Batch Negatives

For aligned user-item pairs:

```text
user_i <-> item_i
```

The diagonal pairs are positive, while off-diagonal items are treated as negatives.

This is efficient, but it can create false negatives when different users in the batch like semantically similar items.

## 6. Hard Negative Mining

A common process:

```text
train current model
-> retrieve high-scoring items
-> remove known positives
-> use remaining items as hard negatives
-> retrain
```

This makes the training set progressively more challenging.

## 7. Semi-Hard Negatives

Semi-hard negatives are difficult but not suspiciously similar.

They often provide useful gradients while reducing false-negative risk.

One practical idea is to keep negatives within a score range:

```text
too easy -> weak signal
too hard -> maybe false negative
middle range -> often useful
```

## 8. Negative Sampling Strategies

Common strategies:

```text
uniform random sampling
popularity-based sampling
same-category sampling
exposed-but-not-clicked sampling
model-based hard negative mining
```

Production systems often mix multiple strategies.

## 9. Sampled Softmax

Full softmax over millions of items is expensive.

Sampled softmax uses:

```text
one positive
+
a subset of negatives
```

to approximate the full classification objective.

## 10. Code Mapping

This day's mini-project maps several sampling strategies to concrete code:

- `uniform_negative_sample(...)`: random negatives from the catalog
- `same_category_negative_sample(...)`: harder negatives within the same semantic bucket
- `popularity_weighted_sample(...)`: sampling biased toward popular items
- `mine_hard_negatives(...)`: selects high-scoring non-positive retrieved items
- `filter_false_negative_candidates(...)`: removes candidates likely to be false negatives
- `mixed_negative_sample(...)`: combines multiple strategies into one training batch

The key training idea is:

```text
negative quality matters
not just negative quantity
```

## 11. ML Systems Connection

Negative sampling connects to:

```text
retrieval training
ANN indexes
implicit feedback
exposure logs
recommendation systems
contrastive learning
distributed training
```

## 12. Commands

Run the demo:

```bash
python days/day38_negative_sampling/negative_sampling.py
```

Run the tests:

```bash
python days/day38_negative_sampling/test_negative_sampling.py
```

## 13. Checklist

- [x] Understand random negatives
- [x] Understand hard negatives
- [x] Understand false negatives
- [x] Understand in-batch negative risk
- [x] Understand hard negative mining
- [x] Understand semi-hard negatives
- [x] Understand sampled softmax motivation
- [x] Run demo
- [x] Run tests
