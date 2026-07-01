# Day 40: Learning to Rank

## 1. Goal

Today's goal is to understand how ranking models are trained after candidate retrieval, and how pointwise, pairwise, and listwise objectives differ.

Key concepts:

```text
pointwise ranking
pairwise ranking
listwise ranking
ranking score
pairwise logistic loss
margin ranking loss
cross features
position bias
```

## 2. Ranking Stage

A recommendation system often uses:

```text
retrieval
-> candidates
-> ranking
-> top-k
```

Retrieval focuses on high recall.

Ranking focuses on placing the best candidates near the top.

## 3. Pointwise Ranking

Pointwise training treats every user-item pair independently.

```text
(user, item) -> relevance label
```

Common losses:

```text
binary cross entropy
mean squared error
cross entropy
```

The predicted probability or score is used for sorting.

## 4. Pairwise Ranking

Pairwise training compares two items for the same query.

```text
positive item
should rank above
negative item
```

A common loss is:

```text
softplus(negative_score - positive_score)
```

The loss decreases when the positive score becomes larger than the negative score.

## 5. Margin Ranking

Margin ranking requires:

```text
positive_score
>=
negative_score + margin
```

Loss:

```text
max(
    0,
    margin - positive_score + negative_score
)
```

## 6. Listwise Ranking

Listwise training considers the complete candidate list.

```text
query
-> multiple candidate scores
-> one list-level objective
```

It attempts to optimize the overall ordering rather than individual labels or pairs.

## 7. Pointwise vs Pairwise vs Listwise

```text
pointwise:
  Is this item relevant?

pairwise:
  Should A rank above B?

listwise:
  What is the correct ordering of the list?
```

## 8. Ranking Features

Ranking models may use:

```text
user features
item features
context features
user-item cross features
```

Examples:

```text
category match
price difference
distance
recent interaction
inventory
device
time
```

## 9. Retrieval vs Ranking

Retrieval models must score a very large catalog efficiently.

Ranking models only score a smaller candidate set and can therefore be more complex.

```text
retrieval:
  millions -> hundreds

ranking:
  hundreds -> top-k
```

## 10. Position Bias

Clicks depend on both:

```text
relevance
and
display position
```

Higher-ranked items receive more exposure and are more likely to be clicked.

Training directly on click labels may reproduce this bias.

## 11. Code Mapping

This day's mini-project maps several ranking objectives to runnable code:

- `RankingModel`: a small pointwise ranking network
- `pointwise_bce_loss(...)`: binary pointwise ranking loss
- `pairwise_logistic_loss(...)`: encourages the positive score to exceed the negative score
- `margin_ranking_loss(...)`: enforces a margin between positive and negative scores
- `listwise_softmax_loss(...)`: simple listwise objective over a candidate list
- `build_pair_features(...)`: concatenates user, item, context, and cross features

The key ranking idea is:

```text
retrieval narrows the set
ranking optimizes the final order
```

## 12. Commands

Run the demo:

```bash
python days/day40_learning_to_rank/ranking_model.py
```

Run the tests:

```bash
python days/day40_learning_to_rank/test_ranking_model.py
```

## 13. Checklist

- [x] Understand pointwise ranking
- [x] Understand pairwise ranking
- [x] Understand listwise ranking
- [x] Understand pairwise logistic loss
- [x] Understand margin ranking loss
- [x] Understand cross features
- [x] Understand retrieval vs ranking
- [x] Understand position bias
- [x] Run demo
- [x] Run tests
