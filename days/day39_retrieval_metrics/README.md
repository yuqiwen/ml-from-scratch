# Day 39: Retrieval Evaluation Metrics

## 1. Goal

Today's goal is to evaluate ranked retrieval results and understand how different offline metrics reflect different system goals.

Key concepts:

```text
Precision@K
Recall@K
Hit Rate@K
Reciprocal Rank
MRR
DCG
IDCG
NDCG
graded relevance
```

## 2. Retrieval Evaluation

A retrieval model returns a ranked list:

```text
[item_1, item_2, ..., item_k]
```

Evaluation asks:

```text
Were relevant items retrieved?
How many relevant items were retrieved?
How early did the first relevant item appear?
Were highly relevant items ranked near the top?
```

## 3. Precision@K

```text
Precision@K
=
number of relevant retrieved items
/
K
```

Precision measures how clean the returned top-k list is.

## 4. Recall@K

```text
Recall@K
=
number of relevant retrieved items
/
total number of relevant items
```

Recall measures how much of the relevant set was recovered.

Retrieval systems often prioritize recall because a ranking model cannot rank an item that was never retrieved.

## 5. Hit Rate@K

```text
Hit@K = 1
```

if at least one relevant item appears in the top-k.

Otherwise:

```text
Hit@K = 0
```

Hit Rate@K is the average hit value across queries.

## 6. Reciprocal Rank and MRR

```text
Reciprocal Rank
=
1 / rank of first relevant item
```

If no relevant item is found:

```text
RR = 0
```

MRR is the mean Reciprocal Rank across queries.

MRR only considers the first relevant result, so it emphasizes how quickly the first good item appears.

## 7. DCG

```text
DCG@K
=
sum(
    (2^relevance_i - 1)
    /
    log2(i + 1)
)
```

Relevant items receive less credit when they appear lower in the ranking.

## 8. NDCG

```text
NDCG@K
=
DCG@K / IDCG@K
```

IDCG is the DCG of the ideal ranking.

NDCG is typically between 0 and 1 and supports graded relevance.

## 9. Retrieval vs Ranking Metrics

Retrieval commonly focuses on:

```text
Recall@K
Hit Rate@K
retrieval latency
ANN recall
```

Ranking commonly focuses on:

```text
NDCG@K
MRR
Precision@K
```

## 10. Offline Evaluation Limitations

Observed interactions are incomplete labels.

```text
not clicked
does not necessarily mean
irrelevant
```

Offline evaluation should be combined with:

```text
exposure logs
time-based splits
online A/B testing
business metrics
```

## 11. Code Mapping

This day's mini-project maps common evaluation metrics to runnable code:

- `precision_at_k(...)`: fraction of top-k results that are relevant
- `recall_at_k(...)`: fraction of all relevant items recovered
- `hit_at_k(...)`: whether at least one relevant result appears in top-k
- `reciprocal_rank(...)`: inverse rank of the first relevant item
- `dcg_at_k(...)` and `ndcg_at_k(...)`: position-aware ranking quality with graded relevance
- `evaluate_dataset(...)`: aggregates metrics across multiple queries

The important evaluation idea is:

```text
different metrics answer
different product questions
```

## 12. Commands

Run the demo:

```bash
python days/day39_retrieval_metrics/retrieval_metrics.py
```

Run the tests:

```bash
python days/day39_retrieval_metrics/test_retrieval_metrics.py
```

## 13. Checklist

- [x] Understand Precision@K
- [x] Understand Recall@K
- [x] Understand Hit Rate@K
- [x] Understand MRR
- [x] Understand DCG
- [x] Understand NDCG
- [x] Understand graded relevance
- [x] Understand offline evaluation limitations
- [x] Run demo
- [x] Run tests
