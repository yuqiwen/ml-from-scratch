# Day 35: Online Inference, Batch Inference, and Two-Stage Recommendation

## 1. Goal

Today's goal is to understand how online and batch inference work together in recommendation systems, especially in a two-stage retrieval-and-ranking pipeline.

Key concepts:

```text
online inference
batch inference
latency
throughput
candidate generation
ranking
precomputation
real-time context
top-k
```

## 2. Online Inference

Online inference runs when a real-time request arrives.

```text
request
-> model inference
-> immediate response
```

It prioritizes:

```text
low latency
availability
tail latency
```

Examples:

```text
recommendation ranking
fraud detection
search ranking
ad prediction
```

## 3. Batch Inference

Batch inference processes a large dataset together.

```text
large dataset
-> batched model forward
-> store predictions
```

It prioritizes:

```text
throughput
resource utilization
cost
total completion time
```

Examples:

```text
daily churn scoring
precomputed recommendations
image tagging
user embedding generation
```

## 4. Training vs Batch Inference

Training:

```text
forward
loss
backward
optimizer step
```

Batch inference:

```text
large batches
forward only
save predictions
```

Batch inference does not update model parameters.

## 5. Two-Stage Recommendation

A recommendation system often uses:

```text
candidate generation
-> ranking
```

Candidate generation reduces:

```text
millions of items
-> hundreds or thousands of candidates
```

Ranking uses a more expensive model to select the final top-k items.

## 6. Candidate Generation

Candidate generation focuses on:

```text
speed
coverage
recall
```

Common techniques:

```text
embedding similarity
collaborative filtering
popular items
rule-based retrieval
nearest-neighbor search
```

## 7. Ranking

Ranking combines:

```text
user features
item features
request context
```

The ranking model outputs one score per candidate.

```text
candidates
-> scores
-> sort
-> top-k
```

## 8. Offline and Online Cooperation

Offline computation may prepare:

```text
user embeddings
item embeddings
candidate lists
long-term user features
popularity statistics
```

Online inference may use:

```text
current request context
latest user behavior
inventory
time
location
device
```

This is the common production split:

```text
offline heavy computation
+
online lightweight reranking
```

## 9. Why Not Precompute Everything?

Precomputed recommendations may become stale because:

```text
user behavior changes
items become unavailable
context changes
trends change
inventory changes
```

Even if offline candidate generation is strong, the final response still benefits from fresh online signals.

## 10. Code Mapping

This day's mini-project mirrors the real system design:

- `run_offline_job(...)`: precomputes a candidate pool for each user
- `BatchCandidateStore`: stores batch-generated candidates
- `serve_recommendation(...)`: represents the online serving path
- `online_rank(...)`: applies request-time context, quality, popularity, and availability filtering

The important systems idea is:

```text
offline retrieval narrows the search space
online ranking adapts to the live request
```

## 11. Commands

Run the demo:

```bash
python days/day35_online_vs_batch_inference/recommender.py
```

Run the tests:

```bash
python days/day35_online_vs_batch_inference/test_recommender.py
```

## 12. Checklist

- [x] Understand online inference
- [x] Understand batch inference
- [x] Understand latency vs throughput
- [x] Understand candidate generation
- [x] Understand ranking
- [x] Understand precomputation
- [x] Understand online reranking
- [x] Run the recommendation demo
- [x] Run tests
