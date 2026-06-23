# Day 36: Embedding Retrieval, Vector Search, and ANN Basics

## 1. Goal

Today's goal is to understand how embedding-based candidate retrieval works, and how exact vector search differs from approximate nearest neighbor search.

Key concepts:

```text
embedding
user embedding
item embedding
dot product
cosine similarity
exact search
approximate nearest neighbor
ANN
vector index
Recall@K
```

## 2. Embedding Retrieval

Embedding retrieval represents users and items as vectors.

```text
user
-> user embedding

item
-> item embedding
```

The system retrieves items whose vectors are most similar to the user vector.

## 3. Dot Product

```text
u dot v
=
sum(u_i * v_i)
```

A larger dot product commonly means greater relevance.

## 4. Cosine Similarity

```text
cosine(u, v)
=
dot(u, v) / (norm(u) * norm(v))
```

Cosine similarity compares vector direction.

If both vectors are normalized to unit length:

```text
cosine similarity = dot product
```

## 5. Exact Search

Exact search compares the query vector against every item vector.

Complexity:

```text
O(number_of_items * embedding_dimension)
```

It gives exact nearest neighbors but can become too slow for large-scale online serving.

## 6. ANN

Approximate nearest neighbor search trades a small amount of accuracy for much faster search.

```text
exact search:
  high accuracy
  high cost

ANN:
  approximate results
  much lower latency
```

The main systems tradeoff is:

```text
more speed
usually means
slightly lower recall
```

## 7. Recall@K

Recall@K measures how many exact top-k neighbors are also found by ANN.

Example:

```text
exact top-10 contains 10 relevant neighbors
ANN finds 8 of them

Recall@10 = 0.8
```

## 8. Vector Index

A vector index organizes vectors so the system does not need to scan every item.

Common approaches:

```text
HNSW
IVF
PQ
LSH
```

This mini-project uses a toy cluster-based index to illustrate the idea without implementing a full production ANN library.

## 9. Recommendation Serving Flow

```text
offline:
  compute item embeddings
  build vector index

online:
  compute user embedding
  search vector index
  retrieve candidates
  rank candidates
  return top-k
```

## 10. Code Mapping

This day's code maps the concepts to a simple retrieval pipeline:

- `dot_product(...)` and `cosine_similarity(...)`: similarity metrics
- `exact_search(...)`: brute-force nearest-neighbor retrieval
- `ToyClusterIndex`: simplified ANN-like index that searches only selected clusters
- `recall_at_k(...)`: evaluates how many exact top-k results are recovered by the approximate search

The important retrieval idea is:

```text
exact search is the quality baseline
ANN search is the latency-friendly serving path
```

## 11. ML Systems Connection

Embedding retrieval connects to:

```text
recommendation systems
semantic search
RAG
vector databases
ANN indexes
ranking systems
retrieval models
```

## 12. Commands

Run the demo:

```bash
python days/day36_embedding_retrieval/vector_retrieval.py
```

Run the tests:

```bash
python days/day36_embedding_retrieval/test_vector_retrieval.py
```

## 13. Checklist

- [x] Understand embeddings
- [x] Understand dot product
- [x] Understand cosine similarity
- [x] Understand exact search
- [x] Understand ANN
- [x] Understand vector indexes
- [x] Understand Recall@K
- [x] Run retrieval demo
- [x] Run tests
