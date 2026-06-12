# Day 25: Dynamic Batching and Continuous Batching Basics

## 1. Goal

Today's goal is to understand batching in model serving systems.

Key concepts:

```text
serving batching
static batching
dynamic batching
max_batch_size
max_wait_ms
latency / throughput tradeoff
continuous batching
LLM serving
```

---

## 2. Why Batching in Serving?

Serving one request at a time can underutilize GPU.

Batching combines multiple requests into one model forward:

```text
multiple requests -> one batched forward
```

Benefits:

```text
higher GPU utilization
higher throughput
less kernel launch overhead per sample
```

---

## 3. Static Batching

Static batching waits for a fixed batch size.

Example:

```text
batch_size = 8
wait until 8 requests arrive
run model forward
```

Problem:

```text
if traffic is low, early requests wait too long
```

So static batching can increase throughput but hurt latency.

---

## 4. Dynamic Batching

Dynamic batching uses rules such as:

```text
max_batch_size = 8
max_wait_ms = 5
```

Meaning:

```text
run immediately if 8 requests are ready
otherwise run after waiting at most 5 ms
```

This balances latency and throughput.

---

## 5. Latency / Throughput Tradeoff

Small wait time:

```text
lower latency
smaller batches
lower throughput
```

Large wait time:

```text
higher latency
larger batches
higher throughput
```

Serving systems tune:

```text
max_batch_size
max_wait_ms
queue policy
timeout
priority
```

---

## 6. Why LLM Serving Is Different

Normal model serving:

```text
one request -> one forward -> one response
```

LLM serving:

```text
request -> prefill -> decode token by token
```

Different requests may generate different numbers of tokens.

Example:

```text
request A: 10 tokens
request B: 200 tokens
request C: 40 tokens
```

This makes batching harder.

---

## 7. Continuous Batching

Continuous batching dynamically updates the batch at each decode step.

Example:

```text
step 1 active batch: [A, B, C]

A finishes
D arrives

step 2 active batch: [B, C, D]
```

The batch is not fixed for the entire generation.

It changes as requests finish and new requests arrive.

---

## 8. KV Cache Connection

In LLM serving, each request has its own KV cache.

Continuous batching must track:

```text
active requests
finished requests
new requests
KV cache location
generated length
remaining budget
```

This connects to systems like:

```text
vLLM
PagedAttention
continuous batching scheduler
KV cache memory manager
```

---

## 9. ML Systems Connection

Dynamic batching is a core serving optimization.

It affects:

```text
latency
throughput
GPU utilization
queueing delay
memory usage
fairness
tail latency
```

For AI infra roles, it is important to understand how requests become batches.

---

## 10. Checklist

- [ ] Understand serving batching
- [ ] Understand static batching
- [ ] Understand dynamic batching
- [ ] Understand max_batch_size
- [ ] Understand max_wait_ms
- [ ] Understand latency / throughput tradeoff
- [ ] Understand continuous batching intuition
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass
