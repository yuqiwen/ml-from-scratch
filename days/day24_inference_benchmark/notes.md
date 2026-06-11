# Day 24: Inference Benchmark, Latency, and Throughput

## 1. Goal

Today's goal is to understand basic inference benchmarking.

Key concepts:

```text
inference benchmark
latency
throughput
warmup
p50 / p95 / p99 latency
batch size
CPU/GPU synchronization
```

---

## 2. Inference Benchmark

Inference benchmark measures how fast a trained model runs during serving.

Common metrics:

```text
latency
throughput
memory usage
batch size scaling
```

Today focuses on latency and throughput.

---

## 3. Latency

Latency means how long one inference request takes.

Example:

```text
latency = 3.2 ms
```

This means one request takes 3.2 milliseconds from input to output.

Latency matters a lot for interactive services.

---

## 4. Throughput

Throughput means how many samples the system can process per second.

Example:

```text
throughput = 5000 samples / second
```

Throughput matters for batch processing and high-volume serving.

---

## 5. Latency vs Throughput Tradeoff

Small batch size:

```text
lower latency
lower throughput
```

Large batch size:

```text
higher throughput
possibly higher latency
```

Serving systems often balance latency and throughput depending on product needs.

---

## 6. Warmup

The first few inference runs can be slower because of:

```text
CUDA initialization
memory allocation
cache warmup
backend setup
```

So benchmark usually does warmup runs before measuring.

Example:

```python
for _ in range(num_warmup):
    model(x)
```

Warmup runs are not included in final metrics.

---

## 7. p50 / p95 / p99 Latency

If we run inference many times, latency varies.

Common percentiles:

```text
p50:
  median latency

p95:
  95% of requests are faster than this

p99:
  99% of requests are faster than this
```

Production serving systems care about p95 and p99 because slow tail requests affect user experience.

---

## 8. CUDA Synchronization

CUDA operations are asynchronous by default.

For accurate timing on GPU, use:

```python
if device.type == "cuda":
    torch.cuda.synchronize()

start = time.perf_counter()

model(x)

if device.type == "cuda":
    torch.cuda.synchronize()

end = time.perf_counter()
```

Without synchronization, timing may only measure launch overhead, not actual GPU execution time.

---

## 9. Benchmark Loop

Basic benchmark structure:

```python
model.eval()

with torch.no_grad():
    for _ in range(num_warmup):
        model(x)

    for _ in range(num_iters):
        start = now()
        model(x)
        end = now()
        record_latency()
```

---

## 10. ML Systems Connection

Inference benchmark is a core AI infrastructure skill.

It connects to:

```text
model serving
latency optimization
throughput optimization
batching
GPU utilization
kernel launch overhead
memory bandwidth
LLM tokens/sec
prefill/decode performance
```

For LLM serving, benchmark often includes:

```text
time to first token
decode tokens/sec
KV cache memory
batch scheduling
```

---

## 11. Checklist

- [ ] Understand latency
- [ ] Understand throughput
- [ ] Understand warmup
- [ ] Understand p50 / p95 / p99
- [ ] Understand CUDA synchronization
- [ ] Understand batch size impact
- [ ] Run `implementation.py`
- [ ] Run `test_implementation.py`
- [ ] Confirm all tests pass
