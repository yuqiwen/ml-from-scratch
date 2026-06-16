# Day 29: Serving Concurrency, Backpressure, and Thread Safety

## 1. Goal

Today's goal is to understand how a model serving system handles multiple concurrent requests safely.

Key concepts:

```text
concurrency
parallelism
thread safety
shared model
semaphore
queue
backpressure
bounded queue
overload protection
```

---

## 2. Concurrency vs Parallelism

Concurrency:

```text
multiple tasks are in progress during the same time period
```

Parallelism:

```text
multiple tasks execute at the same instant
```

A serving system may be concurrent even if all work is not truly parallel.

---

## 3. Shared Model

A model is commonly loaded once:

```python
model = load_model()
model.eval()
```

Multiple requests then share the same model.

Pure inference is usually safer than training because it does not update parameters.

However, the model should not modify shared mutable state during forward.

---

## 4. eval and inference_mode

```python
model.eval()
```

Changes inference behavior of modules such as:

```text
Dropout
BatchNorm
```

```python
with torch.inference_mode():
    output = model(x)
```

Disables gradient tracking.

Neither one limits request concurrency.

---

## 5. Semaphore

A semaphore limits how many tasks can enter a critical section.

Example:

```python
semaphore = asyncio.Semaphore(4)
```

At most four tasks may run inference concurrently.

This helps control:

```text
GPU memory usage
kernel queue pressure
tail latency
service stability
```

---

## 6. Lock vs Semaphore

Lock:

```text
one task at a time
```

Semaphore:

```text
up to N tasks at a time
```

Conceptually:

```text
Lock = Semaphore(1)
```

---

## 7. Backpressure

Backpressure prevents the system from accepting unlimited work.

Without backpressure:

```text
request rate > processing rate
-> queue grows
-> latency grows
-> memory grows
-> service may crash
```

Backpressure strategies:

```text
bounded queue
request timeout
reject overload
return HTTP 429
return gRPC RESOURCE_EXHAUSTED
```

---

## 8. Bounded Queue

```python
queue = asyncio.Queue(maxsize=32)
```

A bounded queue limits the number of waiting requests.

When full, the service can:

```text
wait
reject
timeout
```

---

## 9. ML Systems Connection

Concurrency control affects:

```text
latency
throughput
GPU utilization
memory usage
fairness
tail latency
service reliability
```

Production model serving systems often combine:

```text
request queue
dynamic batching
concurrency limit
timeouts
circuit breaker
autoscaling
```

---

## 10. Commands

Run simulator:

```bash
python concurrency_sim.py
```

Run tests:

```bash
python test_concurrency_sim.py
```

---

## 11. Checklist

- [ ] Understand concurrency
- [ ] Understand parallelism
- [ ] Understand shared model risks
- [ ] Understand semaphore
- [ ] Understand Lock vs Semaphore
- [ ] Understand backpressure
- [ ] Understand bounded queue
- [ ] Run simulator
- [ ] Run tests
