# Day 27: Model Serving Monitoring

## 1. Goal

Today's goal is to understand monitoring for a model serving service.

Key concepts:

```text
request count
success count
error count
error rate
latency
p50 / p95 / p99
health check
readiness check
middleware
metrics endpoint
```

---

## 2. Why Monitoring?

A service can be running but still unhealthy.

Possible problems:

```text
high latency
high error rate
model loading failure
traffic spike
invalid input
resource exhaustion
```

Monitoring helps detect problems before users report them.

---

## 3. Core Metrics

Basic serving metrics:

```text
total requests
successful requests
failed requests
error rate
average latency
p50 latency
p95 latency
p99 latency
```

Error rate:

```text
failed_requests / total_requests
```

---

## 4. Average vs Percentile Latency

Average latency alone can hide slow requests.

Percentiles provide tail latency information:

```text
p50:
  median latency

p95:
  95% of requests are faster than this

p99:
  99% of requests are faster than this
```

Production systems often monitor p95 and p99.

---

## 5. Health vs Readiness

Liveness:

```text
Is the service process alive?
```

Readiness:

```text
Is the service ready to accept prediction requests?
```

Example:

```text
/health -> server is alive
/ready -> model is loaded and ready
```

---

## 6. Middleware

Middleware wraps every HTTP request.

It can:

```text
record start time
execute endpoint
record end time
count success / failure
record latency
```

---

## 7. Metrics Endpoint

This project exposes:

```text
GET /health
GET /ready
GET /metrics
POST /predict
```

Example metrics response:

```json
{
  "total_requests": 10,
  "successful_requests": 9,
  "failed_requests": 1,
  "error_rate": 0.1,
  "average_latency_ms": 3.4,
  "p50_latency_ms": 3.1,
  "p95_latency_ms": 5.8,
  "p99_latency_ms": 6.2
}
```

---

## 8. Commands

Train model:

```bash
python train_model.py
```

Start service:

```bash
uvicorn serve_model:app --reload
```

Call prediction:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/predict `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"features":[1.0,2.0]}'
```

Read metrics:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/metrics
```

Run tests:

```bash
python test_monitoring.py
```

---

## 9. ML Systems Connection

Serving monitoring connects to:

```text
Prometheus
Grafana
OpenTelemetry
logging
alerting
SLO
SLA
autoscaling
incident response
```

Important production metrics often follow:

```text
rate
errors
duration
```

This is sometimes called the RED method:

```text
R = request rate
E = error rate
D = request duration
```

---

## 10. Checklist

- [ ] Understand request count
- [ ] Understand error rate
- [ ] Understand latency percentiles
- [ ] Understand health check
- [ ] Understand readiness check
- [ ] Understand middleware
- [ ] Run model training
- [ ] Start FastAPI service
- [ ] Call prediction endpoint
- [ ] Read metrics endpoint
- [ ] Run tests
