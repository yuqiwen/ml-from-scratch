# Day 30: Model Versioning, Canary Deployment, and Rollback

## 1. Goal

Today's goal is to understand how model versions are safely deployed.

Key concepts:

```text
model version
code version
model registry
canary deployment
shadow deployment
A/B testing
sticky routing
rollback
```

---

## 2. Why Model Versioning?

A model artifact should not be stored only as:

```text
model.pt
```

Better:

```text
model_v1.pt
model_v2.pt
model_v3.pt
```

Versioning makes it possible to:

```text
identify the deployed model
compare versions
preserve old models
rollback quickly
reproduce experiments
```

---

## 3. Code Version vs Model Version

Code version identifies:

```text
serving code
preprocessing code
model architecture
API implementation
```

Model version identifies:

```text
trained weights
training data
hyperparameters
evaluation metrics
artifact
```

Production systems may track:

```text
code_version
model_version
data_version
config_version
```

---

## 4. Model Registry

A model registry stores model artifacts and metadata.

Example metadata:

```text
model name
version
artifact path
training dataset
metrics
creation time
deployment stage
```

Common stages:

```text
development
staging
production
archived
```

---

## 5. Canary Deployment

Canary deployment sends a small percentage of real traffic to a new model.

Example:

```text
v1: 95%
v2: 5%
```

If v2 is healthy:

```text
5% -> 20% -> 50% -> 100%
```

If v2 is unhealthy:

```text
5% -> 0%
```

---

## 6. Shadow Deployment

Shadow deployment sends a copy of traffic to the new model.

```text
request
  -> v1 response returned to user
  -> v2 result recorded only
```

The user does not receive the shadow model result.

---

## 7. Canary vs A/B Testing

Canary deployment focuses on:

```text
safe rollout
latency
errors
stability
resource usage
```

A/B testing focuses on:

```text
user behavior
business metrics
conversion
engagement
```

---

## 8. Sticky Routing

Random routing may send the same user to different models.

Sticky routing uses a stable key:

```text
hash(user_id)
```

This keeps one user on the same model version.

---

## 9. Rollback

Rollback sends traffic back to the stable model.

Example:

```text
before rollback:
  v1 = 50%
  v2 = 50%

after rollback:
  v1 = 100%
  v2 = 0%
```

Rollback should be fast and should not require retraining.

---

## 10. ML Systems Connection

Model versioning connects to:

```text
MLflow Model Registry
deployment pipelines
Kubernetes
feature flags
traffic routing
monitoring
SLOs
CI/CD
reproducibility
```

---

## 11. Commands

Run router demo:

```bash
python model_router.py
```

Run tests:

```bash
python test_model_router.py
```

---

## 12. Checklist

- [ ] Understand model versioning
- [ ] Understand code version vs model version
- [ ] Understand model registry
- [ ] Understand canary deployment
- [ ] Understand shadow deployment
- [ ] Understand A/B testing
- [ ] Understand sticky routing
- [ ] Understand rollback
- [ ] Run router demo
- [ ] Run tests
