# Day 31: Data Drift, Prediction Drift, and Model Monitoring

## 1. Goal

Today's goal is to understand model-specific monitoring after deployment.

Key concepts:

```text
data drift
concept drift
prediction drift
reference distribution
live distribution
missing rate
drift score
model health
```

---

## 2. Service Health vs Model Health

Service health measures:

```text
latency
errors
availability
readiness
```

Model health measures:

```text
input distribution
prediction distribution
accuracy
business quality
drift
```

A service can be technically healthy while model quality is degrading.

---

## 3. Data Drift

Data drift means:

```text
P_train(X) != P_live(X)
```

The live input distribution differs from the training distribution.

Examples:

```text
user age distribution changes
transaction amount changes
image brightness changes
text length changes
```

---

## 4. Concept Drift

Concept drift means:

```text
P_train(Y | X) != P_live(Y | X)
```

The relationship between input and target changes.

Example:

```text
the same customer profile now has a different default risk
```

---

## 5. Prediction Drift

Prediction drift means:

```text
P_train(y_hat) != P_live(y_hat)
```

The model output distribution changes.

Prediction drift may indicate:

```text
real-world behavior change
input drift
preprocessing bug
model bug
feature missing
```

It is an alert signal, not proof that the model is wrong.

---

## 6. Delayed Labels

Online systems often do not receive labels immediately.

Without labels, monitor:

```text
feature drift
prediction drift
missing rate
out-of-range values
```

When labels arrive, monitor:

```text
accuracy
precision
recall
AUC
business metrics
```

---

## 7. Simple Drift Score

A simple standardized mean difference:

```text
abs(live_mean - reference_mean) / reference_std
```

A larger score means stronger mean shift.

This simple metric does not detect every distribution change.

---

## 8. Missing Rate

A feature pipeline can fail without crashing the model service.

Monitor:

```text
reference missing rate
live missing rate
missing rate increase
```

A sudden increase may indicate an upstream data problem.

---

## 9. Production Drift Metrics

Production systems may use:

```text
mean / standard deviation
quantiles
histograms
PSI
KS test
KL divergence
Jensen-Shannon divergence
```

---

## 10. ML Systems Connection

Model monitoring connects to:

```text
feature pipelines
data validation
model retraining
alerting
model registry
rollback
A/B testing
online evaluation
```

A common lifecycle is:

```text
train
-> deploy
-> monitor
-> detect drift
-> investigate
-> retrain or rollback
```

---

## 11. Commands

Run drift demo:

```bash
python drift_detector.py
```

Run tests:

```bash
python test_drift_detector.py
```

---

## 12. Checklist

- [ ] Understand data drift
- [ ] Understand concept drift
- [ ] Understand prediction drift
- [ ] Understand service health vs model health
- [ ] Understand delayed labels
- [ ] Understand standardized mean difference
- [ ] Understand missing-rate monitoring
- [ ] Run drift demo
- [ ] Run tests
