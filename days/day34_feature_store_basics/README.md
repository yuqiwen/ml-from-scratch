# Day 34: Feature Store, Point-in-Time Join, and Feature Freshness

## 1. Goal

Today's goal is to understand how feature stores support consistent ML training and online serving.

Key concepts:

```text
feature definition
feature value
offline store
online store
materialization
point-in-time join
feature freshness
event time
processing time
```

---

## 2. Feature Store

A feature store manages reusable machine learning features.

It may manage:

```text
feature definitions
feature values
schemas
versions
offline storage
online storage
freshness
historical retrieval
```

---

## 3. Feature Definition vs Feature Value

Feature definition:

```text
how a feature is computed
```

Feature value:

```text
the computed value for one entity at one time
```

Example:

```text
definition:
  user_7d_click_count

value:
  user_id=123
  timestamp=2026-06-20
  value=42
```

---

## 4. Offline Store

Offline stores support:

```text
training
historical analysis
batch feature computation
training dataset generation
```

They usually contain historical feature values.

---

## 5. Online Store

Online stores support:

```text
low-latency inference
real-time feature lookup
```

They commonly store the latest feature values for each entity.

---

## 6. Materialization

Materialization copies or publishes computed feature values into the online store.

```text
offline feature values
-> materialization
-> online store
```

---

## 7. Point-in-Time Join

For each training sample, select the latest feature value available at or before the sample timestamp.

```text
feature_time <= sample_time
```

Then choose the feature row with the largest valid feature timestamp.

This prevents future information leakage.

---

## 8. Feature Freshness

Feature freshness measures how old the current feature value is.

```text
freshness lag
=
current time - feature timestamp
```

A feature may be considered stale if its lag exceeds an allowed threshold.

---

## 9. Event Time vs Processing Time

Event time:

```text
when the real-world event happened
```

Processing time:

```text
when the system processed the event
```

Late or out-of-order events make this distinction important.

---

## 10. Training-Serving Consistency

Feature stores help reduce skew by sharing:

```text
feature names
schemas
transformations
default values
window definitions
versions
```

However, freshness and pipeline correctness still need monitoring.

---

## 11. ML Systems Connection

Feature stores connect to:

```text
training pipelines
online serving
stream processing
data warehouses
Redis
point-in-time correctness
feature monitoring
model retraining
```

---

## 12. Commands

Run feature store demo:

```bash
python feature_store.py
```

Run tests:

```bash
python test_feature_store.py
```

---

## 13. Checklist

- [ ] Understand feature definition vs value
- [ ] Understand offline store
- [ ] Understand online store
- [ ] Understand materialization
- [ ] Understand point-in-time joins
- [ ] Understand feature freshness
- [ ] Understand event time vs processing time
- [ ] Run feature store demo
- [ ] Run tests