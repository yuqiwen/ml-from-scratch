# Day 33: Training-Serving Skew and Feature Consistency

## 1. Goal

Today's goal is to understand why training features and serving features must be consistent.

Key concepts:

```text
training-serving skew
offline features
online features
preprocessing mismatch
schema mismatch
category mapping mismatch
point-in-time correctness
data leakage
feature store
```

---

## 2. Training-Serving Skew

Training-serving skew means:

```text
training feature computation
!=
serving feature computation
```

The model may still run successfully while producing poor predictions.

---

## 3. Common Causes

Examples:

```text
different normalization
different missing-value handling
different category mappings
different feature order
different units
different time windows
different tokenization
```

---

## 4. Offline vs Online Features

Offline features are commonly used for training:

```text
batch pipeline
data warehouse
historical tables
```

Online features are used for real-time inference:

```text
request payload
Redis
online feature store
streaming pipeline
```

Both should represent the same feature definition.

---

## 5. Schema Skew

Example:

```text
training:
[age, income, amount]

serving:
[income, age, amount]
```

The shape is valid, but feature meaning is wrong.

Feature order must be versioned and preserved.

---

## 6. Category Mapping Skew

Category encoders must be reused.

Example:

```text
training:
US -> 0
CN -> 1

serving:
CN -> 0
US -> 1
```

This silently changes model input semantics.

---

## 7. Point-in-Time Correctness

Training features must only use information available at prediction time.

Using future information causes data leakage.

Example:

```text
prediction time:
January 1

invalid feature:
data from February 1
```

Offline metrics may look excellent, but online quality will fail.

---

## 8. Feature Store

A feature store helps manage:

```text
feature definitions
offline values
online values
feature versions
point-in-time joins
freshness
```

One goal is reducing offline-online inconsistency.

---

## 9. Skew Detection

For the same entities or requests, compare:

```text
offline feature values
online feature values
```

Metrics:

```text
match rate
absolute difference
relative difference
missing mismatch
```

---

## 10. ML Systems Connection

Training-serving skew connects to:

```text
feature engineering
feature stores
data pipelines
model serving
data validation
model monitoring
reproducibility
```

A common prevention strategy:

```text
shared feature definitions
versioned preprocessing artifacts
offline-online validation
schema checks
```

---

## 11. Commands

Run skew detector demo:

```bash
python skew_detector.py
```

Run tests:

```bash
python test_skew_detector.py
```

---

## 12. Checklist

- [ ] Understand training-serving skew
- [ ] Understand offline vs online features
- [ ] Understand preprocessing mismatch
- [ ] Understand schema mismatch
- [ ] Understand category mapping mismatch
- [ ] Understand point-in-time correctness
- [ ] Understand feature store motivation
- [ ] Run skew detector
- [ ] Run tests