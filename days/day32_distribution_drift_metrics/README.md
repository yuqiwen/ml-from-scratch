# Day 32: PSI, KS Statistic, and Distribution Drift Metrics

## 1. Goal

Today's goal is to compare complete feature distributions rather than only comparing mean and standard deviation.

Key concepts:

```text
histogram
bin
Population Stability Index
PSI
CDF
KS statistic
distribution drift
quantile bins
numerical stability
```

---

## 2. Why Mean and Standard Deviation Are Not Enough

Two datasets can have similar summary statistics while having different distribution shapes.

Therefore production drift monitoring often compares:

```text
histograms
bin proportions
cumulative distributions
```

---

## 3. PSI

Population Stability Index compares reference and live proportions across bins.

```text
PSI = sum(
    (live_i - reference_i)
    * log(live_i / reference_i)
)
```

A PSI close to zero means the distributions are similar.

A larger PSI means stronger distribution shift.

Common heuristic thresholds:

```text
PSI < 0.1:
  little drift

0.1 <= PSI < 0.25:
  moderate drift

PSI >= 0.25:
  significant drift
```

These thresholds are business heuristics, not universal laws.

---

## 4. Binning

PSI requires bins.

Common strategies:

```text
equal-width bins
reference quantile bins
domain-specific bins
```

Quantile bins try to place a similar number of reference samples in each bin.

---

## 5. Numerical Stability

If one bin has zero probability, the logarithm is undefined.

Use a small epsilon:

```python
ratio = max(ratio, epsilon)
```

This prevents division by zero and `log(0)`.

---

## 6. KS Statistic

The KS statistic compares cumulative distribution functions.

```text
KS = max_x |F_reference(x) - F_live(x)|
```

Range:

```text
0 to 1
```

A larger value means stronger distribution difference.

---

## 7. PSI vs KS

PSI:

```text
requires bins
easy to explain by bucket
sensitive to bin boundaries
```

KS:

```text
does not require bins
compares empirical CDFs
captures maximum cumulative difference
```

Neither metric proves that model quality decreased.

---

## 8. Statistical Significance vs Practical Significance

A full KS test may return a p-value.

With very large datasets, a tiny difference may become statistically significant.

Production monitoring should consider:

```text
effect size
business impact
feature importance
sample size
model quality
```

---

## 9. ML Systems Connection

Distribution drift metrics connect to:

```text
feature monitoring
data validation
alerting
model retraining
rollback
A/B testing
model quality monitoring
```

A typical process:

```text
detect drift
-> investigate feature pipeline
-> inspect predictions
-> inspect labels
-> retrain, fix, or rollback
```

---

## 10. Commands

Run drift metric demo:

```bash
python drift_metrics.py
```

Run tests:

```bash
python test_drift_metrics.py
```

---

## 11. Checklist

- [ ] Understand histogram comparison
- [ ] Understand PSI
- [ ] Understand bins
- [ ] Understand epsilon
- [ ] Understand CDF
- [ ] Understand KS statistic
- [ ] Understand PSI vs KS
- [ ] Run drift metric demo
- [ ] Run tests
