# Day 41: Position Bias and Inverse Propensity Weighting

## 1. Goal

Today's goal is to understand why click labels are biased by display position and how inverse propensity weighting can reduce that bias.

Key concepts:

```text
position bias
exposure
examination probability
propensity
inverse propensity weighting
IPS
weight clipping
self-normalized IPS
feedback loop
off-policy evaluation
```

## 2. Position Bias

Higher-ranked items are more likely to be examined and clicked.

Observed clicks depend on both:

```text
item relevance
and
display position
```

A simplified model is:

```text
P(click)
=
P(examine position)
*
P(relevant | examined)
```

## 3. Feedback Loop

Historical ranking affects exposure.

```text
ranked higher
-> more exposure
-> more clicks
-> more positive training labels
-> ranked higher again
```

This can reinforce old model biases.

## 4. Propensity

Propensity represents the probability that an item receives an observation opportunity.

For position bias:

```text
propensity(position)
=
P(user examines this position)
```

Lower positions usually have smaller propensity.

## 5. Inverse Propensity Weighting

IPS uses:

```text
weight = 1 / propensity
```

Low-propensity observations receive larger weights.

Weighted pointwise loss:

```text
weighted_loss
=
mean(
    sample_loss / propensity
)
```

## 6. Propensity Estimation

Propensity may be estimated using:

```text
randomized ranking experiments
position swaps
intervention logs
examination models
```

Observed CTR alone does not cleanly separate relevance from position effects.

## 7. High Variance

Small propensity values create very large weights.

```text
propensity = 0.001
weight = 1000
```

This may cause unstable gradients and high-variance estimates.

## 8. Weight Clipping

Clip large IPS weights:

```text
weight
=
min(
    1 / propensity,
    max_weight
)
```

Clipping reduces variance at the cost of some bias.

## 9. Self-Normalized IPS

```text
SNIPS loss
=
sum(weight * loss)
/
sum(weight)
```

This often improves stability but is not exactly unbiased.

## 10. Pairwise IPS

Pairwise ranking losses may be weighted by the propensity of the clicked item.

```text
IPS pairwise loss
=
pairwise_loss
/
clicked_item_propensity
```

Clicks from lower-exposure positions receive greater importance.

## 11. ML Systems Connection

Position debiasing connects to:

```text
learning to rank
recommendation systems
advertising
search
causal inference
contextual bandits
off-policy evaluation
reinforcement learning
```

## 12. Code Mapping

This day's mini-project maps debiasing ideas to runnable code:

- `position_propensity(...)`: toy examination model over rank positions
- `ips_weights(...)`: inverse propensity weights with optional clipping
- `ips_pointwise_loss(...)`: IPS-weighted pointwise BCE
- `self_normalized_ips_loss(...)`: self-normalized variant for better stability
- `ips_pairwise_loss(...)`: pairwise IPS weighting
- `simulate_click_probability(...)`: illustrates how position and relevance combine

The key debiasing idea is:

```text
clicks are not pure relevance labels
exposure must be accounted for
```

## 13. Commands

Run the demo:

```bash
python days/day41_position_bias_ips/debiased_ranking.py
```

Run the tests:

```bash
python days/day41_position_bias_ips/test_debiased_ranking.py
```

## 14. Checklist

- [x] Understand position bias
- [x] Understand exposure vs relevance
- [x] Understand propensity
- [x] Understand IPS
- [x] Understand weight clipping
- [x] Understand self-normalized IPS
- [x] Understand feedback loops
- [x] Understand randomized interventions
- [x] Run demo
- [x] Run tests
