# Day 43: Upper Confidence Bound Bandit

## 1. Goal

Today's goal is to understand uncertainty-aware exploration with Upper Confidence Bound, and how UCB differs from random exploration strategies like epsilon-greedy.

Key concepts:

```text
empirical mean reward
uncertainty bonus
optimism under uncertainty
UCB score
regret
action count
```

## 2. Core Formula

UCB chooses the action with the largest optimistic score:

```text
UCB(action)
=
estimated mean reward
+
exploration bonus
```

A common bonus is:

```text
sqrt(
    c * log(total_steps) / action_count
)
```

where `c` controls exploration strength.

## 3. Why UCB Explores

An action can be selected for two different reasons:

```text
it already looks strong
or
it has high uncertainty
```

This is the main idea behind optimism under uncertainty.

## 4. Comparison with Epsilon-Greedy

```text
epsilon-greedy:
  sometimes explore randomly

UCB:
  explore based on uncertainty
```

UCB is more directed because rarely tried actions receive a larger exploration bonus.

## 5. Infinite Bonus for Untried Actions

Many UCB implementations assign:

```text
bonus = infinity
```

for actions that have never been tried.

This guarantees every action is explored at least once.

## 6. Regret

As with other bandit methods, regret measures the reward lost relative to always choosing the best action.

```text
regret
=
best expected reward
-
chosen action expected reward
```

A good UCB policy keeps cumulative regret growing slowly over time.

## 7. ML Systems Connection

UCB connects to:

```text
recommendation
news ranking
ad selection
online experimentation
bandit feedback
reinforcement learning
```

## 8. Code Mapping

This day's mini-project maps UCB ideas to simple code:

- `UCBBandit`: tracks counts, reward sums, and total steps
- `estimated_reward(...)`: empirical mean reward
- `exploration_bonus(...)`: uncertainty-based exploration term
- `ucb_score(...)`: optimistic score used for action selection
- `run_simulation(...)`: shows how action counts shift toward the best arm over time

The key exploration idea is:

```text
explore where uncertainty remains
not just at random
```

## 9. Commands

Run the demo:

```bash
python days/day43_ucb_bandit/ucb_bandit.py
```

Run the tests:

```bash
python days/day43_ucb_bandit/test_ucb_bandit.py
```

## 10. Checklist

- [x] Understand empirical mean reward
- [x] Understand the uncertainty bonus
- [x] Understand optimism under uncertainty
- [x] Understand the UCB score
- [x] Understand regret
- [x] Understand how UCB differs from epsilon-greedy
- [x] Run demo
- [x] Run tests
