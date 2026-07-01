# Day 42: Contextual Bandit and Exploration vs Exploitation

## 1. Goal

Today's goal is to understand how a system chooses an action, observes a reward, and balances exploitation with exploration under partial feedback.

Key concepts:

```text
context
action
reward
policy
exploration
exploitation
epsilon-greedy
regret
logged bandit feedback
```

## 2. Contextual Bandit

At each round:

```text
observe context
-> choose one action
-> observe reward only for that action
-> update policy
```

Example:

```text
context:
  user/device/time

actions:
  candidate items

reward:
  click or no click
```

The system does not observe what would have happened for actions it did not choose.

## 3. Exploration vs Exploitation

Exploitation chooses the action with the highest current estimated reward.

Exploration intentionally tries another action to gather information.

Pure exploitation may lock the system into an early mistake.

Pure exploration learns but may give poor user experience.

## 4. Epsilon-Greedy

With probability `epsilon`:

```text
choose a random action
```

With probability `1 - epsilon`:

```text
choose the action with the highest estimated reward
```

Example:

```text
epsilon = 0.1

10% random exploration
90% greedy exploitation
```

## 5. Reward Estimate

For each context bucket and action:

```text
estimated reward
=
total observed reward
/
number of times chosen
```

The estimate improves as the action is selected more often.

## 6. Regret

Regret measures the reward lost compared with always choosing the optimal action.

For one round:

```text
regret
=
best possible expected reward
-
chosen action expected reward
```

Cumulative regret adds this difference over time.

## 7. Logged Bandit Feedback

A logged interaction contains:

```text
context
chosen action
propensity
observed reward
```

Only the reward for the chosen action is observed.

This is different from supervised learning, where labels may be available for every class or example.

## 8. ML Systems Connection

Contextual bandits connect to:

```text
recommendation
ads
ranking
A/B testing
propensity logging
off-policy evaluation
reinforcement learning
```

## 9. Code Mapping

This day's mini-project maps the contextual bandit loop to runnable code:

- `EpsilonGreedyBandit`: maintains per-context action counts and reward estimates
- `choose_action(...)`: returns both the action and its logging propensity
- `update(...)`: updates the running reward estimate after observing feedback
- `simulate_reward(...)`: toy hidden environment
- `run_simulation(...)`: demonstrates learning under repeated interaction

The key bandit idea is:

```text
only chosen actions reveal rewards
so exploration is necessary
```

## 10. Commands

Run the demo:

```bash
python days/day42_contextual_bandit/contextual_bandit.py
```

Run the tests:

```bash
python days/day42_contextual_bandit/test_contextual_bandit.py
```

## 11. Checklist

- [x] Understand context, action, reward
- [x] Understand partial feedback
- [x] Understand exploration vs exploitation
- [x] Understand epsilon-greedy
- [x] Understand reward estimates
- [x] Understand regret
- [x] Understand logged bandit feedback
- [x] Run demo
- [x] Run tests
