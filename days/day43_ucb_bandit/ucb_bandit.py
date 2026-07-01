from __future__ import annotations
import math
import random


class UCBBandit:
    def __init__(self, actions: list[str], exploration_scale: float = 2.0) -> None:
        if not actions:
            raise ValueError("actions must not be empty")
        if exploration_scale <= 0:
            raise ValueError("exploration_scale must be positive")

        self.actions = list(actions)
        self.exploration_scale = exploration_scale
        self.counts = {action: 0 for action in actions}
        self.reward_sums = {action: 0.0 for action in actions}
        self.total_steps = 0

    def estimated_reward(self, action: str) -> float:
        count = self.counts[action]
        return 0.0 if count == 0 else self.reward_sums[action] / count

    def exploration_bonus(self, action: str) -> float:
        count = self.counts[action]
        if count == 0:
            return math.inf
        return math.sqrt(
            self.exploration_scale
            * math.log(max(self.total_steps, 1))
            / count
        )

    def ucb_score(self, action: str) -> float:
        return self.estimated_reward(action) + self.exploration_bonus(action)

    def choose_action(self) -> str:
        return max(self.actions, key=self.ucb_score)

    def update(self, action: str, reward: float) -> None:
        if action not in self.counts:
            raise ValueError("unknown action")
        self.total_steps += 1
        self.counts[action] += 1
        self.reward_sums[action] += reward


def simulate_reward(action: str, rng: random.Random) -> float:
    probabilities = {"camera": 0.70, "travel": 0.50, "food": 0.30}
    return float(rng.random() < probabilities[action])


def run_simulation(rounds: int = 5000, seed: int = 42):
    bandit = UCBBandit(["camera", "travel", "food"])
    rng = random.Random(seed)
    regret = 0.0
    expected = {"camera": 0.70, "travel": 0.50, "food": 0.30}

    for _ in range(rounds):
        action = bandit.choose_action()
        reward = simulate_reward(action, rng)
        bandit.update(action, reward)
        regret += 0.70 - expected[action]

    return bandit, regret


def main() -> None:
    bandit, regret = run_simulation()

    for action in bandit.actions:
        print(
            f"{action:7s} "
            f"count={bandit.counts[action]:4d} "
            f"estimate={bandit.estimated_reward(action):.3f} "
            f"bonus={bandit.exploration_bonus(action):.3f}"
        )

    print(f"cumulative regret: {regret:.2f}")


if __name__ == "__main__":
    main()
