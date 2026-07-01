from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class Interaction:
    context: str
    action: str
    propensity: float
    reward: float


class EpsilonGreedyBandit:
    """
    A toy contextual bandit.

    The context is a simple string bucket, such as:
        "mobile-evening"
        "desktop-morning"

    For every context-action pair, the bandit tracks:
        count
        total reward
        estimated mean reward
    """

    def __init__(
        self,
        actions: list[str],
        epsilon: float = 0.1,
        seed: int = 42,
    ) -> None:
        if not actions:
            raise ValueError("actions must not be empty")

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")

        self.actions = list(actions)
        self.epsilon = epsilon
        self.rng = random.Random(seed)

        self.counts: dict[tuple[str, str], int] = {}
        self.reward_sums: dict[tuple[str, str], float] = {}

    def estimated_reward(
        self,
        context: str,
        action: str,
    ) -> float:
        key = (context, action)
        count = self.counts.get(key, 0)

        if count == 0:
            return 0.0

        return self.reward_sums[key] / count

    def greedy_action(
        self,
        context: str,
    ) -> str:
        """
        Select the action with the highest current estimate.

        Ties are broken randomly to avoid always favoring
        the first action before any evidence exists.
        """
        estimates = {
            action: self.estimated_reward(context, action)
            for action in self.actions
        }

        best_value = max(estimates.values())

        best_actions = [
            action
            for action, value in estimates.items()
            if value == best_value
        ]

        return self.rng.choice(best_actions)

    def choose_action(
        self,
        context: str,
    ) -> tuple[str, float, bool]:
        """
        Return:
            action
            propensity of selecting that action
            whether this decision was exploratory
        """
        should_explore = self.rng.random() < self.epsilon

        if should_explore:
            action = self.rng.choice(self.actions)

            # Under epsilon-greedy, a random action can also
            # coincide with the greedy action.
            propensity = self.epsilon / len(self.actions)

            greedy = self.greedy_action(context)
            if action == greedy:
                propensity += 1.0 - self.epsilon

            return action, propensity, True

        action = self.greedy_action(context)

        propensity = (
            1.0 - self.epsilon
            + self.epsilon / len(self.actions)
        )

        return action, propensity, False

    def update(
        self,
        context: str,
        action: str,
        reward: float,
    ) -> None:
        if action not in self.actions:
            raise ValueError("unknown action")

        key = (context, action)

        self.counts[key] = self.counts.get(key, 0) + 1
        self.reward_sums[key] = (
            self.reward_sums.get(key, 0.0) + reward
        )


def simulate_reward(
    context: str,
    action: str,
    rng: random.Random,
) -> float:
    """
    Hidden environment.

    Different contexts prefer different actions.
    """
    click_probabilities = {
        ("mobile-evening", "camera"): 0.70,
        ("mobile-evening", "travel"): 0.45,
        ("mobile-evening", "food"): 0.30,
        ("desktop-morning", "camera"): 0.25,
        ("desktop-morning", "travel"): 0.40,
        ("desktop-morning", "food"): 0.65,
    }

    probability = click_probabilities[(context, action)]

    return float(rng.random() < probability)


def optimal_expected_reward(
    context: str,
) -> float:
    if context == "mobile-evening":
        return 0.70

    if context == "desktop-morning":
        return 0.65

    raise ValueError("unknown context")


def chosen_expected_reward(
    context: str,
    action: str,
) -> float:
    values = {
        ("mobile-evening", "camera"): 0.70,
        ("mobile-evening", "travel"): 0.45,
        ("mobile-evening", "food"): 0.30,
        ("desktop-morning", "camera"): 0.25,
        ("desktop-morning", "travel"): 0.40,
        ("desktop-morning", "food"): 0.65,
    }

    return values[(context, action)]


def run_simulation(
    rounds: int = 5_000,
    epsilon: float = 0.1,
    seed: int = 7,
) -> tuple[EpsilonGreedyBandit, list[Interaction], float]:
    bandit = EpsilonGreedyBandit(
        actions=["camera", "travel", "food"],
        epsilon=epsilon,
        seed=seed,
    )

    environment_rng = random.Random(seed + 1)
    contexts = [
        "mobile-evening",
        "desktop-morning",
    ]

    logs: list[Interaction] = []
    cumulative_regret = 0.0

    for round_index in range(rounds):
        context = contexts[round_index % len(contexts)]

        action, propensity, _ = bandit.choose_action(context)

        reward = simulate_reward(
            context=context,
            action=action,
            rng=environment_rng,
        )

        bandit.update(
            context=context,
            action=action,
            reward=reward,
        )

        cumulative_regret += (
            optimal_expected_reward(context)
            - chosen_expected_reward(context, action)
        )

        logs.append(
            Interaction(
                context=context,
                action=action,
                propensity=propensity,
                reward=reward,
            )
        )

    return bandit, logs, cumulative_regret


def main() -> None:
    bandit, logs, cumulative_regret = run_simulation()

    for context in [
        "mobile-evening",
        "desktop-morning",
    ]:
        print(f"Context: {context}")

        for action in bandit.actions:
            print(
                f"  action={action:7s} "
                f"count={bandit.counts.get((context, action), 0):4d} "
                f"estimate={bandit.estimated_reward(context, action):.3f}"
            )

        print(
            f"  greedy action: "
            f"{bandit.greedy_action(context)}"
        )
        print()

    print(f"logged interactions: {len(logs)}")
    print(f"cumulative regret: {cumulative_regret:.2f}")


if __name__ == "__main__":
    main()
