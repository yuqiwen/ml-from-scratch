import math
from ucb_bandit import UCBBandit, run_simulation


def test_untried_action_has_infinite_bonus() -> None:
    bandit = UCBBandit(["A", "B"])
    assert math.isinf(bandit.exploration_bonus("A"))


def test_estimated_reward_updates() -> None:
    bandit = UCBBandit(["A"])
    bandit.update("A", 1.0)
    bandit.update("A", 0.0)
    bandit.update("A", 1.0)
    assert abs(bandit.estimated_reward("A") - 2 / 3) < 1e-12


def test_simulation_prefers_best_action() -> None:
    bandit, regret = run_simulation(rounds=10000, seed=7)
    assert regret >= 0.0
    assert bandit.counts["camera"] > bandit.counts["travel"]
    assert bandit.counts["camera"] > bandit.counts["food"]


def main() -> None:
    test_untried_action_has_infinite_bonus()
    test_estimated_reward_updates()
    test_simulation_prefers_best_action()
    print("All Day 43 tests passed.")


if __name__ == "__main__":
    main()
