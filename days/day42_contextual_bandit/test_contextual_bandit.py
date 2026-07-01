from contextual_bandit import (
    EpsilonGreedyBandit,
    run_simulation,
)


def test_estimate_updates() -> None:
    bandit = EpsilonGreedyBandit(
        actions=["A", "B"],
        epsilon=0.1,
    )

    bandit.update("ctx", "A", 1.0)
    bandit.update("ctx", "A", 0.0)
    bandit.update("ctx", "A", 1.0)

    estimate = bandit.estimated_reward("ctx", "A")

    assert abs(estimate - 2 / 3) < 1e-12


def test_greedy_selects_better_action() -> None:
    bandit = EpsilonGreedyBandit(
        actions=["A", "B"],
        epsilon=0.0,
        seed=1,
    )

    for reward in [1.0, 1.0, 1.0]:
        bandit.update("ctx", "A", reward)

    for reward in [0.0, 0.0, 0.0]:
        bandit.update("ctx", "B", reward)

    assert bandit.greedy_action("ctx") == "A"


def test_propensity_is_valid() -> None:
    bandit = EpsilonGreedyBandit(
        actions=["A", "B", "C"],
        epsilon=0.2,
        seed=3,
    )

    for _ in range(20):
        _, propensity, _ = bandit.choose_action("ctx")
        assert 0.0 < propensity <= 1.0


def test_simulation_learns_context_preferences() -> None:
    bandit, logs, regret = run_simulation(
        rounds=10_000,
        epsilon=0.1,
        seed=11,
    )

    assert len(logs) == 10_000
    assert regret >= 0.0

    assert (
        bandit.greedy_action("mobile-evening")
        == "camera"
    )

    assert (
        bandit.greedy_action("desktop-morning")
        == "food"
    )


def main() -> None:
    test_estimate_updates()
    test_greedy_selects_better_action()
    test_propensity_is_valid()
    test_simulation_learns_context_preferences()

    print("All Day 42 tests passed.")


if __name__ == "__main__":
    main()
