import torch

from debiased_ranking import (
    ips_pairwise_loss,
    ips_pointwise_loss,
    ips_weights,
    position_propensity,
    self_normalized_ips_loss,
    simulate_click_probability,
)


def test_position_propensity_decreases() -> None:
    positions = torch.tensor(
        [1, 2, 3, 4]
    )

    propensities = position_propensity(
        positions,
        decay=0.8,
    )

    print(
        "Test 1: propensity decreases"
    )
    print(propensities)

    assert propensities[0] == 1.0
    assert torch.all(
        propensities[:-1]
        > propensities[1:]
    )

    print("Passed.\n")


def test_inverse_propensity_weights() -> None:
    propensities = torch.tensor(
        [1.0, 0.5, 0.25]
    )

    weights = ips_weights(
        propensities
    )

    print(
        "Test 2: inverse propensity weights"
    )
    print(weights)

    expected = torch.tensor(
        [1.0, 2.0, 4.0]
    )

    assert torch.allclose(
        weights,
        expected,
    )

    print("Passed.\n")


def test_weight_clipping() -> None:
    propensities = torch.tensor(
        [1.0, 0.1, 0.001]
    )

    weights = ips_weights(
        propensities,
        max_weight=5.0,
    )

    print(
        "Test 3: weight clipping"
    )
    print(weights)

    assert torch.all(
        weights <= 5.0
    )

    assert weights[-1] == 5.0

    print("Passed.\n")


def test_low_propensity_increases_loss() -> None:
    scores = torch.tensor(
        [0.0]
    )

    labels = torch.tensor(
        [1.0]
    )

    high_propensity_loss = (
        ips_pointwise_loss(
            scores,
            labels,
            propensities=torch.tensor(
                [1.0]
            ),
        )
    )

    low_propensity_loss = (
        ips_pointwise_loss(
            scores,
            labels,
            propensities=torch.tensor(
                [0.1]
            ),
        )
    )

    print(
        "Test 4: low propensity increases loss"
    )
    print(high_propensity_loss)
    print(low_propensity_loss)

    assert (
        low_propensity_loss
        > high_propensity_loss
    )

    print("Passed.\n")


def test_self_normalized_equal_losses() -> None:
    scores = torch.tensor(
        [0.0, 0.0]
    )

    labels = torch.tensor(
        [1.0, 1.0]
    )

    propensities = torch.tensor(
        [1.0, 0.1]
    )

    snips = self_normalized_ips_loss(
        scores,
        labels,
        propensities,
    )

    normal_single_loss = (
        torch.nn.functional
        .binary_cross_entropy_with_logits(
            torch.tensor([0.0]),
            torch.tensor([1.0]),
        )
    )

    print(
        "Test 5: self-normalized equal losses"
    )
    print(snips)
    print(normal_single_loss)

    assert torch.allclose(
        snips,
        normal_single_loss,
    )

    print("Passed.\n")


def test_ips_pairwise_weighting() -> None:
    positive_scores = torch.tensor(
        [2.0, 2.0]
    )

    negative_scores = torch.tensor(
        [1.0, 1.0]
    )

    high_propensity = ips_pairwise_loss(
        positive_scores,
        negative_scores,
        positive_propensities=(
            torch.tensor(
                [1.0, 1.0]
            )
        ),
    )

    low_propensity = ips_pairwise_loss(
        positive_scores,
        negative_scores,
        positive_propensities=(
            torch.tensor(
                [0.1, 0.1]
            )
        ),
    )

    print(
        "Test 6: pairwise IPS weighting"
    )
    print(high_propensity)
    print(low_propensity)

    assert low_propensity > high_propensity

    print("Passed.\n")


def test_click_probability_has_position_bias() -> None:
    relevance = torch.tensor(
        [0.8, 0.8]
    )

    positions = torch.tensor(
        [1, 5]
    )

    click_probability = (
        simulate_click_probability(
            relevance,
            positions,
            decay=0.8,
        )
    )

    print(
        "Test 7: click probability position bias"
    )
    print(click_probability)

    assert (
        click_probability[0]
        > click_probability[1]
    )

    print("Passed.\n")


def main() -> None:
    test_position_propensity_decreases()
    test_inverse_propensity_weights()
    test_weight_clipping()
    test_low_propensity_increases_loss()
    test_self_normalized_equal_losses()
    test_ips_pairwise_weighting()
    test_click_probability_has_position_bias()

    print(
        "All Day 41 tests passed."
    )


if __name__ == "__main__":
    main()