from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LoggedRankingBatch:
    scores: torch.Tensor
    labels: torch.Tensor
    positions: torch.Tensor


def validate_shapes(
    scores: torch.Tensor,
    labels: torch.Tensor,
    propensities: torch.Tensor,
) -> None:
    if not (
        scores.shape
        == labels.shape
        == propensities.shape
    ):
        raise ValueError(
            "scores, labels, and propensities "
            "must have the same shape."
        )


def position_propensity(
    positions: torch.Tensor,
    decay: float = 0.8,
) -> torch.Tensor:
    """
    Toy examination model.

    Position 1 has propensity 1.0.

    Later positions decay geometrically:

        propensity(position)
        =
        decay ** (position - 1)
    """
    if decay <= 0 or decay > 1:
        raise ValueError(
            "decay must be in (0, 1]."
        )

    if torch.any(positions < 1):
        raise ValueError(
            "positions must start from 1."
        )

    return decay ** (
        positions.float() - 1.0
    )


def pointwise_bce_per_sample(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Return one BCE loss value per sample.
    """
    if scores.shape != labels.shape:
        raise ValueError(
            "scores and labels must "
            "have the same shape."
        )

    return F.binary_cross_entropy_with_logits(
        scores,
        labels.float(),
        reduction="none",
    )


def ips_weights(
    propensities: torch.Tensor,
    min_propensity: float = 1e-3,
    max_weight: float | None = None,
) -> torch.Tensor:
    """
    Calculate inverse propensity weights.

    min_propensity avoids division by zero.

    max_weight optionally clips large weights.
    """
    if min_propensity <= 0:
        raise ValueError(
            "min_propensity must be positive."
        )

    safe_propensities = torch.clamp(
        propensities,
        min=min_propensity,
    )

    weights = 1.0 / safe_propensities

    if max_weight is not None:
        if max_weight <= 0:
            raise ValueError(
                "max_weight must be positive."
            )

        weights = torch.clamp(
            weights,
            max=max_weight,
        )

    return weights


def ips_pointwise_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    propensities: torch.Tensor,
    max_weight: float | None = None,
) -> torch.Tensor:
    """
    Standard IPS-weighted BCE loss.
    """
    validate_shapes(
        scores,
        labels,
        propensities,
    )

    losses = pointwise_bce_per_sample(
        scores,
        labels,
    )

    weights = ips_weights(
        propensities,
        max_weight=max_weight,
    )

    return (
        weights * losses
    ).mean()


def self_normalized_ips_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    propensities: torch.Tensor,
    max_weight: float | None = None,
) -> torch.Tensor:
    """
    Self-normalized IPS:

        sum(weight * loss)
        /
        sum(weight)
    """
    validate_shapes(
        scores,
        labels,
        propensities,
    )

    losses = pointwise_bce_per_sample(
        scores,
        labels,
    )

    weights = ips_weights(
        propensities,
        max_weight=max_weight,
    )

    denominator = weights.sum()

    if denominator <= 0:
        raise ValueError(
            "Sum of weights must be positive."
        )

    return (
        weights * losses
    ).sum() / denominator


def pairwise_logistic_loss_per_pair(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
) -> torch.Tensor:
    """
    One pairwise logistic loss per pair.
    """
    if (
        positive_scores.shape
        != negative_scores.shape
    ):
        raise ValueError(
            "positive and negative scores "
            "must have the same shape."
        )

    return F.softplus(
        negative_scores
        - positive_scores
    )


def ips_pairwise_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    positive_propensities: torch.Tensor,
    max_weight: float | None = None,
) -> torch.Tensor:
    """
    Weight pairwise loss using the propensity
    of the clicked or preferred item.
    """
    if not (
        positive_scores.shape
        == negative_scores.shape
        == positive_propensities.shape
    ):
        raise ValueError(
            "All tensors must have "
            "the same shape."
        )

    losses = pairwise_logistic_loss_per_pair(
        positive_scores,
        negative_scores,
    )

    weights = ips_weights(
        positive_propensities,
        max_weight=max_weight,
    )

    return (
        weights * losses
    ).mean()


def simulate_click_probability(
    relevance_probability: torch.Tensor,
    positions: torch.Tensor,
    decay: float = 0.8,
) -> torch.Tensor:
    """
    Toy click model:

        P(click)
        =
        P(relevant)
        *
        P(examine position)
    """
    if torch.any(
        (relevance_probability < 0)
        | (relevance_probability > 1)
    ):
        raise ValueError(
            "relevance_probability must "
            "be in [0, 1]."
        )

    propensities = position_propensity(
        positions,
        decay=decay,
    )

    return (
        relevance_probability
        * propensities
    )


def run_demo() -> None:
    positions = torch.tensor(
        [1, 2, 5, 10]
    )

    propensities = position_propensity(
        positions,
        decay=0.8,
    )

    print("Position propensities")

    for position, propensity in zip(
        positions.tolist(),
        propensities.tolist(),
    ):
        print(
            f"position={position}, "
            f"propensity={propensity:.4f}, "
            f"ips_weight={1.0 / propensity:.4f}"
        )

    print()

    scores = torch.tensor(
        [0.5, 0.5, 0.5, 0.5]
    )

    labels = torch.tensor(
        [1.0, 1.0, 1.0, 1.0]
    )

    normal_loss = (
        pointwise_bce_per_sample(
            scores,
            labels,
        ).mean()
    )

    ips_loss = ips_pointwise_loss(
        scores,
        labels,
        propensities,
    )

    clipped_ips_loss = ips_pointwise_loss(
        scores,
        labels,
        propensities,
        max_weight=5.0,
    )

    snips_loss = self_normalized_ips_loss(
        scores,
        labels,
        propensities,
        max_weight=5.0,
    )

    print(
        f"normal BCE = "
        f"{normal_loss.item():.4f}"
    )

    print(
        f"IPS BCE = "
        f"{ips_loss.item():.4f}"
    )

    print(
        f"clipped IPS BCE = "
        f"{clipped_ips_loss.item():.4f}"
    )

    print(
        f"self-normalized IPS BCE = "
        f"{snips_loss.item():.4f}"
    )


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()