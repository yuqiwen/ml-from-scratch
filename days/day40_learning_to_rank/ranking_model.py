from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RankingBatch:
    features: torch.Tensor
    labels: torch.Tensor


class RankingModel(nn.Module):
    """
    Small pointwise ranking model.

    Input:
        pair_features: (B, feature_dim)

    Output:
        scores: (B,)
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 32,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                feature_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        pair_features: torch.Tensor,
    ) -> torch.Tensor:
        scores = self.network(
            pair_features
        )

        return scores.squeeze(-1)


def pointwise_bce_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Binary classification loss for ranking.

    scores:
        raw logits, shape (B,)

    labels:
        0/1 labels, shape (B,)
    """
    if scores.shape != labels.shape:
        raise ValueError(
            "scores and labels must have "
            "the same shape."
        )

    return F.binary_cross_entropy_with_logits(
        scores,
        labels.float(),
    )


def pairwise_logistic_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
) -> torch.Tensor:
    """
    Encourage positive scores to exceed
    negative scores.

    loss = softplus(
        negative_score - positive_score
    )
    """
    if (
        positive_scores.shape
        != negative_scores.shape
    ):
        raise ValueError(
            "positive_scores and "
            "negative_scores must have "
            "the same shape."
        )

    differences = (
        negative_scores
        - positive_scores
    )

    return F.softplus(
        differences
    ).mean()


def margin_ranking_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Encourage:

        positive_score
        >=
        negative_score + margin
    """
    if margin < 0:
        raise ValueError(
            "margin must be non-negative."
        )

    if (
        positive_scores.shape
        != negative_scores.shape
    ):
        raise ValueError(
            "positive_scores and "
            "negative_scores must have "
            "the same shape."
        )

    losses = F.relu(
        margin
        - positive_scores
        + negative_scores
    )

    return losses.mean()


def listwise_softmax_loss(
    scores: torch.Tensor,
    relevance: torch.Tensor,
) -> torch.Tensor:
    """
    Simple listwise loss.

    scores:
        shape (B, N)

    relevance:
        non-negative relevance values,
        shape (B, N)
    """
    if scores.shape != relevance.shape:
        raise ValueError(
            "scores and relevance must "
            "have the same shape."
        )

    if torch.any(relevance < 0):
        raise ValueError(
            "relevance must be non-negative."
        )

    target_distribution = F.softmax(
        relevance.float(),
        dim=-1,
    )

    log_model_distribution = F.log_softmax(
        scores,
        dim=-1,
    )

    loss_per_query = -(
        target_distribution
        * log_model_distribution
    ).sum(dim=-1)

    return loss_per_query.mean()


def build_pair_features(
    user_features: torch.Tensor,
    item_features: torch.Tensor,
    context_features: torch.Tensor,
) -> torch.Tensor:
    """
    Concatenate user, item, context, and
    simple elementwise cross features.

    All tensors use shape:
        (B, D)
    """
    if not (
        user_features.shape
        == item_features.shape
        == context_features.shape
    ):
        raise ValueError(
            "All feature tensors must "
            "have the same shape."
        )

    cross_features = (
        user_features
        * item_features
    )

    return torch.cat(
        [
            user_features,
            item_features,
            context_features,
            cross_features,
        ],
        dim=-1,
    )


def generate_synthetic_pointwise_batch(
    batch_size: int = 128,
    base_dim: int = 4,
) -> RankingBatch:
    """
    Generate synthetic user-item examples.

    Labels depend on user-item alignment
    and context.
    """
    user = torch.randn(
        batch_size,
        base_dim,
    )

    item = torch.randn(
        batch_size,
        base_dim,
    )

    context = torch.randn(
        batch_size,
        base_dim,
    )

    pair_features = build_pair_features(
        user,
        item,
        context,
    )

    hidden_relevance = (
        (user * item).sum(dim=-1)
        + 0.3 * context.sum(dim=-1)
    )

    labels = (
        hidden_relevance > 0
    ).float()

    return RankingBatch(
        features=pair_features,
        labels=labels,
    )


def pointwise_accuracy(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    predictions = (
        torch.sigmoid(scores) >= 0.5
    ).float()

    return (
        predictions == labels
    ).float().mean().item()


def pairwise_accuracy(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
) -> float:
    return (
        positive_scores
        > negative_scores
    ).float().mean().item()


def train_pointwise_demo(
    num_steps: int = 300,
) -> RankingModel:
    torch.manual_seed(42)

    base_dim = 4
    pair_feature_dim = base_dim * 4

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = RankingModel(
        feature_dim=pair_feature_dim,
        hidden_dim=32,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.003,
    )

    for step in range(num_steps):
        batch = (
            generate_synthetic_pointwise_batch(
                batch_size=128,
                base_dim=base_dim,
            )
        )

        features = batch.features.to(device)
        labels = batch.labels.to(device)

        model.train()

        scores = model(features)

        loss = pointwise_bce_loss(
            scores,
            labels,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            accuracy = pointwise_accuracy(
                scores,
                labels,
            )

            print(
                f"step={step:03d}, "
                f"loss={loss.item():.4f}, "
                f"accuracy={accuracy:.2%}"
            )

    return model


def pairwise_loss_demo() -> None:
    positive_scores = torch.tensor(
        [2.0, 0.5, 1.0]
    )

    negative_scores = torch.tensor(
        [0.0, 1.5, 0.8]
    )

    logistic = pairwise_logistic_loss(
        positive_scores,
        negative_scores,
    )

    margin = margin_ranking_loss(
        positive_scores,
        negative_scores,
        margin=1.0,
    )

    print()
    print("Pairwise loss demo")
    print(
        f"logistic loss = "
        f"{logistic.item():.4f}"
    )
    print(
        f"margin loss = "
        f"{margin.item():.4f}"
    )
    print(
        f"pairwise accuracy = "
        f"{pairwise_accuracy(
            positive_scores,
            negative_scores,
        ):.2%}"
    )


def listwise_loss_demo() -> None:
    scores = torch.tensor(
        [
            [2.0, 0.5, 1.5],
            [0.2, 1.7, 0.8],
        ]
    )

    relevance = torch.tensor(
        [
            [3.0, 0.0, 2.0],
            [0.0, 3.0, 1.0],
        ]
    )

    loss = listwise_softmax_loss(
        scores,
        relevance,
    )

    print()
    print("Listwise loss demo")
    print(
        f"listwise loss = "
        f"{loss.item():.4f}"
    )


def main() -> None:
    train_pointwise_demo()
    pairwise_loss_demo()
    listwise_loss_demo()


if __name__ == "__main__":
    main()