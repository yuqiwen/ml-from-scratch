import torch

from ranking_model import (
    RankingModel,
    build_pair_features,
    listwise_softmax_loss,
    margin_ranking_loss,
    pairwise_accuracy,
    pairwise_logistic_loss,
    pointwise_bce_loss,
)


def test_ranking_model_shape() -> None:
    model = RankingModel(
        feature_dim=16,
        hidden_dim=8,
    )

    features = torch.randn(
        5,
        16,
    )

    scores = model(features)

    print(
        "Test 1: ranking model shape"
    )
    print(scores.shape)

    assert scores.shape == (5,)

    print("Passed.\n")


def test_pair_feature_shape() -> None:
    user = torch.randn(7, 4)
    item = torch.randn(7, 4)
    context = torch.randn(7, 4)

    features = build_pair_features(
        user,
        item,
        context,
    )

    print(
        "Test 2: pair feature shape"
    )
    print(features.shape)

    assert features.shape == (
        7,
        16,
    )

    print("Passed.\n")


def test_pointwise_loss_scalar() -> None:
    scores = torch.tensor(
        [1.0, -1.0, 0.5]
    )

    labels = torch.tensor(
        [1.0, 0.0, 1.0]
    )

    loss = pointwise_bce_loss(
        scores,
        labels,
    )

    print(
        "Test 3: pointwise loss scalar"
    )
    print(loss)

    assert loss.ndim == 0
    assert loss.item() > 0

    print("Passed.\n")


def test_good_pairwise_order_has_lower_loss() -> None:
    good_loss = pairwise_logistic_loss(
        positive_scores=torch.tensor(
            [3.0, 2.0]
        ),
        negative_scores=torch.tensor(
            [0.0, -1.0]
        ),
    )

    bad_loss = pairwise_logistic_loss(
        positive_scores=torch.tensor(
            [0.0, -1.0]
        ),
        negative_scores=torch.tensor(
            [3.0, 2.0]
        ),
    )

    print(
        "Test 4: good pair order lowers loss"
    )
    print(good_loss)
    print(bad_loss)

    assert good_loss < bad_loss

    print("Passed.\n")


def test_margin_loss_zero_when_margin_met() -> None:
    loss = margin_ranking_loss(
        positive_scores=torch.tensor(
            [3.0, 4.0]
        ),
        negative_scores=torch.tensor(
            [1.0, 2.0]
        ),
        margin=1.0,
    )

    print(
        "Test 5: margin requirement met"
    )
    print(loss)

    assert loss.item() == 0.0

    print("Passed.\n")


def test_pairwise_accuracy() -> None:
    accuracy = pairwise_accuracy(
        positive_scores=torch.tensor(
            [3.0, 0.5, 2.0]
        ),
        negative_scores=torch.tensor(
            [1.0, 1.0, 0.0]
        ),
    )

    print(
        "Test 6: pairwise accuracy"
    )
    print(accuracy)

    assert abs(
        accuracy - 2 / 3
    ) < 1e-6

    print("Passed.\n")


def test_listwise_better_order_lower_loss() -> None:
    relevance = torch.tensor(
        [
            [3.0, 2.0, 0.0]
        ]
    )

    good_scores = torch.tensor(
        [
            [3.0, 2.0, 0.0]
        ]
    )

    bad_scores = torch.tensor(
        [
            [0.0, 2.0, 3.0]
        ]
    )

    good_loss = listwise_softmax_loss(
        good_scores,
        relevance,
    )

    bad_loss = listwise_softmax_loss(
        bad_scores,
        relevance,
    )

    print(
        "Test 7: better list order lowers loss"
    )
    print(good_loss)
    print(bad_loss)

    assert good_loss < bad_loss

    print("Passed.\n")


def main() -> None:
    test_ranking_model_shape()
    test_pair_feature_shape()
    test_pointwise_loss_scalar()
    test_good_pairwise_order_has_lower_loss()
    test_margin_loss_zero_when_margin_met()
    test_pairwise_accuracy()
    test_listwise_better_order_lower_loss()

    print(
        "All Day 40 tests passed."
    )


if __name__ == "__main__":
    main()