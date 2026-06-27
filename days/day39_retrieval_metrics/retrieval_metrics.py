from dataclasses import dataclass
import math
from typing import Hashable


ItemId = Hashable


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    predicted_items: list[ItemId]
    relevant_items: set[ItemId]


@dataclass(frozen=True)
class RetrievalMetricReport:
    precision_at_k: float
    recall_at_k: float
    hit_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


def validate_k(
    k: int,
) -> None:
    if k <= 0:
        raise ValueError(
            "k must be positive."
        )


def top_k_items(
    predicted_items: list[ItemId],
    k: int,
) -> list[ItemId]:
    validate_k(k)

    return predicted_items[:k]


def precision_at_k(
    predicted_items: list[ItemId],
    relevant_items: set[ItemId],
    k: int,
) -> float:
    """
    Fraction of the top-k predictions
    that are relevant.
    """
    top_k = top_k_items(
        predicted_items,
        k,
    )

    if not top_k:
        return 0.0

    hits = sum(
        item_id in relevant_items
        for item_id in top_k
    )

    # Common convention:
    # divide by k rather than len(top_k).
    # This assumes the system is expected to
    # return k results.
    return hits / k


def recall_at_k(
    predicted_items: list[ItemId],
    relevant_items: set[ItemId],
    k: int,
) -> float:
    """
    Fraction of all relevant items recovered
    in the top-k predictions.
    """
    if not relevant_items:
        return 0.0

    top_k = top_k_items(
        predicted_items,
        k,
    )

    hits = sum(
        item_id in relevant_items
        for item_id in top_k
    )

    return hits / len(relevant_items)


def hit_at_k(
    predicted_items: list[ItemId],
    relevant_items: set[ItemId],
    k: int,
) -> float:
    """
    Return 1.0 if the top-k contains at least
    one relevant item, otherwise 0.0.
    """
    top_k = top_k_items(
        predicted_items,
        k,
    )

    return float(
        any(
            item_id in relevant_items
            for item_id in top_k
        )
    )


def reciprocal_rank(
    predicted_items: list[ItemId],
    relevant_items: set[ItemId],
) -> float:
    """
    Reciprocal of the first relevant rank.
    """
    for rank, item_id in enumerate(
        predicted_items,
        start=1,
    ):
        if item_id in relevant_items:
            return 1.0 / rank

    return 0.0


def dcg_at_k(
    relevance_scores: list[float],
    k: int,
) -> float:
    """
    Calculate DCG using graded relevance.

    relevance_scores should already follow
    predicted ranking order.
    """
    validate_k(k)

    dcg = 0.0

    for rank, relevance in enumerate(
        relevance_scores[:k],
        start=1,
    ):
        if relevance < 0:
            raise ValueError(
                "Relevance scores must be "
                "non-negative."
            )

        gain = (
            2.0 ** relevance
            - 1.0
        )

        discount = math.log2(
            rank + 1
        )

        dcg += gain / discount

    return dcg


def ndcg_at_k(
    predicted_items: list[ItemId],
    relevance_by_item: dict[
        ItemId,
        float,
    ],
    k: int,
) -> float:
    """
    Calculate NDCG for the predicted ranking.
    """
    validate_k(k)

    predicted_relevance = [
        relevance_by_item.get(
            item_id,
            0.0,
        )
        for item_id
        in predicted_items[:k]
    ]

    actual_dcg = dcg_at_k(
        predicted_relevance,
        k,
    )

    ideal_relevance = sorted(
        relevance_by_item.values(),
        reverse=True,
    )[:k]

    ideal_dcg = dcg_at_k(
        ideal_relevance,
        k,
    )

    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg


def evaluate_query(
    predicted_items: list[ItemId],
    relevant_items: set[ItemId],
    relevance_by_item: dict[
        ItemId,
        float,
    ],
    k: int,
) -> RetrievalMetricReport:
    return RetrievalMetricReport(
        precision_at_k=precision_at_k(
            predicted_items,
            relevant_items,
            k,
        ),
        recall_at_k=recall_at_k(
            predicted_items,
            relevant_items,
            k,
        ),
        hit_at_k=hit_at_k(
            predicted_items,
            relevant_items,
            k,
        ),
        reciprocal_rank=reciprocal_rank(
            predicted_items,
            relevant_items,
        ),
        ndcg_at_k=ndcg_at_k(
            predicted_items,
            relevance_by_item,
            k,
        ),
    )


def mean_metric(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def evaluate_dataset(
    query_results: list[QueryResult],
    k: int,
) -> dict[str, float]:
    """
    Evaluate binary relevance metrics over
    multiple queries.

    For NDCG, relevant items receive score 1.
    """
    reports = []

    for result in query_results:
        relevance_by_item = {
            item_id: 1.0
            for item_id
            in result.relevant_items
        }

        reports.append(
            evaluate_query(
                predicted_items=(
                    result.predicted_items
                ),
                relevant_items=(
                    result.relevant_items
                ),
                relevance_by_item=(
                    relevance_by_item
                ),
                k=k,
            )
        )

    return {
        f"precision@{k}": mean_metric(
            [
                report.precision_at_k
                for report in reports
            ]
        ),
        f"recall@{k}": mean_metric(
            [
                report.recall_at_k
                for report in reports
            ]
        ),
        f"hit_rate@{k}": mean_metric(
            [
                report.hit_at_k
                for report in reports
            ]
        ),
        "mrr": mean_metric(
            [
                report.reciprocal_rank
                for report in reports
            ]
        ),
        f"ndcg@{k}": mean_metric(
            [
                report.ndcg_at_k
                for report in reports
            ]
        ),
    }


def run_demo() -> None:
    predicted = [
        "A",
        "B",
        "C",
        "D",
        "E",
    ]

    relevant = {
        "B",
        "D",
        "X",
    }

    relevance_by_item = {
        "B": 3.0,
        "D": 2.0,
        "X": 1.0,
    }

    report = evaluate_query(
        predicted_items=predicted,
        relevant_items=relevant,
        relevance_by_item=(
            relevance_by_item
        ),
        k=5,
    )

    print("Single query report")
    print(report)
    print()

    dataset = [
        QueryResult(
            query_id="user-1",
            predicted_items=[
                "A",
                "B",
                "C",
            ],
            relevant_items={
                "B",
                "X",
            },
        ),
        QueryResult(
            query_id="user-2",
            predicted_items=[
                "D",
                "E",
                "F",
            ],
            relevant_items={
                "D",
            },
        ),
        QueryResult(
            query_id="user-3",
            predicted_items=[
                "G",
                "H",
                "I",
            ],
            relevant_items={
                "Z",
            },
        ),
    ]

    metrics = evaluate_dataset(
        dataset,
        k=3,
    )

    print("Dataset metrics")

    for name, value in metrics.items():
        print(
            f"{name}: {value:.4f}"
        )


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()