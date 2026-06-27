from retrieval_metrics import (
    QueryResult,
    dcg_at_k,
    evaluate_dataset,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k() -> None:
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

    result = precision_at_k(
        predicted,
        relevant,
        k=5,
    )

    print("Test 1: Precision@K")
    print(result)

    assert abs(
        result - 0.4
    ) < 1e-12

    print("Passed.\n")


def test_recall_at_k() -> None:
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

    result = recall_at_k(
        predicted,
        relevant,
        k=5,
    )

    print("Test 2: Recall@K")
    print(result)

    assert abs(
        result - 2 / 3
    ) < 1e-12

    print("Passed.\n")


def test_hit_at_k() -> None:
    predicted = [
        "A",
        "B",
        "C",
    ]

    relevant = {
        "B",
    }

    result = hit_at_k(
        predicted,
        relevant,
        k=3,
    )

    print("Test 3: Hit@K")
    print(result)

    assert result == 1.0

    print("Passed.\n")


def test_miss_at_k() -> None:
    predicted = [
        "A",
        "B",
        "C",
    ]

    relevant = {
        "X",
    }

    result = hit_at_k(
        predicted,
        relevant,
        k=3,
    )

    print("Test 4: miss at K")
    print(result)

    assert result == 0.0

    print("Passed.\n")


def test_reciprocal_rank() -> None:
    predicted = [
        "A",
        "B",
        "C",
        "D",
    ]

    relevant = {
        "C",
        "D",
    }

    result = reciprocal_rank(
        predicted,
        relevant,
    )

    print("Test 5: reciprocal rank")
    print(result)

    assert abs(
        result - 1 / 3
    ) < 1e-12

    print("Passed.\n")


def test_perfect_ndcg() -> None:
    predicted = [
        "A",
        "B",
        "C",
    ]

    relevance = {
        "A": 3.0,
        "B": 2.0,
        "C": 1.0,
    }

    result = ndcg_at_k(
        predicted,
        relevance,
        k=3,
    )

    print("Test 6: perfect NDCG")
    print(result)

    assert abs(
        result - 1.0
    ) < 1e-12

    print("Passed.\n")


def test_bad_order_has_lower_ndcg() -> None:
    ideal = [
        "A",
        "B",
        "C",
    ]

    reversed_order = [
        "C",
        "B",
        "A",
    ]

    relevance = {
        "A": 3.0,
        "B": 2.0,
        "C": 1.0,
    }

    ideal_score = ndcg_at_k(
        ideal,
        relevance,
        k=3,
    )

    reversed_score = ndcg_at_k(
        reversed_order,
        relevance,
        k=3,
    )

    print(
        "Test 7: worse order lowers NDCG"
    )
    print(ideal_score)
    print(reversed_score)

    assert ideal_score == 1.0
    assert reversed_score < ideal_score

    print("Passed.\n")


def test_dcg_early_result_worth_more() -> None:
    early = dcg_at_k(
        [1.0, 0.0, 0.0],
        k=3,
    )

    late = dcg_at_k(
        [0.0, 0.0, 1.0],
        k=3,
    )

    print(
        "Test 8: early relevance worth more"
    )
    print(early)
    print(late)

    assert early > late

    print("Passed.\n")


def test_dataset_metrics() -> None:
    queries = [
        QueryResult(
            query_id="q1",
            predicted_items=[
                "A",
                "B",
            ],
            relevant_items={
                "A",
            },
        ),
        QueryResult(
            query_id="q2",
            predicted_items=[
                "C",
                "D",
            ],
            relevant_items={
                "X",
            },
        ),
    ]

    metrics = evaluate_dataset(
        queries,
        k=2,
    )

    print("Test 9: dataset metrics")
    print(metrics)

    assert abs(
        metrics["hit_rate@2"]
        - 0.5
    ) < 1e-12

    assert abs(
        metrics["mrr"]
        - 0.5
    ) < 1e-12

    print("Passed.\n")


def main() -> None:
    test_precision_at_k()
    test_recall_at_k()
    test_hit_at_k()
    test_miss_at_k()
    test_reciprocal_rank()
    test_perfect_ndcg()
    test_bad_order_has_lower_ndcg()
    test_dcg_early_result_worth_more()
    test_dataset_metrics()

    print(
        "All Day 39 tests passed."
    )


if __name__ == "__main__":
    main()