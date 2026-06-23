from vector_retrieval import (
    ItemEmbedding,
    ToyClusterIndex,
    cosine_similarity,
    dot_product,
    exact_search,
    normalize,
    recall_at_k,
)


def test_dot_product() -> None:
    result = dot_product(
        [1.0, 2.0],
        [3.0, 4.0],
    )

    print("Test 1: dot product")
    print(result)

    assert result == 11.0

    print("Passed.\n")


def test_cosine_identical_direction() -> None:
    similarity = cosine_similarity(
        [1.0, 2.0],
        [2.0, 4.0],
    )

    print(
        "Test 2: identical direction cosine"
    )
    print(similarity)

    assert abs(similarity - 1.0) < 1e-12

    print("Passed.\n")


def test_normalize() -> None:
    result = normalize(
        [3.0, 4.0]
    )

    print("Test 3: normalize")
    print(result)

    assert abs(result[0] - 0.6) < 1e-12
    assert abs(result[1] - 0.8) < 1e-12

    print("Passed.\n")


def test_exact_search() -> None:
    items = [
        ItemEmbedding(
            item_id=1,
            embedding=[1.0, 0.0],
        ),
        ItemEmbedding(
            item_id=2,
            embedding=[0.8, 0.2],
        ),
        ItemEmbedding(
            item_id=3,
            embedding=[0.0, 1.0],
        ),
    ]

    results = exact_search(
        query_embedding=[1.0, 0.0],
        items=items,
        top_k=2,
    )

    print("Test 4: exact search")
    print(results)

    assert results[0].item_id == 1
    assert results[1].item_id == 2

    print("Passed.\n")


def test_cluster_index() -> None:
    index = ToyClusterIndex(
        cluster_centers={
            "x": [1.0, 0.0],
            "y": [0.0, 1.0],
        }
    )

    index.add_item(
        ItemEmbedding(
            item_id=1,
            embedding=[0.9, 0.1],
        )
    )

    index.add_item(
        ItemEmbedding(
            item_id=2,
            embedding=[0.1, 0.9],
        )
    )

    results = index.search(
        query_embedding=[1.0, 0.0],
        top_k=1,
        num_clusters_to_search=1,
    )

    print("Test 5: cluster index")
    print(results)

    assert results[0].item_id == 1

    print("Passed.\n")


def test_recall_at_k() -> None:
    exact = [
        type("Result", (), {
            "item_id": 1
        })(),
        type("Result", (), {
            "item_id": 2
        })(),
        type("Result", (), {
            "item_id": 3
        })(),
    ]

    approximate = [
        type("Result", (), {
            "item_id": 1
        })(),
        type("Result", (), {
            "item_id": 2
        })(),
        type("Result", (), {
            "item_id": 4
        })(),
    ]

    recall = recall_at_k(
        exact_results=exact,
        approximate_results=approximate,
        k=3,
    )

    print("Test 6: Recall@K")
    print(recall)

    assert abs(
        recall - 2 / 3
    ) < 1e-12

    print("Passed.\n")


def main() -> None:
    test_dot_product()
    test_cosine_identical_direction()
    test_normalize()
    test_exact_search()
    test_cluster_index()
    test_recall_at_k()

    print("All Day 36 tests passed.")


if __name__ == "__main__":
    main()