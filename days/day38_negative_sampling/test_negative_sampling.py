from negative_sampling import (
    Item,
    ScoredItem,
    filter_false_negative_candidates,
    mine_hard_negatives,
    popularity_weighted_sample,
    same_category_negative_sample,
    uniform_negative_sample,
)


def build_items() -> list[Item]:
    return [
        Item(
            item_id=1,
            category="camera",
            popularity=1.0,
        ),
        Item(
            item_id=2,
            category="camera",
            popularity=0.9,
        ),
        Item(
            item_id=3,
            category="camera",
            popularity=0.8,
        ),
        Item(
            item_id=4,
            category="food",
            popularity=0.7,
        ),
        Item(
            item_id=5,
            category="travel",
            popularity=0.6,
        ),
    ]


def test_uniform_sampling_excludes_positive() -> None:
    negatives = uniform_negative_sample(
        all_item_ids=[
            1,
            2,
            3,
            4,
            5,
        ],
        positive_item_ids={
            1,
            3,
        },
        num_negatives=2,
        seed=42,
    )

    print(
        "Test 1: uniform sampling excludes positives"
    )
    print(negatives)

    assert 1 not in negatives
    assert 3 not in negatives
    assert len(negatives) == 2

    print("Passed.\n")


def test_same_category_sampling() -> None:
    negatives = (
        same_category_negative_sample(
            items=build_items(),
            positive_item_id=1,
            positive_item_ids={1},
            num_negatives=2,
            seed=42,
        )
    )

    print(
        "Test 2: same-category sampling"
    )
    print(negatives)

    assert set(negatives) == {
        2,
        3,
    }

    print("Passed.\n")


def test_hard_negative_mining() -> None:
    retrieved = [
        ScoredItem(
            item_id=1,
            score=0.99,
        ),
        ScoredItem(
            item_id=2,
            score=0.95,
        ),
        ScoredItem(
            item_id=3,
            score=0.90,
        ),
        ScoredItem(
            item_id=4,
            score=0.40,
        ),
    ]

    negatives = mine_hard_negatives(
        retrieved_items=retrieved,
        known_positive_item_ids={1},
        num_negatives=2,
    )

    print(
        "Test 3: hard negative mining"
    )
    print(negatives)

    assert [
        item.item_id
        for item in negatives
    ] == [
        2,
        3,
    ]

    print("Passed.\n")


def test_semi_hard_score_range() -> None:
    retrieved = [
        ScoredItem(
            item_id=1,
            score=0.99,
        ),
        ScoredItem(
            item_id=2,
            score=0.94,
        ),
        ScoredItem(
            item_id=3,
            score=0.80,
        ),
        ScoredItem(
            item_id=4,
            score=0.30,
        ),
    ]

    negatives = mine_hard_negatives(
        retrieved_items=retrieved,
        known_positive_item_ids={1},
        num_negatives=5,
        minimum_score=0.50,
        maximum_score=0.90,
    )

    print(
        "Test 4: semi-hard score range"
    )
    print(negatives)

    assert [
        item.item_id
        for item in negatives
    ] == [
        3,
    ]

    print("Passed.\n")


def test_filter_false_negatives() -> None:
    candidates = [
        2,
        3,
        4,
        5,
        6,
    ]

    known_positives = {
        1,
        5,
    }

    equivalent_groups = [
        {
            1,
            2,
            3,
        },
        {
            7,
            8,
        },
    ]

    filtered = (
        filter_false_negative_candidates(
            candidate_item_ids=candidates,
            user_history_positive_ids=(
                known_positives
            ),
            semantically_equivalent_groups=(
                equivalent_groups
            ),
        )
    )

    print(
        "Test 5: false negatives filtered"
    )
    print(filtered)

    assert filtered == [
        4,
        6,
    ]

    print("Passed.\n")


def test_popularity_sampling_unique() -> None:
    negatives = popularity_weighted_sample(
        items=build_items(),
        positive_item_ids={1},
        num_negatives=3,
        seed=42,
    )

    print(
        "Test 6: popularity sampling"
    )
    print(negatives)

    assert len(negatives) == 3
    assert len(set(negatives)) == 3
    assert 1 not in negatives

    print("Passed.\n")


def main() -> None:
    test_uniform_sampling_excludes_positive()
    test_same_category_sampling()
    test_hard_negative_mining()
    test_semi_hard_score_range()
    test_filter_false_negatives()
    test_popularity_sampling_unique()

    print(
        "All Day 38 tests passed."
    )


if __name__ == "__main__":
    main()