from dataclasses import dataclass
import random


@dataclass(frozen=True)
class Item:
    item_id: int
    category: str
    popularity: float


@dataclass(frozen=True)
class ScoredItem:
    item_id: int
    score: float


def uniform_negative_sample(
    all_item_ids: list[int],
    positive_item_ids: set[int],
    num_negatives: int,
    seed: int | None = None,
) -> list[int]:
    """
    Uniformly sample negatives while excluding
    known positive items.
    """
    if num_negatives <= 0:
        raise ValueError(
            "num_negatives must be positive."
        )

    candidates = [
        item_id
        for item_id in all_item_ids
        if item_id not in positive_item_ids
    ]

    if num_negatives > len(candidates):
        raise ValueError(
            "Not enough negative candidates."
        )

    rng = random.Random(seed)

    return rng.sample(
        candidates,
        k=num_negatives,
    )


def same_category_negative_sample(
    items: list[Item],
    positive_item_id: int,
    positive_item_ids: set[int],
    num_negatives: int,
    seed: int | None = None,
) -> list[int]:
    """
    Sample negatives from the same category
    as the positive item.
    """
    if num_negatives <= 0:
        raise ValueError(
            "num_negatives must be positive."
        )

    items_by_id = {
        item.item_id: item
        for item in items
    }

    if positive_item_id not in items_by_id:
        raise KeyError(
            "positive_item_id not found."
        )

    positive_category = (
        items_by_id[
            positive_item_id
        ].category
    )

    candidates = [
        item.item_id
        for item in items
        if (
            item.category
            == positive_category
            and item.item_id
            not in positive_item_ids
        )
    ]

    if num_negatives > len(candidates):
        raise ValueError(
            "Not enough same-category negatives."
        )

    rng = random.Random(seed)

    return rng.sample(
        candidates,
        k=num_negatives,
    )


def popularity_weighted_sample(
    items: list[Item],
    positive_item_ids: set[int],
    num_negatives: int,
    seed: int | None = None,
) -> list[int]:
    """
    Sample without replacement using item
    popularity as the sampling weight.
    """
    if num_negatives <= 0:
        raise ValueError(
            "num_negatives must be positive."
        )

    candidates = [
        item
        for item in items
        if item.item_id
        not in positive_item_ids
    ]

    if num_negatives > len(candidates):
        raise ValueError(
            "Not enough negative candidates."
        )

    if any(
        item.popularity < 0
        for item in candidates
    ):
        raise ValueError(
            "Popularity must be non-negative."
        )

    rng = random.Random(seed)
    selected: list[int] = []
    remaining = list(candidates)

    for _ in range(num_negatives):
        total_weight = sum(
            item.popularity
            for item in remaining
        )

        if total_weight <= 0:
            chosen = rng.choice(
                remaining
            )
        else:
            threshold = (
                rng.random()
                * total_weight
            )

            cumulative = 0.0
            chosen = remaining[-1]

            for item in remaining:
                cumulative += (
                    item.popularity
                )

                if cumulative >= threshold:
                    chosen = item
                    break

        selected.append(
            chosen.item_id
        )

        remaining.remove(chosen)

    return selected


def mine_hard_negatives(
    retrieved_items: list[ScoredItem],
    known_positive_item_ids: set[int],
    num_negatives: int,
    minimum_score: float | None = None,
    maximum_score: float | None = None,
) -> list[ScoredItem]:
    """
    Select high-scoring retrieved items that are
    not known positives.

    Optional score boundaries can be used for
    semi-hard negative mining.
    """
    if num_negatives <= 0:
        raise ValueError(
            "num_negatives must be positive."
        )

    candidates = []

    for item in retrieved_items:
        if (
            item.item_id
            in known_positive_item_ids
        ):
            continue

        if (
            minimum_score is not None
            and item.score < minimum_score
        ):
            continue

        if (
            maximum_score is not None
            and item.score > maximum_score
        ):
            continue

        candidates.append(item)

    candidates.sort(
        key=lambda item: item.score,
        reverse=True,
    )

    return candidates[:num_negatives]


def filter_false_negative_candidates(
    candidate_item_ids: list[int],
    user_history_positive_ids: set[int],
    semantically_equivalent_groups: list[
        set[int]
    ],
) -> list[int]:
    """
    Remove likely false negatives.

    An item is removed if:
    - the user has already interacted positively
    - it belongs to an equivalence group containing
      one of the user's known positives
    """
    equivalent_to_positive: set[int] = set()

    for group in (
        semantically_equivalent_groups
    ):
        if (
            group
            & user_history_positive_ids
        ):
            equivalent_to_positive.update(
                group
            )

    return [
        item_id
        for item_id in candidate_item_ids
        if (
            item_id
            not in user_history_positive_ids
            and item_id
            not in equivalent_to_positive
        )
    ]


def mixed_negative_sample(
    items: list[Item],
    positive_item_id: int,
    known_positive_item_ids: set[int],
    random_count: int,
    same_category_count: int,
    hard_negatives: list[ScoredItem],
    hard_count: int,
    seed: int = 42,
) -> list[int]:
    """
    Combine several negative sampling strategies.
    """
    sampled: list[int] = []

    if random_count > 0:
        sampled.extend(
            uniform_negative_sample(
                all_item_ids=[
                    item.item_id
                    for item in items
                ],
                positive_item_ids=(
                    known_positive_item_ids
                ),
                num_negatives=random_count,
                seed=seed,
            )
        )

    excluded = (
        known_positive_item_ids
        | set(sampled)
    )

    if same_category_count > 0:
        sampled.extend(
            same_category_negative_sample(
                items=items,
                positive_item_id=(
                    positive_item_id
                ),
                positive_item_ids=excluded,
                num_negatives=(
                    same_category_count
                ),
                seed=seed + 1,
            )
        )

    excluded = (
        known_positive_item_ids
        | set(sampled)
    )

    if hard_count > 0:
        mined = mine_hard_negatives(
            retrieved_items=hard_negatives,
            known_positive_item_ids=excluded,
            num_negatives=hard_count,
        )

        sampled.extend(
            item.item_id
            for item in mined
        )

    return sampled


def build_demo_items() -> list[Item]:
    return [
        Item(
            item_id=1,
            category="camera",
            popularity=0.95,
        ),
        Item(
            item_id=2,
            category="camera",
            popularity=0.85,
        ),
        Item(
            item_id=3,
            category="camera",
            popularity=0.75,
        ),
        Item(
            item_id=4,
            category="food",
            popularity=0.90,
        ),
        Item(
            item_id=5,
            category="travel",
            popularity=0.70,
        ),
        Item(
            item_id=6,
            category="camera",
            popularity=0.65,
        ),
        Item(
            item_id=7,
            category="home",
            popularity=0.50,
        ),
    ]


def run_demo() -> None:
    items = build_demo_items()

    known_positives = {
        1,
    }

    print("Uniform negatives")
    print(
        uniform_negative_sample(
            all_item_ids=[
                item.item_id
                for item in items
            ],
            positive_item_ids=(
                known_positives
            ),
            num_negatives=3,
            seed=1,
        )
    )
    print()

    print("Same-category negatives")
    print(
        same_category_negative_sample(
            items=items,
            positive_item_id=1,
            positive_item_ids=(
                known_positives
            ),
            num_negatives=2,
            seed=1,
        )
    )
    print()

    retrieved = [
        ScoredItem(
            item_id=1,
            score=0.97,
        ),
        ScoredItem(
            item_id=2,
            score=0.92,
        ),
        ScoredItem(
            item_id=3,
            score=0.89,
        ),
        ScoredItem(
            item_id=4,
            score=0.50,
        ),
        ScoredItem(
            item_id=5,
            score=0.35,
        ),
    ]

    print("Hard negatives")
    print(
        mine_hard_negatives(
            retrieved_items=retrieved,
            known_positive_item_ids=(
                known_positives
            ),
            num_negatives=2,
        )
    )
    print()

    print("Semi-hard negatives")
    print(
        mine_hard_negatives(
            retrieved_items=retrieved,
            known_positive_item_ids=(
                known_positives
            ),
            num_negatives=2,
            minimum_score=0.5,
            maximum_score=0.90,
        )
    )


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()