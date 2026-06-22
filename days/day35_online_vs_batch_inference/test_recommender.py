from recommender import (
    BatchCandidateStore,
    Item,
    RequestContext,
    UserProfile,
    offline_candidate_generation,
    online_rank,
    run_offline_job,
    serve_recommendation,
)


def build_items() -> list[Item]:
    return [
        Item(
            item_id=1,
            category="camera",
            popularity=0.8,
            quality=0.9,
        ),
        Item(
            item_id=2,
            category="travel",
            popularity=0.9,
            quality=0.7,
        ),
        Item(
            item_id=3,
            category="camera",
            popularity=0.6,
            quality=0.95,
        ),
        Item(
            item_id=4,
            category="food",
            popularity=1.0,
            quality=0.8,
        ),
        Item(
            item_id=5,
            category="camera",
            popularity=0.95,
            quality=0.6,
            available=False,
        ),
    ]


def build_user() -> UserProfile:
    return UserProfile(
        user_id="user-1",
        preferred_categories=(
            "camera",
            "travel",
        ),
    )


def test_offline_candidate_generation() -> None:
    items = build_items()
    user = build_user()

    candidates = (
        offline_candidate_generation(
            user=user,
            items=items,
            max_candidates=3,
        )
    )

    print(
        "Test 1: offline candidate generation"
    )
    print(candidates)

    assert len(candidates) == 3
    assert 1 in candidates
    assert 2 in candidates
    assert 5 in candidates

    print("Passed.\n")


def test_batch_candidate_store() -> None:
    store = BatchCandidateStore()

    store.write_candidates(
        "user-1",
        [1, 2, 3],
    )

    result = store.get_candidates(
        "user-1"
    )

    print("Test 2: candidate store")
    print(result)

    assert result == [1, 2, 3]

    # Returned list is a copy.
    result.append(99)

    assert store.get_candidates(
        "user-1"
    ) == [1, 2, 3]

    print("Passed.\n")


def test_online_rank_filters_unavailable() -> None:
    items = build_items()
    items_by_id = {
        item.item_id: item
        for item in items
    }

    user = build_user()

    context = RequestContext(
        current_category="camera",
        freshness_weight=0.2,
    )

    results = online_rank(
        candidate_ids=[1, 2, 5],
        items_by_id=items_by_id,
        user=user,
        context=context,
        top_k=3,
    )

    result_ids = [
        result.item_id
        for result in results
    ]

    print(
        "Test 3: unavailable item filtered"
    )
    print(result_ids)

    assert 5 not in result_ids

    print("Passed.\n")


def test_online_context_changes_order() -> None:
    items = build_items()
    items_by_id = {
        item.item_id: item
        for item in items
    }

    user = build_user()

    camera_results = online_rank(
        candidate_ids=[1, 2, 3],
        items_by_id=items_by_id,
        user=user,
        context=RequestContext(
            current_category="camera",
            freshness_weight=0.0,
        ),
        top_k=3,
    )

    travel_results = online_rank(
        candidate_ids=[1, 2, 3],
        items_by_id=items_by_id,
        user=user,
        context=RequestContext(
            current_category="travel",
            freshness_weight=0.0,
        ),
        top_k=3,
    )

    print(
        "Test 4: request context changes ranking"
    )
    print(camera_results)
    print(travel_results)

    assert (
        camera_results[0].item_id
        in {1, 3}
    )

    assert travel_results[0].item_id == 2

    print("Passed.\n")


def test_full_offline_online_flow() -> None:
    items = build_items()
    user = build_user()

    store = BatchCandidateStore()

    run_offline_job(
        users=[user],
        items=items,
        store=store,
    )

    results = serve_recommendation(
        user=user,
        context=RequestContext(
            current_category="camera",
            freshness_weight=0.2,
        ),
        items_by_id={
            item.item_id: item
            for item in items
        },
        candidate_store=store,
        top_k=2,
    )

    print(
        "Test 5: full offline-online flow"
    )
    print(results)

    assert len(results) == 2

    assert all(
        result.item_id != 5
        for result in results
    )

    print("Passed.\n")


def main() -> None:
    test_offline_candidate_generation()
    test_batch_candidate_store()
    test_online_rank_filters_unavailable()
    test_online_context_changes_order()
    test_full_offline_online_flow()

    print("All Day 35 tests passed.")


if __name__ == "__main__":
    main()