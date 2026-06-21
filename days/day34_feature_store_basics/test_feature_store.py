from datetime import datetime, timedelta

from feature_store import (
    FeatureValue,
    ToyFeatureStore,
    TrainingSample,
)


FEATURE_NAME = "user_7d_click_count"


def build_store() -> ToyFeatureStore:
    store = ToyFeatureStore()

    store.write_offline(
        FeatureValue(
            entity_id="user-1",
            feature_name=FEATURE_NAME,
            event_time=datetime(
                2026,
                1,
                1,
            ),
            value=10.0,
        )
    )

    store.write_offline(
        FeatureValue(
            entity_id="user-1",
            feature_name=FEATURE_NAME,
            event_time=datetime(
                2026,
                1,
                10,
            ),
            value=20.0,
        )
    )

    store.write_offline(
        FeatureValue(
            entity_id="user-1",
            feature_name=FEATURE_NAME,
            event_time=datetime(
                2026,
                1,
                20,
            ),
            value=30.0,
        )
    )

    return store


def test_point_in_time_lookup() -> None:
    store = build_store()

    sample = TrainingSample(
        sample_id="sample-1",
        entity_id="user-1",
        sample_time=datetime(
            2026,
            1,
            15,
        ),
    )

    result = store.point_in_time_lookup(
        sample,
        FEATURE_NAME,
    )

    print("Test 1: point-in-time lookup")
    print(result)

    assert result.feature_value == 20.0
    assert result.feature_time == datetime(
        2026,
        1,
        10,
    )

    print("Passed.\n")


def test_future_value_not_used() -> None:
    store = build_store()

    sample = TrainingSample(
        sample_id="sample-2",
        entity_id="user-1",
        sample_time=datetime(
            2026,
            1,
            5,
        ),
    )

    result = store.point_in_time_lookup(
        sample,
        FEATURE_NAME,
    )

    print("Test 2: future value not used")
    print(result)

    assert result.feature_value == 10.0

    print("Passed.\n")


def test_no_historical_value() -> None:
    store = build_store()

    sample = TrainingSample(
        sample_id="sample-3",
        entity_id="user-1",
        sample_time=datetime(
            2025,
            12,
            1,
        ),
    )

    result = store.point_in_time_lookup(
        sample,
        FEATURE_NAME,
    )

    print("Test 3: no historical value")
    print(result)

    assert result.feature_value is None
    assert result.feature_time is None

    print("Passed.\n")


def test_materialize_latest() -> None:
    store = build_store()

    online = store.materialize_latest(
        entity_id="user-1",
        feature_name=FEATURE_NAME,
    )

    print("Test 4: materialize latest")
    print(online)

    assert online.value == 30.0
    assert online.event_time == datetime(
        2026,
        1,
        20,
    )

    fetched = store.get_online(
        "user-1",
        FEATURE_NAME,
    )

    assert fetched == online

    print("Passed.\n")


def test_stale_feature() -> None:
    store = build_store()

    store.materialize_latest(
        entity_id="user-1",
        feature_name=FEATURE_NAME,
    )

    stale = store.is_stale(
        entity_id="user-1",
        feature_name=FEATURE_NAME,
        current_time=datetime(
            2026,
            1,
            20,
            5,
            0,
        ),
        maximum_age=timedelta(
            hours=1
        ),
    )

    print("Test 5: stale feature")
    print(f"stale = {stale}")

    assert stale is True

    print("Passed.\n")


def test_fresh_feature() -> None:
    store = build_store()

    store.materialize_latest(
        entity_id="user-1",
        feature_name=FEATURE_NAME,
    )

    stale = store.is_stale(
        entity_id="user-1",
        feature_name=FEATURE_NAME,
        current_time=datetime(
            2026,
            1,
            20,
            0,
            30,
        ),
        maximum_age=timedelta(
            hours=1
        ),
    )

    print("Test 6: fresh feature")
    print(f"stale = {stale}")

    assert stale is False

    print("Passed.\n")


def main() -> None:
    test_point_in_time_lookup()
    test_future_value_not_used()
    test_no_historical_value()
    test_materialize_latest()
    test_stale_feature()
    test_fresh_feature()

    print("All Day 34 tests passed.")


if __name__ == "__main__":
    main()