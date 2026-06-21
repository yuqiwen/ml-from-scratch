from feature_pipeline import (
    DEFAULT_CONFIG,
    RawUserFeatures,
    preprocess_shared,
)
from skew_detector import (
    compare_processed_features,
    detect_pipeline_skew,
)


def test_shared_pipeline_matches() -> None:
    raw = RawUserFeatures(
        age=40.0,
        income_usd=80_000.0,
        country="US",
    )

    offline = preprocess_shared(
        raw,
        DEFAULT_CONFIG,
    )

    online = preprocess_shared(
        raw,
        DEFAULT_CONFIG,
    )

    report = compare_processed_features(
        offline,
        online,
    )

    print("Test 1: shared pipeline matches")
    print(report)

    assert report.match_rate == 1.0
    assert report.skew_detected is False

    print("Passed.\n")


def test_buggy_pipeline_detected() -> None:
    raw = RawUserFeatures(
        age=45.0,
        income_usd=100_000.0,
        country="US",
    )

    report = detect_pipeline_skew(
        raw=raw,
        use_buggy_online=True,
    )

    print("Test 2: buggy pipeline detected")

    for comparison in report.comparisons:
        print(comparison)

    assert report.skew_detected is True
    assert report.match_rate < 1.0

    print("Passed.\n")


def test_feature_names_preserved() -> None:
    raw = RawUserFeatures(
        age=45.0,
        income_usd=100_000.0,
        country="US",
    )

    report = detect_pipeline_skew(
        raw=raw,
        use_buggy_online=True,
    )

    names = {
        comparison.feature_name
        for comparison in report.comparisons
    }

    print("Test 3: feature names preserved")
    print(names)

    assert names == {
        "age_normalized",
        "income_normalized",
        "country_id",
    }

    print("Passed.\n")


def test_missing_values_consistent() -> None:
    raw = RawUserFeatures(
        age=None,
        income_usd=None,
        country=None,
    )

    report = detect_pipeline_skew(
        raw=raw,
        use_buggy_online=False,
    )

    print(
        "Test 4: missing values are consistent"
    )
    print(report)

    assert report.skew_detected is False

    print("Passed.\n")


def test_tolerance() -> None:
    raw = RawUserFeatures(
        age=40.0,
        income_usd=80_000.0,
        country="CN",
    )

    offline = preprocess_shared(
        raw,
        DEFAULT_CONFIG,
    )

    online = type(offline)(
        age_normalized=(
            offline.age_normalized
            + 1e-7
        ),
        income_normalized=(
            offline.income_normalized
        ),
        country_id=offline.country_id,
    )

    report = compare_processed_features(
        offline,
        online,
        tolerance=1e-6,
    )

    print("Test 5: numeric tolerance")
    print(report)

    assert report.skew_detected is False

    print("Passed.\n")


def main() -> None:
    test_shared_pipeline_matches()
    test_buggy_pipeline_detected()
    test_feature_names_preserved()
    test_missing_values_consistent()
    test_tolerance()

    print("All Day 33 tests passed.")


if __name__ == "__main__":
    main()