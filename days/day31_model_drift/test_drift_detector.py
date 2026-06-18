import math

from drift_detector import (
    calculate_stats,
    detect_drift,
    standardized_mean_difference,
)


def test_calculate_stats() -> None:
    values = [
        1.0,
        2.0,
        None,
        3.0,
    ]

    stats = calculate_stats(values)

    print("Test 1: calculate stats")
    print(stats)

    assert stats.count == 3
    assert stats.mean == 2.0
    assert stats.minimum == 1.0
    assert stats.maximum == 3.0
    assert stats.missing_rate == 0.25

    print("Passed.\n")


def test_standardized_mean_difference() -> None:
    score = standardized_mean_difference(
        reference_mean=10.0,
        live_mean=14.0,
        reference_std=2.0,
    )

    print(
        "Test 2: standardized mean difference"
    )
    print(f"score = {score}")

    assert score == 2.0

    print("Passed.\n")


def test_stable_distribution() -> None:
    reference = [
        1,
        2,
        3,
        4,
        5,
    ]

    live = [
        1.1,
        2.1,
        3.0,
        3.9,
        4.9,
    ]

    report = detect_drift(
        reference,
        live,
    )

    print("Test 3: stable distribution")
    print(report)

    assert report.drift_detected is False

    print("Passed.\n")


def test_mean_drift_detected() -> None:
    reference = [
        8,
        9,
        10,
        11,
        12,
    ]

    live = [
        18,
        19,
        20,
        21,
        22,
    ]

    report = detect_drift(
        reference,
        live,
    )

    print("Test 4: mean drift")
    print(report)

    assert report.mean_drift_score > 1.0
    assert report.drift_detected is True

    print("Passed.\n")


def test_missing_rate_drift_detected() -> None:
    reference = [
        1,
        2,
        3,
        4,
        5,
    ]

    live = [
        1,
        None,
        None,
        None,
        5,
    ]

    report = detect_drift(
        reference,
        live,
        mean_drift_threshold=100.0,
        std_ratio_lower=0.0,
        std_ratio_upper=100.0,
        missing_rate_change_threshold=0.2,
    )

    print("Test 5: missing-rate drift")
    print(report)

    assert (
        report.missing_rate_change
        > 0.2
    )

    assert report.drift_detected is True

    print("Passed.\n")


def test_all_missing_rejected() -> None:
    print("Test 6: all-missing live data")

    try:
        detect_drift(
            reference_values=[
                1,
                2,
                3,
            ],
            live_values=[
                None,
                None,
            ],
        )
        raised = False
    except ValueError:
        raised = True

    assert raised

    print("Passed.\n")


def main() -> None:
    test_calculate_stats()
    test_standardized_mean_difference()
    test_stable_distribution()
    test_mean_drift_detected()
    test_missing_rate_drift_detected()
    test_all_missing_rejected()

    print("All Day 31 tests passed.")


if __name__ == "__main__":
    main()