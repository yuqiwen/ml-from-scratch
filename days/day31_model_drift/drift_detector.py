from dataclasses import dataclass
import math
import statistics


@dataclass(frozen=True)
class DistributionStats:
    count: int
    mean: float
    std: float
    minimum: float
    maximum: float
    missing_rate: float


@dataclass(frozen=True)
class DriftReport:
    reference: DistributionStats
    live: DistributionStats
    mean_drift_score: float
    std_ratio: float
    missing_rate_change: float
    drift_detected: bool


def calculate_stats(
    values: list[float | None],
) -> DistributionStats:
    """
    Calculate basic distribution statistics.

    None represents a missing value.
    """
    if not values:
        raise ValueError(
            "values must not be empty."
        )

    valid_values = [
        float(value)
        for value in values
        if value is not None
    ]

    missing_count = (
        len(values) - len(valid_values)
    )

    missing_rate = (
        missing_count / len(values)
    )

    if not valid_values:
        return DistributionStats(
            count=0,
            mean=math.nan,
            std=math.nan,
            minimum=math.nan,
            maximum=math.nan,
            missing_rate=missing_rate,
        )

    mean = statistics.mean(valid_values)

    std = (
        statistics.pstdev(valid_values)
        if len(valid_values) > 1
        else 0.0
    )

    return DistributionStats(
        count=len(valid_values),
        mean=mean,
        std=std,
        minimum=min(valid_values),
        maximum=max(valid_values),
        missing_rate=missing_rate,
    )


def standardized_mean_difference(
    reference_mean: float,
    live_mean: float,
    reference_std: float,
    epsilon: float = 1e-8,
) -> float:
    """
    Measure how far the live mean moved,
    expressed in reference standard deviations.
    """
    denominator = max(
        abs(reference_std),
        epsilon,
    )

    return abs(
        live_mean - reference_mean
    ) / denominator


def safe_std_ratio(
    reference_std: float,
    live_std: float,
    epsilon: float = 1e-8,
) -> float:
    """
    Compare live spread with reference spread.
    """
    denominator = max(
        abs(reference_std),
        epsilon,
    )

    return live_std / denominator


def detect_drift(
    reference_values: list[float | None],
    live_values: list[float | None],
    mean_drift_threshold: float = 1.0,
    std_ratio_lower: float = 0.5,
    std_ratio_upper: float = 2.0,
    missing_rate_change_threshold: float = 0.1,
) -> DriftReport:
    """
    Detect simple drift using:

    - standardized mean shift
    - standard deviation ratio
    - missing-rate increase
    """
    reference = calculate_stats(
        reference_values
    )
    live = calculate_stats(
        live_values
    )

    if reference.count == 0:
        raise ValueError(
            "reference data must contain "
            "at least one non-missing value."
        )

    if live.count == 0:
        raise ValueError(
            "live data must contain "
            "at least one non-missing value."
        )

    mean_drift_score = (
        standardized_mean_difference(
            reference_mean=reference.mean,
            live_mean=live.mean,
            reference_std=reference.std,
        )
    )

    std_ratio = safe_std_ratio(
        reference_std=reference.std,
        live_std=live.std,
    )

    missing_rate_change = (
        live.missing_rate
        - reference.missing_rate
    )

    mean_drift = (
        mean_drift_score
        > mean_drift_threshold
    )

    spread_drift = (
        std_ratio < std_ratio_lower
        or std_ratio > std_ratio_upper
    )

    missing_drift = (
        missing_rate_change
        > missing_rate_change_threshold
    )

    drift_detected = (
        mean_drift
        or spread_drift
        or missing_drift
    )

    return DriftReport(
        reference=reference,
        live=live,
        mean_drift_score=mean_drift_score,
        std_ratio=std_ratio,
        missing_rate_change=missing_rate_change,
        drift_detected=drift_detected,
    )


def print_report(
    name: str,
    report: DriftReport,
) -> None:
    print(f"Feature: {name}")
    print(
        f"reference mean="
        f"{report.reference.mean:.4f}"
    )
    print(
        f"live mean="
        f"{report.live.mean:.4f}"
    )
    print(
        f"mean drift score="
        f"{report.mean_drift_score:.4f}"
    )
    print(
        f"std ratio="
        f"{report.std_ratio:.4f}"
    )
    print(
        f"missing rate change="
        f"{report.missing_rate_change:.4f}"
    )
    print(
        f"drift detected="
        f"{report.drift_detected}"
    )
    print()


def run_demo() -> None:
    reference_age = [
        24,
        27,
        29,
        31,
        33,
        35,
        37,
        39,
    ]

    stable_live_age = [
        25,
        28,
        30,
        31,
        34,
        35,
        37,
        38,
    ]

    drifted_live_age = [
        44,
        46,
        48,
        50,
        52,
        54,
        56,
        58,
    ]

    missing_live_age = [
        25,
        None,
        None,
        None,
        35,
        None,
        None,
        40,
    ]

    stable_report = detect_drift(
        reference_age,
        stable_live_age,
    )

    drifted_report = detect_drift(
        reference_age,
        drifted_live_age,
    )

    missing_report = detect_drift(
        reference_age,
        missing_live_age,
    )

    print_report(
        "stable_age",
        stable_report,
    )

    print_report(
        "drifted_age",
        drifted_report,
    )

    print_report(
        "missing_age",
        missing_report,
    )


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()