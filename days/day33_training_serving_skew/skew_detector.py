from dataclasses import dataclass, fields
from typing import Any

from feature_pipeline import (
    DEFAULT_CONFIG,
    ProcessedFeatures,
    RawUserFeatures,
    preprocess_buggy_online,
    preprocess_shared,
)


@dataclass(frozen=True)
class FeatureComparison:
    feature_name: str
    offline_value: float | int
    online_value: float | int
    absolute_difference: float
    matched: bool


@dataclass(frozen=True)
class SkewReport:
    comparisons: list[FeatureComparison]
    total_features: int
    matched_features: int
    match_rate: float
    skew_detected: bool


def compare_processed_features(
    offline: ProcessedFeatures,
    online: ProcessedFeatures,
    tolerance: float = 1e-6,
) -> SkewReport:
    """
    Compare two processed feature objects field by field.
    """
    if tolerance < 0:
        raise ValueError(
            "tolerance must be non-negative."
        )

    comparisons: list[
        FeatureComparison
    ] = []

    for feature_field in fields(
        ProcessedFeatures
    ):
        name = feature_field.name

        offline_value = getattr(
            offline,
            name,
        )

        online_value = getattr(
            online,
            name,
        )

        absolute_difference = abs(
            float(offline_value)
            - float(online_value)
        )

        matched = (
            absolute_difference
            <= tolerance
        )

        comparisons.append(
            FeatureComparison(
                feature_name=name,
                offline_value=offline_value,
                online_value=online_value,
                absolute_difference=(
                    absolute_difference
                ),
                matched=matched,
            )
        )

    matched_features = sum(
        comparison.matched
        for comparison in comparisons
    )

    total_features = len(comparisons)

    match_rate = (
        matched_features / total_features
        if total_features > 0
        else 1.0
    )

    return SkewReport(
        comparisons=comparisons,
        total_features=total_features,
        matched_features=matched_features,
        match_rate=match_rate,
        skew_detected=(
            matched_features
            != total_features
        ),
    )


def detect_pipeline_skew(
    raw: RawUserFeatures,
    use_buggy_online: bool,
) -> SkewReport:
    """
    Compute the same raw sample through offline and online pipelines.
    """
    offline_features = preprocess_shared(
        raw,
        DEFAULT_CONFIG,
    )

    if use_buggy_online:
        online_features = (
            preprocess_buggy_online(
                raw,
                DEFAULT_CONFIG,
            )
        )
    else:
        online_features = preprocess_shared(
            raw,
            DEFAULT_CONFIG,
        )

    return compare_processed_features(
        offline=offline_features,
        online=online_features,
    )


def print_report(
    name: str,
    report: SkewReport,
) -> None:
    print(f"Sample: {name}")
    print(
        f"match_rate="
        f"{report.match_rate:.2%}"
    )
    print(
        f"skew_detected="
        f"{report.skew_detected}"
    )

    for comparison in report.comparisons:
        print(
            f"{comparison.feature_name}: "
            f"offline={comparison.offline_value}, "
            f"online={comparison.online_value}, "
            f"diff="
            f"{comparison.absolute_difference:.6f}, "
            f"matched={comparison.matched}"
        )

    print()


def run_demo() -> None:
    raw = RawUserFeatures(
        age=45.0,
        income_usd=100_000.0,
        country="US",
    )

    consistent_report = detect_pipeline_skew(
        raw=raw,
        use_buggy_online=False,
    )

    buggy_report = detect_pipeline_skew(
        raw=raw,
        use_buggy_online=True,
    )

    print_report(
        "consistent_pipeline",
        consistent_report,
    )

    print_report(
        "buggy_online_pipeline",
        buggy_report,
    )


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()