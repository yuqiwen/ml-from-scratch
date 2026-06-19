from dataclasses import dataclass
import bisect
import math
import statistics


@dataclass(frozen=True)
class PSIResult:
    psi: float
    bin_edges: list[float]
    reference_ratios: list[float]
    live_ratios: list[float]
    bin_contributions: list[float]


@dataclass(frozen=True)
class DistributionDriftReport:
    psi_result: PSIResult
    ks_statistic: float


def validate_numeric_values(
    values: list[float],
    name: str,
) -> None:
    if not values:
        raise ValueError(
            f"{name} must not be empty."
        )

    for value in values:
        if not math.isfinite(value):
            raise ValueError(
                f"{name} must contain only "
                "finite numeric values."
            )


def percentile(
    sorted_values: list[float],
    percent: float,
) -> float:
    """
    Compute a percentile with linear interpolation.

    sorted_values must already be sorted.
    """
    if not sorted_values:
        raise ValueError(
            "sorted_values must not be empty."
        )

    if percent < 0 or percent > 100:
        raise ValueError(
            "percent must be between 0 and 100."
        )

    if len(sorted_values) == 1:
        return sorted_values[0]

    index = (
        len(sorted_values) - 1
    ) * percent / 100.0

    lower_index = int(index)
    upper_index = min(
        lower_index + 1,
        len(sorted_values) - 1,
    )

    weight = index - lower_index

    return (
        sorted_values[lower_index]
        * (1.0 - weight)
        + sorted_values[upper_index]
        * weight
    )


def build_quantile_bin_edges(
    reference_values: list[float],
    num_bins: int,
) -> list[float]:
    """
    Build bin boundaries from reference quantiles.

    Returned edges include:
        -inf
        internal quantile boundaries
        +inf
    """
    validate_numeric_values(
        reference_values,
        "reference_values",
    )

    if num_bins < 2:
        raise ValueError(
            "num_bins must be at least 2."
        )

    sorted_reference = sorted(
        reference_values
    )

    internal_edges = []

    for bin_index in range(
        1,
        num_bins,
    ):
        percent = (
            bin_index
            / num_bins
            * 100.0
        )

        edge = percentile(
            sorted_reference,
            percent,
        )

        internal_edges.append(edge)

    # Repeated values can create duplicate edges.
    # Remove duplicates while preserving order.
    unique_internal_edges = []

    for edge in internal_edges:
        if (
            not unique_internal_edges
            or edge
            > unique_internal_edges[-1]
        ):
            unique_internal_edges.append(
                edge
            )

    return [
        -math.inf,
        *unique_internal_edges,
        math.inf,
    ]


def count_bins(
    values: list[float],
    bin_edges: list[float],
) -> list[int]:
    """
    Count values in intervals:

        [edge_0, edge_1)
        [edge_1, edge_2)
        ...
        [edge_n-1, edge_n]

    The final edge is typically +inf.
    """
    validate_numeric_values(
        values,
        "values",
    )

    if len(bin_edges) < 3:
        raise ValueError(
            "bin_edges must define "
            "at least two bins."
        )

    counts = [
        0
        for _ in range(
            len(bin_edges) - 1
        )
    ]

    internal_edges = bin_edges[1:-1]

    for value in values:
        bin_index = bisect.bisect_right(
            internal_edges,
            value,
        )

        counts[bin_index] += 1

    return counts


def counts_to_ratios(
    counts: list[int],
) -> list[float]:
    total = sum(counts)

    if total <= 0:
        raise ValueError(
            "counts must contain "
            "at least one sample."
        )

    return [
        count / total
        for count in counts
    ]


def calculate_psi(
    reference_values: list[float],
    live_values: list[float],
    num_bins: int = 10,
    epsilon: float = 1e-6,
) -> PSIResult:
    """
    Calculate PSI using reference quantile bins.
    """
    validate_numeric_values(
        reference_values,
        "reference_values",
    )
    validate_numeric_values(
        live_values,
        "live_values",
    )

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be positive."
        )

    bin_edges = build_quantile_bin_edges(
        reference_values,
        num_bins,
    )

    reference_counts = count_bins(
        reference_values,
        bin_edges,
    )

    live_counts = count_bins(
        live_values,
        bin_edges,
    )

    reference_ratios = counts_to_ratios(
        reference_counts
    )

    live_ratios = counts_to_ratios(
        live_counts
    )

    contributions = []

    for reference_ratio, live_ratio in zip(
        reference_ratios,
        live_ratios,
    ):
        safe_reference = max(
            reference_ratio,
            epsilon,
        )

        safe_live = max(
            live_ratio,
            epsilon,
        )

        contribution = (
            safe_live - safe_reference
        ) * math.log(
            safe_live / safe_reference
        )

        contributions.append(
            contribution
        )

    return PSIResult(
        psi=sum(contributions),
        bin_edges=bin_edges,
        reference_ratios=reference_ratios,
        live_ratios=live_ratios,
        bin_contributions=contributions,
    )


def empirical_cdf(
    sorted_values: list[float],
    value: float,
) -> float:
    """
    Fraction of samples <= value.
    """
    count = bisect.bisect_right(
        sorted_values,
        value,
    )

    return count / len(sorted_values)


def calculate_ks_statistic(
    reference_values: list[float],
    live_values: list[float],
) -> float:
    """
    Calculate the two-sample KS statistic.

    This returns only the maximum CDF difference,
    not a p-value.
    """
    validate_numeric_values(
        reference_values,
        "reference_values",
    )
    validate_numeric_values(
        live_values,
        "live_values",
    )

    sorted_reference = sorted(
        reference_values
    )
    sorted_live = sorted(
        live_values
    )

    combined_values = sorted(
        set(
            sorted_reference
            + sorted_live
        )
    )

    max_difference = 0.0

    for value in combined_values:
        reference_cdf = empirical_cdf(
            sorted_reference,
            value,
        )

        live_cdf = empirical_cdf(
            sorted_live,
            value,
        )

        difference = abs(
            reference_cdf
            - live_cdf
        )

        max_difference = max(
            max_difference,
            difference,
        )

    return max_difference


def build_drift_report(
    reference_values: list[float],
    live_values: list[float],
    num_bins: int = 10,
) -> DistributionDriftReport:
    psi_result = calculate_psi(
        reference_values,
        live_values,
        num_bins=num_bins,
    )

    ks_statistic = calculate_ks_statistic(
        reference_values,
        live_values,
    )

    return DistributionDriftReport(
        psi_result=psi_result,
        ks_statistic=ks_statistic,
    )


def classify_psi(
    psi: float,
) -> str:
    """
    Heuristic classification only.
    """
    if psi < 0.1:
        return "little_or_no_drift"

    if psi < 0.25:
        return "moderate_drift"

    return "significant_drift"


def print_report(
    name: str,
    report: DistributionDriftReport,
) -> None:
    print(f"Distribution: {name}")
    print(
        f"PSI = "
        f"{report.psi_result.psi:.6f}"
    )
    print(
        f"PSI classification = "
        f"{classify_psi(report.psi_result.psi)}"
    )
    print(
        f"KS statistic = "
        f"{report.ks_statistic:.6f}"
    )
    print()

    print("Bin details")

    for index, (
        reference_ratio,
        live_ratio,
        contribution,
    ) in enumerate(
        zip(
            report.psi_result.reference_ratios,
            report.psi_result.live_ratios,
            report.psi_result.bin_contributions,
        )
    ):
        print(
            f"bin={index:02d}, "
            f"reference={reference_ratio:.4f}, "
            f"live={live_ratio:.4f}, "
            f"contribution={contribution:.6f}"
        )

    print()


def run_demo() -> None:
    reference = [
        float(value)
        for value in range(100)
    ]

    stable_live = [
        float(value) + 1.0
        for value in range(100)
    ]

    shifted_live = [
        float(value) + 50.0
        for value in range(100)
    ]

    concentrated_live = [
        50.0
        for _ in range(100)
    ]

    stable_report = build_drift_report(
        reference,
        stable_live,
        num_bins=10,
    )

    shifted_report = build_drift_report(
        reference,
        shifted_live,
        num_bins=10,
    )

    concentrated_report = build_drift_report(
        reference,
        concentrated_live,
        num_bins=10,
    )

    print_report(
        "stable_live",
        stable_report,
    )

    print_report(
        "shifted_live",
        shifted_report,
    )

    print_report(
        "concentrated_live",
        concentrated_report,
    )


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()