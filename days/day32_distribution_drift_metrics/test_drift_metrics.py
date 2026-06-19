import math

from drift_metrics import (
    build_quantile_bin_edges,
    calculate_ks_statistic,
    calculate_psi,
    classify_psi,
    count_bins,
)


def test_quantile_bin_edges() -> None:
    reference = [
        float(value)
        for value in range(100)
    ]

    edges = build_quantile_bin_edges(
        reference,
        num_bins=5,
    )

    print("Test 1: quantile bin edges")
    print(edges)

    assert edges[0] == -math.inf
    assert edges[-1] == math.inf
    assert len(edges) == 6

    print("Passed.\n")


def test_count_bins() -> None:
    values = [
        1.0,
        2.0,
        5.0,
        8.0,
    ]

    edges = [
        -math.inf,
        3.0,
        6.0,
        math.inf,
    ]

    counts = count_bins(
        values,
        edges,
    )

    print("Test 2: count bins")
    print(counts)

    assert counts == [
        2,
        1,
        1,
    ]

    print("Passed.\n")


def test_identical_distribution_has_zero_psi() -> None:
    reference = [
        float(value)
        for value in range(100)
    ]

    result = calculate_psi(
        reference,
        reference.copy(),
        num_bins=10,
    )

    print(
        "Test 3: identical distribution PSI"
    )
    print(f"PSI = {result.psi}")

    assert abs(result.psi) < 1e-12

    print("Passed.\n")


def test_shifted_distribution_has_larger_psi() -> None:
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

    stable_result = calculate_psi(
        reference,
        stable_live,
        num_bins=10,
    )

    shifted_result = calculate_psi(
        reference,
        shifted_live,
        num_bins=10,
    )

    print(
        "Test 4: shifted distribution PSI"
    )
    print(
        f"stable PSI = {stable_result.psi}"
    )
    print(
        f"shifted PSI = {shifted_result.psi}"
    )

    assert (
        shifted_result.psi
        > stable_result.psi
    )

    print("Passed.\n")


def test_identical_distribution_has_zero_ks() -> None:
    values = [
        1.0,
        2.0,
        3.0,
        4.0,
    ]

    ks = calculate_ks_statistic(
        values,
        values.copy(),
    )

    print(
        "Test 5: identical distribution KS"
    )
    print(f"KS = {ks}")

    assert ks == 0.0

    print("Passed.\n")


def test_separated_distributions_have_large_ks() -> None:
    reference = [
        0.0,
        1.0,
        2.0,
        3.0,
    ]

    live = [
        10.0,
        11.0,
        12.0,
        13.0,
    ]

    ks = calculate_ks_statistic(
        reference,
        live,
    )

    print(
        "Test 6: separated distributions KS"
    )
    print(f"KS = {ks}")

    assert ks == 1.0

    print("Passed.\n")


def test_classify_psi() -> None:
    print("Test 7: classify PSI")

    assert (
        classify_psi(0.05)
        == "little_or_no_drift"
    )

    assert (
        classify_psi(0.15)
        == "moderate_drift"
    )

    assert (
        classify_psi(0.30)
        == "significant_drift"
    )

    print("Passed.\n")


def main() -> None:
    test_quantile_bin_edges()
    test_count_bins()
    test_identical_distribution_has_zero_psi()
    test_shifted_distribution_has_larger_psi()
    test_identical_distribution_has_zero_ks()
    test_separated_distributions_have_large_ks()
    test_classify_psi()

    print("All Day 32 tests passed.")


if __name__ == "__main__":
    main()