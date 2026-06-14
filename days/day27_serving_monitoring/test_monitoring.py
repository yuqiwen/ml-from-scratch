from metrics import MetricsStore, percentile


def test_percentile() -> None:
    values = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]

    print("Test 1: percentile")

    assert percentile(values, 50) == 3.0
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 5.0

    print("Passed.\n")


def test_record_success() -> None:
    store = MetricsStore()

    store.record_success(10.0)
    store.record_success(20.0)

    snapshot = store.snapshot()

    print("Test 2: record success")
    print(snapshot)

    assert snapshot["total_requests"] == 2
    assert snapshot["successful_requests"] == 2
    assert snapshot["failed_requests"] == 0
    assert snapshot["error_rate"] == 0.0
    assert snapshot["average_latency_ms"] == 15.0

    print("Passed.\n")


def test_record_failure() -> None:
    store = MetricsStore()

    store.record_success(10.0)
    store.record_failure(30.0)

    snapshot = store.snapshot()

    print("Test 3: record failure")
    print(snapshot)

    assert snapshot["total_requests"] == 2
    assert snapshot["successful_requests"] == 1
    assert snapshot["failed_requests"] == 1
    assert snapshot["error_rate"] == 0.5

    print("Passed.\n")


def test_latency_percentiles() -> None:
    store = MetricsStore()

    latencies = [
        5.0,
        10.0,
        15.0,
        20.0,
        100.0,
    ]

    for latency in latencies:
        store.record_success(latency)

    snapshot = store.snapshot()

    print("Test 4: latency percentiles")
    print(snapshot)

    assert snapshot["p50_latency_ms"] == 15.0
    assert (
        snapshot["p95_latency_ms"]
        >= snapshot["p50_latency_ms"]
    )
    assert (
        snapshot["p99_latency_ms"]
        >= snapshot["p95_latency_ms"]
    )

    print("Passed.\n")


def test_empty_metrics() -> None:
    store = MetricsStore()

    snapshot = store.snapshot()

    print("Test 5: empty metrics")
    print(snapshot)

    assert snapshot["total_requests"] == 0
    assert snapshot["error_rate"] == 0.0
    assert snapshot["average_latency_ms"] == 0.0

    print("Passed.\n")


def main() -> None:
    test_percentile()
    test_record_success()
    test_record_failure()
    test_latency_percentiles()
    test_empty_metrics()

    print("All Day 27 tests passed.")


if __name__ == "__main__":
    main()