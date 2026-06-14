from dataclasses import dataclass, field
from threading import Lock


def percentile(
    values: list[float],
    percent: float,
) -> float:
    """
    Compute a percentile using linear interpolation.
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)

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

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]

    return (
        lower_value * (1.0 - weight)
        + upper_value * weight
    )


@dataclass
class MetricsStore:
    """
    In-memory serving metrics.

    Lock is used because multiple requests may update
    metrics concurrently.
    """

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    latencies_ms: list[float] = field(
        default_factory=list
    )
    lock: Lock = field(
        default_factory=Lock
    )

    def record_success(
        self,
        latency_ms: float,
    ) -> None:
        with self.lock:
            self.total_requests += 1
            self.successful_requests += 1
            self.latencies_ms.append(latency_ms)

    def record_failure(
        self,
        latency_ms: float,
    ) -> None:
        with self.lock:
            self.total_requests += 1
            self.failed_requests += 1
            self.latencies_ms.append(latency_ms)

    def snapshot(self) -> dict[str, float | int]:
        """
        Return a consistent copy of current metrics.
        """
        with self.lock:
            total = self.total_requests
            success = self.successful_requests
            failure = self.failed_requests
            latencies = list(self.latencies_ms)

        error_rate = (
            failure / total
            if total > 0
            else 0.0
        )

        average_latency = (
            sum(latencies) / len(latencies)
            if latencies
            else 0.0
        )

        return {
            "total_requests": total,
            "successful_requests": success,
            "failed_requests": failure,
            "error_rate": error_rate,
            "average_latency_ms": average_latency,
            "p50_latency_ms": percentile(
                latencies,
                50,
            ),
            "p95_latency_ms": percentile(
                latencies,
                95,
            ),
            "p99_latency_ms": percentile(
                latencies,
                99,
            ),
        }