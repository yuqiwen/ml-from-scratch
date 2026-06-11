import math
import time
from dataclasses import dataclass

import torch
from torch import nn


class TinyInferenceModel(nn.Module):
    """
    Small MLP used for inference benchmarking.

    Input:
        X: (B, input_dim)

    Output:
        y_hat: (B, output_dim)
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 10,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X)


@dataclass
class BenchmarkResult:
    batch_size: int
    num_iters: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_samples_per_sec: float


def synchronize_if_cuda(device: torch.device) -> None:
    """
    Synchronize CUDA device for accurate timing.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()


def percentile(values: list[float], q: float) -> float:
    """
    Compute percentile from a list of numeric values.
    """
    if not values:
        raise ValueError("values must not be empty.")
    if q < 0 or q > 100:
        raise ValueError("q must be in [0, 100].")

    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (q / 100.0)

    lower = math.floor(rank)
    upper = math.ceil(rank)

    if lower == upper:
        return float(sorted_values[lower])

    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    weight = rank - lower

    return float(lower_value + (upper_value - lower_value) * weight)


@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    batch_size: int,
    input_dim: int,
    device: torch.device,
    num_warmup: int = 10,
    num_iters: int = 50,
) -> BenchmarkResult:
    """
    Benchmark model inference latency and throughput.
    """
    model.eval()

    x = torch.randn(batch_size, input_dim, device=device)

    for _ in range(num_warmup):
        _ = model(x)

    synchronize_if_cuda(device)

    latencies_ms: list[float] = []

    for _ in range(num_iters):
        synchronize_if_cuda(device)
        start = time.perf_counter()
        _ = model(x)
        synchronize_if_cuda(device)
        end = time.perf_counter()

        latencies_ms.append((end - start) * 1000.0)

    avg_latency_ms = sum(latencies_ms) / len(latencies_ms)
    throughput_samples_per_sec = batch_size / (avg_latency_ms / 1000.0)

    return BenchmarkResult(
        batch_size=batch_size,
        num_iters=num_iters,
        avg_latency_ms=avg_latency_ms,
        p50_latency_ms=percentile(latencies_ms, 50),
        p95_latency_ms=percentile(latencies_ms, 95),
        p99_latency_ms=percentile(latencies_ms, 99),
        throughput_samples_per_sec=throughput_samples_per_sec,
    )


def benchmark_demo() -> None:
    """
    Run a simple benchmark with several batch sizes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyInferenceModel(
        input_dim=128,
        hidden_dim=256,
        output_dim=10,
    ).to(device)

    print(f"device = {device}")

    for batch_size in [1, 8, 32]:
        result = benchmark_inference(
            model=model,
            batch_size=batch_size,
            input_dim=128,
            device=device,
            num_warmup=5,
            num_iters=20,
        )

        print(
            f"batch_size={result.batch_size}, "
            f"avg={result.avg_latency_ms:.3f} ms, "
            f"p50={result.p50_latency_ms:.3f} ms, "
            f"p95={result.p95_latency_ms:.3f} ms, "
            f"p99={result.p99_latency_ms:.3f} ms, "
            f"throughput={result.throughput_samples_per_sec:.2f} samples/s"
        )


def main() -> None:
    benchmark_demo()


if __name__ == "__main__":
    main()
