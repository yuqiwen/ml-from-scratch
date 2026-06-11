import torch

from implementation import (
    BenchmarkResult,
    TinyInferenceModel,
    benchmark_inference,
    percentile,
    synchronize_if_cuda,
)


def test_percentile_basic():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    p50 = percentile(values, 50)
    p100 = percentile(values, 100)
    p0 = percentile(values, 0)

    print("Test 1: percentile")
    print(f"p0 = {p0}")
    print(f"p50 = {p50}")
    print(f"p100 = {p100}")

    assert p0 == 1.0
    assert p50 == 3.0
    assert p100 == 5.0

    print("Passed.\n")


def test_model_output_shape():
    model = TinyInferenceModel(
        input_dim=128,
        hidden_dim=64,
        output_dim=10,
    )

    x = torch.randn(4, 128)
    y = model(x)

    print("Test 2: model output shape")
    print(f"y shape = {y.shape}")

    assert y.shape == (4, 10)

    print("Passed.\n")


def test_benchmark_returns_result():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyInferenceModel(
        input_dim=32,
        hidden_dim=64,
        output_dim=5,
    ).to(device)

    result = benchmark_inference(
        model=model,
        batch_size=4,
        input_dim=32,
        device=device,
        num_warmup=2,
        num_iters=5,
    )

    print("Test 3: benchmark returns result")
    print(result)

    assert isinstance(result, BenchmarkResult)
    assert result.batch_size == 4
    assert result.num_iters == 5
    assert result.avg_latency_ms > 0
    assert result.throughput_samples_per_sec > 0

    print("Passed.\n")


def test_p95_greater_than_or_equal_p50():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyInferenceModel(
        input_dim=16,
        hidden_dim=32,
        output_dim=2,
    ).to(device)

    result = benchmark_inference(
        model=model,
        batch_size=2,
        input_dim=16,
        device=device,
        num_warmup=2,
        num_iters=10,
    )

    print("Test 4: p95 >= p50")
    print(f"p50 = {result.p50_latency_ms}")
    print(f"p95 = {result.p95_latency_ms}")

    assert result.p95_latency_ms >= result.p50_latency_ms

    print("Passed.\n")


def test_synchronize_if_cuda_runs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Test 5: synchronize_if_cuda runs")
    print(f"device = {device}")

    synchronize_if_cuda(device)

    print("Passed.\n")


def main():
    test_percentile_basic()
    test_model_output_shape()
    test_benchmark_returns_result()
    test_p95_greater_than_or_equal_p50()
    test_synchronize_if_cuda_runs()

    print("All Day 24 tests passed.")


if __name__ == "__main__":
    main()