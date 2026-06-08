import torch
from torch import nn

from implementation import (
    TinyMLP,
    evaluate,
    generate_data,
    get_amp_context,
    train_one_epoch_amp,
    train_one_epoch_fp32,
)


def test_dtype_memory_sizes():
    x_fp32 = torch.zeros(10, dtype=torch.float32)
    x_fp16 = torch.zeros(10, dtype=torch.float16)
    x_bf16 = torch.zeros(10, dtype=torch.bfloat16)

    print("Test 1: dtype element sizes")
    print(f"fp32 element_size = {x_fp32.element_size()}")
    print(f"fp16 element_size = {x_fp16.element_size()}")
    print(f"bf16 element_size = {x_bf16.element_size()}")

    assert x_fp32.element_size() == 4
    assert x_fp16.element_size() == 2
    assert x_bf16.element_size() == 2

    print("Passed.\n")


def test_fp32_training_reduces_loss():
    X, y = generate_data(n_samples=256, seed=1)

    model = TinyMLP(hidden_dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    initial_loss = evaluate(model, X, y, loss_fn)

    for _ in range(5):
        train_one_epoch_fp32(model, X, y, optimizer, loss_fn)

    final_loss = evaluate(model, X, y, loss_fn)

    print("Test 2: FP32 training reduces loss")
    print(f"initial_loss = {initial_loss:.6f}")
    print(f"final_loss = {final_loss:.6f}")

    assert final_loss < initial_loss

    print("Passed.\n")


def test_amp_training_runs_and_reduces_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = generate_data(n_samples=256, seed=2)
    X = X.to(device)
    y = y.to(device)

    model = TinyMLP(hidden_dim=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    initial_loss = evaluate(model, X, y, loss_fn)

    amp_dtype = torch.bfloat16

    for _ in range(5):
        train_one_epoch_amp(
            model=model,
            X=X,
            y=y,
            optimizer=optimizer,
            loss_fn=loss_fn,
            amp_dtype=amp_dtype,
            use_grad_scaler=False,
        )

    final_loss = evaluate(model, X, y, loss_fn)

    print("Test 3: AMP training runs and reduces loss")
    print(f"device = {device}")
    print(f"initial_loss = {initial_loss:.6f}")
    print(f"final_loss = {final_loss:.6f}")

    assert final_loss < initial_loss

    print("Passed.\n")


def test_autocast_context_runs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.randn(4, 2).to(device)

    with get_amp_context(device=device, dtype=torch.bfloat16):
        Y = X @ X.T

    print("Test 4: autocast context runs")
    print(f"Y dtype = {Y.dtype}")

    assert Y.shape == (4, 4)

    print("Passed.\n")


def test_amp_with_grad_scaler_flag_runs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = generate_data(n_samples=128, seed=3)
    X = X.to(device)
    y = y.to(device)

    model = TinyMLP(hidden_dim=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    loss = train_one_epoch_amp(
        model=model,
        X=X,
        y=y,
        optimizer=optimizer,
        loss_fn=loss_fn,
        amp_dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        use_grad_scaler=True,
    )

    print("Test 5: AMP with GradScaler flag runs")
    print(f"device = {device}")
    print(f"loss = {loss:.6f}")

    assert loss > 0

    print("Passed.\n")


def main():
    test_dtype_memory_sizes()
    test_fp32_training_reduces_loss()
    test_amp_training_runs_and_reduces_loss()
    test_autocast_context_runs()
    test_amp_with_grad_scaler_flag_runs()

    print("All Day 21 tests passed.")


if __name__ == "__main__":
    main()