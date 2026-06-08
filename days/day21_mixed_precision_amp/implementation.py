import contextlib

import torch
from torch import nn


class TinyMLP(nn.Module):
    """
    Small MLP for regression.

    Input:
        X: (B, 2)

    Output:
        y_hat: (B, 1)
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X)


def generate_data(
    n_samples: int = 512,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate nonlinear regression data.

    y = x1^2 + 0.5*x2 + noise
    """
    torch.manual_seed(seed)

    X = torch.randn(n_samples, 2)
    noise = 0.1 * torch.randn(n_samples, 1)

    y = X[:, [0]] ** 2 + 0.5 * X[:, [1]] + noise

    return X, y


def get_amp_context(
    device: torch.device,
    dtype: torch.dtype | None,
):
    """
    Return autocast context if dtype is provided and supported.

    For CPU tests, this still works for bfloat16 on many PyTorch builds,
    but we keep fallback behavior simple.
    """
    if dtype is None:
        return contextlib.nullcontext()

    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
    )


def train_one_epoch_fp32(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    """
    Standard FP32 training step.
    """
    model.train()

    y_hat = model(X)
    loss = loss_fn(y_hat, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


def train_one_epoch_amp(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_grad_scaler: bool = False,
) -> float:
    """
    AMP training step.

    If use_grad_scaler is True and CUDA is available, use GradScaler.
    Otherwise, use normal backward.
    """
    model.train()

    device = X.device

    # GradScaler is mainly for CUDA FP16.
    scaler_enabled = use_grad_scaler and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    optimizer.zero_grad()

    with get_amp_context(device=device, dtype=amp_dtype):
        y_hat = model(X)
        loss = loss_fn(y_hat, y)

    if scaler_enabled:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return float(loss.item())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
) -> float:
    """
    Evaluate model.
    """
    model.eval()
    y_hat = model(X)
    loss = loss_fn(y_hat, y)
    return float(loss.item())


def compare_fp32_and_amp_demo() -> None:
    """
    Compare FP32 and AMP training loops.

    On CPU, AMP may not be faster.
    On CUDA, AMP can reduce memory and improve speed depending on hardware.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = generate_data(n_samples=512, seed=1)
    X = X.to(device)
    y = y.to(device)

    loss_fn = nn.MSELoss()

    fp32_model = TinyMLP().to(device)
    amp_model = TinyMLP().to(device)

    # Start from same weights for comparison.
    amp_model.load_state_dict(fp32_model.state_dict())

    fp32_optimizer = torch.optim.AdamW(fp32_model.parameters(), lr=0.01)
    amp_optimizer = torch.optim.AdamW(amp_model.parameters(), lr=0.01)

    print(f"Using device: {device}")

    print("Training FP32 model")
    for epoch in range(5):
        loss = train_one_epoch_fp32(
            model=fp32_model,
            X=X,
            y=y,
            optimizer=fp32_optimizer,
            loss_fn=loss_fn,
        )
        print(f"fp32 epoch={epoch:02d}, loss={loss:.6f}")

    print()

    # Prefer BF16 for portability/stability if available.
    amp_dtype = torch.bfloat16 if device.type in {"cuda", "cpu"} else None

    print(f"Training AMP model with dtype={amp_dtype}")
    for epoch in range(5):
        loss = train_one_epoch_amp(
            model=amp_model,
            X=X,
            y=y,
            optimizer=amp_optimizer,
            loss_fn=loss_fn,
            amp_dtype=amp_dtype,
            use_grad_scaler=False,
        )
        print(f"amp epoch={epoch:02d}, loss={loss:.6f}")

    print()

    fp32_eval = evaluate(fp32_model, X, y, loss_fn)
    amp_eval = evaluate(amp_model, X, y, loss_fn)

    print("Final evaluation")
    print(f"fp32_eval_loss = {fp32_eval:.6f}")
    print(f"amp_eval_loss  = {amp_eval:.6f}")


def dtype_memory_demo() -> None:
    """
    Show memory difference between FP32 and FP16/BF16 tensors.
    """
    x_fp32 = torch.zeros(1024, 1024, dtype=torch.float32)
    x_fp16 = torch.zeros(1024, 1024, dtype=torch.float16)
    x_bf16 = torch.zeros(1024, 1024, dtype=torch.bfloat16)

    print("Dtype memory demo")
    print(f"FP32 bytes = {x_fp32.element_size() * x_fp32.numel()}")
    print(f"FP16 bytes = {x_fp16.element_size() * x_fp16.numel()}")
    print(f"BF16 bytes = {x_bf16.element_size() * x_bf16.numel()}")
    print()


def main() -> None:
    dtype_memory_demo()
    compare_fp32_and_amp_demo()


if __name__ == "__main__":
    main()
