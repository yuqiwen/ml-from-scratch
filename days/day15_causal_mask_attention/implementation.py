import math

import torch
from torch import nn


def create_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Create a lower-triangular causal mask.

    Returns:
        mask shape: (seq_len, seq_len)

    mask[i, j] = True if token i can attend to token j.
    """
    return torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    )


def apply_causal_mask(scores: torch.Tensor) -> torch.Tensor:
    """
    Apply causal mask to attention scores.

    Args:
        scores shape: (B, H, T, T)

    Returns:
        masked_scores shape: (B, H, T, T)
    """
    seq_len = scores.shape[-1]
    mask = create_causal_mask(seq_len, device=scores.device)

    # Broadcast mask from (T, T) to (B, H, T, T).
    masked_scores = scores.masked_fill(~mask, float("-inf"))

    return masked_scores


class CausalMultiHeadSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Input:
        X shape: (B, T, D)

    Output:
        output shape: (B, T, D)
        attention_weights shape: (B, H, T, T)
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, X: torch.Tensor) -> torch.Tensor:
        """
        Convert:
            (B, T, D) -> (B, H, T, Hd)
        """
        batch_size, seq_len, _ = X.shape

        X = X.view(batch_size, seq_len, self.num_heads, self.head_dim)
        X = X.transpose(1, 2)

        return X

    def concat_heads(self, X: torch.Tensor) -> torch.Tensor:
        """
        Convert:
            (B, H, T, Hd) -> (B, T, D)
        """
        batch_size, num_heads, seq_len, head_dim = X.shape

        X = X.transpose(1, 2).contiguous()
        X = X.view(batch_size, seq_len, num_heads * head_dim)

        return X

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute causal multi-head self-attention.
        """
        Q = self.split_heads(self.q_proj(X))
        K = self.split_heads(self.k_proj(X))
        V = self.split_heads(self.v_proj(X))

        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        scores = apply_causal_mask(scores)

        attention_weights = torch.softmax(scores, dim=-1)

        head_output = attention_weights @ V

        concat_output = self.concat_heads(head_output)

        output = self.out_proj(concat_output)

        return output, attention_weights


def causal_mask_demo() -> None:
    """
    Print a causal mask.
    """
    seq_len = 5
    mask = create_causal_mask(seq_len)

    print("Causal mask demo")
    print(mask.int())
    print()


def causal_attention_shape_demo() -> None:
    """
    Demonstrate causal attention shapes.
    """
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2

    X = torch.randn(batch_size, seq_len, d_model)

    attention = CausalMultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
    )

    output, attention_weights = attention(X)

    print("Causal multi-head attention shape demo")
    print(f"X shape                 = {X.shape}")
    print(f"output shape            = {output.shape}")
    print(f"attention_weights shape = {attention_weights.shape}")
    print()


def main() -> None:
    causal_mask_demo()
    causal_attention_shape_demo()


if __name__ == "__main__":
    main()