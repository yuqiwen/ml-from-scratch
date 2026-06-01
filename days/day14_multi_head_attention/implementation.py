import math

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention from scratch.

    Input:
        X shape: (batch_size, seq_len, d_model)

    Output:
        output shape: (batch_size, seq_len, d_model)
        attention_weights shape: (batch_size, num_heads, seq_len, seq_len)
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
            X: (B, T, D)

        Into:
            X_heads: (B, H, T, Hd)
        """
        batch_size, seq_len, d_model = X.shape

        X = X.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        X = X.transpose(1, 2)

        return X

    def concat_heads(self, X: torch.Tensor) -> torch.Tensor:
        """
        Convert:
            X: (B, H, T, Hd)

        Into:
            X_concat: (B, T, D)
        """
        batch_size, num_heads, seq_len, head_dim = X.shape

        X = X.transpose(1, 2)

        # After transpose, tensor may be non-contiguous.
        # contiguous() makes memory layout compatible with view().
        X = X.contiguous()

        X = X.view(
            batch_size,
            seq_len,
            num_heads * head_dim,
        )

        return X

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head self-attention.
        """
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        attention_weights = torch.softmax(scores, dim=-1)

        head_output = attention_weights @ V

        concat_output = self.concat_heads(head_output)

        output = self.out_proj(concat_output)

        return output, attention_weights


def shape_demo() -> None:
    """
    Demonstrate multi-head attention shapes.
    """
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2

    X = torch.randn(batch_size, seq_len, d_model)

    attention = MultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
    )

    output, attention_weights = attention(X)

    print("Multi-head attention shape demo")
    print(f"X shape                 = {X.shape}")
    print(f"output shape            = {output.shape}")
    print(f"attention_weights shape = {attention_weights.shape}")
    print(f"head_dim                = {attention.head_dim}")
    print()


def split_concat_demo() -> None:
    """
    Demonstrate split_heads and concat_heads.
    """
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2

    X = torch.randn(batch_size, seq_len, d_model)

    attention = MultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
    )

    X_heads = attention.split_heads(X)
    X_concat = attention.concat_heads(X_heads)

    print("Split / concat demo")
    print(f"original X shape = {X.shape}")
    print(f"split shape      = {X_heads.shape}")
    print(f"concat shape     = {X_concat.shape}")
    print(f"concat close to original: {torch.allclose(X, X_concat)}")
    print()


def main() -> None:
    shape_demo()
    split_concat_demo()


if __name__ == "__main__":
    main()