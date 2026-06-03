import math

import torch
from torch import nn


def create_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Create a lower-triangular causal mask.

    mask[i, j] = True means token i can attend to token j.
    """
    return torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    )


class CausalMultiHeadSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Input:
        X: (B, T, D)

    Output:
        output: (B, T, D)
        attention_weights: (B, H, T, T)
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
            (B, H, T, Hd) -> (B, T, D)
        """
        batch_size, num_heads, seq_len, head_dim = X.shape

        X = X.transpose(1, 2).contiguous()

        X = X.view(
            batch_size,
            seq_len,
            num_heads * head_dim,
        )

        return X

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute causal multi-head self-attention.
        """
        batch_size, seq_len, _ = X.shape

        Q = self.split_heads(self.q_proj(X))
        K = self.split_heads(self.k_proj(X))
        V = self.split_heads(self.v_proj(X))

        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        mask = create_causal_mask(seq_len, device=X.device)
        scores = scores.masked_fill(~mask, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)

        head_output = attention_weights @ V

        concat_output = self.concat_heads(head_output)

        output = self.out_proj(concat_output)

        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    """
    Transformer feed-forward network.

    Structure:
        Linear(D, ffn_dim)
        GELU
        Linear(ffn_dim, D)

    Input:
        X: (B, T, D)

    Output:
        Y: (B, T, D)
    """

    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X)


class TransformerBlock(nn.Module):
    """
    A basic GPT-style Pre-LN Transformer block.

    Structure:
        x = x + attention(layer_norm(x))
        x = x + ffn(layer_norm(x))

    Input:
        x: (B, T, D)

    Output:
        x: (B, T, D)
        attention_weights: (B, H, T, T)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attention = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            ffn_dim=ffn_dim,
        )

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        """
        attention_input = self.ln1(X)
        attention_output, attention_weights = self.attention(attention_input)

        X = X + attention_output

        ffn_input = self.ln2(X)
        ffn_output = self.ffn(ffn_input)

        X = X + ffn_output

        return X, attention_weights


def transformer_block_shape_demo() -> None:
    """
    Demonstrate Transformer block shapes.
    """
    batch_size = 2
    seq_len = 5
    d_model = 16
    num_heads = 4
    ffn_dim = 64

    X = torch.randn(batch_size, seq_len, d_model)

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
    )

    output, attention_weights = block(X)

    print("Transformer block shape demo")
    print(f"input shape             = {X.shape}")
    print(f"output shape            = {output.shape}")
    print(f"attention_weights shape = {attention_weights.shape}")
    print()


def layernorm_demo() -> None:
    """
    Demonstrate LayerNorm shape behavior.
    """
    X = torch.randn(2, 5, 16)

    layer_norm = nn.LayerNorm(16)

    Y = layer_norm(X)

    print("LayerNorm demo")
    print(f"input shape  = {X.shape}")
    print(f"output shape = {Y.shape}")
    print()


def ffn_demo() -> None:
    """
    Demonstrate FFN shape behavior.
    """
    X = torch.randn(2, 5, 16)

    ffn = FeedForwardNetwork(
        d_model=16,
        ffn_dim=64,
    )

    Y = ffn(X)

    print("FFN demo")
    print(f"input shape  = {X.shape}")
    print(f"output shape = {Y.shape}")
    print()


def main() -> None:
    layernorm_demo()
    ffn_demo()
    transformer_block_shape_demo()


if __name__ == "__main__":
    main()