import math

import torch
from torch import nn


class SingleHeadSelfAttention(nn.Module):
    """
    Single-head self-attention.

    Input:
        X shape: (batch_size, seq_len, d_model)

    Output:
        output shape: (batch_size, seq_len, d_model)
        attention_weights shape: (batch_size, seq_len, seq_len)
    """

    def __init__(self, d_model: int):
        super().__init__()

        self.d_model = d_model

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product self-attention.
        """
        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.d_model)

        attention_weights = torch.softmax(scores, dim=-1)

        output = attention_weights @ V

        return output, attention_weights


class TinyTransformerEmbeddingDemo(nn.Module):
    """
    Token embedding + single-head self-attention demo.

    This is not a full Transformer block.
    It only shows:
        token IDs -> embeddings -> self-attention
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attention = SingleHeadSelfAttention(d_model=d_model)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_ids shape: (batch_size, seq_len)

        Returns:
            output shape: (batch_size, seq_len, d_model)
            attention_weights shape: (batch_size, seq_len, seq_len)
        """
        X = self.embedding(token_ids)
        output, attention_weights = self.attention(X)
        return output, attention_weights


def manual_attention(
    X: torch.Tensor,
    Wq: torch.Tensor,
    Wk: torch.Tensor,
    Wv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Manual attention calculation using raw weight matrices.

    Args:
        X:  (B, T, D)
        Wq: (D, D)
        Wk: (D, D)
        Wv: (D, D)

    Returns:
        output: (B, T, D)
        attention_weights: (B, T, T)
    """
    d_model = X.shape[-1]

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(d_model)

    attention_weights = torch.softmax(scores, dim=-1)

    output = attention_weights @ V

    return output, attention_weights


def self_attention_shape_demo() -> None:
    """
    Demonstrate self-attention shapes.
    """
    batch_size = 2
    seq_len = 4
    d_model = 8

    X = torch.randn(batch_size, seq_len, d_model)

    attention = SingleHeadSelfAttention(d_model=d_model)

    output, attention_weights = attention(X)

    print("Self-attention shape demo")
    print(f"X shape                 = {X.shape}")
    print(f"output shape            = {output.shape}")
    print(f"attention_weights shape = {attention_weights.shape}")
    print()


def embedding_attention_demo() -> None:
    """
    Demonstrate token IDs -> embeddings -> self-attention.
    """
    vocab_size = 20
    d_model = 8

    token_ids = torch.tensor(
        [
            [2, 5, 9, 4],
            [1, 3, 7, 8],
        ],
        dtype=torch.long,
    )

    model = TinyTransformerEmbeddingDemo(
        vocab_size=vocab_size,
        d_model=d_model,
    )

    output, attention_weights = model(token_ids)

    print("Embedding + attention demo")
    print(f"token_ids shape         = {token_ids.shape}")
    print(f"output shape            = {output.shape}")
    print(f"attention_weights shape = {attention_weights.shape}")
    print()

    print("Attention weights for first sequence:")
    print(attention_weights[0])
    print("Each row should sum to 1:")
    print(attention_weights[0].sum(dim=-1))
    print()


def main() -> None:
    self_attention_shape_demo()
    embedding_attention_demo()


if __name__ == "__main__":
    main()