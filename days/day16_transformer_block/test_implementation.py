import torch
from torch import nn

from implementation import (
    CausalMultiHeadSelfAttention,
    FeedForwardNetwork,
    TransformerBlock,
)


def test_ffn_output_shape():
    batch_size = 2
    seq_len = 5
    d_model = 16
    ffn_dim = 64

    X = torch.randn(batch_size, seq_len, d_model)

    ffn = FeedForwardNetwork(
        d_model=d_model,
        ffn_dim=ffn_dim,
    )

    Y = ffn(X)

    print("Test 1: FFN output shape")
    print(f"Y shape = {Y.shape}")

    assert Y.shape == (batch_size, seq_len, d_model)

    print("Passed.\n")


def test_layernorm_shape():
    X = torch.randn(2, 5, 16)

    ln = nn.LayerNorm(16)

    Y = ln(X)

    print("Test 2: LayerNorm shape")
    print(f"Y shape = {Y.shape}")

    assert Y.shape == X.shape

    print("Passed.\n")


def test_causal_attention_output_shape():
    batch_size = 2
    seq_len = 5
    d_model = 16
    num_heads = 4

    X = torch.randn(batch_size, seq_len, d_model)

    attention = CausalMultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
    )

    output, weights = attention(X)

    print("Test 3: causal attention output shape")
    print(f"output shape = {output.shape}")
    print(f"weights shape = {weights.shape}")

    assert output.shape == (batch_size, seq_len, d_model)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)

    print("Passed.\n")


def test_transformer_block_output_shape():
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

    output, weights = block(X)

    print("Test 4: Transformer block output shape")
    print(f"output shape = {output.shape}")
    print(f"weights shape = {weights.shape}")

    assert output.shape == X.shape
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)

    print("Passed.\n")


def test_residual_connection_changes_but_preserves_shape():
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

    output, _ = block(X)

    print("Test 5: residual preserves shape")
    print(f"input shape = {X.shape}")
    print(f"output shape = {output.shape}")

    assert output.shape == X.shape

    # The output should usually not be exactly equal to input
    # because attention and FFN add transformations.
    assert not torch.allclose(output, X)

    print("Passed.\n")


def test_transformer_block_backward_runs():
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

    output, _ = block(X)

    loss = output.mean()
    loss.backward()

    print("Test 6: Transformer block backward runs")

    has_grad = False
    for param in block.parameters():
        if param.grad is not None:
            has_grad = True
            break

    assert has_grad

    print("Passed.\n")


def main():
    test_ffn_output_shape()
    test_layernorm_shape()
    test_causal_attention_output_shape()
    test_transformer_block_output_shape()
    test_residual_connection_changes_but_preserves_shape()
    test_transformer_block_backward_runs()

    print("All Day 16 tests passed.")


if __name__ == "__main__":
    main()