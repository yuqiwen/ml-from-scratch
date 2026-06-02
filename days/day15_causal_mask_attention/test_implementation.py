import torch

from implementation import (
    CausalMultiHeadSelfAttention,
    apply_causal_mask,
    create_causal_mask,
)


def test_causal_mask_shape_and_values():
    mask = create_causal_mask(seq_len=4)

    expected = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )

    print("Test 1: causal mask values")
    print(mask.int())

    assert mask.shape == (4, 4)
    assert torch.equal(mask.cpu(), expected)

    print("Passed.\n")


def test_apply_causal_mask_sets_future_to_negative_infinity():
    scores = torch.zeros(1, 1, 4, 4)

    masked_scores = apply_causal_mask(scores)

    print("Test 2: apply causal mask")
    print(masked_scores)

    assert torch.isinf(masked_scores[0, 0, 0, 1])
    assert masked_scores[0, 0, 0, 1] < 0

    assert masked_scores[0, 0, 3, 0] == 0

    print("Passed.\n")


def test_attention_weights_future_positions_are_zero():
    batch_size = 1
    seq_len = 4
    d_model = 8
    num_heads = 2

    X = torch.randn(batch_size, seq_len, d_model)

    attention = CausalMultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
    )

    _, weights = attention(X)

    print("Test 3: future attention weights are zero")
    print(weights[0, 0])

    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert torch.isclose(weights[0, 0, i, j], torch.tensor(0.0), atol=1e-6)

    print("Passed.\n")


def test_attention_weights_rows_sum_to_one():
    X = torch.randn(2, 5, 12)

    attention = CausalMultiHeadSelfAttention(
        d_model=12,
        num_heads=3,
    )

    _, weights = attention(X)

    row_sums = weights.sum(dim=-1)

    print("Test 4: attention row sums")
    print(row_sums)

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    print("Passed.\n")


def test_causal_attention_output_shape():
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2

    X = torch.randn(batch_size, seq_len, d_model)

    attention = CausalMultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
    )

    output, weights = attention(X)

    print("Test 5: causal attention output shape")
    print(f"output shape = {output.shape}")
    print(f"weights shape = {weights.shape}")

    assert output.shape == (batch_size, seq_len, d_model)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)

    print("Passed.\n")


def main():
    test_causal_mask_shape_and_values()
    test_apply_causal_mask_sets_future_to_negative_infinity()
    test_attention_weights_future_positions_are_zero()
    test_attention_weights_rows_sum_to_one()
    test_causal_attention_output_shape()

    print("All Day 15 tests passed.")


if __name__ == "__main__":
    main()