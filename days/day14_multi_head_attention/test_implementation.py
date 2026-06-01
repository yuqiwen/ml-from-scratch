import torch

from implementation import MultiHeadSelfAttention


def test_output_shape():
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

    print("Test 1: output shape")
    print(f"output shape = {output.shape}")
    print(f"attention_weights shape = {attention_weights.shape}")

    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)

    print("Passed.\n")


def test_attention_weights_sum_to_one():
    batch_size = 2
    seq_len = 5
    d_model = 12
    num_heads = 3

    X = torch.randn(batch_size, seq_len, d_model)

    attention = MultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
    )

    _, attention_weights = attention(X)

    row_sums = attention_weights.sum(dim=-1)

    print("Test 2: attention weights sum to one")
    print(row_sums)

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    print("Passed.\n")


def test_split_and_concat_restore_shape_and_values():
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

    print("Test 3: split and concat restore values")
    print(f"X_heads shape = {X_heads.shape}")
    print(f"X_concat shape = {X_concat.shape}")

    assert X_heads.shape == (batch_size, num_heads, seq_len, d_model // num_heads)
    assert X_concat.shape == X.shape
    assert torch.allclose(X, X_concat)

    print("Passed.\n")


def test_invalid_head_configuration_raises_error():
    print("Test 4: invalid head configuration raises error")

    try:
        MultiHeadSelfAttention(
            d_model=10,
            num_heads=3,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid head configuration.")

    print("Passed.\n")


def test_output_changes_with_input():
    batch_size = 1
    seq_len = 4
    d_model = 8
    num_heads = 2

    attention = MultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
    )

    X1 = torch.randn(batch_size, seq_len, d_model)
    X2 = X1.clone()
    X2[:, 0, :] += 1.0

    output1, weights1 = attention(X1)
    output2, weights2 = attention(X2)

    print("Test 5: output changes with input")

    assert not torch.allclose(output1, output2)
    assert not torch.allclose(weights1, weights2)

    print("Passed.\n")


def main():
    test_output_shape()
    test_attention_weights_sum_to_one()
    test_split_and_concat_restore_shape_and_values()
    test_invalid_head_configuration_raises_error()
    test_output_changes_with_input()

    print("All Day 14 tests passed.")


if __name__ == "__main__":
    main()
