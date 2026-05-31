import torch

from implementation import (
    SingleHeadSelfAttention,
    TinyTransformerEmbeddingDemo,
    manual_attention,
)


def test_single_head_attention_output_shape():
    batch_size = 2
    seq_len = 4
    d_model = 8

    X = torch.randn(batch_size, seq_len, d_model)

    attention = SingleHeadSelfAttention(d_model=d_model)

    output, attention_weights = attention(X)

    print("Test 1: single-head attention output shape")
    print(f"output shape = {output.shape}")
    print(f"attention_weights shape = {attention_weights.shape}")

    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, seq_len, seq_len)

    print("Passed.\n")


def test_attention_weights_sum_to_one():
    batch_size = 2
    seq_len = 4
    d_model = 8

    X = torch.randn(batch_size, seq_len, d_model)

    attention = SingleHeadSelfAttention(d_model=d_model)

    _, attention_weights = attention(X)

    row_sums = attention_weights.sum(dim=-1)

    print("Test 2: attention weights row sums")
    print(row_sums)

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    print("Passed.\n")


def test_manual_attention_matches_module_when_weights_same():
    batch_size = 2
    seq_len = 3
    d_model = 4

    X = torch.randn(batch_size, seq_len, d_model)

    attention = SingleHeadSelfAttention(d_model=d_model)

    # nn.Linear stores weight as (out_features, in_features).
    # For X @ W, we need W.T.
    Wq = attention.Wq.weight.detach().T
    Wk = attention.Wk.weight.detach().T
    Wv = attention.Wv.weight.detach().T

    module_output, module_weights = attention(X)
    manual_output, manual_weights = manual_attention(X, Wq, Wk, Wv)

    print("Test 3: manual attention matches module attention")
    print(f"module_output shape = {module_output.shape}")
    print(f"manual_output shape = {manual_output.shape}")

    assert torch.allclose(module_output, manual_output, atol=1e-6)
    assert torch.allclose(module_weights, manual_weights, atol=1e-6)

    print("Passed.\n")


def test_embedding_attention_shapes():
    batch_size = 2
    seq_len = 5
    vocab_size = 30
    d_model = 8

    token_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
    )

    model = TinyTransformerEmbeddingDemo(
        vocab_size=vocab_size,
        d_model=d_model,
    )

    output, attention_weights = model(token_ids)

    print("Test 4: embedding + attention shapes")
    print(f"output shape = {output.shape}")
    print(f"attention_weights shape = {attention_weights.shape}")

    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, seq_len, seq_len)

    print("Passed.\n")


def test_attention_output_depends_on_input():
    batch_size = 1
    seq_len = 4
    d_model = 8

    attention = SingleHeadSelfAttention(d_model=d_model)

    X1 = torch.randn(batch_size, seq_len, d_model)
    X2 = X1.clone()
    X2[:, 0, :] += 1.0

    output1, weights1 = attention(X1)
    output2, weights2 = attention(X2)

    print("Test 5: attention output depends on input")

    assert not torch.allclose(output1, output2)
    assert not torch.allclose(weights1, weights2)

    print("Passed.\n")


def main():
    test_single_head_attention_output_shape()
    test_attention_weights_sum_to_one()
    test_manual_attention_matches_module_when_weights_same()
    test_embedding_attention_shapes()
    test_attention_output_depends_on_input()

    print("All Day 13 tests passed.")


if __name__ == "__main__":
    main()