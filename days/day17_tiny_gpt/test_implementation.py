import torch

from implementation import (
    TinyGPT,
    compute_lm_loss,
    create_next_token_inputs_targets,
)


def test_next_token_shift():
    token_ids = torch.tensor(
        [
            [10, 20, 30, 40, 50],
        ]
    )

    input_ids, target_ids = create_next_token_inputs_targets(token_ids)

    print("Test 1: next-token shift")
    print(f"input_ids = {input_ids}")
    print(f"target_ids = {target_ids}")

    assert torch.equal(input_ids, torch.tensor([[10, 20, 30, 40]]))
    assert torch.equal(target_ids, torch.tensor([[20, 30, 40, 50]]))

    print("Passed.\n")


def test_tiny_gpt_output_shape():
    vocab_size = 50
    max_seq_len = 16
    d_model = 32
    num_heads = 4
    ffn_dim = 128
    num_layers = 2

    model = TinyGPT(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
    )

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(4, 8),
    )

    logits = model(input_ids)

    print("Test 2: TinyGPT output shape")
    print(f"logits shape = {logits.shape}")

    assert logits.shape == (4, 8, vocab_size)

    print("Passed.\n")


def test_lm_loss_is_scalar():
    vocab_size = 20
    logits = torch.randn(2, 5, vocab_size)
    targets = torch.randint(0, vocab_size, size=(2, 5))

    loss = compute_lm_loss(logits, targets)

    print("Test 3: LM loss is scalar")
    print(f"loss shape = {loss.shape}")
    print(f"loss = {loss.item():.6f}")

    assert loss.ndim == 0
    assert loss.item() > 0

    print("Passed.\n")


def test_backward_runs():
    vocab_size = 30
    max_seq_len = 16
    d_model = 24
    num_heads = 4
    ffn_dim = 96
    num_layers = 2

    model = TinyGPT(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
    )

    token_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(3, 8),
    )

    input_ids, target_ids = create_next_token_inputs_targets(token_ids)

    logits = model(input_ids)
    loss = compute_lm_loss(logits, target_ids)

    loss.backward()

    print("Test 4: backward runs")
    print(f"loss = {loss.item():.6f}")

    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            break

    assert has_grad

    print("Passed.\n")


def test_sequence_too_long_raises_error():
    vocab_size = 30
    max_seq_len = 4
    d_model = 16
    num_heads = 4
    ffn_dim = 64
    num_layers = 1

    model = TinyGPT(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
    )

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(2, 5),
    )

    print("Test 5: sequence too long raises error")

    try:
        model(input_ids)
        raised = False
    except ValueError:
        raised = True

    assert raised

    print("Passed.\n")


def main():
    test_next_token_shift()
    test_tiny_gpt_output_shape()
    test_lm_loss_is_scalar()
    test_backward_runs()
    test_sequence_too_long_raises_error()

    print("All Day 17 tests passed.")


if __name__ == "__main__":
    main()