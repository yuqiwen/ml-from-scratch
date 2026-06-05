import torch

from implementation import CachedSelfAttention, KVCache


def test_empty_cache_length_is_zero():
    cache = KVCache()

    print("Test 1: empty cache length")
    print(f"cache length = {cache.length}")

    assert cache.length == 0

    print("Passed.\n")


def test_cache_append_increases_length():
    cache = KVCache()

    K1 = torch.randn(1, 2, 4, 8)
    V1 = torch.randn(1, 2, 4, 8)

    cache.append(K1, V1)

    print("Test 2: cache append initial")
    print(f"cache length = {cache.length}")

    assert cache.length == 4

    K2 = torch.randn(1, 2, 1, 8)
    V2 = torch.randn(1, 2, 1, 8)

    cache.append(K2, V2)

    print("After appending one token")
    print(f"cache length = {cache.length}")

    assert cache.length == 5

    print("Passed.\n")


def test_prefill_cache_shape():
    B = 2
    T = 4
    D = 8
    H = 2

    attention = CachedSelfAttention(d_model=D, num_heads=H)
    cache = KVCache()

    X = torch.randn(B, T, D)

    output, cache, weights = attention(
        X=X,
        cache=cache,
        is_prefill=True,
    )

    print("Test 3: prefill cache shape")
    print(f"output shape = {output.shape}")
    print(f"K shape = {cache.K.shape}")
    print(f"V shape = {cache.V.shape}")
    print(f"weights shape = {weights.shape}")

    assert output.shape == (B, T, D)
    assert cache.K.shape == (B, H, T, D // H)
    assert cache.V.shape == (B, H, T, D // H)
    assert weights.shape == (B, H, T, T)

    print("Passed.\n")


def test_decode_appends_one_token():
    B = 1
    prompt_len = 4
    D = 8
    H = 2

    attention = CachedSelfAttention(d_model=D, num_heads=H)
    cache = KVCache()

    prompt = torch.randn(B, prompt_len, D)

    _, cache, _ = attention(
        X=prompt,
        cache=cache,
        is_prefill=True,
    )

    new_token = torch.randn(B, 1, D)

    output, cache, weights = attention(
        X=new_token,
        cache=cache,
        is_prefill=False,
    )

    print("Test 4: decode appends one token")
    print(f"output shape = {output.shape}")
    print(f"cache length = {cache.length}")
    print(f"weights shape = {weights.shape}")

    assert output.shape == (B, 1, D)
    assert cache.length == prompt_len + 1
    assert weights.shape == (B, H, 1, prompt_len + 1)

    print("Passed.\n")


def test_multiple_decode_steps_grow_cache():
    B = 1
    prompt_len = 3
    D = 12
    H = 3

    attention = CachedSelfAttention(d_model=D, num_heads=H)
    cache = KVCache()

    prompt = torch.randn(B, prompt_len, D)

    _, cache, _ = attention(
        X=prompt,
        cache=cache,
        is_prefill=True,
    )

    for _ in range(5):
        new_token = torch.randn(B, 1, D)
        _, cache, _ = attention(
            X=new_token,
            cache=cache,
            is_prefill=False,
        )

    print("Test 5: multiple decode steps grow cache")
    print(f"cache length = {cache.length}")

    assert cache.length == prompt_len + 5

    print("Passed.\n")


def main():
    test_empty_cache_length_is_zero()
    test_cache_append_increases_length()
    test_prefill_cache_shape()
    test_decode_appends_one_token()
    test_multiple_decode_steps_grow_cache()

    print("All Day 18 tests passed.")


if __name__ == "__main__":
    main()