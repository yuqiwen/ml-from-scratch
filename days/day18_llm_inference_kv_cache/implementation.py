import math

import torch
from torch import nn


class KVCache:
    """
    A simple KV cache for one Transformer layer.

    Stores:
        K_cache: (B, H, past_len, Hd)
        V_cache: (B, H, past_len, Hd)
    """

    def __init__(self):
        self.K: torch.Tensor | None = None
        self.V: torch.Tensor | None = None

    def append(
        self,
        K_new: torch.Tensor,
        V_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new K/V to cache.

        Args:
            K_new: (B, H, new_len, Hd)
            V_new: (B, H, new_len, Hd)

        Returns:
            K_cache: (B, H, past_len + new_len, Hd)
            V_cache: (B, H, past_len + new_len, Hd)
        """
        if self.K is None:
            self.K = K_new
            self.V = V_new
        else:
            self.K = torch.cat([self.K, K_new], dim=2)
            self.V = torch.cat([self.V, V_new], dim=2)

        return self.K, self.V

    @property
    def length(self) -> int:
        """
        Number of cached tokens.
        """
        if self.K is None:
            return 0
        return self.K.shape[2]


class CachedSelfAttention(nn.Module):
    """
    A minimal causal self-attention module with KV cache support.

    This is simplified for inference explanation.

    Input:
        X: (B, T_new, D)

    During prefill:
        T_new = prompt length

    During decode:
        T_new = 1
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
        (B, T, D) -> (B, H, T, Hd)
        """
        B, T, _ = X.shape
        X = X.view(B, T, self.num_heads, self.head_dim)
        X = X.transpose(1, 2)
        return X

    def concat_heads(self, X: torch.Tensor) -> torch.Tensor:
        """
        (B, H, T, Hd) -> (B, T, D)
        """
        B, H, T, Hd = X.shape
        X = X.transpose(1, 2).contiguous()
        X = X.view(B, T, H * Hd)
        return X

    def forward(
        self,
        X: torch.Tensor,
        cache: KVCache,
        is_prefill: bool,
    ) -> tuple[torch.Tensor, KVCache, torch.Tensor]:
        """
        Args:
            X: (B, T_new, D)
            cache: KVCache
            is_prefill: whether this is the full prompt stage

        Returns:
            output: (B, T_new, D)
            cache: updated KVCache
            attention_weights: (B, H, T_new, total_len)
        """
        B, T_new, _ = X.shape

        Q_new = self.split_heads(self.q_proj(X))
        K_new = self.split_heads(self.k_proj(X))
        V_new = self.split_heads(self.v_proj(X))

        K_cache, V_cache = cache.append(K_new, V_new)

        scores = Q_new @ K_cache.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        # During prefill, T_new == total_len, so we need a causal mask.
        # During decode, T_new is usually 1, and K_cache only contains past + current,
        # so there are no future tokens to mask.
        if is_prefill:
            total_len = K_cache.shape[2]
            causal_mask = torch.tril(
                torch.ones(T_new, total_len, dtype=torch.bool, device=X.device)
            )
            scores = scores.masked_fill(~causal_mask, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)

        head_output = attention_weights @ V_cache
        concat_output = self.concat_heads(head_output)
        output = self.out_proj(concat_output)

        return output, cache, attention_weights


def prefill_decode_demo() -> None:
    """
    Demonstrate prefill and decode cache growth.
    """
    torch.manual_seed(42)

    B = 1
    prompt_len = 4
    D = 8
    H = 2

    attention = CachedSelfAttention(d_model=D, num_heads=H)
    cache = KVCache()

    prompt_hidden = torch.randn(B, prompt_len, D)

    print("Prefill")
    output, cache, weights = attention(
        X=prompt_hidden,
        cache=cache,
        is_prefill=True,
    )

    print(f"prompt_hidden shape = {prompt_hidden.shape}")
    print(f"output shape        = {output.shape}")
    print(f"K cache shape       = {cache.K.shape}")
    print(f"V cache shape       = {cache.V.shape}")
    print(f"cache length        = {cache.length}")
    print(f"attention shape     = {weights.shape}")
    print()

    for step in range(3):
        new_token_hidden = torch.randn(B, 1, D)

        output, cache, weights = attention(
            X=new_token_hidden,
            cache=cache,
            is_prefill=False,
        )

        print(f"Decode step {step + 1}")
        print(f"new token hidden shape = {new_token_hidden.shape}")
        print(f"output shape           = {output.shape}")
        print(f"K cache shape          = {cache.K.shape}")
        print(f"cache length           = {cache.length}")
        print(f"attention shape        = {weights.shape}")
        print()


def main() -> None:
    prefill_decode_demo()


if __name__ == "__main__":
    main()