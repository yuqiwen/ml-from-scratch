import math

import torch
from torch import nn


def create_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Create lower-triangular causal mask.

    Shape:
        (seq_len, seq_len)
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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, T, _ = X.shape

        Q = self.split_heads(self.q_proj(X))
        K = self.split_heads(self.k_proj(X))
        V = self.split_heads(self.v_proj(X))

        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        mask = create_causal_mask(T, device=X.device)
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)

        head_output = weights @ V
        concat_output = self.concat_heads(head_output)
        output = self.out_proj(concat_output)

        return output


class FeedForwardNetwork(nn.Module):
    """
    Transformer FFN:
        Linear(D, ffn_dim)
        GELU
        Linear(ffn_dim, D)
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
    GPT-style Pre-LN Transformer block.

    x = x + attention(layer_norm(x))
    x = x + ffn(layer_norm(x))
    """

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int):
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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X + self.attention(self.ln1(X))
        X = X + self.ffn(self.ln2(X))
        return X


class TinyGPT(nn.Module):
    """
    A tiny GPT-style decoder-only language model.

    Input:
        input_ids: (B, T)

    Output:
        logits: (B, T, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_ln = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T)

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape

        if T > self.max_seq_len:
            raise ValueError("Sequence length exceeds max_seq_len.")

        positions = torch.arange(T, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(B, T)

        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)

        X = token_emb + pos_emb

        for block in self.blocks:
            X = block(X)

        X = self.final_ln(X)

        logits = self.lm_head(X)

        return logits


def create_next_token_inputs_targets(token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create input and target tensors for next-token prediction.

    Example:
        token_ids: [10, 20, 30, 40, 50]

        input_ids:  [10, 20, 30, 40]
        target_ids: [20, 30, 40, 50]

    Args:
        token_ids: (B, T)

    Returns:
        input_ids:  (B, T-1)
        target_ids: (B, T-1)
    """
    input_ids = token_ids[:, :-1]
    target_ids = token_ids[:, 1:]

    return input_ids, target_ids


def compute_lm_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute language modeling cross entropy loss.

    Args:
        logits:  (B, T, vocab_size)
        targets: (B, T)

    Returns:
        scalar loss
    """
    B, T, vocab_size = logits.shape

    logits_flat = logits.view(B * T, vocab_size)
    targets_flat = targets.reshape(B * T)

    loss_fn = nn.CrossEntropyLoss()

    loss = loss_fn(logits_flat, targets_flat)

    return loss


def tiny_gpt_demo() -> None:
    """
    Run a tiny GPT forward and loss demo.
    """
    torch.manual_seed(42)

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

    token_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(4, 8),
    )

    input_ids, target_ids = create_next_token_inputs_targets(token_ids)

    logits = model(input_ids)
    loss = compute_lm_loss(logits, target_ids)

    print("TinyGPT demo")
    print(f"token_ids shape  = {token_ids.shape}")
    print(f"input_ids shape  = {input_ids.shape}")
    print(f"target_ids shape = {target_ids.shape}")
    print(f"logits shape     = {logits.shape}")
    print(f"loss             = {loss.item():.6f}")


def main() -> None:
    tiny_gpt_demo()


if __name__ == "__main__":
    main()