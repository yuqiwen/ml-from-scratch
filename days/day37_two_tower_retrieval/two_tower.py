from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RetrievalBatch:
    user_features: torch.Tensor
    item_features: torch.Tensor


class Tower(nn.Module):
    """
    Small MLP tower.

    Input:
        features: (B, input_dim)

    Output:
        embedding: (B, embedding_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.network(features)

        return F.normalize(
            embedding,
            p=2,
            dim=-1,
        )


class TwoTowerModel(nn.Module):
    """
    Two-Tower retrieval model.

    User and item towers are separate networks,
    but both output embedding_dim vectors.
    """

    def __init__(
        self,
        user_input_dim: int,
        item_input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        temperature: float = 0.1,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError(
                "temperature must be positive."
            )

        self.user_tower = Tower(
            input_dim=user_input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
        )

        self.item_tower = Tower(
            input_dim=item_input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
        )

        self.temperature = temperature

    def encode_users(
        self,
        user_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.user_tower(
            user_features
        )

    def encode_items(
        self,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.item_tower(
            item_features
        )

    def similarity_matrix(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return all pairwise user-item similarities.

        user embeddings:
            (B, D)

        item embeddings:
            (B, D)

        logits:
            (B, B)
        """
        user_embeddings = self.encode_users(
            user_features
        )

        item_embeddings = self.encode_items(
            item_features
        )

        logits = (
            user_embeddings
            @ item_embeddings.T
        ) / self.temperature

        return logits

    def forward(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        logits = self.similarity_matrix(
            user_features,
            item_features,
        )

        batch_size = logits.shape[0]

        labels = torch.arange(
            batch_size,
            device=logits.device,
        )

        user_to_item_loss = (
            F.cross_entropy(
                logits,
                labels,
            )
        )

        item_to_user_loss = (
            F.cross_entropy(
                logits.T,
                labels,
            )
        )

        loss = (
            user_to_item_loss
            + item_to_user_loss
        ) / 2.0

        return loss, logits


def generate_synthetic_batch(
    batch_size: int = 64,
    feature_dim: int = 8,
    noise_std: float = 0.05,
) -> RetrievalBatch:
    """
    Generate aligned user-item positive pairs.

    Each item feature is close to its matching
    user feature.
    """
    latent = torch.randn(
        batch_size,
        feature_dim,
    )

    user_features = (
        latent
        + noise_std
        * torch.randn_like(latent)
    )

    item_features = (
        latent
        + noise_std
        * torch.randn_like(latent)
    )

    return RetrievalBatch(
        user_features=user_features,
        item_features=item_features,
    )


def retrieval_accuracy(
    logits: torch.Tensor,
) -> float:
    """
    Top-1 retrieval accuracy for aligned pairs.
    """
    predictions = logits.argmax(
        dim=1
    )

    labels = torch.arange(
        logits.shape[0],
        device=logits.device,
    )

    return (
        predictions == labels
    ).float().mean().item()


def train_demo(
    num_steps: int = 300,
) -> TwoTowerModel:
    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = TwoTowerModel(
        user_input_dim=8,
        item_input_dim=8,
        hidden_dim=32,
        embedding_dim=16,
        temperature=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.003,
    )

    for step in range(num_steps):
        batch = generate_synthetic_batch(
            batch_size=64,
            feature_dim=8,
        )

        user_features = (
            batch.user_features.to(device)
        )

        item_features = (
            batch.item_features.to(device)
        )

        model.train()

        loss, logits = model(
            user_features,
            item_features,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            accuracy = retrieval_accuracy(
                logits
            )

            print(
                f"step={step:03d}, "
                f"loss={loss.item():.4f}, "
                f"retrieval_accuracy="
                f"{accuracy:.2%}"
            )

    return model


@torch.inference_mode()
def offline_encode_items(
    model: TwoTowerModel,
    item_features: torch.Tensor,
) -> torch.Tensor:
    """
    Simulate offline item embedding generation.
    """
    model.eval()

    device = next(
        model.parameters()
    ).device

    return model.encode_items(
        item_features.to(device)
    )


@torch.inference_mode()
def retrieve_top_k(
    model: TwoTowerModel,
    user_features: torch.Tensor,
    item_embeddings: torch.Tensor,
    top_k: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Exact retrieval from precomputed item embeddings.

    user_features:
        (1, user_input_dim)

    item_embeddings:
        (N, embedding_dim)
    """
    if top_k <= 0:
        raise ValueError(
            "top_k must be positive."
        )

    model.eval()

    device = next(
        model.parameters()
    ).device

    user_embedding = model.encode_users(
        user_features.to(device)
    )

    item_embeddings = (
        item_embeddings.to(device)
    )

    scores = (
        user_embedding
        @ item_embeddings.T
    ).squeeze(0)

    top_k = min(
        top_k,
        scores.numel(),
    )

    top_scores, top_indices = (
        torch.topk(
            scores,
            k=top_k,
        )
    )

    return top_indices, top_scores


def main() -> None:
    model = train_demo()

    catalog = generate_synthetic_batch(
        batch_size=20,
        feature_dim=8,
        noise_std=0.02,
    )

    item_embeddings = offline_encode_items(
        model,
        catalog.item_features,
    )

    query_user = (
        catalog.user_features[0:1]
    )

    top_indices, top_scores = (
        retrieve_top_k(
            model=model,
            user_features=query_user,
            item_embeddings=item_embeddings,
            top_k=5,
        )
    )

    print()
    print("Top-k retrieval")
    print(
        "indices =",
        top_indices.tolist(),
    )
    print(
        "scores =",
        [
            round(score, 4)
            for score
            in top_scores.tolist()
        ],
    )

    print(
        "Expected matching item index = 0"
    )


if __name__ == "__main__":
    main()