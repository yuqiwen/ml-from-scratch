from dataclasses import dataclass
import math


Vector = list[float]


@dataclass(frozen=True)
class ItemEmbedding:
    item_id: int
    embedding: Vector


@dataclass(frozen=True)
class SearchResult:
    item_id: int
    score: float


def validate_vector(
    vector: Vector,
    name: str,
) -> None:
    if not vector:
        raise ValueError(
            f"{name} must not be empty."
        )

    for value in vector:
        if not math.isfinite(value):
            raise ValueError(
                f"{name} must contain only "
                "finite values."
            )


def dot_product(
    left: Vector,
    right: Vector,
) -> float:
    """
    Calculate vector dot product.
    """
    validate_vector(left, "left")
    validate_vector(right, "right")

    if len(left) != len(right):
        raise ValueError(
            "Vectors must have the same dimension."
        )

    return sum(
        left_value * right_value
        for left_value, right_value
        in zip(left, right)
    )


def l2_norm(
    vector: Vector,
) -> float:
    """
    Calculate Euclidean vector norm.
    """
    validate_vector(vector, "vector")

    return math.sqrt(
        sum(
            value * value
            for value in vector
        )
    )


def normalize(
    vector: Vector,
    epsilon: float = 1e-12,
) -> Vector:
    """
    Return an L2-normalized vector.
    """
    norm = l2_norm(vector)

    if norm <= epsilon:
        raise ValueError(
            "Cannot normalize a zero vector."
        )

    return [
        value / norm
        for value in vector
    ]


def cosine_similarity(
    left: Vector,
    right: Vector,
) -> float:
    """
    Calculate cosine similarity.
    """
    denominator = (
        l2_norm(left)
        * l2_norm(right)
    )

    if denominator <= 1e-12:
        raise ValueError(
            "Cosine similarity is undefined "
            "for a zero vector."
        )

    return (
        dot_product(left, right)
        / denominator
    )


def exact_search(
    query_embedding: Vector,
    items: list[ItemEmbedding],
    top_k: int,
    metric: str = "cosine",
) -> list[SearchResult]:
    """
    Scan every item and return the top-k results.

    This is exact brute-force search.
    """
    if top_k <= 0:
        raise ValueError(
            "top_k must be positive."
        )

    if metric not in {
        "cosine",
        "dot",
    }:
        raise ValueError(
            "metric must be 'cosine' or 'dot'."
        )

    results: list[SearchResult] = []

    for item in items:
        if metric == "cosine":
            score = cosine_similarity(
                query_embedding,
                item.embedding,
            )
        else:
            score = dot_product(
                query_embedding,
                item.embedding,
            )

        results.append(
            SearchResult(
                item_id=item.item_id,
                score=score,
            )
        )

    results.sort(
        key=lambda result: result.score,
        reverse=True,
    )

    return results[:top_k]


class ToyClusterIndex:
    """
    A very small ANN-like index.

    This is not a production ANN algorithm.

    It groups items into predefined clusters and
    searches only the most similar cluster.
    """

    def __init__(
        self,
        cluster_centers: dict[str, Vector],
    ):
        if not cluster_centers:
            raise ValueError(
                "cluster_centers must not be empty."
            )

        self.cluster_centers = (
            cluster_centers
        )

        self.items_by_cluster: dict[
            str,
            list[ItemEmbedding],
        ] = {
            cluster_name: []
            for cluster_name
            in cluster_centers
        }

    def add_item(
        self,
        item: ItemEmbedding,
    ) -> str:
        """
        Assign an item to its most similar cluster.
        """
        best_cluster = max(
            self.cluster_centers,
            key=lambda cluster_name:
                cosine_similarity(
                    item.embedding,
                    self.cluster_centers[
                        cluster_name
                    ],
                ),
        )

        self.items_by_cluster[
            best_cluster
        ].append(item)

        return best_cluster

    def search(
        self,
        query_embedding: Vector,
        top_k: int,
        num_clusters_to_search: int = 1,
    ) -> list[SearchResult]:
        """
        Search only the nearest clusters.

        Searching fewer clusters is faster but may
        miss relevant items.
        """
        if num_clusters_to_search <= 0:
            raise ValueError(
                "num_clusters_to_search "
                "must be positive."
            )

        cluster_scores = []

        for cluster_name, center in (
            self.cluster_centers.items()
        ):
            score = cosine_similarity(
                query_embedding,
                center,
            )

            cluster_scores.append(
                (
                    score,
                    cluster_name,
                )
            )

        cluster_scores.sort(
            reverse=True
        )

        selected_clusters = [
            cluster_name
            for _, cluster_name
            in cluster_scores[
                :num_clusters_to_search
            ]
        ]

        candidate_items: list[
            ItemEmbedding
        ] = []

        for cluster_name in selected_clusters:
            candidate_items.extend(
                self.items_by_cluster[
                    cluster_name
                ]
            )

        return exact_search(
            query_embedding=query_embedding,
            items=candidate_items,
            top_k=top_k,
            metric="cosine",
        )


def recall_at_k(
    exact_results: list[SearchResult],
    approximate_results: list[SearchResult],
    k: int,
) -> float:
    """
    Compare approximate top-k with exact top-k.
    """
    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    exact_ids = {
        result.item_id
        for result in exact_results[:k]
    }

    approximate_ids = {
        result.item_id
        for result
        in approximate_results[:k]
    }

    if not exact_ids:
        return 0.0

    return (
        len(
            exact_ids
            & approximate_ids
        )
        / len(exact_ids)
    )


def build_demo_items() -> list[ItemEmbedding]:
    return [
        ItemEmbedding(
            item_id=1,
            embedding=[0.95, 0.05, 0.10],
        ),
        ItemEmbedding(
            item_id=2,
            embedding=[0.85, 0.10, 0.20],
        ),
        ItemEmbedding(
            item_id=3,
            embedding=[0.10, 0.90, 0.10],
        ),
        ItemEmbedding(
            item_id=4,
            embedding=[0.05, 0.85, 0.20],
        ),
        ItemEmbedding(
            item_id=5,
            embedding=[0.20, 0.10, 0.90],
        ),
        ItemEmbedding(
            item_id=6,
            embedding=[0.30, 0.20, 0.80],
        ),
    ]


def run_demo() -> None:
    items = build_demo_items()

    query = [
        0.90,
        0.10,
        0.15,
    ]

    exact_results = exact_search(
        query_embedding=query,
        items=items,
        top_k=3,
        metric="cosine",
    )

    print("Exact search")
    for result in exact_results:
        print(result)
    print()

    index = ToyClusterIndex(
        cluster_centers={
            "camera": [1.0, 0.0, 0.0],
            "food": [0.0, 1.0, 0.0],
            "travel": [0.0, 0.0, 1.0],
        }
    )

    for item in items:
        cluster = index.add_item(item)

        print(
            f"item={item.item_id} "
            f"assigned_to={cluster}"
        )

    print()

    approximate_results = index.search(
        query_embedding=query,
        top_k=3,
        num_clusters_to_search=1,
    )

    print("Approximate cluster search")
    for result in approximate_results:
        print(result)

    print()

    recall = recall_at_k(
        exact_results=exact_results,
        approximate_results=(
            approximate_results
        ),
        k=3,
    )

    print(f"Recall@3 = {recall:.2f}")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()