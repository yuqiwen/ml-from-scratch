import torch

from two_tower import (
    TwoTowerModel,
    generate_synthetic_batch,
    offline_encode_items,
    retrieval_accuracy,
    retrieve_top_k,
)


def build_model() -> TwoTowerModel:
    return TwoTowerModel(
        user_input_dim=8,
        item_input_dim=8,
        hidden_dim=16,
        embedding_dim=4,
        temperature=0.1,
    )


def test_tower_output_shapes() -> None:
    model = build_model()

    user_features = torch.randn(
        5,
        8,
    )

    item_features = torch.randn(
        5,
        8,
    )

    user_embeddings = model.encode_users(
        user_features
    )

    item_embeddings = model.encode_items(
        item_features
    )

    print("Test 1: tower output shapes")
    print(user_embeddings.shape)
    print(item_embeddings.shape)

    assert user_embeddings.shape == (
        5,
        4,
    )

    assert item_embeddings.shape == (
        5,
        4,
    )

    print("Passed.\n")


def test_embeddings_are_normalized() -> None:
    model = build_model()

    embeddings = model.encode_users(
        torch.randn(6, 8)
    )

    norms = torch.linalg.vector_norm(
        embeddings,
        dim=1,
    )

    print(
        "Test 2: embeddings normalized"
    )
    print(norms)

    assert torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=1e-5,
    )

    print("Passed.\n")


def test_similarity_matrix_shape() -> None:
    model = build_model()

    logits = model.similarity_matrix(
        torch.randn(7, 8),
        torch.randn(7, 8),
    )

    print(
        "Test 3: similarity matrix shape"
    )
    print(logits.shape)

    assert logits.shape == (
        7,
        7,
    )

    print("Passed.\n")


def test_loss_is_scalar() -> None:
    model = build_model()

    loss, logits = model(
        torch.randn(4, 8),
        torch.randn(4, 8),
    )

    print("Test 4: loss is scalar")
    print(loss)

    assert loss.ndim == 0
    assert logits.shape == (
        4,
        4,
    )

    print("Passed.\n")


def test_retrieval_accuracy() -> None:
    logits = torch.tensor(
        [
            [5.0, 1.0, 0.0],
            [0.0, 4.0, 1.0],
            [1.0, 0.0, 3.0],
        ]
    )

    accuracy = retrieval_accuracy(
        logits
    )

    print(
        "Test 5: retrieval accuracy"
    )
    print(accuracy)

    assert accuracy == 1.0

    print("Passed.\n")


def test_offline_item_encoding_shape() -> None:
    model = build_model()

    item_features = torch.randn(
        10,
        8,
    )

    embeddings = offline_encode_items(
        model,
        item_features,
    )

    print(
        "Test 6: offline item encoding"
    )
    print(embeddings.shape)

    assert embeddings.shape == (
        10,
        4,
    )

    print("Passed.\n")


def test_retrieve_top_k() -> None:
    torch.manual_seed(0)

    model = build_model()

    batch = generate_synthetic_batch(
        batch_size=5,
        feature_dim=8,
    )

    item_embeddings = offline_encode_items(
        model,
        batch.item_features,
    )

    indices, scores = retrieve_top_k(
        model=model,
        user_features=(
            batch.user_features[0:1]
        ),
        item_embeddings=item_embeddings,
        top_k=3,
    )

    print("Test 7: retrieve top-k")
    print(indices)
    print(scores)

    assert indices.shape == (3,)
    assert scores.shape == (3,)

    print("Passed.\n")


def main() -> None:
    test_tower_output_shapes()
    test_embeddings_are_normalized()
    test_similarity_matrix_shape()
    test_loss_is_scalar()
    test_retrieval_accuracy()
    test_offline_item_encoding_shape()
    test_retrieve_top_k()

    print(
        "All Day 37 tests passed."
    )


if __name__ == "__main__":
    main()