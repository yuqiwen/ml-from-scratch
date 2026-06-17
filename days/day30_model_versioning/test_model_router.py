from model_router import ModelRouter
from model_versions import (
    MODEL_V1,
    MODEL_V2,
)


def test_invalid_canary_percentage() -> None:
    print("Test 1: invalid canary percentage")

    try:
        ModelRouter(
            stable_model=MODEL_V1,
            canary_model=MODEL_V2,
            canary_percentage=-1,
        )
        raised_negative = False
    except ValueError:
        raised_negative = True

    try:
        ModelRouter(
            stable_model=MODEL_V1,
            canary_model=MODEL_V2,
            canary_percentage=101,
        )
        raised_large = False
    except ValueError:
        raised_large = True

    assert raised_negative
    assert raised_large

    print("Passed.\n")


def test_zero_percent_routes_to_stable() -> None:
    router = ModelRouter(
        stable_model=MODEL_V1,
        canary_model=MODEL_V2,
        canary_percentage=0.0,
    )

    print("Test 2: zero percent routes to stable")

    for index in range(100):
        result = router.route_sticky(
            f"user-{index}"
        )

        assert (
            result.model_version
            == MODEL_V1.version
        )

    print("Passed.\n")


def test_hundred_percent_routes_to_canary() -> None:
    router = ModelRouter(
        stable_model=MODEL_V1,
        canary_model=MODEL_V2,
        canary_percentage=100.0,
    )

    print(
        "Test 3: hundred percent routes to canary"
    )

    for index in range(100):
        result = router.route_sticky(
            f"user-{index}"
        )

        assert (
            result.model_version
            == MODEL_V2.version
        )

    print("Passed.\n")


def test_sticky_routing_is_stable() -> None:
    router = ModelRouter(
        stable_model=MODEL_V1,
        canary_model=MODEL_V2,
        canary_percentage=25.0,
    )

    user_id = "same-user"

    versions = {
        router.route_sticky(
            user_id
        ).model_version
        for _ in range(20)
    }

    print("Test 4: sticky routing is stable")
    print(versions)

    assert len(versions) == 1

    print("Passed.\n")


def test_rollback() -> None:
    router = ModelRouter(
        stable_model=MODEL_V1,
        canary_model=MODEL_V2,
        canary_percentage=50.0,
    )

    router.rollback()

    print("Test 5: rollback")

    assert router.canary_percentage == 0.0

    for index in range(100):
        result = router.route_sticky(
            f"user-{index}"
        )

        assert (
            result.model_version
            == MODEL_V1.version
        )

    print("Passed.\n")


def test_promote_canary() -> None:
    router = ModelRouter(
        stable_model=MODEL_V1,
        canary_model=MODEL_V2,
        canary_percentage=50.0,
    )

    router.promote_canary()

    print("Test 6: promote canary")

    assert (
        router.stable_model.version
        == MODEL_V2.version
    )

    assert router.canary_percentage == 0.0

    for index in range(100):
        result = router.route_sticky(
            f"user-{index}"
        )

        assert (
            result.model_version
            == MODEL_V2.version
        )

    print("Passed.\n")


def main() -> None:
    test_invalid_canary_percentage()
    test_zero_percent_routes_to_stable()
    test_hundred_percent_routes_to_canary()
    test_sticky_routing_is_stable()
    test_rollback()
    test_promote_canary()

    print("All Day 30 tests passed.")


if __name__ == "__main__":
    main()