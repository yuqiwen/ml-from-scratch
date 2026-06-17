import hashlib
import random
from dataclasses import dataclass

from model_versions import (
    MODEL_V1,
    MODEL_V2,
    ModelVersion,
)


@dataclass
class RoutingResult:
    user_id: str
    model_version: str
    routing_strategy: str


class ModelRouter:
    """
    Toy router supporting:

    - random canary routing
    - sticky routing
    - rollback
    """

    def __init__(
        self,
        stable_model: ModelVersion,
        canary_model: ModelVersion,
        canary_percentage: float = 0.0,
    ):
        self.stable_model = stable_model
        self.canary_model = canary_model

        self.set_canary_percentage(
            canary_percentage
        )

    def set_canary_percentage(
        self,
        percentage: float,
    ) -> None:
        """
        Set traffic percentage sent to canary.

        Valid range:
            0.0 to 100.0
        """
        if percentage < 0 or percentage > 100:
            raise ValueError(
                "percentage must be between 0 and 100."
            )

        self.canary_percentage = percentage

    def route_random(
        self,
        user_id: str,
    ) -> RoutingResult:
        """
        Random routing.

        The same user may receive different versions
        across different requests.
        """
        draw = random.random() * 100

        if draw < self.canary_percentage:
            selected = self.canary_model
        else:
            selected = self.stable_model

        return RoutingResult(
            user_id=user_id,
            model_version=selected.version,
            routing_strategy="random",
        )

    def route_sticky(
        self,
        user_id: str,
    ) -> RoutingResult:
        """
        Sticky routing based on stable hash.

        The same user_id always maps to the same bucket.
        """
        if not user_id:
            raise ValueError(
                "user_id must not be empty."
            )

        digest = hashlib.sha256(
            user_id.encode("utf-8")
        ).hexdigest()

        bucket = int(
            digest[:8],
            16,
        ) % 10000

        canary_threshold = int(
            self.canary_percentage * 100
        )

        if bucket < canary_threshold:
            selected = self.canary_model
        else:
            selected = self.stable_model

        return RoutingResult(
            user_id=user_id,
            model_version=selected.version,
            routing_strategy="sticky",
        )

    def rollback(self) -> None:
        """
        Remove all traffic from canary model.
        """
        self.canary_percentage = 0.0

    def promote_canary(self) -> None:
        """
        Promote canary to stable.

        Old canary becomes the new stable model.
        """
        self.stable_model = self.canary_model
        self.canary_percentage = 0.0


def random_routing_demo() -> None:
    router = ModelRouter(
        stable_model=MODEL_V1,
        canary_model=MODEL_V2,
        canary_percentage=20.0,
    )

    counts = {
        "v1": 0,
        "v2": 0,
    }

    for request_id in range(1000):
        result = router.route_random(
            user_id=f"user-{request_id}"
        )

        counts[result.model_version] += 1

    print("Random canary routing demo")
    print(counts)
    print()


def sticky_routing_demo() -> None:
    router = ModelRouter(
        stable_model=MODEL_V1,
        canary_model=MODEL_V2,
        canary_percentage=20.0,
    )

    user_id = "user-123"

    results = [
        router.route_sticky(user_id)
        for _ in range(5)
    ]

    print("Sticky routing demo")

    for result in results:
        print(result)

    print()


def rollback_demo() -> None:
    router = ModelRouter(
        stable_model=MODEL_V1,
        canary_model=MODEL_V2,
        canary_percentage=50.0,
    )

    print("Before rollback")
    print(
        f"canary_percentage="
        f"{router.canary_percentage}"
    )

    router.rollback()

    print("After rollback")
    print(
        f"canary_percentage="
        f"{router.canary_percentage}"
    )
    print()


def main() -> None:
    random_routing_demo()
    sticky_routing_demo()
    rollback_demo()


if __name__ == "__main__":
    main()