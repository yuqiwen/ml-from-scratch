from dataclasses import dataclass


@dataclass(frozen=True)
class ModelVersion:
    """
    Metadata describing one model version.
    """

    name: str
    version: str
    accuracy: float
    latency_ms: float
    stage: str


MODEL_V1 = ModelVersion(
    name="tiny-regression",
    version="v1",
    accuracy=0.90,
    latency_ms=3.0,
    stage="production",
)


MODEL_V2 = ModelVersion(
    name="tiny-regression",
    version="v2",
    accuracy=0.93,
    latency_ms=4.5,
    stage="canary",
)