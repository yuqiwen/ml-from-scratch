from dataclasses import dataclass


@dataclass(frozen=True)
class RawUserFeatures:
    age: float | None
    income_usd: float | None
    country: str | None


@dataclass(frozen=True)
class ProcessedFeatures:
    age_normalized: float
    income_normalized: float
    country_id: int


@dataclass(frozen=True)
class FeatureConfig:
    age_mean: float
    age_std: float
    income_mean: float
    income_std: float
    country_mapping: dict[str, int]
    unknown_country_id: int


DEFAULT_CONFIG = FeatureConfig(
    age_mean=35.0,
    age_std=10.0,
    income_mean=70_000.0,
    income_std=30_000.0,
    country_mapping={
        "US": 0,
        "CN": 1,
        "IN": 2,
    },
    unknown_country_id=3,
)


def safe_normalize(
    value: float | None,
    mean: float,
    std: float,
    missing_value: float = 0.0,
) -> float:
    """
    Normalize one numeric feature.

    Missing values use a configured default.
    """
    if value is None:
        return missing_value

    if std <= 0:
        raise ValueError(
            "std must be positive."
        )

    return (value - mean) / std


def preprocess_shared(
    raw: RawUserFeatures,
    config: FeatureConfig = DEFAULT_CONFIG,
) -> ProcessedFeatures:
    """
    Correct preprocessing shared by training and serving.
    """
    age_normalized = safe_normalize(
        value=raw.age,
        mean=config.age_mean,
        std=config.age_std,
    )

    income_normalized = safe_normalize(
        value=raw.income_usd,
        mean=config.income_mean,
        std=config.income_std,
    )

    country_id = config.country_mapping.get(
        raw.country or "",
        config.unknown_country_id,
    )

    return ProcessedFeatures(
        age_normalized=age_normalized,
        income_normalized=income_normalized,
        country_id=country_id,
    )


def preprocess_buggy_online(
    raw: RawUserFeatures,
    config: FeatureConfig = DEFAULT_CONFIG,
) -> ProcessedFeatures:
    """
    Intentionally buggy online preprocessing.

    Bugs:
    - age is not normalized
    - income is interpreted as cents
    - country mapping is inconsistent
    """
    age_normalized = (
        0.0
        if raw.age is None
        else raw.age
    )

    income_in_cents = (
        None
        if raw.income_usd is None
        else raw.income_usd * 100.0
    )

    income_normalized = safe_normalize(
        value=income_in_cents,
        mean=config.income_mean,
        std=config.income_std,
    )

    buggy_mapping = {
        "CN": 0,
        "US": 1,
        "IN": 2,
    }

    country_id = buggy_mapping.get(
        raw.country or "",
        config.unknown_country_id,
    )

    return ProcessedFeatures(
        age_normalized=age_normalized,
        income_normalized=income_normalized,
        country_id=country_id,
    )