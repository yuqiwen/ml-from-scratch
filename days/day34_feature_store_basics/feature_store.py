from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class FeatureValue:
    """
    One historical value for one entity.

    entity_id:
        identifier such as user_id

    feature_name:
        feature being stored

    event_time:
        time when this feature value became valid

    value:
        numeric feature value
    """

    entity_id: str
    feature_name: str
    event_time: datetime
    value: float


@dataclass(frozen=True)
class TrainingSample:
    """
    One training example requiring historical features.
    """

    sample_id: str
    entity_id: str
    sample_time: datetime


@dataclass(frozen=True)
class PointInTimeResult:
    sample_id: str
    entity_id: str
    sample_time: datetime
    feature_name: str
    feature_time: datetime | None
    feature_value: float | None


@dataclass(frozen=True)
class OnlineFeature:
    entity_id: str
    feature_name: str
    event_time: datetime
    value: float


class ToyFeatureStore:
    """
    Toy feature store with:

    - historical offline values
    - latest online values
    - point-in-time lookup
    - freshness checks
    """

    def __init__(self) -> None:
        self.offline_values: list[
            FeatureValue
        ] = []

        self.online_values: dict[
            tuple[str, str],
            OnlineFeature,
        ] = {}

    def write_offline(
        self,
        feature_value: FeatureValue,
    ) -> None:
        """
        Store one historical feature value.
        """
        self.offline_values.append(
            feature_value
        )

    def materialize_latest(
        self,
        entity_id: str,
        feature_name: str,
    ) -> OnlineFeature:
        """
        Publish the latest historical value
        into the online store.
        """
        candidates = [
            value
            for value in self.offline_values
            if (
                value.entity_id == entity_id
                and value.feature_name
                == feature_name
            )
        ]

        if not candidates:
            raise KeyError(
                "No offline feature values found."
            )

        latest = max(
            candidates,
            key=lambda value: value.event_time,
        )

        online_feature = OnlineFeature(
            entity_id=latest.entity_id,
            feature_name=latest.feature_name,
            event_time=latest.event_time,
            value=latest.value,
        )

        key = (
            entity_id,
            feature_name,
        )

        self.online_values[key] = (
            online_feature
        )

        return online_feature

    def get_online(
        self,
        entity_id: str,
        feature_name: str,
    ) -> OnlineFeature:
        """
        Fetch latest online feature.
        """
        key = (
            entity_id,
            feature_name,
        )

        if key not in self.online_values:
            raise KeyError(
                "Online feature not found."
            )

        return self.online_values[key]

    def point_in_time_lookup(
        self,
        sample: TrainingSample,
        feature_name: str,
    ) -> PointInTimeResult:
        """
        Select the latest feature value whose
        event_time is <= sample_time.
        """
        candidates = [
            value
            for value in self.offline_values
            if (
                value.entity_id
                == sample.entity_id
                and value.feature_name
                == feature_name
                and value.event_time
                <= sample.sample_time
            )
        ]

        if not candidates:
            return PointInTimeResult(
                sample_id=sample.sample_id,
                entity_id=sample.entity_id,
                sample_time=sample.sample_time,
                feature_name=feature_name,
                feature_time=None,
                feature_value=None,
            )

        selected = max(
            candidates,
            key=lambda value: value.event_time,
        )

        return PointInTimeResult(
            sample_id=sample.sample_id,
            entity_id=sample.entity_id,
            sample_time=sample.sample_time,
            feature_name=feature_name,
            feature_time=selected.event_time,
            feature_value=selected.value,
        )

    def is_stale(
        self,
        entity_id: str,
        feature_name: str,
        current_time: datetime,
        maximum_age: timedelta,
    ) -> bool:
        """
        Return True when the online value is older
        than the maximum allowed age.
        """
        online_feature = self.get_online(
            entity_id,
            feature_name,
        )

        age = (
            current_time
            - online_feature.event_time
        )

        return age > maximum_age
    
def run_demo() -> None:
    store = ToyFeatureStore()

    feature_name = "user_7d_click_count"

    store.write_offline(
        FeatureValue(
            entity_id="user-123",
            feature_name=feature_name,
            event_time=datetime(
                2026,
                1,
                5,
                10,
                0,
            ),
            value=10.0,
        )
    )

    store.write_offline(
        FeatureValue(
            entity_id="user-123",
            feature_name=feature_name,
            event_time=datetime(
                2026,
                1,
                15,
                10,
                0,
            ),
            value=20.0,
        )
    )

    store.write_offline(
        FeatureValue(
            entity_id="user-123",
            feature_name=feature_name,
            event_time=datetime(
                2026,
                1,
                25,
                10,
                0,
            ),
            value=30.0,
        )
    )

    sample_1 = TrainingSample(
        sample_id="sample-1",
        entity_id="user-123",
        sample_time=datetime(
            2026,
            1,
            10,
            12,
            0,
        ),
    )

    sample_2 = TrainingSample(
        sample_id="sample-2",
        entity_id="user-123",
        sample_time=datetime(
            2026,
            1,
            20,
            12,
            0,
        ),
    )

    result_1 = store.point_in_time_lookup(
        sample_1,
        feature_name,
    )

    result_2 = store.point_in_time_lookup(
        sample_2,
        feature_name,
    )

    print("Point-in-time lookup")
    print(result_1)
    print(result_2)
    print()

    online_feature = (
        store.materialize_latest(
            entity_id="user-123",
            feature_name=feature_name,
        )
    )

    print("Materialized online feature")
    print(online_feature)
    print()

    stale = store.is_stale(
        entity_id="user-123",
        feature_name=feature_name,
        current_time=datetime(
            2026,
            1,
            25,
            12,
            30,
        ),
        maximum_age=timedelta(
            hours=1
        ),
    )

    print(f"is_stale = {stale}")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()