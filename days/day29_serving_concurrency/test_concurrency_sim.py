import asyncio

from concurrency_sim import (
    InferenceService,
    Request,
)


async def test_single_request_accepted() -> None:
    service = InferenceService(
        max_queue_size=2,
        max_concurrency=1,
    )

    await service.start(
        num_workers=1
    )

    result = await service.submit(
        Request(
            request_id=1,
            processing_time_ms=10,
        )
    )

    print("Test 1: single request accepted")
    print(result)

    assert result.accepted is True
    assert result.error is None
    assert result.start_time_ms is not None
    assert result.end_time_ms is not None

    await service.stop()

    print("Passed.\n")


async def test_invalid_configuration() -> None:
    print("Test 2: invalid configuration")

    try:
        InferenceService(
            max_queue_size=0,
            max_concurrency=1,
        )
        raised_queue = False
    except ValueError:
        raised_queue = True

    try:
        InferenceService(
            max_queue_size=1,
            max_concurrency=0,
        )
        raised_concurrency = False
    except ValueError:
        raised_concurrency = True

    assert raised_queue
    assert raised_concurrency

    print("Passed.\n")


async def test_concurrency_limit() -> None:
    service = InferenceService(
        max_queue_size=10,
        max_concurrency=2,
    )

    await service.start(
        num_workers=4
    )

    requests = [
        Request(
            request_id=i,
            processing_time_ms=50,
        )
        for i in range(4)
    ]

    tasks = [
        asyncio.create_task(
            service.submit(request)
        )
        for request in requests
    ]

    results = await asyncio.gather(
        *tasks
    )

    print("Test 3: concurrency limit")
    for result in results:
        print(result)

    assert all(
        result.accepted
        for result in results
    )

    intervals = [
        (
            result.start_time_ms,
            result.end_time_ms,
        )
        for result in results
    ]

    max_overlap = 0

    for i, (
        start_i,
        end_i,
    ) in enumerate(intervals):
        overlap = 0

        for start_j, end_j in intervals:
            if (
                start_j < end_i
                and end_j > start_i
            ):
                overlap += 1

        max_overlap = max(
            max_overlap,
            overlap,
        )

    assert max_overlap <= 2

    await service.stop()

    print("Passed.\n")


async def test_queue_overload_rejection() -> None:
    service = InferenceService(
        max_queue_size=1,
        max_concurrency=1,
    )

    # No worker is started yet.
    # First request fills the queue.
    first_task = asyncio.create_task(
        service.submit(
            Request(
                request_id=1,
                processing_time_ms=100,
            )
        )
    )

    await asyncio.sleep(0)

    second_result = await service.submit(
        Request(
            request_id=2,
            processing_time_ms=100,
        )
    )

    print("Test 4: queue overload")
    print(second_result)

    assert second_result.accepted is False
    assert second_result.error == "queue_full"

    await service.start(
        num_workers=1
    )

    first_result = await first_task

    assert first_result.accepted is True

    await service.stop()

    print("Passed.\n")


async def run_all_tests() -> None:
    await test_single_request_accepted()
    await test_invalid_configuration()
    await test_concurrency_limit()
    await test_queue_overload_rejection()

    print("All Day 29 tests passed.")


def main() -> None:
    asyncio.run(run_all_tests())


if __name__ == "__main__":
    main()