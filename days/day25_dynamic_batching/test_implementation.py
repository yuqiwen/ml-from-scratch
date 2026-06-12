from implementation import (
    DynamicBatcher,
    LLMRequest,
    Request,
    simulate_continuous_batching,
    simulate_dynamic_batching,
)


def test_dispatch_when_max_batch_size_reached():
    batcher = DynamicBatcher(max_batch_size=3, max_wait_ms=10)

    batcher.add_request(Request(request_id=1, arrival_time_ms=0))
    batcher.add_request(Request(request_id=2, arrival_time_ms=1))
    batcher.add_request(Request(request_id=3, arrival_time_ms=2))

    print("Test 1: dispatch when max batch size reached")
    print(f"should_dispatch = {batcher.should_dispatch(2)}")

    assert batcher.should_dispatch(2)

    batch = batcher.dispatch(2)

    assert batch.request_ids == [1, 2, 3]
    assert batch.dispatch_time_ms == 2

    print("Passed.\n")


def test_dispatch_when_max_wait_reached():
    batcher = DynamicBatcher(max_batch_size=4, max_wait_ms=5)

    batcher.add_request(Request(request_id=1, arrival_time_ms=0))

    print("Test 2: dispatch when max wait reached")
    print(f"should at 4ms = {batcher.should_dispatch(4)}")
    print(f"should at 5ms = {batcher.should_dispatch(5)}")

    assert not batcher.should_dispatch(4)
    assert batcher.should_dispatch(5)

    print("Passed.\n")


def test_simulate_dynamic_batching_batches_requests():
    requests = [
        Request(request_id=1, arrival_time_ms=0),
        Request(request_id=2, arrival_time_ms=1),
        Request(request_id=3, arrival_time_ms=2),
        Request(request_id=4, arrival_time_ms=10),
    ]

    batches = simulate_dynamic_batching(
        requests=requests,
        max_batch_size=3,
        max_wait_ms=5,
    )

    print("Test 3: simulate dynamic batching")
    for batch in batches:
        print(batch)

    assert batches[0].request_ids == [1, 2, 3]
    assert batches[1].request_ids == [4]

    print("Passed.\n")


def test_continuous_batching_removes_finished_requests():
    initial_requests = [
        LLMRequest(request_id=1, remaining_tokens=1),
        LLMRequest(request_id=2, remaining_tokens=3),
    ]

    steps = simulate_continuous_batching(
        initial_requests=initial_requests,
        max_batch_size=2,
    )

    print("Test 4: continuous batching removes finished requests")
    for step in steps:
        print(step)

    assert steps[0].active_request_ids == [1, 2]
    assert steps[1].active_request_ids == [2]
    assert steps[2].active_request_ids == [2]

    print("Passed.\n")


def test_continuous_batching_adds_new_requests():
    initial_requests = [
        LLMRequest(request_id=1, remaining_tokens=2),
    ]

    arriving = {
        1: [LLMRequest(request_id=2, remaining_tokens=1)]
    }

    steps = simulate_continuous_batching(
        initial_requests=initial_requests,
        arriving_requests_by_step=arriving,
        max_batch_size=2,
    )

    print("Test 5: continuous batching adds new requests")
    for step in steps:
        print(step)

    assert steps[0].active_request_ids == [1]
    assert steps[1].active_request_ids == [1, 2]

    print("Passed.\n")


def main():
    test_dispatch_when_max_batch_size_reached()
    test_dispatch_when_max_wait_reached()
    test_simulate_dynamic_batching_batches_requests()
    test_continuous_batching_removes_finished_requests()
    test_continuous_batching_adds_new_requests()

    print("All Day 25 tests passed.")


if __name__ == "__main__":
    main()