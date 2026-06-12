from dataclasses import dataclass
from collections import deque


@dataclass
class Request:
    """
    A toy request for batching simulation.

    arrival_time_ms:
        when the request arrives

    request_id:
        unique request id
    """
    request_id: int
    arrival_time_ms: int


@dataclass
class Batch:
    """
    A batch created by the dynamic batching scheduler.
    """
    request_ids: list[int]
    dispatch_time_ms: int


class DynamicBatcher:
    """
    A simple dynamic batching scheduler.

    Rules:
        1. collect requests in a queue
        2. if queue size reaches max_batch_size, dispatch immediately
        3. otherwise, dispatch when the oldest request waited max_wait_ms
    """

    def __init__(self, max_batch_size: int, max_wait_ms: int):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive.")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative.")

        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue: deque[Request] = deque()

    def add_request(self, request: Request) -> None:
        """
        Add one request to queue.
        """
        self.queue.append(request)

    def should_dispatch(self, current_time_ms: int) -> bool:
        """
        Decide whether to dispatch a batch now.
        """
        if not self.queue:
            return False

        if len(self.queue) >= self.max_batch_size:
            return True

        oldest_request = self.queue[0]
        waited_ms = current_time_ms - oldest_request.arrival_time_ms

        return waited_ms >= self.max_wait_ms

    def dispatch(self, current_time_ms: int) -> Batch:
        """
        Dispatch up to max_batch_size requests.
        """
        if not self.queue:
            raise ValueError("Cannot dispatch empty queue.")

        request_ids = []

        while self.queue and len(request_ids) < self.max_batch_size:
            request = self.queue.popleft()
            request_ids.append(request.request_id)

        return Batch(
            request_ids=request_ids,
            dispatch_time_ms=current_time_ms,
        )


def simulate_dynamic_batching(
    requests: list[Request],
    max_batch_size: int,
    max_wait_ms: int,
) -> list[Batch]:
    """
    Simulate dynamic batching over request arrival events.

    This simulator processes arrival times in order.

    After all requests arrive, it flushes the remaining queue.
    """
    batcher = DynamicBatcher(
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
    )

    batches: list[Batch] = []

    sorted_requests = sorted(requests, key=lambda r: r.arrival_time_ms)

    for request in sorted_requests:
        current_time = request.arrival_time_ms

        batcher.add_request(request)

        if batcher.should_dispatch(current_time):
            batches.append(batcher.dispatch(current_time))

    # Flush remaining queue at the time when the oldest request reaches max wait.
    while batcher.queue:
        oldest = batcher.queue[0]
        flush_time = oldest.arrival_time_ms + max_wait_ms
        batches.append(batcher.dispatch(flush_time))

    return batches


@dataclass
class LLMRequest:
    """
    A toy LLM request for continuous batching simulation.

    remaining_tokens:
        how many decode steps this request still needs
    """
    request_id: int
    remaining_tokens: int


@dataclass
class DecodeStep:
    """
    One decode step's active batch.
    """
    step_id: int
    active_request_ids: list[int]


def simulate_continuous_batching(
    initial_requests: list[LLMRequest],
    arriving_requests_by_step: dict[int, list[LLMRequest]] | None = None,
    max_batch_size: int = 4,
) -> list[DecodeStep]:
    """
    Simulate a toy continuous batching loop.

    At each step:
        1. add newly arrived requests
        2. choose up to max_batch_size active requests
        3. decode one token for each active request
        4. remove completed requests

    This is only a scheduler intuition demo, not real LLM execution.
    """
    if arriving_requests_by_step is None:
        arriving_requests_by_step = {}

    waiting_queue: deque[LLMRequest] = deque(initial_requests)
    active: list[LLMRequest] = []
    steps: list[DecodeStep] = []

    step_id = 0

    while waiting_queue or active or step_id in arriving_requests_by_step:
        for req in arriving_requests_by_step.get(step_id, []):
            waiting_queue.append(req)

        while waiting_queue and len(active) < max_batch_size:
            active.append(waiting_queue.popleft())

        if not active:
            step_id += 1
            continue

        steps.append(
            DecodeStep(
                step_id=step_id,
                active_request_ids=[req.request_id for req in active],
            )
        )

        for req in active:
            req.remaining_tokens -= 1

        active = [req for req in active if req.remaining_tokens > 0]

        step_id += 1

    return steps


def dynamic_batching_demo() -> None:
    """
    Demonstrate dynamic batching.
    """
    requests = [
        Request(request_id=1, arrival_time_ms=0),
        Request(request_id=2, arrival_time_ms=1),
        Request(request_id=3, arrival_time_ms=2),
        Request(request_id=4, arrival_time_ms=10),
        Request(request_id=5, arrival_time_ms=11),
    ]

    batches = simulate_dynamic_batching(
        requests=requests,
        max_batch_size=3,
        max_wait_ms=5,
    )

    print("Dynamic batching demo")
    for batch in batches:
        print(
            f"dispatch_time={batch.dispatch_time_ms}ms, "
            f"request_ids={batch.request_ids}"
        )
    print()


def continuous_batching_demo() -> None:
    """
    Demonstrate toy continuous batching.
    """
    initial_requests = [
        LLMRequest(request_id=1, remaining_tokens=2),
        LLMRequest(request_id=2, remaining_tokens=4),
        LLMRequest(request_id=3, remaining_tokens=1),
    ]

    arriving = {
        1: [LLMRequest(request_id=4, remaining_tokens=2)],
        3: [LLMRequest(request_id=5, remaining_tokens=1)],
    }

    steps = simulate_continuous_batching(
        initial_requests=initial_requests,
        arriving_requests_by_step=arriving,
        max_batch_size=3,
    )

    print("Continuous batching demo")
    for step in steps:
        print(
            f"step={step.step_id}, "
            f"active={step.active_request_ids}"
        )
    print()


def main() -> None:
    dynamic_batching_demo()
    continuous_batching_demo()


if __name__ == "__main__":
    main()