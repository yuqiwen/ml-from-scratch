import asyncio
import time
from dataclasses import dataclass


@dataclass
class Request:
    request_id: int
    processing_time_ms: int


@dataclass
class RequestResult:
    request_id: int
    accepted: bool
    start_time_ms: float | None
    end_time_ms: float | None
    error: str | None


class InferenceService:
    """
    Toy inference service with:

    - bounded queue
    - limited inference concurrency
    - overload rejection
    """

    def __init__(
        self,
        max_queue_size: int,
        max_concurrency: int,
    ):
        if max_queue_size <= 0:
            raise ValueError(
                "max_queue_size must be positive."
            )

        if max_concurrency <= 0:
            raise ValueError(
                "max_concurrency must be positive."
            )

        self.queue: asyncio.Queue[
            tuple[Request, asyncio.Future]
        ] = asyncio.Queue(
            maxsize=max_queue_size
        )

        self.semaphore = asyncio.Semaphore(
            max_concurrency
        )

        self.worker_tasks: list[
            asyncio.Task
        ] = []

        self.running = False

    async def start(
        self,
        num_workers: int,
    ) -> None:
        if num_workers <= 0:
            raise ValueError(
                "num_workers must be positive."
            )

        self.running = True

        for worker_id in range(num_workers):
            task = asyncio.create_task(
                self._worker_loop(worker_id)
            )

            self.worker_tasks.append(task)

    async def stop(self) -> None:
        self.running = False

        for task in self.worker_tasks:
            task.cancel()

        await asyncio.gather(
            *self.worker_tasks,
            return_exceptions=True,
        )

        self.worker_tasks.clear()

    async def submit(
        self,
        request: Request,
    ) -> RequestResult:
        """
        Submit one request.

        If queue is full, reject immediately.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        try:
            self.queue.put_nowait(
                (request, future)
            )
        except asyncio.QueueFull:
            return RequestResult(
                request_id=request.request_id,
                accepted=False,
                start_time_ms=None,
                end_time_ms=None,
                error="queue_full",
            )

        result = await future

        return result

    async def _worker_loop(
        self,
        worker_id: int,
    ) -> None:
        while True:
            request, future = (
                await self.queue.get()
            )

            try:
                result = await self._run_inference(
                    request
                )

                if not future.done():
                    future.set_result(result)

            except Exception as exc:
                if not future.done():
                    future.set_result(
                        RequestResult(
                            request_id=request.request_id,
                            accepted=True,
                            start_time_ms=None,
                            end_time_ms=None,
                            error=str(exc),
                        )
                    )

            finally:
                self.queue.task_done()

    async def _run_inference(
        self,
        request: Request,
    ) -> RequestResult:
        """
        Simulate model inference under
        semaphore-based concurrency control.
        """
        async with self.semaphore:
            start = time.perf_counter()

            await asyncio.sleep(
                request.processing_time_ms
                / 1000.0
            )

            end = time.perf_counter()

        return RequestResult(
            request_id=request.request_id,
            accepted=True,
            start_time_ms=start * 1000,
            end_time_ms=end * 1000,
            error=None,
        )


async def run_demo() -> None:
    service = InferenceService(
        max_queue_size=4,
        max_concurrency=2,
    )

    await service.start(
        num_workers=4
    )

    requests = [
        Request(
            request_id=i,
            processing_time_ms=100,
        )
        for i in range(10)
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

    for result in results:
        if result.accepted:
            latency_ms = (
                result.end_time_ms
                - result.start_time_ms
            )

            print(
                f"request={result.request_id}, "
                f"accepted=True, "
                f"processing_latency="
                f"{latency_ms:.2f}ms"
            )
        else:
            print(
                f"request={result.request_id}, "
                f"accepted=False, "
                f"error={result.error}"
            )

    await service.stop()


def main() -> None:
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()