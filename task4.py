"""Task 4: static chunks versus work stealing on a skewed loop."""

import os
import platform
import threading
import time
from collections import deque

STUDENT_ID = 230103163
ITEM_COUNT = 1000
WORKER_COUNT = 4
CHUNK_SIZE = 5


def cpu_model() -> str:
    """Return the CPU marketing name, then fall back to platform details."""
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            name = str(name).strip()
            if name:
                return name
        except OSError:
            pass

    processor = platform.processor().strip()
    machine = platform.machine().strip()
    if processor and machine and machine not in processor:
        return f"{processor} ({machine})"
    return processor or machine or "unknown"


def item_cost(index: int) -> float:
    """Irregular cost: sleep(0.0001 * (i % 20))."""
    return 0.0001 * (index % 20)


def run_item(index: int) -> None:
    delay = item_cost(index)
    if delay > 0.0:
        time.sleep(delay)


def static_blocks(item_count: int, workers: int) -> list[list[int]]:
    blocks = []
    for worker in range(workers):
        start = (worker * item_count) // workers
        end = ((worker + 1) * item_count) // workers
        blocks.append(list(range(start, end)))
    return blocks


def chunk_items(indices: list[int], chunk_size: int) -> list[list[int]]:
    return [
        indices[offset : offset + chunk_size]
        for offset in range(0, len(indices), chunk_size)
    ]


class WorkStealingQueue:
    """One deque per worker. Owners pop from the right; thieves pop from the left."""

    def __init__(self, blocks: list[list[int]], chunk_size: int) -> None:
        self._deques = [deque(chunk_items(block, chunk_size)) for block in blocks]
        self._locks = [threading.Lock() for _ in blocks]

    def take_local(self, worker: int) -> list[int] | None:
        with self._locks[worker]:
            if self._deques[worker]:
                return self._deques[worker].pop()
        return None

    def steal(self, worker: int) -> list[int] | None:
        worker_count = len(self._deques)
        for offset in range(1, worker_count):
            victim = (worker + offset) % worker_count
            with self._locks[victim]:
                if self._deques[victim]:
                    return self._deques[victim].popleft()
        return None


def _worker(worker: int, barrier: threading.Barrier, body, starts: list[float], finishes: list[float], counts: list[int]) -> None:
    barrier.wait()
    starts[worker] = time.perf_counter()
    counts[worker] = body(worker)
    finishes[worker] = time.perf_counter()


def run_strategy(worker_count: int, body) -> dict[str, float | list[float] | list[int]]:
    starts = [0.0] * worker_count
    finishes = [0.0] * worker_count
    counts = [0] * worker_count
    barrier = threading.Barrier(worker_count)
    threads = [
        threading.Thread(
            target=_worker,
            args=(worker, barrier, body, starts, finishes, counts),
        )
        for worker in range(worker_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    origin = min(starts)
    finish_ms = [(stamp - origin) * 1000.0 for stamp in finishes]
    first_ms = min(finish_ms)
    last_ms = max(finish_ms)
    return {
        "wall_ms": last_ms,
        "first_ms": first_ms,
        "last_ms": last_ms,
        "delta_ms": last_ms - first_ms,
        "finish_ms": finish_ms,
        "counts": counts,
    }


def static_body(blocks: list[list[int]]):
    def body(worker: int) -> int:
        for index in blocks[worker]:
            run_item(index)
        return len(blocks[worker])

    return body


def stealing_body(queue: WorkStealingQueue):
    def body(worker: int) -> int:
        done = 0
        while True:
            chunk = queue.take_local(worker)
            if chunk is None:
                chunk = queue.steal(worker)
            if chunk is None:
                return done
            for index in chunk:
                run_item(index)
            done += len(chunk)

    return body


def print_result(title: str, result: dict) -> None:
    print(title)
    print(f"Wall time (ms): {result['wall_ms']:.3f}")
    print(f"First thread finish (ms): {result['first_ms']:.3f}")
    print(f"Last thread finish (ms): {result['last_ms']:.3f}")
    print(f"Completion delta (ms): {result['delta_ms']:.3f}")
    for worker, (finish_ms, count) in enumerate(zip(result["finish_ms"], result["counts"])):
        print(f"Worker {worker}: finish {finish_ms:.3f} ms, items {count}")
    print()


def main() -> None:
    blocks = static_blocks(ITEM_COUNT, WORKER_COUNT)
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU model: {cpu_model()}")
    print(f"Logical cores: {os.cpu_count()}")
    print(f"Items: {ITEM_COUNT}")
    print(f"Workers: {WORKER_COUNT}")
    print("Item cost: sleep(0.0001 * (i % 20))")
    print(f"Dynamic chunk size: {CHUNK_SIZE}")
    print()
    print("Static block cost units (sum of i % 20):")
    for worker, block in enumerate(blocks):
        units = sum(index % 20 for index in block)
        print(f"Worker {worker}: indices {block[0]}-{block[-1]}, cost units {units}")
    print()

    static_result = run_strategy(WORKER_COUNT, static_body(blocks))
    stealing_result = run_strategy(
        WORKER_COUNT,
        stealing_body(WorkStealingQueue(blocks, CHUNK_SIZE)),
    )

    print_result("Strategy A: Static chunks", static_result)
    print_result("Strategy B: Work stealing", stealing_result)


if __name__ == "__main__":
    main()
