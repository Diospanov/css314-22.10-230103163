"""Task 2: multi-core summation scaling compared with Amdahl's Law."""

import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor

# One thread per worker. Extra OpenMP threads would oversubscribe the cores.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np

STUDENT_ID = 230103163
N = 20_000_000
PARALLEL_FRACTION = 0.95


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


def chunk_ranges(n: int, workers: int) -> list[tuple[int, int]]:
    """Split [0, n) into contiguous slices, one per worker."""
    ranges = []
    for i in range(workers):
        start = (i * n) // workers
        end = ((i + 1) * n) // workers
        if start < end:
            ranges.append((start, end))
    return ranges


def sum_slice(data: np.ndarray, start: int, end: int) -> float:
    return float(np.sum(data[start:end], dtype=np.float64))


def parallel_sum(data: np.ndarray, ranges: list[tuple[int, int]], executor: ThreadPoolExecutor | None) -> float:
    """Sum the array. One worker runs inline; more workers share `executor`."""
    if executor is None:
        start, end = ranges[0]
        return sum_slice(data, start, end)

    parts = executor.map(lambda bounds: sum_slice(data, bounds[0], bounds[1]), ranges)
    return float(sum(parts))


def time_sum(data: np.ndarray, workers: int) -> tuple[float, float]:
    """Warm up once, then return one wall-clock time in milliseconds and the sum."""
    ranges = chunk_ranges(data.size, workers)
    executor = ThreadPoolExecutor(max_workers=len(ranges)) if len(ranges) > 1 else None
    try:
        parallel_sum(data, ranges, executor)
        start = time.perf_counter()
        total = parallel_sum(data, ranges, executor)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return elapsed_ms, total
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


def amdahl_upper_bound(workers: int, parallel_fraction: float = PARALLEL_FRACTION) -> float:
    """S(p) = 1 / ((1 - P) + P / p)."""
    serial_fraction = 1.0 - parallel_fraction
    return 1.0 / (serial_fraction + parallel_fraction / workers)


def main() -> None:
    logical_cores = os.cpu_count() or 1
    worker_counts = [1, 2, 4, logical_cores, logical_cores * 2]

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU model: {cpu_model()}")
    print(f"Logical cores: {logical_cores}")
    print(f"Parallel fraction P: {PARALLEL_FRACTION}")

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.random(N, dtype=np.float64)
    print(f"Array length: {data.size}")
    print(f"Dtype: {data.dtype}")
    print(f"Seed: {STUDENT_ID}")

    reference = float(np.sum(data, dtype=np.float64))
    print(f"Serial reference sum: {reference:.10f}")
    print("Wall time unit: milliseconds")
    print()

    timed_runs: list[tuple[int, float, float]] = []
    baseline_ms = 0.0
    for workers in worker_counts:
        elapsed_ms, total = time_sum(data, workers)
        if abs(total - reference) > 1e-4:
            raise SystemExit(
                f"Sum mismatch at {workers} workers: {total} != {reference}"
            )
        if workers == worker_counts[0]:
            baseline_ms = elapsed_ms
        timed_runs.append((workers, elapsed_ms, total))

    header = (
        f"{'Cores':>5} | {'Wall Time':>12} | {'Speedup':>8} | {'Amdahl Upper Bound':>19}"
    )
    print(header)
    print("-" * len(header))
    for workers, elapsed_ms, _total in timed_runs:
        speedup = baseline_ms / elapsed_ms
        bound = amdahl_upper_bound(workers)
        print(
            f"{workers:5d} | {elapsed_ms:12.3f} | {speedup:8.3f} | {bound:19.3f}"
        )


if __name__ == "__main__":
    main()
