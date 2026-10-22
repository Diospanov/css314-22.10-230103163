"""Task 5: find where parallel saxpy overhead exceeds the serial run."""

import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import median

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np

STUDENT_ID = 230103163
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
A = 2.5
B = -1.25
# Ignore gaps smaller than this so timer noise is not called a crossover.
MARGIN = 0.02


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


def chunk_ranges(length: int, workers: int) -> list[tuple[int, int]]:
    ranges = []
    for worker in range(workers):
        start = (worker * length) // workers
        end = ((worker + 1) * length) // workers
        if start < end:
            ranges.append((start, end))
    return ranges


def saxpy_chunk(x: np.ndarray, y: np.ndarray, start: int, end: int) -> None:
    y[start:end] = A * x[start:end] + B


def repeat_count(length: int) -> int:
    if length <= 10_000:
        return 80
    if length <= 100_000:
        return 50
    if length <= 1_000_000:
        return 30
    return 12


def measure(x: np.ndarray, y: np.ndarray, pool: ThreadPoolExecutor, workers: int, repeats: int) -> tuple[float, float]:
    """Return median serial and parallel times in milliseconds."""
    ranges = chunk_ranges(x.size, workers)

    def run_serial() -> None:
        y[:] = A * x + B

    def run_parallel() -> None:
        futures = [pool.submit(saxpy_chunk, x, y, start, end) for start, end in ranges]
        for future in futures:
            future.result()

    for _ in range(5):
        run_serial()
        run_parallel()

    expected = A * x + B
    if not np.allclose(y, expected):
        raise SystemExit(f"Parallel saxpy mismatch at N={x.size}")

    serial_samples = []
    parallel_samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        run_serial()
        serial_samples.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        run_parallel()
        parallel_samples.append((time.perf_counter() - started) * 1000.0)
    return median(serial_samples), median(parallel_samples)


def classify(serial_ms: float, parallel_ms: float) -> str:
    ratio = parallel_ms / serial_ms
    if ratio > 1.0 + MARGIN:
        return "SLOWER"
    if ratio < 1.0 - MARGIN:
        return "FASTER"
    return "TIE"


def log_points(low: int, high: int, count: int = 6) -> list[int]:
    if high <= low + 1:
        return []
    grid = np.linspace(np.log10(low), np.log10(high), count + 2)[1:-1]
    points = sorted({int(round(10 ** value)) for value in grid})
    return [point for point in points if low < point < high]


def crossover_bounds(rows: list[tuple[int, str]]) -> tuple[int, int] | None:
    """Return (last slower N, first N of the stable faster suffix)."""
    stable_fast = None
    for index, (_length, label) in enumerate(rows):
        if all(item[1] == "FASTER" for item in rows[index:]):
            stable_fast = index
            break
    if stable_fast is None or stable_fast == 0:
        return None
    slower = [length for length, label in rows[:stable_fast] if label == "SLOWER"]
    if not slower:
        return None
    return slower[-1], rows[stable_fast][0]


def print_table(title: str, rows: list[tuple[int, float, float, str]]) -> None:
    print(title)
    header = f"{'N':>10} | {'Serial (ms)':>12} | {'Parallel (ms)':>14} | {'Speedup':>8} | {'Parallel':>8}"
    print(header)
    print("-" * len(header))
    for length, serial_ms, parallel_ms, label in rows:
        speedup = serial_ms / parallel_ms
        print(
            f"{length:10d} | {serial_ms:12.4f} | {parallel_ms:14.4f} | {speedup:8.3f} | {label:>8}"
        )
    print()


def main() -> None:
    workers = os.cpu_count() or 1
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU model: {cpu_model()}")
    print(f"Logical cores: {workers}")
    print(f"Parallel workers: {workers}")
    print(f"Operation: y[i] = {A} * x[i] + ({B})")
    print("Parallel pool: persistent threads, timed section is dispatch plus saxpy")
    print()

    rng = np.random.default_rng(STUDENT_ID)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        warmup_x = np.ones(10_000, dtype=np.float64)
        warmup_y = np.empty_like(warmup_x)
        measure(warmup_x, warmup_y, pool, workers, repeats=3)

        required = []
        for length in SIZES:
            x = rng.random(length, dtype=np.float64)
            y = np.empty(length, dtype=np.float64)
            serial_ms, parallel_ms = measure(x, y, pool, workers, repeat_count(length))
            required.append((length, serial_ms, parallel_ms, classify(serial_ms, parallel_ms)))

        print_table("Required sizes", required)

        labels = [(length, label) for length, _serial, _parallel, label in required]
        bounds = crossover_bounds(labels)
        if bounds is None:
            print("Crossover range: no shift from SLOWER to FASTER inside the required sizes.")
            return

        low, high = bounds
        refined = []
        for length in log_points(low, high):
            x = rng.random(length, dtype=np.float64)
            y = np.empty(length, dtype=np.float64)
            serial_ms, parallel_ms = measure(x, y, pool, workers, repeat_count(length))
            refined.append((length, serial_ms, parallel_ms, classify(serial_ms, parallel_ms)))

        if refined:
            print_table("Refined sizes inside the required-size bracket", refined)

        combined = sorted(required + refined, key=lambda row: row[0])
        combined_labels = [(length, label) for length, _serial, _parallel, label in combined]
        exact = crossover_bounds(combined_labels)
        if exact is None:
            print(
                f"Crossover range: multi-threading shifts from SLOWER to FASTER between N={low} and N={high}."
            )
            return

        slow_until, fast_from = exact
        print(
            "Crossover range: multi-threading shifts from SLOWER than serial "
            f"to FASTER than serial for N in ({slow_until}, {fast_from})."
        )
        slower_holds = all(
            label == "SLOWER" for length, label in combined_labels if length <= slow_until
        )
        faster_holds = all(
            label == "FASTER" for length, label in combined_labels if length >= fast_from
        )
        if slower_holds:
            print(f"Measured N <= {slow_until} are SLOWER.")
        if faster_holds:
            print(f"Measured N >= {fast_from} are FASTER.")
    finally:
        pool.shutdown()


if __name__ == "__main__":
    main()
