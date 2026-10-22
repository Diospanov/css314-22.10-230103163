"""Task 7: memory-bound copy versus compute-bound trigonometry."""

import math
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
from numba import njit

STUDENT_ID = 230103163
N = 30_000_000


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


def copy_chunk(source: np.ndarray, dest: np.ndarray, start: int, end: int) -> None:
    dest[start:end] = source[start:end]


@njit(nogil=True)
def trig_chunk(source, dest, start, end):
    for index in range(start, end):
        value = source[index]
        dest[index] = math.sin(value) * math.cos(value) + math.tan(value)


def time_kernel(worker, source, dest, workers, pool, repeats) -> float:
    ranges = chunk_ranges(source.size, workers)

    def once() -> None:
        futures = [
            pool.submit(worker, source, dest, start, end) for start, end in ranges
        ]
        for future in futures:
            future.result()

    once()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        once()
        samples.append((time.perf_counter() - started) * 1000.0)
    return median(samples)


def scaling_rows(worker, source, dest, core_counts, pool, repeats):
    rows = []
    baseline_ms = None
    for cores in core_counts:
        elapsed_ms = time_kernel(worker, source, dest, cores, pool, repeats)
        if baseline_ms is None:
            baseline_ms = elapsed_ms
        speedup = baseline_ms / elapsed_ms
        efficiency = speedup / cores
        rows.append((cores, elapsed_ms, speedup, efficiency))
    return rows


def print_table(title: str, rows) -> None:
    print(title)
    header = f"{'Cores':>5} | {'Wall (ms)':>12} | {'Speedup':>8} | {'Efficiency':>10}"
    print(header)
    print("-" * len(header))
    for cores, elapsed_ms, speedup, efficiency in rows:
        print(
            f"{cores:5d} | {elapsed_ms:12.3f} | {speedup:8.3f} | {efficiency:10.3f}"
        )
    print()


def main() -> None:
    max_cores = os.cpu_count() or 1
    core_counts = [1, 2, max_cores]
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU model: {cpu_model()}")
    print(f"Logical cores: {max_cores}")
    print(f"Array length: {N}")
    print("Kernel A: B[i] = A[i]")
    print("Kernel B: B[i] = sin(A[i]) * cos(A[i]) + tan(A[i])")
    print("Efficiency = Speedup / Cores")
    print()

    rng = np.random.default_rng(STUDENT_ID)
    source = rng.uniform(-1.0, 1.0, N)
    dest = np.empty_like(source)
    warmup = np.zeros(16, dtype=np.float64)
    warmup_out = np.empty_like(warmup)
    trig_chunk(warmup, warmup_out, 0, warmup.size)

    pool = ThreadPoolExecutor(max_workers=max_cores)
    try:
        copy_rows = scaling_rows(copy_chunk, source, dest, core_counts, pool, repeats=5)
        if not np.array_equal(dest, source):
            raise SystemExit("Kernel A copy did not match the source array")

        trig_rows = scaling_rows(trig_chunk, source, dest, core_counts, pool, repeats=3)
        sample = source[0]
        expected = math.sin(sample) * math.cos(sample) + math.tan(sample)
        if not math.isclose(dest[0], expected, rel_tol=1e-9, abs_tol=1e-9):
            raise SystemExit("Kernel B result did not match the scalar formula")
    finally:
        pool.shutdown()

    print_table("Kernel A (memory-bound copy)", copy_rows)
    print_table("Kernel B (compute-bound trig)", trig_rows)

    copy_efficiency = copy_rows[-1][3]
    trig_efficiency = trig_rows[-1][3]
    if copy_efficiency < trig_efficiency:
        stalled = "Kernel A (memory-bound vector copy)"
    else:
        stalled = "Kernel B (compute-bound trigonometry)"
    print(
        f"{stalled} stalled in scaling due to memory channel saturation."
    )
    print(
        f"Efficiency at {max_cores} cores: Kernel A {copy_efficiency:.3f}, "
        f"Kernel B {trig_efficiency:.3f}."
    )


if __name__ == "__main__":
    main()
