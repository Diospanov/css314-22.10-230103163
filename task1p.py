"""Task 1P: row-major versus column-major parallel cache traversal."""

import ctypes
import math
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from ctypes import wintypes
from statistics import median

# One thread per worker. Extra OpenMP threads would oversubscribe the cores.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
from numba import njit

STUDENT_ID = 230103163
ROWS = 8_000
COLS = 8_000
REPEATS = 3


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


def physical_core_count() -> int:
    """Count physical cores. Logical processors include hyper-threads."""
    logical = os.cpu_count() or 1
    if os.name != "nt":
        return logical

    relation_processor_core = 0
    query = ctypes.windll.kernel32.GetLogicalProcessorInformationEx
    query.argtypes = [wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    query.restype = wintypes.BOOL

    length = wintypes.DWORD(0)
    query(relation_processor_core, None, ctypes.byref(length))
    if length.value <= 0:
        return logical

    buffer = (ctypes.c_byte * length.value)()
    if not query(relation_processor_core, ctypes.byref(buffer), ctypes.byref(length)):
        return logical

    count = 0
    offset = 0
    while offset + 8 <= length.value:
        relationship = int.from_bytes(bytes(buffer[offset : offset + 4]), "little")
        size = int.from_bytes(bytes(buffer[offset + 4 : offset + 8]), "little")
        if size <= 0:
            break
        if relationship == relation_processor_core:
            count += 1
        offset += size
    return count or logical


def chunk_ranges(length: int, workers: int) -> list[tuple[int, int]]:
    """Split [0, length) into contiguous slices, one per worker."""
    ranges = []
    for worker in range(workers):
        start = (worker * length) // workers
        end = ((worker + 1) * length) // workers
        if start < end:
            ranges.append((start, end))
    return ranges


@njit(nogil=True)
def sum_row_slice(matrix, row_start, row_end):
    """Sum a row slice. The inner loop walks contiguous columns."""
    total = 0.0
    cols = matrix.shape[1]
    for row in range(row_start, row_end):
        for col in range(cols):
            total += matrix[row, col]
    return total


@njit(nogil=True)
def sum_col_slice(matrix, col_start, col_end):
    """Sum a column slice. The inner loop jumps one full row per element."""
    total = 0.0
    rows = matrix.shape[0]
    for col in range(col_start, col_end):
        for row in range(rows):
            total += matrix[row, col]
    return total


def parallel_sum(worker, matrix: np.ndarray, ranges: list[tuple[int, int]], pool: ThreadPoolExecutor) -> float:
    futures = [pool.submit(worker, matrix, start, end) for start, end in ranges]
    return float(sum(future.result() for future in futures))


def time_mode(worker, matrix: np.ndarray, ranges: list[tuple[int, int]], pool: ThreadPoolExecutor) -> tuple[float, float]:
    """Warm up once, then return the median wall-clock time in milliseconds and the last sum."""
    parallel_sum(worker, matrix, ranges, pool)
    samples = []
    total = 0.0
    for _ in range(REPEATS):
        started = time.perf_counter()
        total = parallel_sum(worker, matrix, ranges, pool)
        samples.append((time.perf_counter() - started) * 1000.0)
    return median(samples), total


def main() -> None:
    logical_cores = os.cpu_count() or 1
    physical_cores = physical_core_count()

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU model: {cpu_model()}")
    print(f"Logical cores: {logical_cores}")
    print(f"Physical cores: {physical_cores}")
    print(f"Worker threads: {physical_cores}")
    print(f"Matrix shape: {ROWS} x {COLS}")
    print("Dtype: float64")
    print(f"Seed: {STUDENT_ID}")
    print("Mode A: row slices, inner loop across contiguous columns")
    print("Mode B: column slices, inner loop across rows (strided)")
    print(f"Repeats: {REPEATS} (median after one warmup)")
    print()

    rng = np.random.default_rng(STUDENT_ID)
    matrix = np.ascontiguousarray(rng.random((ROWS, COLS), dtype=np.float64))
    if matrix.shape != (ROWS, COLS) or matrix.dtype != np.float64:
        raise SystemExit("matrix is not 8000 x 8000 float64")
    if not matrix.flags.c_contiguous:
        raise SystemExit("matrix is not row-major")

    # Compile both kernels before either timed run.
    warmup = np.zeros((2, 2), dtype=np.float64)
    sum_row_slice(warmup, 0, 2)
    sum_col_slice(warmup, 0, 2)

    row_ranges = chunk_ranges(ROWS, physical_cores)
    col_ranges = chunk_ranges(COLS, physical_cores)
    pool = ThreadPoolExecutor(max_workers=physical_cores)
    try:
        mode_a_ms, mode_a_sum = time_mode(sum_row_slice, matrix, row_ranges, pool)
        mode_b_ms, mode_b_sum = time_mode(sum_col_slice, matrix, col_ranges, pool)
    finally:
        pool.shutdown()

    reference = float(np.sum(matrix, dtype=np.float64))
    if not math.isclose(mode_a_sum, reference, rel_tol=1e-8, abs_tol=1e-4):
        raise SystemExit(f"Mode A sum {mode_a_sum} did not match reference {reference}")
    if not math.isclose(mode_b_sum, reference, rel_tol=1e-8, abs_tol=1e-4):
        raise SystemExit(f"Mode B sum {mode_b_sum} did not match reference {reference}")

    penalty = mode_b_ms / mode_a_ms
    print(f"Mode A (Row-Major) runtime (ms): {mode_a_ms:.3f}")
    print(f"Mode B (Column-Major) runtime (ms): {mode_b_ms:.3f}")
    print(f"Cache penalty factor: {penalty:.3f}")
    print(f"Reference sum: {reference:.10f}")
    print(
        "Mode B steps across rows, so each load skips to a new cache line "
        "while Mode A consumes a full line before moving on."
    )


if __name__ == "__main__":
    main()
