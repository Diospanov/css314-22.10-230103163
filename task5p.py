"""Task 5P: thread-pool oversubscription on a batch of small hashes."""

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
TASKS = 10_000
BLOCK_BYTES = 1024
ROUNDS = 500
REPEATS = 3
FNV_OFFSET = np.uint64(14695981039346656037)
FNV_PRIME = np.uint64(1099511628211)


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


@njit(nogil=True)
def hash_block(data, rounds):
    """FNV-1a over a 1KB block, repeated. The GIL stays released."""
    acc = FNV_OFFSET
    n = data.size
    for _ in range(rounds):
        for index in range(n):
            acc ^= np.uint64(data[index])
            acc *= FNV_PRIME
    return acc


def pool_sizes(cores: int) -> list[int]:
    """1, Cores/2, Cores, Cores*2, Cores*4, Cores*16, 256. Drop duplicate sizes."""
    requested = [
        1,
        max(cores // 2, 1),
        cores,
        cores * 2,
        cores * 4,
        cores * 16,
        256,
    ]
    sizes = []
    for size in requested:
        if size not in sizes:
            sizes.append(size)
    return sizes


def run_batch(pool: ThreadPoolExecutor, block: np.ndarray) -> int:
    futures = [pool.submit(hash_block, block, ROUNDS) for _ in range(TASKS)]
    checksum = 0
    for future in futures:
        checksum ^= int(future.result())
    return checksum


def time_pool(size: int, block: np.ndarray) -> tuple[float, int]:
    """Warm up so threads already exist, then return the median batch time and checksum."""
    pool = ThreadPoolExecutor(max_workers=size)
    try:
        expected = run_batch(pool, block)
        samples = []
        checksum = expected
        for _ in range(REPEATS):
            started = time.perf_counter()
            checksum = run_batch(pool, block)
            samples.append((time.perf_counter() - started) * 1000.0)
            if checksum != expected:
                raise SystemExit(f"pool size {size} checksum changed between repeats")
        return median(samples), checksum
    finally:
        pool.shutdown()


def mark_statuses(sizes: list[int], times: list[float], cores: int) -> tuple[list[str], int | None]:
    """Pool where the curve turns up after the fastest run."""
    del cores
    best_index = times.index(min(times))
    threshold = None
    for index in range(best_index + 1, len(sizes)):
        if times[index] > times[index - 1]:
            threshold = sizes[index]
            break

    statuses = []
    for index, size in enumerate(sizes):
        if size == threshold:
            statuses.append("threshold")
        elif size == 1:
            statuses.append("baseline")
        elif index == best_index:
            statuses.append("best")
        elif threshold is not None and size > threshold:
            statuses.append("oversubscribed")
        else:
            statuses.append("scaling")
    return statuses, threshold


def main() -> None:
    cores = os.cpu_count() or 1
    sizes = pool_sizes(cores)

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU model: {cpu_model()}")
    print(f"Logical cores: {cores}")
    print(f"Tasks: {TASKS}")
    print(f"Task: FNV-1a of a {BLOCK_BYTES}-byte block, {ROUNDS} rounds")
    print(f"Seed: {STUDENT_ID}")
    print("Cores in the pool-size list are logical cores.")
    print(
        "Requested pools: 1, Cores/2, Cores, Cores * 2, Cores * 4, Cores * 16, 256"
    )
    if cores * 16 == 256:
        print("Cores * 16 and 256 are the same pool, so that size is measured once.")
    print(f"Pool sizes: {', '.join(str(size) for size in sizes)}")
    print(f"Repeats: {REPEATS} (median after one warmup)")
    print(
        "Threshold rule: first pool after the fastest run whose execution time increases."
    )
    print()

    rng = np.random.default_rng(STUDENT_ID)
    block = rng.integers(0, 256, size=BLOCK_BYTES, dtype=np.uint8)
    hash_block(block, 1)

    rows = []
    checksums = []
    for size in sizes:
        elapsed_ms, checksum = time_pool(size, block)
        rows.append((size, elapsed_ms))
        checksums.append(checksum)

    if any(checksum != checksums[0] for checksum in checksums):
        raise SystemExit("batch checksum differed across pool sizes")

    times = [elapsed for _, elapsed in rows]
    statuses, threshold = mark_statuses(sizes, times, cores)
    baseline_ms = times[0]

    header = (
        f"{'Pool Size':>9} | {'Execution Time (ms)':>19} | "
        f"{'Speedup relative to 1 thread':>30} | Status"
    )
    print(header)
    print("-" * len(header))
    for (size, elapsed_ms), status in zip(rows, statuses):
        speedup = baseline_ms / elapsed_ms
        print(f"{size:9d} | {elapsed_ms:19.3f} | {speedup:30.3f} | {status}")
    print()
    if threshold is None:
        print("Execution time did not turn upward after the fastest pool.")
    else:
        print(f"Context-switch threshold: pool size {threshold}.")
        print(
            "Execution time starts worsening there from OS thread context switching."
        )


if __name__ == "__main__":
    main()
