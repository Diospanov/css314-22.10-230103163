"""Task 2P: Monte Carlo pi with a shared PRNG versus per-thread PRNG state."""

import math
import os
import platform
import threading
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
from numba import njit, types
from numba.extending import intrinsic

STUDENT_ID = 230103163
THREAD_COUNT = 4
SAMPLES = 40_000_000

SPLITMIX_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
SPLITMIX_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
SPLITMIX_MUL2 = np.uint64(0x94D049BB133111EB)
XORSHIFT_MUL = np.uint64(0x2545F4914F6CDD1D)
UNIT_INTERVAL = np.float64(1.0 / 9007199254740992.0)


@intrinsic
def atomic_exchange(typingctx, values, index, new_value):
    """Swap `new_value` into an int64 slot and return the previous value."""
    signature = types.int64(
        types.Array(types.int64, 1, "C"),
        types.int64,
        types.int64,
    )

    def codegen(context, builder, sig, args):
        array_type, _, _ = sig.args
        array, index, new_value = args
        array_struct = context.make_array(array_type)(context, builder, value=array)
        slot = builder.gep(array_struct.data, [index])
        return builder.atomic_rmw("xchg", slot, new_value, ordering="acq_rel")

    return signature, codegen


@njit(nogil=True)
def mix_seed(seed):
    """SplitMix64. Keeps a zero input from collapsing xorshift64*."""
    x = np.uint64(seed) + SPLITMIX_GOLDEN
    z = x
    z = (z ^ (z >> np.uint64(30))) * SPLITMIX_MUL1
    z = (z ^ (z >> np.uint64(27))) * SPLITMIX_MUL2
    z ^= z >> np.uint64(31)
    if z == 0:
        z = SPLITMIX_GOLDEN
    return z


@njit(nogil=True)
def next_unit(state):
    """Advance xorshift64* and return a float64 in [0, 1)."""
    x = state[0]
    x ^= x >> np.uint64(12)
    x ^= x << np.uint64(25)
    x ^= x >> np.uint64(27)
    state[0] = x
    mixed = x * XORSHIFT_MUL
    return np.float64(mixed >> np.uint64(11)) * UNIT_INTERVAL


@njit(nogil=True)
def acquire(lock):
    """Test-and-set spin mutex. 0 is free, 1 is held."""
    while atomic_exchange(lock, 0, 1) != 0:
        pass


@njit(nogil=True)
def release(lock):
    atomic_exchange(lock, 0, 0)


@njit(nogil=True)
def monte_carlo_shared(lock, state, samples, slot, hits):
    """Draw every point from one generator while holding the mutex."""
    count = 0
    for _ in range(samples):
        acquire(lock)
        x = next_unit(state)
        y = next_unit(state)
        release(lock)
        if x * x + y * y <= 1.0:
            count += 1
    hits[slot] = count


@njit(nogil=True)
def monte_carlo_independent(seed, samples, slot, hits):
    """Draw from a private generator. No shared state and no mutex."""
    state = np.empty(1, dtype=np.uint64)
    state[0] = mix_seed(seed)
    count = 0
    for _ in range(samples):
        x = next_unit(state)
        y = next_unit(state)
        if x * x + y * y <= 1.0:
            count += 1
    hits[slot] = count


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


def thread_seed(index: int) -> int:
    return STUDENT_ID * 10 + index


def run_threads(workers: list[threading.Thread]) -> float:
    started = time.perf_counter()
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    return (time.perf_counter() - started) * 1000.0


def estimate_pi(hits: np.ndarray) -> float:
    return 4.0 * float(np.sum(hits, dtype=np.int64)) / SAMPLES


def require_plausible(label: str, estimate: float) -> None:
    error = abs(estimate - math.pi)
    if error > 0.01:
        raise SystemExit(f"{label} pi {estimate} is too far from math.pi (error {error})")


def main() -> None:
    if SAMPLES % THREAD_COUNT != 0:
        raise SystemExit("sample count must divide evenly across 4 threads")
    samples_per_thread = SAMPLES // THREAD_COUNT
    seeds = [thread_seed(index) for index in range(THREAD_COUNT)]

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU model: {cpu_model()}")
    print(f"Logical cores: {os.cpu_count()}")
    print(f"Threads: {THREAD_COUNT}")
    print(f"Samples: {SAMPLES}")
    print(f"Samples per thread: {samples_per_thread}")
    print(f"Shared PRNG seed: {STUDENT_ID}")
    print("Independent PRNG seed: (Student_ID * 10) + thread_index")
    print(f"Independent seeds: {', '.join(str(seed) for seed in seeds)}")
    print("Shared guard: spin mutex around each generator draw")
    print()

    lock = np.zeros(1, dtype=np.int64)
    state = np.zeros(1, dtype=np.uint64)
    hits = np.zeros(THREAD_COUNT, dtype=np.int64)

    # Compile both kernels before either timed run.
    state[0] = mix_seed(np.uint64(STUDENT_ID))
    monte_carlo_shared(lock, state, 1, 0, hits)
    monte_carlo_independent(np.uint64(seeds[0]), 1, 0, hits)
    lock[0] = 0
    hits[:] = 0

    state[0] = mix_seed(np.uint64(STUDENT_ID))
    shared_workers = [
        threading.Thread(
            target=monte_carlo_shared,
            args=(lock, state, samples_per_thread, index, hits),
        )
        for index in range(THREAD_COUNT)
    ]
    shared_ms = run_threads(shared_workers)
    if int(lock[0]) != 0:
        raise SystemExit("shared mutex was left locked")
    shared_pi = estimate_pi(hits)
    require_plausible("Shared PRNG", shared_pi)
    shared_error = abs(shared_pi - math.pi)

    independent_hits = np.zeros(THREAD_COUNT, dtype=np.int64)
    independent_workers = [
        threading.Thread(
            target=monte_carlo_independent,
            args=(np.uint64(seeds[index]), samples_per_thread, index, independent_hits),
        )
        for index in range(THREAD_COUNT)
    ]
    independent_ms = run_threads(independent_workers)
    independent_pi = estimate_pi(independent_hits)
    require_plausible("Independent PRNG", independent_pi)
    independent_error = abs(independent_pi - math.pi)

    speedup = shared_ms / independent_ms
    print("Shared PRNG")
    print(f"Computed pi: {shared_pi:.10f}")
    print(f"Absolute error: {shared_error:.10f}")
    print(f"Wall time (ms): {shared_ms:.3f}")
    print()
    print("Independent PRNG")
    print(f"Computed pi: {independent_pi:.10f}")
    print(f"Absolute error: {independent_error:.10f}")
    print(f"Wall time (ms): {independent_ms:.3f}")
    print()
    print(f"Speedup factor: {speedup:.3f}")
    print(
        "Removing the mutex lets each thread keep its own generator state, "
        "so the four draws no longer wait on one lock."
    )


if __name__ == "__main__":
    main()
