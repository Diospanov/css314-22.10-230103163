"""Task 6P: OS mutex versus hardware compare-and-swap under contention."""

import os
import platform
import threading
import time

import numpy as np
from numba import njit, types
from numba.extending import intrinsic

STUDENT_ID = 230103163
THREAD_COUNT = 8
INCREMENTS_PER_THREAD = 1_000_000
TOTAL_OPS = THREAD_COUNT * INCREMENTS_PER_THREAD


@intrinsic
def cas_once(typingctx, counter, expected, desired):
    """Hardware compare-and-swap. Returns the word that was in memory."""
    signature = types.int64(
        types.Array(types.int64, 1, "C"),
        types.int64,
        types.int64,
    )

    def codegen(context, builder, sig, args):
        array_type = sig.args[0]
        array, expected, desired = args
        array_struct = context.make_array(array_type)(context, builder, value=array)
        zero = context.get_constant(types.int64, 0)
        slot = builder.gep(array_struct.data, [zero])
        swapped = builder.cmpxchg(slot, expected, desired, "acq_rel", "acquire")
        return builder.extract_value(swapped, 0)

    return signature, codegen


@njit(nogil=True)
def cas_loop(counter, repeats):
    """Increment by spinning until compare-and-swap publishes this thread's +1."""
    for _ in range(repeats):
        expected = counter[0]
        while True:
            desired = expected + 1
            old = cas_once(counter, expected, desired)
            if old == expected:
                break
            expected = old


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


def mutex_worker(counter: list[int], lock: threading.Lock) -> None:
    for _ in range(INCREMENTS_PER_THREAD):
        with lock:
            counter[0] += 1


def run_mutex() -> tuple[int, float]:
    counter = [0]
    lock = threading.Lock()
    threads = [
        threading.Thread(target=mutex_worker, args=(counter, lock))
        for _ in range(THREAD_COUNT)
    ]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return counter[0], elapsed_ms


def run_cas() -> tuple[int, float]:
    counter = np.zeros(1, dtype=np.int64)
    threads = [
        threading.Thread(target=cas_loop, args=(counter, INCREMENTS_PER_THREAD))
        for _ in range(THREAD_COUNT)
    ]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return int(counter[0]), elapsed_ms


def main() -> None:
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU model: {cpu_model()}")
    print(f"Logical cores: {os.cpu_count()}")
    print(f"Threads: {THREAD_COUNT}")
    print(f"Increments per thread: {INCREMENTS_PER_THREAD}")
    print(f"Total operations: {TOTAL_OPS}")
    print("Primitive 1: threading.Lock (OS mutex)")
    print("Primitive 2: LLVM cmpxchg retry loop (hardware CAS spin)")
    print()

    warmup = np.zeros(1, dtype=np.int64)
    cas_loop(warmup, 1)
    if int(warmup[0]) != 1:
        raise SystemExit(f"CAS warmup left the counter at {int(warmup[0])}")

    mutex_value, mutex_ms = run_mutex()
    if mutex_value != TOTAL_OPS:
        raise SystemExit(f"Mutex counter ended at {mutex_value}, expected {TOTAL_OPS}")

    cas_value, cas_ms = run_cas()
    if cas_value != TOTAL_OPS:
        raise SystemExit(f"CAS counter ended at {cas_value}, expected {TOTAL_OPS}")

    mutex_ops = TOTAL_OPS / (mutex_ms / 1000.0)
    cas_ops = TOTAL_OPS / (cas_ms / 1000.0)
    ratio = cas_ops / mutex_ops

    print(f"Mutex final value: {mutex_value}")
    print(f"Atomic CAS final value: {cas_value}")
    print()
    print(f"Mutex Execution Time (ms): {mutex_ms:.3f}")
    print(f"Mutex Throughput (ops/sec): {mutex_ops:.3f}")
    print(f"Atomic CAS Execution Time (ms): {cas_ms:.3f}")
    print(f"Atomic CAS Throughput (ops/sec): {cas_ops:.3f}")
    print(f"Throughput Ratio (CAS / Mutex): {ratio:.3f}")


if __name__ == "__main__":
    main()
