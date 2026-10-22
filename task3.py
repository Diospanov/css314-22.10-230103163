"""Task 3: cache-line contention from false sharing."""

import ctypes
import os
import platform
import threading
import time
from ctypes import wintypes

import numpy as np
from numba import njit, types
from numba.extending import intrinsic

STUDENT_ID = 230103163
ITERATIONS = 50_000_000
CACHE_LINE_BYTES = 64
INT64_BYTES = 8
PADDED_STRIDE = CACHE_LINE_BYTES // INT64_BYTES


@intrinsic
def atomic_increment(typingctx, counts, index):
    """Add 1 with an atomic read-modify-write so each iteration stores to memory.

    A plain ``counts[index] += 1`` is legal for LLVM to keep in a register and
    store once, which erases the false-sharing traffic this task measures.
    """
    signature = types.int64(types.Array(types.int64, 1, "C"), types.int64)

    def codegen(context, builder, sig, args):
        array_type, _index_type = sig.args
        array, index = args
        array_struct = context.make_array(array_type)(context, builder, value=array)
        slot = builder.gep(array_struct.data, [index])
        one = context.get_constant(types.int64, 1)
        return builder.atomic_rmw("add", slot, one, ordering="monotonic")

    return signature, codegen


@njit(nogil=True)
def increment_unpadded(shared_counts, index, repeats):
    for _ in range(repeats):
        atomic_increment(shared_counts, index)


@njit(nogil=True)
def increment_padded(padded_counts, shared_counts, index, repeats):
    """Increment a private 64-byte slot, then publish the total once."""
    slot = index * PADDED_STRIDE
    for _ in range(repeats):
        atomic_increment(padded_counts, slot)
    shared_counts[index] = padded_counts[slot]


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


def run_workers(worker, worker_count: int, repeats: int, padded: bool) -> tuple[float, np.ndarray]:
    shared_counts = np.zeros(worker_count, dtype=np.int64)
    if padded:
        padded_counts = np.zeros(worker_count * PADDED_STRIDE, dtype=np.int64)
        thread_args = [
            (padded_counts, shared_counts, index, repeats) for index in range(worker_count)
        ]
    else:
        thread_args = [(shared_counts, index, repeats) for index in range(worker_count)]

    threads = [
        threading.Thread(target=worker, args=args) for args in thread_args
    ]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return elapsed_ms, shared_counts


def expect_full_count(shared_counts: np.ndarray, repeats: int, label: str) -> None:
    if not np.all(shared_counts == repeats):
        raise SystemExit(f"{label} counts were not {repeats}: {shared_counts.tolist()}")


def main() -> None:
    logical_cores = os.cpu_count() or 1
    physical_cores = physical_core_count()

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU model: {cpu_model()}")
    print(f"Physical cores: {physical_cores}")
    print(f"Logical cores: {logical_cores}")
    print(f"Worker threads: {physical_cores}")
    print(f"Iterations per thread: {ITERATIONS}")
    print(f"shared_counts length: {physical_cores}")
    print(f"Padding stride: {CACHE_LINE_BYTES} bytes")

    # Compile both kernels before either timed run.
    warmup_counts = np.zeros(physical_cores, dtype=np.int64)
    warmup_padded = np.zeros(physical_cores * PADDED_STRIDE, dtype=np.int64)
    increment_unpadded(warmup_counts, 0, 1)
    increment_padded(warmup_padded, warmup_counts, 0, 1)

    unpadded_ms, unpadded_counts = run_workers(
        increment_unpadded, physical_cores, ITERATIONS, padded=False
    )
    expect_full_count(unpadded_counts, ITERATIONS, "Unpadded Shared")

    padded_ms, padded_counts = run_workers(
        increment_padded, physical_cores, ITERATIONS, padded=True
    )
    expect_full_count(padded_counts, ITERATIONS, "Padded/Local")

    ratio = unpadded_ms / padded_ms
    print()
    print(f"Unpadded Shared (ms): {unpadded_ms:.3f}")
    print(f"Padded/Local (ms): {padded_ms:.3f}")
    print(f"Execution ratio (unpadded / padded): {ratio:.3f}")


if __name__ == "__main__":
    main()
