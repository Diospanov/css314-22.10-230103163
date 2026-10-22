"""Task 6: lost updates without synchronization, and the cost of a lock."""

import os
import platform
import threading
import time

import numpy as np
from numba import njit, types
from numba.extending import intrinsic

STUDENT_ID = 230103163
THREAD_COUNT = 4
INCREMENTS_PER_THREAD = 2_000_000
EXPECTED = THREAD_COUNT * INCREMENTS_PER_THREAD


@intrinsic
def racy_increment(typingctx, counter):
    """Non-atomic load/add/store. Volatile keeps every increment in memory."""
    signature = types.void(types.Array(types.int64, 1, "C"))

    def codegen(context, builder, sig, args):
        array_type = sig.args[0]
        array_struct = context.make_array(array_type)(context, builder, value=args[0])
        zero = context.get_constant(types.int64, 0)
        slot = builder.gep(array_struct.data, [zero])
        loaded = builder.load(slot)
        one = context.get_constant(types.int64, 1)
        updated = builder.add(loaded, one)
        stored = builder.store(updated, slot)

        def load_descr(buf, instruction=loaded):
            ptr = instruction.operands[0]
            buf.append(
                f"load volatile {instruction.type}, {ptr.type} {ptr.get_reference()}\n"
            )

        def store_descr(buf, instruction=stored):
            value, ptr = instruction.operands
            buf.append(
                "store volatile {0} {1}, {2} {3}\n".format(
                    value.type,
                    value.get_reference(),
                    ptr.type,
                    ptr.get_reference(),
                )
            )

        loaded.descr = load_descr
        stored.descr = store_descr

    return signature, codegen


@njit(nogil=True)
def racy_loop(counter, repeats):
    for _ in range(repeats):
        racy_increment(counter)


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


def run_unsynchronized() -> tuple[int, float]:
    counter = np.zeros(1, dtype=np.int64)
    racy_loop(counter, 1)
    counter[0] = 0
    threads = [
        threading.Thread(target=racy_loop, args=(counter, INCREMENTS_PER_THREAD))
        for _ in range(THREAD_COUNT)
    ]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return int(counter[0]), elapsed_ms


def locked_worker(shared: list[int], lock: threading.Lock) -> None:
    for _ in range(INCREMENTS_PER_THREAD):
        with lock:
            shared[0] += 1


def run_locked() -> tuple[int, float]:
    shared = [0]
    lock = threading.Lock()
    threads = [
        threading.Thread(target=locked_worker, args=(shared, lock))
        for _ in range(THREAD_COUNT)
    ]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return shared[0], elapsed_ms


def main() -> None:
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU model: {cpu_model()}")
    print(f"Logical cores: {os.cpu_count()}")
    print(f"Threads: {THREAD_COUNT}")
    print(f"Increments per thread: {INCREMENTS_PER_THREAD}")
    print(f"Expected final value: {EXPECTED}")
    print()

    actual, unsync_ms = run_unsynchronized()
    lost = EXPECTED - actual
    lost_pct = 100.0 * lost / EXPECTED
    print("Unsynchronized counter")
    print(f"Actual final value: {actual}")
    print(f"Updates lost: {lost}")
    print(f"Percentage of updates lost: {lost_pct:.3f}%")
    print(f"Wall time (ms): {unsync_ms:.3f}")
    print()

    locked_value, locked_ms = run_locked()
    if locked_value != EXPECTED:
        raise SystemExit(f"Locked counter ended at {locked_value}, expected {EXPECTED}")
    penalty_ms = locked_ms - unsync_ms
    print("Locked counter")
    print(f"Actual final value: {locked_value}")
    print(f"Wall time (ms): {locked_ms:.3f}")
    print(f"Lock time penalty (ms): {penalty_ms:.3f}")


if __name__ == "__main__":
    main()
