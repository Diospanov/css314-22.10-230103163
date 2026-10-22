"""Task 4P: pickled pipe transfer versus zero-copy shared memory."""

import math
import multiprocessing
import os
import platform
import time
from multiprocessing import shared_memory
from statistics import median

import numpy as np

STUDENT_ID = 230103163
SIZES = (100_000, 1_000_000, 10_000_000)
MAX_N = SIZES[-1]
DTYPE = np.float64
REPEATS = 5


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


def probe(array: np.ndarray) -> tuple[float, float, float, int]:
    """Fingerprint the payload without reducing the whole array."""
    return (
        float(array[0]),
        float(array[array.size // 2]),
        float(array[-1]),
        int(array.size),
    )


def expect_probe(got: tuple[float, float, float, int], source: np.ndarray, n: int) -> None:
    first, mid, last, size = got
    if size != n:
        raise SystemExit(f"transferred length {size} != {n}")
    same = (
        math.isclose(first, float(source[0]), rel_tol=0.0, abs_tol=0.0)
        and math.isclose(mid, float(source[n // 2]), rel_tol=0.0, abs_tol=0.0)
        and math.isclose(last, float(source[n - 1]), rel_tol=0.0, abs_tol=0.0)
    )
    if not same:
        raise SystemExit(f"transferred values did not match the source for N={n}")


def ipc_worker(conn) -> None:
    """Receive either a pickled array or a look at the shared buffer."""
    shm = None
    attached = None
    try:
        while True:
            message = conn.recv()
            kind = message[0]
            if kind == "stop":
                if shm is not None:
                    shm.close()
                    shm = None
                conn.send("stopped")
                return
            if kind == "attach":
                if shm is not None:
                    shm.close()
                shm = shared_memory.SharedMemory(name=message[1])
                attached = np.ndarray((MAX_N,), dtype=DTYPE, buffer=shm.buf)
                conn.send("attached")
            elif kind == "pipe":
                conn.send(probe(message[1]))
            elif kind == "shm":
                n = message[1]
                conn.send(probe(attached[:n]))
            else:
                raise RuntimeError(f"unknown message {kind}")
    finally:
        if shm is not None:
            shm.close()


def time_transfer(action, repeats: int = REPEATS) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        action()
        samples.append((time.perf_counter() - started) * 1000.0)
    return median(samples)


def main() -> None:
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    item_bytes = np.dtype(DTYPE).itemsize
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU model: {cpu_model()}")
    print(f"Logical cores: {os.cpu_count()}")
    print(f"Dtype: {np.dtype(DTYPE).name}")
    print(f"Seed: {STUDENT_ID}")
    print(f"Sizes: {', '.join(str(n) for n in SIZES)}")
    print("Path A: multiprocessing.Pipe, array pickled on send")
    print("Path B: shared_memory zero-copy; timed section is the ready/done handshake")
    print(f"Repeats: {REPEATS} (median after one warmup)")
    print()

    rng = np.random.default_rng(STUDENT_ID)
    source = rng.random(MAX_N, dtype=DTYPE)
    shm = shared_memory.SharedMemory(create=True, size=source.nbytes)
    shared = np.ndarray((MAX_N,), dtype=DTYPE, buffer=shm.buf)
    shared[:] = source

    parent, child = multiprocessing.Pipe()
    process = multiprocessing.Process(target=ipc_worker, args=(child,))
    process.start()
    child.close()
    rows = []
    try:
        parent.send(("attach", shm.name))
        if parent.recv() != "attached":
            raise SystemExit("worker failed to attach shared memory")

        for n in SIZES:
            payload = np.ascontiguousarray(source[:n])

            def path_a(payload=payload, n=n) -> None:
                parent.send(("pipe", payload))
                expect_probe(parent.recv(), payload, n)

            def path_b(n=n) -> None:
                parent.send(("shm", n))
                expect_probe(parent.recv(), source, n)

            path_a()
            path_b()
            path_a_ms = time_transfer(path_a)
            path_b_ms = time_transfer(path_b)
            rows.append((n, n * item_bytes, path_a_ms, path_b_ms))
    finally:
        try:
            parent.send(("stop",))
            parent.recv()
        except (OSError, EOFError, BrokenPipeError):
            process.terminate()
        process.join(timeout=10)
        parent.close()
        shm.close()
        shm.unlink()

    header = (
        f"{'N':>10} | {'Payload (bytes)':>16} | {'Path A (ms)':>12} | "
        f"{'Path B (ms)':>12} | {'A / B':>8}"
    )
    print(header)
    print("-" * len(header))
    for n, nbytes, path_a_ms, path_b_ms in rows:
        print(
            f"{n:10d} | {nbytes:16d} | {path_a_ms:12.3f} | "
            f"{path_b_ms:12.3f} | {path_a_ms / path_b_ms:8.3f}"
        )
    print()
    print(
        "Path A copies every element through the pipe. Path B only exchanges "
        "a small ready/done message, because both processes already map the same buffer."
    )


if __name__ == "__main__":
    main()
