"""Task 3P: serial inclusive prefix sum versus a two-pass parallel scan."""

import ctypes
import os
import platform
import sys
import threading
import time
from ctypes import wintypes
from statistics import median

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
from numba import njit

STUDENT_ID = 230103163
N = 2_000_000
REPEATS = 10


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
def serial_scan(data, output):
    """Inclusive prefix: output[i] = output[i - 1] + input[i]."""
    output[0] = data[0]
    for index in range(1, data.size):
        output[index] = output[index - 1] + data[index]


@njit(nogil=True)
def local_scan(data, output, start, end, chunk_sums, slot):
    """Inclusive prefix of one chunk, as if that chunk started at zero."""
    running = data[start]
    output[start] = running
    for index in range(start + 1, end):
        running = running + data[index]
        output[index] = running
    chunk_sums[slot] = running


@njit(nogil=True)
def exclusive_offsets(chunk_sums, offsets):
    """offsets[k] is the sum of every chunk before k."""
    running = np.int64(0)
    for index in range(chunk_sums.size):
        offsets[index] = running
        running += chunk_sums[index]


@njit(nogil=True)
def apply_offset(output, start, end, offset):
    for index in range(start, end):
        output[index] += offset


class ScanTeam:
    """Persistent workers so the timed region is the scan, not thread startup."""

    def __init__(self, data: np.ndarray, output: np.ndarray, workers: int) -> None:
        self.data = data
        self.output = output
        self.ranges = chunk_ranges(data.size, workers)
        parties = len(self.ranges)
        self.chunk_sums = np.zeros(parties, dtype=np.int64)
        self.offsets = np.zeros(parties, dtype=np.int64)
        self.mid_barrier = threading.Barrier(parties)
        self.offset_barrier = threading.Barrier(parties)
        self.lock = threading.Lock()
        self.go = threading.Condition(self.lock)
        self.generation = 0
        self.finished = 0
        self.stop = False
        self.threads = [
            threading.Thread(target=self._worker, args=(index, start, end), daemon=True)
            for index, (start, end) in enumerate(self.ranges)
        ]
        for thread in self.threads:
            thread.start()

    def _worker(self, index: int, start: int, end: int) -> None:
        seen = 0
        parties = len(self.ranges)
        while True:
            with self.lock:
                while self.generation == seen and not self.stop:
                    self.go.wait()
                if self.stop:
                    return
                seen = self.generation
            local_scan(self.data, self.output, start, end, self.chunk_sums, index)
            self.mid_barrier.wait()
            if index == 0:
                exclusive_offsets(self.chunk_sums, self.offsets)
            self.offset_barrier.wait()
            apply_offset(self.output, start, end, self.offsets[index])
            with self.lock:
                self.finished += 1
                if self.finished == parties:
                    self.go.notify()

    def run_once(self) -> float:
        parties = len(self.ranges)
        with self.lock:
            self.finished = 0
            self.generation += 1
            started = time.perf_counter()
            self.go.notify_all()
            while self.finished < parties:
                if not self.go.wait(timeout=60):
                    raise SystemExit("parallel scan timed out")
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        return elapsed_ms

    def close(self) -> None:
        with self.lock:
            self.stop = True
            self.generation += 1
            self.go.notify_all()
        for thread in self.threads:
            thread.join(timeout=5)


def time_serial(data: np.ndarray, output: np.ndarray) -> float:
    samples = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        serial_scan(data, output)
        samples.append((time.perf_counter() - started) * 1000.0)
    return median(samples)


def time_parallel(team: ScanTeam) -> float:
    samples = [team.run_once() for _ in range(REPEATS)]
    return median(samples)


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
    print(f"Array length: {N}")
    print("Dtype: int64")
    print("Value range: 1 to 10")
    print(f"Seed: {STUDENT_ID}")
    print("Serial: output[i] = output[i - 1] + input[i]")
    print("Parallel: local chunk scan, exclusive offset reduction, add offsets")
    print(f"Repeats: {REPEATS} (median after one warmup)")
    print()

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.integers(1, 11, size=N, dtype=np.int64)
    if data.shape != (N,) or data.dtype != np.int64:
        raise SystemExit("input array is not 2000000 int64 values")
    if int(data.min()) < 1 or int(data.max()) > 10:
        raise SystemExit("input values are outside 1 to 10")

    serial_output = np.empty(N, dtype=np.int64)
    parallel_output = np.empty(N, dtype=np.int64)

    # Compile both kernels before either timed run.
    warmup_in = np.array([1, 2, 3, 4], dtype=np.int64)
    warmup_out = np.empty(4, dtype=np.int64)
    warmup_sums = np.zeros(1, dtype=np.int64)
    warmup_offsets = np.zeros(1, dtype=np.int64)
    serial_scan(warmup_in, warmup_out)
    local_scan(warmup_in, warmup_out, 0, 4, warmup_sums, 0)
    exclusive_offsets(warmup_sums, warmup_offsets)
    apply_offset(warmup_out, 0, 4, np.int64(0))

    team = ScanTeam(data, parallel_output, physical_cores)
    try:
        serial_scan(data, serial_output)
        team.run_once()
        serial_ms = time_serial(data, serial_output)
        parallel_ms = time_parallel(team)
    finally:
        team.close()

    total = int(np.sum(data, dtype=np.int64))
    last_ok = parallel_output[-1] == serial_output[-1]
    full_ok = np.array_equal(parallel_output, serial_output) and int(serial_output[-1]) == total
    status = "PASS" if last_ok and full_ok else "FAIL"
    speedup = serial_ms / parallel_ms

    header = "Serial Time (ms) | Parallel Time (ms) | Speedup | Verification Status (PASS/FAIL)"
    print(header)
    print("-" * len(header))
    print(f"{serial_ms:16.3f} | {parallel_ms:18.3f} | {speedup:7.3f} | {status}")
    print()
    if speedup >= 1.0:
        print(
            "Each worker's add chain is only one chunk long, so the shorter "
            "dependency chain beats the extra offset pass."
        )
    else:
        print(
            "The offset pass moves the array a second time, and that extra work "
            "outweighs the shorter per-chunk dependency chain."
        )

    assert parallel_output[-1] == serial_output[-1]
    if not full_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
