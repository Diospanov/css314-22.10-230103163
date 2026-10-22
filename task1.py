"""Task 1: hardware probe and single-thread serial baseline."""

import os
import platform
import time

# Keep numeric kernels on one thread. Must be set before NumPy imports.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np

STUDENT_ID = 230103163
N = 20_000_000


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


def main() -> None:
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU model: {cpu_model()}")
    print(f"Logical cores: {os.cpu_count()}")

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.random(N, dtype=np.float64)
    print(f"Array length: {data.size}")
    print(f"Dtype: {data.dtype}")
    print(f"Seed: {STUDENT_ID}")

    # Population mean and standard deviation, one thread, no worker pool.
    start = time.perf_counter()
    total = float(np.sum(data, dtype=np.float64))
    mean = total / N
    variance = float(np.sum((data - mean) ** 2, dtype=np.float64) / N)
    std = float(np.sqrt(variance))
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print(f"Sum: {total:.10f}")
    print(f"Standard deviation: {std:.10f}")
    print(f"Wall-clock time (ms): {elapsed_ms:.3f}")


if __name__ == "__main__":
    main()
