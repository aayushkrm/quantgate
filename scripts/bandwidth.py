#!/usr/bin/env python3
"""STREAM-style copy and scale bandwidth, printed as JSON (experiment E4, docs/prereg.md §5).

Kernels follow McCalpin's STREAM definitions, with numpy arrays of float64:
  copy:  c[:] = a          bytes moved = 2 * N * 8   (read a, write c)
  scale: b[:] = s * c      bytes moved = 2 * N * 8   (read c, write b)
Each kernel is split into slices across --threads worker threads (numpy releases the GIL for
these copies). Reported is the best of --repeats runs after one warm-up run, in decimal GB/s
(1 GB = 1e9 bytes), plus the median for reference.

Optional E4 check: pass --decode-tok-s and --model-file-bytes to add a ceiling_check block
(decode tok/s times model file size, against the best bandwidth measured).

Caveat: this measures sustained bandwidth for this machine and this thread count. It is not a
claim about other hardware.
"""

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from quantgate.perf import GB, bandwidth_ceiling_check

BYTES_PER_ELEMENT = 8  # float64
SCALE = 3.0

_Kernel = Callable[[np.ndarray, np.ndarray, slice], None]


def _slices(n: int, parts: int) -> list[slice]:
    """Split range(n) into `parts` contiguous slices of near-equal size."""
    bounds = np.linspace(0, n, parts + 1, dtype=np.int64)
    return [slice(int(lo), int(hi)) for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)]


def _copy_part(dst: np.ndarray, src: np.ndarray, sl: slice) -> None:
    """dst[sl] = src[sl]."""
    np.copyto(dst[sl], src[sl])


def _scale_part(dst: np.ndarray, src: np.ndarray, sl: slice) -> None:
    """dst[sl] = SCALE * src[sl]."""
    np.multiply(src[sl], SCALE, out=dst[sl])


def _run_kernel(
    pool: ThreadPoolExecutor,
    kernel: _Kernel,
    dst: np.ndarray,
    src: np.ndarray,
    slices: list[slice],
) -> float:
    """Run one kernel across the slices and return its wall time in seconds."""
    start = time.perf_counter()
    list(pool.map(lambda sl: kernel(dst, src, sl), slices))
    return time.perf_counter() - start


def measure_bandwidth(mb: float, repeats: int, threads: int) -> dict[str, float | int]:
    """Copy and scale bandwidth over arrays of about `mb` megabytes each.

    Returns best and median GB/s for each kernel, plus the array size and element count.
    """
    if mb <= 0 or repeats <= 0 or threads <= 0:
        raise ValueError("mb, repeats and threads must be positive")
    n = max(1, int(mb * 1e6 / BYTES_PER_ELEMENT))
    a = np.ones(n, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    slices = _slices(n, threads)
    kernel_bytes = 2 * n * BYTES_PER_ELEMENT

    times: dict[str, list[float]] = {"copy": [], "scale": []}
    with ThreadPoolExecutor(max_workers=threads) as pool:
        # One warm-up pass per kernel, not recorded.
        _run_kernel(pool, _copy_part, c, a, slices)
        _run_kernel(pool, _scale_part, b, c, slices)
        for _ in range(repeats):
            times["copy"].append(_run_kernel(pool, _copy_part, c, a, slices))
            times["scale"].append(_run_kernel(pool, _scale_part, b, c, slices))

    result: dict[str, float | int] = {
        "array_mb": round(n * BYTES_PER_ELEMENT / 1e6, 3),
        "elements": n,
        "threads": threads,
        "repeats": repeats,
    }
    for name, seconds in times.items():
        rates = [kernel_bytes / t / GB for t in seconds]
        result[f"{name}_gb_s"] = max(rates)
        result[f"{name}_gb_s_median"] = statistics.median(rates)
    return result


def main(argv: list[str] | None = None) -> int:
    """Print the bandwidth JSON. Optional E4 ceiling check when both decode flags are given."""
    parser = argparse.ArgumentParser(description="STREAM-style copy and scale bandwidth (GB/s).")
    parser.add_argument("--mb", type=float, default=256.0, help="megabytes per array (default 256)")
    parser.add_argument("--repeats", type=int, default=5, help="timed repeats per kernel")
    parser.add_argument("--threads", type=int, default=4, help="worker threads (default 4)")
    parser.add_argument("--decode-tok-s", type=float, default=None, help="measured decode tok/s")
    parser.add_argument(
        "--model-file-bytes", type=int, default=None, help="GGUF file size in bytes"
    )
    args = parser.parse_args(argv)
    if (args.decode_tok_s is None) != (args.model_file_bytes is None):
        print(
            "bandwidth: pass both --decode-tok-s and --model-file-bytes, or neither",
            file=sys.stderr,
        )
        return 2
    try:
        result: dict[str, Any] = dict(measure_bandwidth(args.mb, args.repeats, args.threads))
    except ValueError as exc:
        print(f"bandwidth: {exc}", file=sys.stderr)
        return 2
    if args.decode_tok_s is not None and args.model_file_bytes is not None:
        best = max(float(result["copy_gb_s"]), float(result["scale_gb_s"]))
        result["ceiling_check"] = bandwidth_ceiling_check(
            args.decode_tok_s, args.model_file_bytes, best
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
