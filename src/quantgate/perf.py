"""CPU speed and memory measurement.

Each variant is measured in a fresh subprocess, so that memory and warm-up are not shared
between variants. Prefill and decode tok/s are reported as p50 and min-max over repeats. Peak
RSS comes from resource.getrusage (ru_maxrss is bytes on macOS, kilobytes on Linux). Power
and thermal state (pmset -g batt, pmset -g therm) are recorded alongside the numbers.
"""

from pathlib import Path

from quantgate.schema import Backend, PerfConfig


def peak_rss_bytes() -> int:
    """Peak resident set size of the current process, in bytes, normalised across platforms."""
    raise NotImplementedError


def power_state() -> dict[str, str]:
    """Output of `pmset -g batt` and `pmset -g therm` on macOS. Empty strings elsewhere."""
    raise NotImplementedError


def measure_in_process(backend: Backend, cfg: PerfConfig) -> dict[str, float]:
    """Warm-up runs, then repeats. Timings come from GenResult.prefill_s and decode_s.

    Returns prefill_tok_s_p50, decode_tok_s_p50, and decode min/max.
    """
    raise NotImplementedError


def run_perf_subprocess(
    model_path: Path,
    cfg: PerfConfig,
    timeout_s: float = 600.0,
) -> dict[str, float]:
    """Run measure_in_process in a fresh Python subprocess for one GGUF file and return its JSON."""
    raise NotImplementedError
