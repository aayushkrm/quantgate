"""CPU speed and memory measurement.

Each variant is measured in a fresh subprocess (``python -m quantgate.perf``), so that memory and
warm-up are not shared between variants. Prefill and decode tok/s are reported as p50 and
min-max over repeats. Peak RSS comes from resource.getrusage (ru_maxrss is bytes on macOS,
kilobytes on Linux). Power and thermal state (pmset -g batt, pmset -g therm) are recorded
before and after each variant. On Linux the pmset fields are empty strings.

The child uses LlamaCppBackend by default. For tests, the environment variable
QUANTGATE_PERF_BACKEND can name an importable factory as ``module:callable``; the factory is
called as ``factory(model_path, threads)`` and must return a Backend. The CLI never sets it.

Machine-state caveat: the numbers describe the machine they were measured on (CPU-only). They
are not a claim about other hardware.
"""

import argparse
import importlib
import json
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from quantgate.backend import LlamaCppBackend
from quantgate.schema import Backend, GenResult, PerfConfig

BACKEND_ENV = "QUANTGATE_PERF_BACKEND"
GB = 1e9  # decimal gigabytes, the STREAM convention

BackendFactory = Callable[[Path, int], Backend]

_FILLER = "The engineer reviewed the quarterly numbers and wrote a short note about each trend. "
_BATT_SOURCE_RE = re.compile(r"drawing from '([^']+)'")
_BATT_PERCENT_RE = re.compile(r"(\d{1,3})%")
_CPU_SPEED_RE = re.compile(r"CPU_Speed_Limit\s*=\s*(\d+)")
_CPU_SCHED_RE = re.compile(r"CPU_Scheduler_Limit\s*=\s*(\d+)")


@dataclass(frozen=True)
class BatteryState:
    """Parsed `pmset -g batt`: power source name (e.g. 'AC Power') and charge percent."""

    source: str | None
    percent: int | None


@dataclass(frozen=True)
class ThermalState:
    """Parsed `pmset -g therm`. Limits are percentages (100 means unthrottled)."""

    warning_recorded: bool
    performance_warning_recorded: bool
    cpu_speed_limit: int | None
    cpu_scheduler_limit: int | None


@dataclass(frozen=True)
class PerfResult:
    """Measurement for one variant. Contains the file name only, never the full path."""

    model_file: str
    config: dict[str, Any]
    metrics: dict[str, float]
    peak_rss_bytes: int
    power_before: dict[str, Any] = field(default_factory=dict)
    power_after: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for JSON output."""
        return asdict(self)


def peak_rss_bytes() -> int:
    """Peak resident set size of the current process, in bytes, normalised across platforms.

    macOS reports ru_maxrss in bytes; Linux reports kilobytes.
    """
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _run_text(cmd: list[str]) -> str:
    """Stdout of a short command, or an empty string if it fails or is missing."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def power_state(
    run_cmd: Callable[[list[str]], str] | None = None,
    system: str | None = None,
) -> dict[str, str]:
    """Output of `pmset -g batt` and `pmset -g therm` on macOS. Empty strings elsewhere.

    run_cmd and system are injection points for tests.
    """
    current = system if system is not None else platform.system()
    if current != "Darwin":
        return {"pmset_batt": "", "pmset_therm": ""}
    runner = run_cmd if run_cmd is not None else _run_text
    return {
        "pmset_batt": runner(["pmset", "-g", "batt"]),
        "pmset_therm": runner(["pmset", "-g", "therm"]),
    }


def parse_pmset_batt(text: str) -> BatteryState:
    """Parse `pmset -g batt` output. Desktops without a battery give source only or nothing.

    Example line: "Now drawing from 'AC Power'". A battery line carries "NN%; ...".
    """
    source_match = _BATT_SOURCE_RE.search(text)
    percent_match = _BATT_PERCENT_RE.search(text)
    return BatteryState(
        source=source_match.group(1) if source_match else None,
        percent=int(percent_match.group(1)) if percent_match else None,
    )


def _warning_recorded(text: str, kind: str) -> bool:
    """True when pmset reports a recorded warning of this kind ('thermal' or 'performance')."""
    lower = text.lower()
    return f"{kind} warning level has been recorded" in lower and (
        f"no {kind} warning level" not in lower
    )


def parse_pmset_therm(text: str) -> ThermalState:
    """Parse `pmset -g therm` output: warning notes plus CPU speed and scheduler limits."""
    speed = _CPU_SPEED_RE.search(text)
    sched = _CPU_SCHED_RE.search(text)
    return ThermalState(
        warning_recorded=_warning_recorded(text, "thermal"),
        performance_warning_recorded=_warning_recorded(text, "performance"),
        cpu_speed_limit=int(speed.group(1)) if speed else None,
        cpu_scheduler_limit=int(sched.group(1)) if sched else None,
    )


def summarize_power_state(raw: dict[str, str]) -> dict[str, Any]:
    """Parsed AC/battery and thermal summary, the form stored in results (no raw text)."""
    batt = parse_pmset_batt(raw.get("pmset_batt", ""))
    therm = parse_pmset_therm(raw.get("pmset_therm", ""))
    return {
        "pmset_available": bool(raw.get("pmset_batt") or raw.get("pmset_therm")),
        "power_source": batt.source,
        "battery_percent": batt.percent,
        "thermal_warning_recorded": therm.warning_recorded,
        "performance_warning_recorded": therm.performance_warning_recorded,
        "cpu_speed_limit": therm.cpu_speed_limit,
        "cpu_scheduler_limit": therm.cpu_scheduler_limit,
    }


_FILLER_REPEATS = 64  # enough words for any prompt_tokens used here (one word per token minimum)


def build_prompt_text(backend: Backend, target_tokens: int) -> str:
    """Fixed filler text with at most target_tokens tokens, as counted by backend.tokenize.

    Binary search on the word count. The result is deterministic for a given backend.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    words = (_FILLER * (target_tokens // 8 + _FILLER_REPEATS)).split()
    lo, hi = 1, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(backend.tokenize(" ".join(words[:mid]))) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])


def _summary(prefix: str, values: Sequence[float]) -> dict[str, float]:
    """p50, min and max of a list of repeats, keyed <prefix>_p50, _min, _max."""
    return {
        f"{prefix}_p50": float(statistics.median(values)),
        f"{prefix}_min": float(min(values)),
        f"{prefix}_max": float(max(values)),
    }


def _rate(count: int, seconds: float, what: str) -> float:
    """tokens per second, rejecting non-positive timings from the backend."""
    if seconds <= 0.0:
        raise ValueError(f"backend returned a non-positive {what} time: {seconds}")
    return count / seconds


def measure_in_process(backend: Backend, cfg: PerfConfig) -> dict[str, float]:
    """Warm-up runs, then repeats. Timings come from GenResult.prefill_s and decode_s.

    Returns prefill_tok_s and decode_tok_s as _p50, _min and _max over the repeats, plus
    prefill_s (time to first token, p50), the actual prompt and generated token counts, and
    gen_short, which is 1.0 when any repeat stopped before gen_tokens (EOS).
    """
    if cfg.repeats <= 0:
        raise ValueError("repeats must be positive")
    text = build_prompt_text(backend, cfg.prompt_tokens)
    messages = [{"role": "user", "content": text}]

    def run() -> GenResult:
        return backend.generate(messages, max_tokens=cfg.gen_tokens, temperature=0.0, seed=0)

    for _ in range(cfg.warmup):
        run()

    prefill_tok_s: list[float] = []
    decode_tok_s: list[float] = []
    prefill_s: list[float] = []
    n_prompt: list[int] = []
    n_gen: list[int] = []
    for _ in range(cfg.repeats):
        res = run()
        if res.n_gen_tokens <= 0:
            raise ValueError("backend generated no tokens; decode rate is undefined")
        prefill_tok_s.append(_rate(res.n_prompt_tokens, res.prefill_s, "prefill"))
        decode_tok_s.append(_rate(res.n_gen_tokens, res.decode_s, "decode"))
        prefill_s.append(res.prefill_s)
        n_prompt.append(res.n_prompt_tokens)
        n_gen.append(res.n_gen_tokens)

    out: dict[str, float] = {}
    out.update(_summary("prefill_tok_s", prefill_tok_s))
    out.update(_summary("decode_tok_s", decode_tok_s))
    out["prefill_s_p50"] = float(statistics.median(prefill_s))
    out["n_prompt_tokens_p50"] = float(statistics.median(n_prompt))
    out["n_gen_tokens_min"] = float(min(n_gen))
    out["n_gen_tokens_max"] = float(max(n_gen))
    out["gen_short"] = 1.0 if min(n_gen) < cfg.gen_tokens else 0.0
    out["warmup"] = float(cfg.warmup)
    out["repeats"] = float(cfg.repeats)
    return out


def _default_factory(model_path: Path, threads: int) -> Backend:
    """The production backend: LlamaCppBackend with the requested thread count."""
    return LlamaCppBackend(model_path, n_threads=threads)


def _load_backend_factory() -> BackendFactory:
    """The factory named by QUANTGATE_PERF_BACKEND (module:callable), else the llama default."""
    spec = os.environ.get(BACKEND_ENV, "").strip()
    if not spec:
        return _default_factory
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"{BACKEND_ENV} must look like module:callable, got {spec!r}")
    module = importlib.import_module(module_name)
    return cast(BackendFactory, getattr(module, attr))


def run_perf_subprocess(
    model_path: Path,
    cfg: PerfConfig,
    timeout_s: float = 600.0,
) -> dict[str, float]:
    """Run measure_in_process in a fresh Python subprocess for one GGUF file.

    Returns the metrics dict of the child, plus peak_rss_bytes measured inside the child.
    Raises RuntimeError on timeout or a non-zero exit.
    """
    cmd = [
        sys.executable,
        "-m",
        "quantgate.perf",
        "--model",
        str(model_path),
        "--prompt-tokens",
        str(cfg.prompt_tokens),
        "--gen-tokens",
        str(cfg.gen_tokens),
        "--warmup",
        str(cfg.warmup),
        "--repeats",
        str(cfg.repeats),
        "--threads",
        str(cfg.threads),
        "--json",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"perf subprocess timed out after {timeout_s} s") from exc
    if proc.returncode != 0:
        tail = proc.stderr.strip()[-2000:]
        raise RuntimeError(f"perf subprocess exited with {proc.returncode}: {tail}")
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("perf subprocess printed no JSON")
    payload = json.loads(lines[-1])
    return {str(k): float(v) for k, v in payload["metrics"].items()}


def measure_variant(
    model_path: Path,
    prompt_tokens: int = 512,
    gen_tokens: int = 128,
    warmup: int = 2,
    repeats: int = 5,
    threads: int = 4,
    timeout_s: float = 600.0,
) -> PerfResult:
    """Measure one GGUF file in a fresh subprocess, with power state before and after."""
    cfg = PerfConfig(
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
        warmup=warmup,
        repeats=repeats,
        threads=threads,
    )
    before = summarize_power_state(power_state())
    metrics = run_perf_subprocess(model_path, cfg, timeout_s=timeout_s)
    after = summarize_power_state(power_state())
    rss = int(metrics.pop("peak_rss_bytes", 0.0))
    return PerfResult(
        model_file=model_path.name,
        config=cfg.model_dump(),
        metrics=metrics,
        peak_rss_bytes=rss,
        power_before=before,
        power_after=after,
    )


def bandwidth_ceiling_check(
    decode_tok_s: float,
    model_file_bytes: int,
    measured_gb_s: float,
) -> dict[str, float | bool]:
    """E4 check: decode tok/s times model file size should not exceed measured bandwidth.

    Decode of a dense model reads roughly the whole file per token, so
    implied_gb_s = decode_tok_s * model_GB. A ratio above 1.0 means the decode number or the
    bandwidth measurement is wrong (DECISION.md E4, research/03 section 6.3).
    """
    if measured_gb_s <= 0.0:
        raise ValueError("measured bandwidth must be positive")
    model_gb = model_file_bytes / GB
    implied = decode_tok_s * model_gb
    ratio = implied / measured_gb_s
    return {
        "model_gb": model_gb,
        "implied_gb_s": implied,
        "measured_gb_s": measured_gb_s,
        "ratio": ratio,
        "within_ceiling": ratio <= 1.0,
    }


def main(argv: list[str] | None = None) -> int:
    """Child entrypoint: measure one model in this process and print JSON. Exit 2 on error."""
    parser = argparse.ArgumentParser(prog="python -m quantgate.perf")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--json", action="store_true", help="print a JSON object")
    args = parser.parse_args(argv)
    try:
        cfg = PerfConfig(
            prompt_tokens=args.prompt_tokens,
            gen_tokens=args.gen_tokens,
            warmup=args.warmup,
            repeats=args.repeats,
            threads=args.threads,
        )
        factory = _load_backend_factory()
        backend = factory(args.model, args.threads)
        try:
            metrics = measure_in_process(backend, cfg)
            metrics["peak_rss_bytes"] = float(peak_rss_bytes())
        finally:
            backend.close()
    except Exception as exc:  # usage or runtime error: report on stderr, exit 2
        print(f"quantgate.perf: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    payload = {"model_file": args.model.name, "metrics": metrics}
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        for key in sorted(metrics):
            print(f"{key}: {metrics[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
