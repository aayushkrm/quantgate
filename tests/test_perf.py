"""Perf measurement with a deterministic fake backend, pmset parsing, and the bandwidth script."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import quantgate.perf as perf
from quantgate.perf import (
    bandwidth_ceiling_check,
    build_prompt_text,
    measure_in_process,
    measure_variant,
    parse_pmset_batt,
    parse_pmset_therm,
    peak_rss_bytes,
    power_state,
    run_perf_subprocess,
    summarize_power_state,
)
from quantgate.schema import GenResult, PerfConfig

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent

# Fixtures: the shapes of `pmset -g batt` and `pmset -g therm` on macOS. Battery device ids
# and timestamps are removed, so nothing machine-specific is committed.
BATT_AC = (
    "Now drawing from 'AC Power'\n"
    " -InternalBattery-0 (id=0000000)\t71%; AC attached; not charging present: true\n"
)
BATT_BATTERY = (
    "Now drawing from 'Battery Power'\n"
    " -InternalBattery-0 (id=0000000)\t58%; discharging; 3:12 remaining present: true\n"
)
BATT_DESKTOP = "Now drawing from 'AC Power'\n"
THERM_OK = (
    "Note: No thermal warning level has been recorded\n"
    "Note: No performance warning level has been recorded\n"
    "CPU_Scheduler_Limit \t= 100\n"
    "CPU_Available_CPUs \t= 8\n"
    "CPU_Speed_Limit \t= 100\n"
)
THERM_THROTTLED = (
    "Note: Thermal warning level has been recorded\n"
    "Note: Performance warning level has been recorded\n"
    "CPU_Scheduler_Limit \t= 100\n"
    "CPU_Available_CPUs \t= 4\n"
    "CPU_Speed_Limit \t= 62\n"
)

PROMPT_TOKENS = 64
GEN_TOKENS = 16
FAKE_PREFILL_TOK_S = 1000.0
FAKE_DECODE_TOK_S = 50.0


class _TimedFake:
    """Deterministic backend: prefill runs at 1000 tok/s, decode at 50 tok/s. Whitespace tokens."""

    n_vocab = 32

    def __init__(self) -> None:
        self.generate_calls = 0
        self.closed = False

    def tokenize(self, text: str) -> list[int]:
        return list(range(len(text.split())))

    def logprobs(self, tokens: list[int]) -> object:
        raise NotImplementedError

    def score_continuations(self, prompt: str, options: list[str]) -> list[float]:
        raise NotImplementedError

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> GenResult:
        self.generate_calls += 1
        n_prompt = len(self.tokenize(messages[0]["content"]))
        return GenResult(
            text="ok",
            n_prompt_tokens=n_prompt,
            n_gen_tokens=max_tokens,
            prefill_s=n_prompt / FAKE_PREFILL_TOK_S,
            decode_s=max_tokens / FAKE_DECODE_TOK_S,
        )

    def close(self) -> None:
        self.closed = True


def make_timed_fake(model_path: Path, threads: int) -> _TimedFake:
    """Factory used by the subprocess tests via QUANTGATE_PERF_BACKEND=test_perf:make_timed_fake."""
    del model_path, threads
    return _TimedFake()


def _cfg(warmup: int = 2, repeats: int = 5) -> PerfConfig:
    return PerfConfig(
        prompt_tokens=PROMPT_TOKENS,
        gen_tokens=GEN_TOKENS,
        warmup=warmup,
        repeats=repeats,
        threads=4,
    )


def test_build_prompt_text_hits_the_token_budget() -> None:
    backend = _TimedFake()
    text = build_prompt_text(backend, 512)
    assert len(backend.tokenize(text)) == 512
    assert len(backend.tokenize(build_prompt_text(backend, 7))) == 7


def test_measure_in_process_uses_warmup_and_repeats() -> None:
    backend = _TimedFake()
    metrics = measure_in_process(backend, _cfg(warmup=2, repeats=5))
    assert backend.generate_calls == 2 + 5
    assert metrics["prefill_tok_s_p50"] == pytest.approx(FAKE_PREFILL_TOK_S)
    assert metrics["prefill_tok_s_min"] == pytest.approx(FAKE_PREFILL_TOK_S)
    assert metrics["prefill_tok_s_max"] == pytest.approx(FAKE_PREFILL_TOK_S)
    assert metrics["decode_tok_s_p50"] == pytest.approx(FAKE_DECODE_TOK_S)
    assert metrics["decode_tok_s_min"] == pytest.approx(FAKE_DECODE_TOK_S)
    assert metrics["decode_tok_s_max"] == pytest.approx(FAKE_DECODE_TOK_S)
    assert metrics["n_prompt_tokens_p50"] == PROMPT_TOKENS
    assert metrics["n_gen_tokens_min"] == GEN_TOKENS
    assert metrics["gen_short"] == 0.0
    assert metrics["repeats"] == 5.0


def test_measure_in_process_flags_early_stop() -> None:
    class _Short(_TimedFake):
        def generate(self, messages, max_tokens, temperature, seed):  # type: ignore[no-untyped-def]
            res = super().generate(messages, max_tokens, temperature, seed)
            return GenResult(
                text=res.text,
                n_prompt_tokens=res.n_prompt_tokens,
                n_gen_tokens=max_tokens // 2,
                prefill_s=res.prefill_s,
                decode_s=(max_tokens // 2) / FAKE_DECODE_TOK_S,
            )

    metrics = measure_in_process(_Short(), _cfg(warmup=0, repeats=2))
    assert metrics["gen_short"] == 1.0
    assert metrics["decode_tok_s_p50"] == pytest.approx(FAKE_DECODE_TOK_S)


def test_measure_in_process_rejects_zero_timing() -> None:
    class _Zero(_TimedFake):
        def generate(self, messages, max_tokens, temperature, seed):  # type: ignore[no-untyped-def]
            return GenResult(
                text="x", n_prompt_tokens=3, n_gen_tokens=3, prefill_s=0.0, decode_s=0.1
            )

    with pytest.raises(ValueError, match="non-positive"):
        measure_in_process(_Zero(), _cfg(warmup=0, repeats=1))


def test_peak_rss_bytes_normalises_macos_and_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(ru_maxrss=1000)
    monkeypatch.setattr(perf.resource, "getrusage", lambda who: fake)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert peak_rss_bytes() == 1000
    monkeypatch.setattr(sys, "platform", "linux")
    assert peak_rss_bytes() == 1000 * 1024


def test_peak_rss_bytes_is_plausible_for_this_process() -> None:
    assert peak_rss_bytes() > 10_000_000


def test_power_state_is_empty_off_macos() -> None:
    def _must_not_run(cmd: list[str]) -> str:
        raise AssertionError(f"pmset must not run on this platform: {cmd}")

    assert power_state(run_cmd=_must_not_run, system="Linux") == {
        "pmset_batt": "",
        "pmset_therm": "",
    }


def test_power_state_calls_pmset_on_macos() -> None:
    calls: list[list[str]] = []

    def _fake(cmd: list[str]) -> str:
        calls.append(cmd)
        return "out"

    state = power_state(run_cmd=_fake, system="Darwin")
    assert state == {"pmset_batt": "out", "pmset_therm": "out"}
    assert calls == [["pmset", "-g", "batt"], ["pmset", "-g", "therm"]]


def test_parse_pmset_batt_fixtures() -> None:
    ac = parse_pmset_batt(BATT_AC)
    assert (ac.source, ac.percent) == ("AC Power", 71)
    bat = parse_pmset_batt(BATT_BATTERY)
    assert (bat.source, bat.percent) == ("Battery Power", 58)
    desk = parse_pmset_batt(BATT_DESKTOP)
    assert (desk.source, desk.percent) == ("AC Power", None)
    assert parse_pmset_batt("") == perf.BatteryState(source=None, percent=None)


def test_parse_pmset_therm_fixtures() -> None:
    ok = parse_pmset_therm(THERM_OK)
    assert ok.warning_recorded is False
    assert ok.performance_warning_recorded is False
    assert (ok.cpu_speed_limit, ok.cpu_scheduler_limit) == (100, 100)

    hot = parse_pmset_therm(THERM_THROTTLED)
    assert hot.warning_recorded is True
    assert hot.performance_warning_recorded is True
    assert (hot.cpu_speed_limit, hot.cpu_scheduler_limit) == (62, 100)


def test_summarize_power_state_has_no_raw_text() -> None:
    raw = {"pmset_batt": BATT_AC, "pmset_therm": THERM_OK}
    summary = summarize_power_state(raw)
    assert summary["pmset_available"] is True
    assert summary["power_source"] == "AC Power"
    assert summary["battery_percent"] == 71
    assert summary["cpu_speed_limit"] == 100
    assert "id=" not in json.dumps(summary)
    off = summarize_power_state({"pmset_batt": "", "pmset_therm": ""})
    assert off["pmset_available"] is False and off["power_source"] is None


def test_bandwidth_ceiling_check_exact_values() -> None:
    ok = bandwidth_ceiling_check(
        decode_tok_s=10.0, model_file_bytes=2_000_000_000, measured_gb_s=40.0
    )
    assert ok["model_gb"] == pytest.approx(2.0)
    assert ok["implied_gb_s"] == pytest.approx(20.0)
    assert ok["ratio"] == pytest.approx(0.5)
    assert ok["within_ceiling"] is True

    over = bandwidth_ceiling_check(
        decode_tok_s=10.0, model_file_bytes=2_000_000_000, measured_gb_s=10.0
    )
    assert over["ratio"] == pytest.approx(2.0)
    assert over["within_ceiling"] is False
    with pytest.raises(ValueError):
        bandwidth_ceiling_check(10.0, 1, 0.0)


def _subprocess_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(perf.BACKEND_ENV, "test_perf:make_timed_fake")
    monkeypatch.setenv(perf.TESTING_ENV, "1")
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(p for p in (str(TESTS_DIR), existing) if p))


def test_run_perf_subprocess_returns_metrics_and_rss(monkeypatch: pytest.MonkeyPatch) -> None:
    _subprocess_env(monkeypatch)
    metrics = run_perf_subprocess(Path("/nonexistent/fake.gguf"), _cfg(warmup=1, repeats=3))
    assert metrics["decode_tok_s_p50"] == pytest.approx(FAKE_DECODE_TOK_S)
    assert metrics["prefill_tok_s_p50"] == pytest.approx(FAKE_PREFILL_TOK_S)
    assert metrics["peak_rss_bytes"] > 10_000_000
    assert metrics["repeats"] == 3.0


def test_measure_variant_stores_file_name_not_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _subprocess_env(monkeypatch)
    result = measure_variant(
        Path("/Users/someone/models/fake-q4_k_m.gguf"),
        prompt_tokens=PROMPT_TOKENS,
        gen_tokens=GEN_TOKENS,
        warmup=1,
        repeats=2,
        threads=2,
    )
    assert result.model_file == "fake-q4_k_m.gguf"
    assert "peak_rss_bytes" not in result.metrics
    assert result.peak_rss_bytes > 0
    assert result.metrics["decode_tok_s_p50"] == pytest.approx(FAKE_DECODE_TOK_S)
    assert result.config["threads"] == 2
    assert "/Users/" not in json.dumps(result.to_dict())


def test_entrypoint_json_uses_factory_and_hides_model_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(perf.BACKEND_ENV, "test_perf:make_timed_fake")
    monkeypatch.setenv(perf.TESTING_ENV, "1")
    code = perf.main(
        [
            "--model",
            "/secret/dir/m.gguf",
            "--prompt-tokens",
            "32",
            "--gen-tokens",
            "8",
            "--warmup",
            "0",
            "--repeats",
            "2",
            "--threads",
            "1",
            "--json",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(out)
    assert payload["model_file"] == "m.gguf"
    assert "/secret" not in out
    assert payload["metrics"]["n_prompt_tokens_p50"] == 32
    assert payload[perf.PEAK_RSS_KEY] > 0
    assert perf.PEAK_RSS_KEY not in payload["metrics"]


def test_entrypoint_bad_factory_spec_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(perf.BACKEND_ENV, "no_colon_here")
    monkeypatch.setenv(perf.TESTING_ENV, "1")
    assert perf.main(["--model", "x.gguf", "--json"]) == 2
    assert "module:callable" in capsys.readouterr().err


def test_bandwidth_script_smoke_test() -> None:
    script = ROOT / "scripts" / "bandwidth.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--mb", "4", "--repeats", "2", "--threads", "2"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["copy_gb_s"] > 0
    assert data["scale_gb_s"] > 0
    assert data["threads"] == 2
    assert "ceiling_check" not in data


def test_bandwidth_script_ceiling_check_flags_over_ceiling() -> None:
    script = ROOT / "scripts" / "bandwidth.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mb",
            "4",
            "--repeats",
            "1",
            "--threads",
            "1",
            "--decode-tok-s",
            "1e9",
            "--model-file-bytes",
            "1000000000",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["ceiling_check"]["within_ceiling"] is False


def test_bandwidth_script_requires_both_decode_flags() -> None:
    script = ROOT / "scripts" / "bandwidth.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--mb", "1", "--repeats", "1", "--decode-tok-s", "5"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 2


def test_factory_seam_is_ignored_without_testing_flag() -> None:
    factory, note = perf._load_backend_factory({perf.BACKEND_ENV: "test_perf:make_timed_fake"})
    assert factory is perf._default_factory
    assert "ignored" in note and perf.TESTING_ENV in note


def test_factory_seam_is_used_only_with_testing_flag() -> None:
    factory, note = perf._load_backend_factory(
        {perf.BACKEND_ENV: "test_perf:make_timed_fake", perf.TESTING_ENV: "1"}
    )
    assert factory.__name__ == "make_timed_fake"
    assert "testing only" in note


def test_default_factory_loads_without_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class _Spy:
        def __init__(self, path: Path, **kwargs: object) -> None:
            seen["path"] = path
            seen.update(kwargs)

    monkeypatch.setattr(perf, "LlamaCppBackend", _Spy)
    perf._default_factory(Path("m.gguf"), 4)
    assert seen["logits_all"] is False
    assert seen["n_threads"] == 4
    assert seen["n_ctx"] == perf.PERF_N_CTX


def test_run_perf_subprocess_requires_rss_in_child_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = json.dumps({"model_file": "m.gguf", "metrics": {"decode_tok_s_p50": 1.0}})

    def _fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(perf.subprocess, "run", _fake_run)
    with pytest.raises(RuntimeError, match="contract"):
        run_perf_subprocess(Path("m.gguf"), _cfg(warmup=0, repeats=1))


def test_run_perf_subprocess_adds_top_level_rss(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = json.dumps(
        {
            "model_file": "m.gguf",
            "metrics": {"decode_tok_s_p50": 2.0},
            perf.PEAK_RSS_KEY: 123456.0,
        }
    )

    def _fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(perf.subprocess, "run", _fake_run)
    metrics = run_perf_subprocess(Path("m.gguf"), _cfg(warmup=0, repeats=1))
    assert metrics[perf.PEAK_RSS_KEY] == 123456.0
    assert metrics["decode_tok_s_p50"] == 2.0
