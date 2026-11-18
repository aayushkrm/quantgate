"""The CLI: help text, exit codes (0 pass, 1 check failed, 2 usage or runtime error), --json, and
clean errors. Runs on SYNTHETIC fixtures only. No network, no model files."""

import json
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from quantgate.cli import _perf_metrics, _take, app, load_suite_config, scale_suite
from quantgate.report import load_report
from quantgate.schema import Variant

runner = CliRunner()

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONFIG = REPO / "configs" / "variants.yaml"
POLICY = REPO / "configs" / "gate.yaml"
COMMANDS = ["fetch", "eval", "gate", "report", "demo"]


def _invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


# --- help --------------------------------------------------------------------------------------


def test_top_level_help_lists_every_command() -> None:
    result = _invoke("--help")
    assert result.exit_code == 0
    for name in COMMANDS:
        assert name in result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_each_command_help(command: str) -> None:
    result = _invoke(command, "--help")
    assert result.exit_code == 0
    assert "--json" in result.output


# --- gate: exit codes on fixtures --------------------------------------------------------------


def test_gate_exit_0_when_every_variant_passes() -> None:
    result = _invoke("gate", str(FIXTURES / "report_all_pass.json"), "--policy", str(POLICY))
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output
    assert "Recommended: q3_k_m" in result.output


def test_gate_exit_1_when_a_rule_fails() -> None:
    result = _invoke("gate", str(FIXTURES / "report_mixed.json"), "--policy", str(POLICY))
    assert result.exit_code == 1
    assert "FAIL" in result.output
    assert "kl_mean:abs_max" in result.output


def test_gate_exit_1_on_hidden_danger() -> None:
    result = _invoke(
        "gate", str(FIXTURES / "report_hidden_danger.json"), "--policy", str(POLICY), "--json"
    )
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["hidden_danger"] == ["q3_k_m"]
    assert payload["recommended"] == "q4_k_m"
    assert payload["exit_code"] == 1


def test_gate_json_on_pass_has_exit_code_zero() -> None:
    result = _invoke(
        "gate", str(FIXTURES / "report_all_pass.json"), "--policy", str(POLICY), "--json"
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["exit_code"] == 0
    assert payload["stored_decisions_agree"] is True


def test_gate_accepts_results_directory(tmp_path: Path) -> None:
    (tmp_path / "report.json").write_text(
        (FIXTURES / "report_all_pass.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    result = _invoke("gate", str(tmp_path), "--policy", str(POLICY))
    assert result.exit_code == 0


def test_gate_exit_2_on_missing_report(tmp_path: Path) -> None:
    result = _invoke("gate", str(tmp_path / "absent.json"), "--policy", str(POLICY))
    assert result.exit_code == 2
    assert "Traceback" not in result.output
    assert "error" in result.output


def test_gate_exit_2_json_error_object(tmp_path: Path) -> None:
    result = _invoke("gate", str(tmp_path / "absent.json"), "--policy", str(POLICY), "--json")
    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["exit_code"] == 2
    assert payload["error"]


def test_gate_exit_2_on_wrong_schema_version(tmp_path: Path) -> None:
    data = json.loads((FIXTURES / "report_all_pass.json").read_text(encoding="utf-8"))
    data["schema_version"] = "9"
    path = tmp_path / "report.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    result = _invoke("gate", str(path), "--policy", str(POLICY))
    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_gate_exit_2_on_bad_policy(tmp_path: Path) -> None:
    policy = tmp_path / "gate.yaml"
    policy.write_text("rules: not-a-list\n", encoding="utf-8")
    result = _invoke("gate", str(FIXTURES / "report_all_pass.json"), "--policy", str(policy))
    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_gate_exit_2_when_reference_missing_from_report(tmp_path: Path) -> None:
    data = json.loads((FIXTURES / "report_all_pass.json").read_text(encoding="utf-8"))
    data["reference"] = "not-a-variant"
    path = tmp_path / "report.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    result = _invoke("gate", str(path), "--policy", str(POLICY))
    assert result.exit_code == 2
    assert "reference" in result.output


# --- report ------------------------------------------------------------------------------------


def test_report_writes_site_and_plots_from_fixture(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "report.json").write_text(
        (FIXTURES / "report_mixed.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    out = tmp_path / "site"
    result = _invoke("report", str(results), "--out-dir", str(out))
    assert result.exit_code == 0, result.output
    for name in ("index.html", "report.md", "report.json", "pareto_size_vs_arc.png"):
        path = out / name
        assert path.is_file(), name
        assert path.stat().st_size > 0
    assert (out / "pareto_size_vs_arc.svg").stat().st_size > 0


def test_report_json_mode_lists_written_files(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "report.json").write_text(
        (FIXTURES / "report_all_pass.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    result = _invoke("report", str(results), "--out-dir", str(tmp_path / "s"), "--json")
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert any(p.endswith("index.html") for p in payload["written"])


def test_report_splices_readme_block_between_markers(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "report.json").write_text(
        (FIXTURES / "report_mixed.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    readme = tmp_path / "README.md"
    readme.write_text(
        "head\n<!-- results:start -->\nold text\n<!-- results:end -->\ntail\n", encoding="utf-8"
    )
    result = _invoke(
        "report", str(results), "--out-dir", str(tmp_path / "s"), "--readme", str(readme)
    )
    assert result.exit_code == 0, result.output
    text = readme.read_text(encoding="utf-8")
    assert "old text" not in text
    assert "<!-- results:start -->" in text and "<!-- results:end -->" in text
    assert "Recommended variant: q4_k_m" in text


def test_report_exit_2_without_report(tmp_path: Path) -> None:
    result = _invoke("report", str(tmp_path), "--out-dir", str(tmp_path / "s"))
    assert result.exit_code == 2
    assert "Traceback" not in result.output


# --- eval: usage errors and missing inputs (no model is ever loaded here) ----------------------


def test_eval_rejects_unknown_suite() -> None:
    result = _invoke("eval", "-c", str(CONFIG), "--suite", "kl,bogus", "--json")
    assert result.exit_code == 2
    assert "bogus" in json.loads(result.output)["error"]


def test_eval_rejects_empty_suite() -> None:
    result = _invoke("eval", "-c", str(CONFIG), "--suite", " , ")
    assert result.exit_code == 2


def test_eval_rejects_unknown_only_variant(tmp_path: Path) -> None:
    result = _invoke(
        "eval",
        "-c",
        str(CONFIG),
        "--policy",
        str(POLICY),
        "--only",
        "no_such_quant",
        "--models-dir",
        str(tmp_path),
        "--out",
        str(tmp_path / "out"),
    )
    assert result.exit_code == 2
    assert "no_such_quant" in result.output


@pytest.mark.parametrize("bad", ["0", "-0.5", "1.5"])
def test_eval_rejects_out_of_range_n_scale(tmp_path: Path, bad: str) -> None:
    result = _invoke(
        "eval",
        "-c",
        str(CONFIG),
        "--policy",
        str(POLICY),
        "--n-scale",
        bad,
        "--models-dir",
        str(tmp_path),
        "--out",
        str(tmp_path / "out"),
    )
    assert result.exit_code == 2


def test_eval_exit_2_when_model_files_missing(tmp_path: Path) -> None:
    result = _invoke(
        "eval",
        "-c",
        str(CONFIG),
        "--policy",
        str(POLICY),
        "--suite",
        "arc",
        "--models-dir",
        str(tmp_path),
        "--out",
        str(tmp_path / "out"),
        "--json",
    )
    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert "quantgate fetch" in payload["error"]
    assert not (tmp_path / "out").exists()


def test_fetch_exit_2_on_missing_registry(tmp_path: Path) -> None:
    result = _invoke("fetch", "-c", str(tmp_path / "nope.yaml"), "--models-dir", str(tmp_path))
    assert result.exit_code == 2
    assert "Traceback" not in result.output


# --- eval helpers ------------------------------------------------------------------------------


def test_scale_suite_shrinks_counts_and_keeps_perf() -> None:
    base = load_suite_config(CONFIG)
    half = scale_suite(base, 0.5)
    assert half.arc.n == 150
    assert half.gsm8k.n == 50
    assert half.kl.n_passages == 25
    assert half.refusal.n_benign == 25
    assert half.kl.ctx == base.kl.ctx
    assert half.perf == base.perf


def test_scale_suite_is_identity_at_one() -> None:
    base = load_suite_config(CONFIG)
    assert scale_suite(base, 1.0) == base


def test_scaled_counts_never_drop_below_one() -> None:
    base = load_suite_config(CONFIG)
    tiny = scale_suite(base, 0.0001)
    assert tiny.arc.n == 1 and tiny.refusal.n_benign == 1


def test_take_is_seeded_and_keeps_order() -> None:
    items = list(range(50))
    first = _take(items, 10, seed=0)
    assert first == _take(items, 10, seed=0)
    assert first == sorted(first)
    assert len(first) == 10
    assert _take(items, 100, seed=0) == items


def test_perf_metrics_maps_child_keys_and_rss() -> None:
    raw = {
        "decode_tok_s_p50": 20.0,
        "decode_tok_s_min": 19.0,
        "decode_tok_s_max": 21.0,
        "prefill_tok_s_p50": 60.0,
        "peak_rss_bytes": 2 * 2**20,
    }
    metrics = {m.name: m for m in _perf_metrics(raw, 5, None, need_rss=True)}
    assert metrics["decode_tok_s"].value == 20.0
    assert (metrics["decode_tok_s"].ci_low, metrics["decode_tok_s"].ci_high) == (19.0, 21.0)
    assert metrics["peak_rss_mb"].value == pytest.approx(2.0)
    assert "file_mb" not in metrics


def test_perf_metrics_requires_rss_when_asked() -> None:
    from quantgate.cli import CliError

    raw = {"decode_tok_s_p50": 1.0, "prefill_tok_s_p50": 1.0}
    with pytest.raises(CliError, match="peak RSS"):
        _perf_metrics(raw, 2, None, need_rss=True)


# --- demo --------------------------------------------------------------------------------------


def test_demo_runs_on_fake_backend_and_is_labelled_synthetic() -> None:
    result = _invoke("demo", "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["synthetic"] is True
    assert payload["report"]["env_info"]["synthetic"] is True
    assert "SYNTHETIC" in payload["report"]["env_info"]["label"]


def test_demo_text_output_is_labelled() -> None:
    result = _invoke("demo")
    assert result.exit_code == 0
    assert "SYNTHETIC" in result.output


# --- eval end to end on FakeBackend (real orchestration, no model files, no network) -----------


@pytest.fixture
def fake_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Patch model loading and data loaders so `eval` runs for real on FakeBackend."""
    from quantgate import cli as cli_module
    from quantgate.backend import FakeBackend
    from quantgate.evals.mcq import McqItem
    from quantgate.evals.refusal import Prompt, PromptSet
    from quantgate.models import load_registry

    registry = load_registry(CONFIG)
    models = tmp_path / "models"
    for variant in registry.variants:
        if variant.name in ("fp16", "q4_k_m"):
            path = models / variant.repo_id / variant.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic gguf stand-in\n" * 8)

    def load_backend(path: Path, cfg: object) -> FakeBackend:
        perturbation = 0.0 if "fp16" in path.name else 0.05
        return FakeBackend(seed=0, perturbation=perturbation)

    mcq = [
        McqItem(
            item_id=f"s-{i}",
            question=f"synthetic question {i}?",
            choices=["alpha", "beta", "gamma", "delta"],
            labels=["A", "B", "C", "D"],
            answer_key="A",
        )
        for i in range(12)
    ]
    gsm = [(f"synthetic problem {i}", "work\n#### 7") for i in range(6)]
    passages = [f"synthetic passage {i} for drift measurement." for i in range(6)]

    def prompt_set(name: str, prefix: str, n: int) -> PromptSet:
        prompts = [Prompt(f"{prefix}-{i:04d}", f"synthetic {prefix} prompt {i}") for i in range(n)]
        return PromptSet(
            name=name,
            source="synthetic-test",
            fallback_reason=None,
            seed=0,
            total_available=n,
            selected_indices=list(range(n)),
            prompts=prompts,
        )

    monkeypatch.setattr(cli_module, "_load_backend", load_backend)
    monkeypatch.setattr(cli_module, "load_arc_easy", lambda: mcq)
    monkeypatch.setattr(cli_module, "load_gsm8k", lambda: gsm)
    monkeypatch.setattr(cli_module, "load_wikitext_passages", lambda: passages)
    monkeypatch.setattr(
        cli_module,
        "load_harmful_prompts",
        lambda n, seed: prompt_set("harmful", "harm", 8),
    )
    monkeypatch.setattr(
        cli_module,
        "load_benign_prompts",
        lambda n, seed: prompt_set("benign", "benign", 4),
    )
    return {"models": models, "out": tmp_path / "results"}


def test_eval_end_to_end_then_resume_skips_finished_work(
    fake_eval: dict[str, Path],
) -> None:
    args = [
        "eval",
        "-c",
        str(CONFIG),
        "--policy",
        str(POLICY),
        "--only",
        "q4_k_m",
        "--suite",
        "kl,arc,gsm8k,refusal",
        "--models-dir",
        str(fake_eval["models"]),
        "--out",
        str(fake_eval["out"]),
        "--json",
    ]
    first = _invoke(*args)
    assert first.exit_code == 0, first.output
    payload = json.loads(first.output)
    assert payload["ok"] is True
    assert "fp16/kl" in payload["ran"] or "fp16/arc" in payload["ran"]
    assert "q4_k_m/refusal" in payload["ran"]

    out = fake_eval["out"]
    for variant in ("fp16", "q4_k_m"):
        for suite in ("kl", "arc", "gsm8k", "refusal"):
            assert (out / variant / f"{suite}.json").is_file(), (variant, suite)
    assert (out / "env.json").is_file()
    assert (out / "q4_k_m" / "variant.json").is_file()

    report = load_report(out / "report.json")
    assert report.reference == "fp16"
    assert {vr.variant.name for vr in report.variants} == {"fp16", "q4_k_m"}
    assert [d.variant for d in report.decisions] == ["q4_k_m"]
    assert report.env_info["run"]["suites"] == ["kl", "arc", "gsm8k", "refusal"]
    assert report.env_info["run"]["n_scale"] == 1.0
    assert "/Users/" not in (out / "report.json").read_text(encoding="utf-8")

    second = _invoke(*args)
    assert second.exit_code == 0, second.output
    again = json.loads(second.output)
    assert again["ran"] == []
    assert len(again["skipped"]) == 8


def test_eval_n_scale_is_recorded_in_env(fake_eval: dict[str, Path]) -> None:
    result = _invoke(
        "eval",
        "-c",
        str(CONFIG),
        "--policy",
        str(POLICY),
        "--only",
        "q4_k_m",
        "--suite",
        "arc",
        "--n-scale",
        "0.5",
        "--models-dir",
        str(fake_eval["models"]),
        "--out",
        str(fake_eval["out"]),
    )
    assert result.exit_code == 0, result.output
    env = json.loads((fake_eval["out"] / "env.json").read_text(encoding="utf-8"))
    assert env["run"]["n_scale"] == 0.5
    assert env["run"]["suite"]["arc"]["n"] == 150


# --- fetch writes the lock file; eval uses it; quality backend limits ----------------------------


def test_fetch_writes_sidecar_lock_without_touching_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quantgate import cli as cli_module
    from quantgate.models import read_lock, sha256_file

    gguf = tmp_path / "x.gguf"
    gguf.write_bytes(b"observed bytes")
    variant = Variant(name="q8_0", repo_id="org/repo", filename="x.gguf")
    filled = variant.model_copy(
        update={"sha256": sha256_file(gguf), "file_bytes": gguf.stat().st_size}
    )
    monkeypatch.setattr(cli_module, "fetch_registry", lambda cfg, models: [(filled, gguf)])
    registry = tmp_path / "variants.yaml"
    registry.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    before = registry.read_text(encoding="utf-8")

    result = _invoke("fetch", "-c", str(registry), "--models-dir", str(tmp_path), "--json")
    assert result.exit_code == 0, result.output
    assert registry.read_text(encoding="utf-8") == before, "registry YAML must not be rewritten"
    lock = read_lock(tmp_path / "variants.lock.json")
    assert lock["q8_0"]["sha256"] == filled.sha256
    assert str(tmp_path) not in (tmp_path / "variants.lock.json").read_text(encoding="utf-8")


def test_load_backend_refuses_kl_context_that_does_not_fit() -> None:
    from quantgate import cli as cli_module

    cfg = load_suite_config(CONFIG)
    too_long = scale_suite(cfg, 1.0).model_copy(
        update={"kl": cfg.kl.model_copy(update={"ctx": 1024})}
    )
    with pytest.raises(cli_module.CliError, match="n_ctx"):
        cli_module._load_backend(Path("unused.gguf"), too_long)


def test_quality_backend_is_512_tokens_with_one_batch() -> None:
    from quantgate import cli as cli_module

    assert cli_module.CTX_BACKEND == 512
    assert cli_module.CTX_BACKEND >= load_suite_config(CONFIG).kl.ctx + cli_module.KL_CTX_MARGIN
