"""Report JSON, Markdown, HTML, README block and the committed JSON schema.

Inputs are SYNTHETIC fixtures in tests/fixtures/. No network, no model files.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from quantgate.gate import evaluate, load_policy, recommend
from quantgate.report import (
    README_END,
    README_START,
    build_report,
    load_report,
    render_html,
    render_markdown,
    render_readme_block,
    update_readme,
    write_report,
    write_site,
)
from quantgate.schema import MetricValue, QuantReport, Variant, VariantResult

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SCHEMA_FILE = REPO / "schemas" / "quant_report.schema.json"


@pytest.fixture
def mixed() -> QuantReport:
    return load_report(FIXTURES / "report_mixed.json")


@pytest.fixture
def hidden() -> QuantReport:
    return load_report(FIXTURES / "report_hidden_danger.json")


@pytest.fixture
def all_pass() -> QuantReport:
    return load_report(FIXTURES / "report_all_pass.json")


def test_fixtures_are_synthetic_labelled(mixed: QuantReport, hidden: QuantReport) -> None:
    for report in (mixed, hidden):
        assert report.env_info["synthetic"] is True
        assert "SYNTHETIC" in report.env_info["label"]


def test_write_then_load_round_trip(tmp_path: Path, mixed: QuantReport) -> None:
    path = tmp_path / "nested" / "report.json"
    write_report(mixed, path)
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert load_report(path) == mixed


def test_load_rejects_wrong_schema_version(tmp_path: Path, mixed: QuantReport) -> None:
    data = json.loads(mixed.model_dump_json())
    data["schema_version"] = "2"
    path = tmp_path / "report.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_report(path)


def test_build_report_assembles_fields(mixed: QuantReport) -> None:
    rebuilt = build_report(
        model_family=mixed.model_family,
        reference=mixed.reference,
        results=mixed.variants,
        decisions=mixed.decisions,
        recommended=mixed.recommended,
        env_info=mixed.env_info,
    )
    assert rebuilt == mixed


def test_fixture_recommendation_is_consistent_with_gate(
    mixed: QuantReport, hidden: QuantReport, all_pass: QuantReport
) -> None:
    policy = load_policy(REPO / "configs" / "gate.yaml")
    for report in (mixed, hidden, all_pass):
        decisions = evaluate(report.variants, report.reference, policy)
        assert recommend(decisions, report.variants) == report.recommended


def test_markdown_has_tables_gate_and_environment(mixed: QuantReport) -> None:
    text = render_markdown(mixed)
    assert "# quantgate report: synthetic-qwen-fixture" in text
    assert "> **SYNTHETIC.**" in text
    assert "## Quality metrics" in text
    assert "## Speed and size" in text
    assert "| q2_k | FAIL | no |" in text
    assert "Recommended variant: q4_k_m" in text
    assert "## Environment" in text
    assert "SYNTHETIC CPU" in text


def test_markdown_shows_confidence_intervals(mixed: QuantReport) -> None:
    text = render_markdown(mixed)
    assert "59.5" in text  # value from the q8_0 fixture
    assert "[54.5, 64.5] (n=300)" in text  # 95% CI bracket with n


def test_markdown_perf_table_excludes_quality_metrics(mixed: QuantReport) -> None:
    text = render_markdown(mixed)
    perf_section = text.split("## Speed and size", 1)[1].split("## Gate verdict", 1)[0]
    assert "Decode tok/s" in perf_section
    assert "ARC-Easy" not in perf_section
    quality_section = text.split("## Quality metrics", 1)[1].split("## Speed and size", 1)[0]
    assert "Decode tok/s" not in quality_section


def test_markdown_hidden_danger_rows_listed(hidden: QuantReport) -> None:
    text = render_markdown(hidden)
    section = text.split("## Hidden-danger rows", 1)[1].split("## Environment", 1)[0]
    assert "q3_k_m" in section
    assert "No hidden-danger row" not in section


def test_markdown_null_result_is_stated(all_pass: QuantReport) -> None:
    text = render_markdown(all_pass)
    assert "No hidden-danger row under this policy" in text
    assert "Recommended variant: q3_k_m" in text


def test_markdown_states_no_recommendation_when_none_passes() -> None:
    ref = VariantResult(
        variant=Variant(name="fp16", repo_id="x/y", filename="a.gguf", is_reference=True),
        metrics={"kl_mean": MetricValue(name="kl_mean", value=0.0, n=10)},
        raw_path=None,
    )
    cand = VariantResult(
        variant=Variant(name="q2", repo_id="x/y", filename="b.gguf", file_bytes=5),
        metrics={"kl_mean": MetricValue(name="kl_mean", value=0.9, n=10)},
        raw_path=None,
    )
    policy = load_policy(REPO / "configs" / "gate.yaml")
    decisions = evaluate([ref, cand], "fp16", policy)
    report = build_report("fam", "fp16", [ref, cand], decisions, None, {})
    text = render_markdown(report)
    assert "Recommended variant: none" in text
    assert "No environment record" in text


def test_markdown_escapes_pipes_in_cells() -> None:
    ref = VariantResult(
        variant=Variant(name="fp16", repo_id="x/y", filename="a.gguf", is_reference=True),
        metrics={},
        raw_path=None,
    )
    cand = VariantResult(
        variant=Variant(name="a|b", repo_id="x/y", filename="b.gguf", file_bytes=5),
        metrics={"kl_mean": MetricValue(name="kl_mean", value=0.5, n=10)},
        raw_path=None,
    )
    policy = load_policy(REPO / "configs" / "gate.yaml")
    decisions = evaluate([ref, cand], "fp16", policy)
    text = render_markdown(build_report("fam", "fp16", [ref, cand], decisions, None, {}))
    assert "a\\|b" in text


def test_html_contains_tables_and_escapes_names() -> None:
    ref = VariantResult(
        variant=Variant(name="fp16", repo_id="x/y", filename="a.gguf", is_reference=True),
        metrics={},
        raw_path=None,
    )
    cand = VariantResult(
        variant=Variant(name="<script>x</script>", repo_id="x/y", filename="b.gguf", file_bytes=5),
        metrics={"kl_mean": MetricValue(name="kl_mean", value=0.5, n=10)},
        raw_path=None,
    )
    policy = load_policy(REPO / "configs" / "gate.yaml")
    decisions = evaluate([ref, cand], "fp16", policy)
    report = build_report("fam", "fp16", [ref, cand], decisions, None, {})
    page = render_html(report)
    assert "<script>x</script>" not in page
    assert "&lt;script&gt;" in page
    assert page.startswith("<!doctype html>")


def test_html_page_has_dark_mode_tokens_and_plot_links(mixed: QuantReport) -> None:
    page = render_html(mixed, plot_paths=["pareto_size_vs_arc.svg"])
    assert "prefers-color-scheme: dark" in page
    assert 'data-theme="dark"' in page
    assert '<img src="pareto_size_vs_arc.svg"' in page
    assert "<table>" in page


def test_readme_block_links_env_and_is_compact(mixed: QuantReport) -> None:
    block = render_readme_block(mixed)
    assert "results/env.json" in block
    assert "| Variant | Verdict |" in block
    assert "PASS" in block and "FAIL" in block
    assert "SYNTHETIC" in block


def test_readme_block_numbers_come_from_report(mixed: QuantReport) -> None:
    block = render_readme_block(mixed)
    assert "58.0 [53.0, 63.0] (n=300)" in block  # q4_k_m arc_acc from the fixture


def test_update_readme_replaces_between_markers() -> None:
    text = f"intro\n{README_START}\nold\n{README_END}\noutro\n"
    updated = update_readme(text, "NEW BLOCK\n")
    assert updated == f"intro\n{README_START}\nNEW BLOCK\n{README_END}\noutro\n"
    assert "old" not in updated


def test_update_readme_requires_markers() -> None:
    with pytest.raises(ValueError, match="results:start"):
        update_readme("no markers here", "x")


def test_update_readme_rejects_reversed_markers() -> None:
    with pytest.raises(ValueError):
        update_readme(f"{README_END}\n{README_START}", "x")


def test_write_site_writes_pages_and_links_plots(tmp_path: Path, mixed: QuantReport) -> None:
    plot = tmp_path / "pareto_size_vs_arc.svg"
    plot.write_text("<svg></svg>", encoding="utf-8")
    written = write_site(mixed, tmp_path, [plot])
    names = {p.name for p in written}
    assert {"report.json", "report.md", "index.html"} <= names
    for path in written:
        assert path.stat().st_size > 0
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert 'src="pareto_size_vs_arc.svg"' in index


def test_committed_schema_equals_generated() -> None:
    """schemas/quant_report.schema.json must match QuantReport.model_json_schema() exactly."""
    generated = QuantReport.model_json_schema()
    generated = {"$schema": "https://json-schema.org/draft/2020-12/schema", **generated}
    committed = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    assert committed == generated


def test_committed_schema_file_is_canonically_formatted() -> None:
    text = SCHEMA_FILE.read_text(encoding="utf-8")
    generated = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **QuantReport.model_json_schema(),
    }
    assert text == json.dumps(generated, indent=2, sort_keys=True) + "\n"
