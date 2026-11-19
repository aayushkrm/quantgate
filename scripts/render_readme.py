#!/usr/bin/env python3
"""Render the generated blocks of README.md from committed results, or check them.

Two blocks are generated, both only from results/report.json:
- the results block between ``<!-- results:start -->`` and ``<!-- results:end -->``, rendered by
  quantgate.report.render_readme_block;
- the laptop block between ``<!-- laptop:start -->`` and ``<!-- laptop:end -->``, which answers
  "which quant should I use on an old Intel laptop" from the report alone.

Without a committed report both blocks say so, and no number is shown.

    uv run python scripts/render_readme.py           # rewrite the blocks in README.md
    uv run python scripts/render_readme.py --check   # exit 1 if README.md is out of date

--check also scans README.md and docs/*.md for the banned words of AGENTS.md (invariant 8) and
exits 1 on a match. Nothing is typed by hand: the numbers come from the JSON report.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
REPORT = REPO / "results" / "report.json"
HEADLINE = REPO / "results" / "headline.json"
BANDWIDTH = REPO / "results" / "bandwidth.json"
DOCS = REPO / "docs"

LAPTOP_START = "<!-- laptop:start -->"
LAPTOP_END = "<!-- laptop:end -->"
POWER_START = "<!-- power:start -->"
POWER_END = "<!-- power:end -->"
INTERP_START = "<!-- interpretation:start -->"
INTERP_END = "<!-- interpretation:end -->"
AUDIT_START = "<!-- audit:start -->"
AUDIT_END = "<!-- audit:end -->"
LIMIT_AUDIT_START = "<!-- audit-validation:start -->"
LIMIT_AUDIT_END = "<!-- audit-validation:end -->"
SENS_START = "<!-- sensitivity:start -->"
SENS_END = "<!-- sensitivity:end -->"
# Post-hoc sensitivity: the labelled 150-prompt samples that match each audited rung.
SENS_LABELS: dict[str, Path] = {
    "q5_k_m": REPO / "audit" / "refusal_audit_ai_labels_q5_k_m_150_prompts.csv",
    "q2_k": REPO / "audit" / "refusal_audit_ai_labels_q2_k_150_prompts.csv",
}
LIMITATIONS = REPO / "docs" / "limitations.md"
# Every labelled sample that is committed. Each has a labels file and the score JSON that
# `quantgate audit score --json` wrote for it. The titles say what the sample is, not its numbers.
AUDIT_SAMPLES: tuple[tuple[str, Path, Path], ...] = (
    (
        "Sample A: 40 prompts, labelled once per variant",
        REPO / "results" / "audit" / "refusal_audit_score.json",
        REPO / "audit" / "refusal_audit_ai_labels.csv",
    ),
    (
        "Sample B: all 150 prompts, one variant",
        REPO / "results" / "audit" / "refusal_audit_score_q5_k_m_150_prompts.json",
        REPO / "audit" / "refusal_audit_ai_labels_q5_k_m_150_prompts.csv",
    ),
    (
        "Sample C: all 150 prompts, one variant",
        REPO / "results" / "audit" / "refusal_audit_score_q2_k_150_prompts.json",
        REPO / "audit" / "refusal_audit_ai_labels_q2_k_150_prompts.csv",
    ),
)

BANNED = (
    "production-ready",
    "state-of-the-art",
    "SOTA",
    "blazing",
    "first",
    "novel",
    "robust",
    "enterprise",
)

sys.path.insert(0, str(REPO / "src"))

from quantgate.cli import load_suite_config  # noqa: E402
from quantgate.evals import wilson_ci  # noqa: E402
from quantgate.gate import gate_label  # noqa: E402
from quantgate.headline import (  # noqa: E402
    Headline,
    build_headline,
    headline_text,
    load_bandwidth,
    write_headline,
)
from quantgate.interpret import (  # noqa: E402
    audit_coverage_text,
    audit_validation_block,
    interpretation_block,
    load_audit,
    privacy_problems,
)
from quantgate.models import load_registry  # noqa: E402
from quantgate.report import (  # noqa: E402
    load_report,
    render_readme_block,
    update_readme,
)
from quantgate.resources import config_path  # noqa: E402
from quantgate.schema import QuantReport  # noqa: E402
from quantgate.sensitivity import sensitivity_block, sensitivity_rows  # noqa: E402

NO_RESULTS_RESULTS = (
    "_No committed results yet. This block is rendered from `results/report.json` by "
    "`scripts/render_readme.py` once the pre-registered experiments have run._"
)
NO_RESULTS_LAPTOP = (
    "_No committed results yet, so this question has no answer. The answer is rendered from "
    "`results/report.json` once it exists._"
)


def _fmt(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}"


def _metric_text(report: QuantReport, variant: str, metric: str, decimals: int = 1) -> str:
    """A metric of one variant with its interval and n, from the report. 'n/a' if absent."""
    vr = next((v for v in report.variants if v.variant.name == variant), None)
    mv = None if vr is None else vr.metrics.get(metric)
    if mv is None:
        return "n/a"
    text = _fmt(mv.value, decimals)
    if mv.ci_low is not None and mv.ci_high is not None:
        text += f" [{_fmt(mv.ci_low, decimals)}, {_fmt(mv.ci_high, decimals)}]"
    return f"{text} (n={mv.n})"


def _range_text(report: QuantReport, variant: str, metric: str, decimals: int = 1) -> str:
    """A metric of one variant as its value and its min-max over repeats. Not an interval."""
    vr = next((v for v in report.variants if v.variant.name == variant), None)
    mv = None if vr is None else vr.metrics.get(metric)
    if mv is None:
        return "n/a"
    text = _fmt(mv.value, decimals)
    if mv.ci_low is not None and mv.ci_high is not None:
        text += f" (min {_fmt(mv.ci_low, decimals)}, max {_fmt(mv.ci_high, decimals)})"
    return f"{text} over {mv.n} repeats"


def laptop_answer(report: QuantReport | None) -> str:
    """The answer for an old Intel laptop, built from the report only.

    The answer is the recommended variant: the smallest file that passes every rule and is not
    a hidden-danger row. The text lists its size, decode speed and quality metrics, and the
    failing candidates, so a reader can see what was traded.
    """
    if report is None:
        return NO_RESULTS_LAPTOP
    lines: list[str] = []
    if report.env_info.get("synthetic") is True:
        lines += ["> **SYNTHETIC.** Not a measurement.", ""]
    if report.recommended is None:
        lines.append(
            "No candidate passes every rule under the committed policy, so the gate recommends "
            "no quant for this laptop. Read the gate table below before choosing one."
        )
        return "\n".join(lines) + "\n"
    rec = report.recommended
    vr = next(v for v in report.variants if v.variant.name == rec)
    size_mb = vr.metrics.get("file_mb")
    size_text = "size not recorded" if size_mb is None else f"{_fmt(size_mb.value, 0)} MiB"
    lines.append(
        f"On this laptop class the gate recommends **{rec}**: the smallest file that passes "
        f"every rule and is not a hidden-danger row ({size_text})."
    )
    lines.append("")
    lines.append(
        f"- Decode: {_range_text(report, rec, 'decode_tok_s')} tok/s (min to max over repeats)"
    )
    lines.append(f"- KL vs fp16: {_metric_text(report, rec, 'kl_mean', 4)} nats/token")
    lines.append(f"- ARC-Easy: {_metric_text(report, rec, 'arc_acc')} %")
    lines.append(f"- Refusal, harmful: {_metric_text(report, rec, 'refusal_rate')} %")
    lines.append("")
    others = [d for d in report.decisions if d.variant != rec]
    if others:
        lines.append("Other candidates under the same policy:")
        lines.append("")
        for d in others:
            lines.append(f"- {d.variant}: {gate_label(d)}")
    return "\n".join(lines) + "\n"


Z_ALPHA_TWO_SIDED = 1.959963984540054
Z_POWER_80 = 0.8416212335729143


def power_block() -> str:
    """What the pre-registered sample sizes can detect, computed from configs and the Wilson
    interval. These are design numbers, not measured results, so they can be rendered before any
    run. They are worst-case (p = 50%).

    Single rate: the half-width of its 95% Wilson interval. Difference of two independent rates:
    the smallest difference detected with 80% power at alpha 0.05, by the normal approximation.
    A paired design (same items) can detect smaller differences; that is not computed here.
    """
    suite = load_suite_config(config_path("variants.yaml"))
    rows = [
        ("ARC-Easy accuracy", suite.arc.n),
        ("GSM8K accuracy", suite.gsm8k.n),
        ("Refusal, harmful prompts", suite.refusal.n_harmful),
        ("Refusal, benign prompts (over-refusal)", suite.refusal.n_benign),
    ]
    lines = [
        "| Metric | n | 95% half-width of one rate (p = 50%) | Smallest difference detected, "
        "independent samples (80% power) |",
        "|---|---|---|---|",
    ]
    for name, n in rows:
        low, high = wilson_ci(n // 2, n)
        half = 100.0 * (high - low) / 2.0
        mde = 100.0 * (Z_ALPHA_TWO_SIDED + Z_POWER_80) * (2.0 * 0.25 / n) ** 0.5
        lines.append(f"| {name} | {n} | ±{half:.1f} pts | {mde:.1f} pts |")
    return "\n".join(lines) + "\n"


def _replace_between(text: str, start_marker: str, end_marker: str, body: str) -> str:
    """Replace the text between two markers with body, which always ends with a newline.

    Raises ValueError if the markers are missing or out of order.
    """
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"file must contain {start_marker} and {end_marker} in order")
    head = text[: start + len(start_marker)]
    if not body.endswith("\n"):
        body += "\n"
    return f"{head}\n{body}{text[end:]}"


def render_interpretation_body(report: QuantReport | None, bandwidth: dict | None) -> str:
    if report is None:
        return "_No results are committed yet, so there is no interpretation._\n"
    return interpretation_block(report, bandwidth)


def load_all_audits() -> list[tuple[str, dict, list[dict[str, str]]]]:
    """Every committed audit sample, as (title, score, labels). Samples whose files are absent are
    left out, so a missing file is never filled in."""
    found = []
    for title, score_path, labels_path in AUDIT_SAMPLES:
        loaded = load_audit(score_path, labels_path)
        if loaded is not None:
            score, labels = loaded
            found.append((title, score, labels))
    return found


def all_variant_names() -> list[str]:
    """The rungs of the ladder, from the variant registry."""
    return [v.name for v in load_registry(config_path("variants.yaml")).variants]


def render_audit_body(audits: list[tuple[str, dict, list[dict[str, str]]]] | None) -> str:
    if not audits:
        return "_No refusal-classifier validation is committed yet._\n"
    parts = [audit_coverage_text(audits, all_variant_names())]
    for title, score, labels in audits:
        parts.append(audit_validation_block(score, labels, title))
    return "\n".join(parts)


def render_limitations(
    text: str, audits: list[tuple[str, dict, list[dict[str, str]]]] | None
) -> str:
    """docs/limitations.md with the validation blocks replaced from the score JSON and labels."""
    text = _replace_between(text, LIMIT_AUDIT_START, LIMIT_AUDIT_END, render_audit_body(audits))
    return _replace_between(text, SENS_START, SENS_END, render_sensitivity_body())


def render_sensitivity_body() -> str:
    """The post-hoc sensitivity table, computed from committed results and labels files."""
    rows = sensitivity_rows(REPO / "results", SENS_LABELS)
    if not rows:
        return "_No committed refusal results yet, so there is no sensitivity table._\n"
    return sensitivity_block(rows)


def render_power(text: str) -> str:
    """README text with the power block between its markers replaced. Raises if absent."""
    start = text.find(POWER_START)
    end = text.find(POWER_END)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"README must contain {POWER_START} and {POWER_END} in order")
    head = text[: start + len(POWER_START)]
    return f"{head}\n{power_block()}{text[end:]}"


def render_blocks(report: QuantReport | None) -> tuple[str, str]:
    """The results block and the laptop block, as they appear between the markers."""
    results = NO_RESULTS_RESULTS if report is None else render_readme_block(report)
    return results, laptop_answer(report)


def render_readme(
    text: str,
    report: QuantReport | None,
    bandwidth: dict | None = None,
    audit: list[tuple[str, dict, list[dict[str, str]]]] | None = None,
) -> str:
    """README text with every generated block replaced. Raises ValueError if a marker is absent."""
    results, laptop = render_blocks(report)
    text = render_power(update_readme(text, results))
    text = _replace_between(text, LAPTOP_START, LAPTOP_END, laptop)
    text = _replace_between(
        text, INTERP_START, INTERP_END, render_interpretation_body(report, bandwidth)
    )
    text = _replace_between(text, AUDIT_START, AUDIT_END, render_audit_body(audit))
    return _replace_between(text, SENS_START, SENS_END, render_sensitivity_body())


def banned_word_hits(paths: list[Path]) -> list[str]:
    """Banned words found in the given files, as 'file:line: word' strings (whole words only)."""
    hits: list[str] = []
    patterns = [
        (w, re.compile(rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])", re.IGNORECASE)) for w in BANNED
    ]
    for path in paths:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for word, pattern in patterns:
                if pattern.search(line):
                    hits.append(f"{path.relative_to(REPO)}:{lineno}: {word}")
    return hits


def _write_or_check_headline(expected: Headline | None, check: bool) -> int:
    """Write results/headline.json from the committed results, or check that it is current.

    With no committed report there must be no headline file, since a headline built from nothing
    would be invented. Returns 0 when fine, 1 when a check fails.
    """
    current = HEADLINE.read_text(encoding="utf-8") if HEADLINE.is_file() else None
    if expected is None:
        if current is not None and check:
            print(
                "results/headline.json exists but no results/report.json does: remove it",
                file=sys.stderr,
            )
            return 1
        if current is not None and not check:
            HEADLINE.unlink()
            print("Removed results/headline.json (no committed report)")
        return 0
    text = headline_text(expected)
    if check:
        if current != text:
            print(
                "results/headline.json is out of date: run `make report`",
                file=sys.stderr,
            )
            return 1
        return 0
    if current != text:
        write_headline(expected, HEADLINE)
        print(f"Wrote {HEADLINE}")
    else:
        print("results/headline.json is up to date")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if a generated file is stale")
    parser.add_argument(
        "--headline",
        action="store_true",
        help="write only results/headline.json (used by make report)",
    )
    args = parser.parse_args(argv)

    report = load_report(REPORT) if REPORT.is_file() else None
    bandwidth = load_bandwidth(BANDWIDTH)
    expected = build_headline(report, bandwidth)
    audit = load_all_audits()

    if args.headline:
        return _write_or_check_headline(expected, check=False)

    current = README.read_text(encoding="utf-8")
    current_lim = LIMITATIONS.read_text(encoding="utf-8")
    try:
        rendered = render_readme(current, report, bandwidth, audit)
        rendered_lim = render_limitations(current_lim, audit)
    except ValueError as exc:
        print(f"render_readme: {exc}", file=sys.stderr)
        return 2

    if not args.check:
        if rendered != current:
            README.write_text(rendered, encoding="utf-8")
            print(f"Rewrote {README.relative_to(REPO)}")
        else:
            print("README is up to date")
        if rendered_lim != current_lim:
            LIMITATIONS.write_text(rendered_lim, encoding="utf-8")
            print(f"Rewrote {LIMITATIONS.relative_to(REPO)}")
        return _write_or_check_headline(expected, check=False)

    status = 0
    if rendered != current:
        print(
            "README.md is out of date: run `uv run python scripts/render_readme.py`",
            file=sys.stderr,
        )
        status = 1
    if rendered_lim != current_lim:
        print("docs/limitations.md is out of date: run the renderer", file=sys.stderr)
        status = 1
    committed = sorted(
        [
            *(REPO / "results").rglob("*.json"),
            *(REPO / "audit").glob("*.csv"),
        ]
    )
    problems = privacy_problems(committed)
    for problem in problems:
        print(f"privacy: {problem}", file=sys.stderr)
    if problems:
        status = 1
    status = max(status, _write_or_check_headline(expected, check=True))
    banned_paths = [README, *sorted(DOCS.glob("*.md"))]
    hits = banned_word_hits(banned_paths)
    for hit in hits:
        print(f"banned word: {hit}", file=sys.stderr)
    if hits:
        status = 1
    if status == 0:
        print("README blocks are current and no banned words were found")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
