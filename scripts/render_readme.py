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
DOCS = REPO / "docs"

LAPTOP_START = "<!-- laptop:start -->"
LAPTOP_END = "<!-- laptop:end -->"

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

from quantgate.gate import gate_label  # noqa: E402
from quantgate.report import (  # noqa: E402
    load_report,
    render_readme_block,
    update_readme,
)
from quantgate.schema import QuantReport  # noqa: E402

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
    lines.append(f"- Decode: {_metric_text(report, rec, 'decode_tok_s')} tok/s")
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


def render_blocks(report: QuantReport | None) -> tuple[str, str]:
    """The results block and the laptop block, as they appear between the markers."""
    results = NO_RESULTS_RESULTS if report is None else render_readme_block(report)
    return results, laptop_answer(report)


def render_readme(text: str, report: QuantReport | None) -> str:
    """README text with both generated blocks replaced. Raises ValueError if a marker is absent."""
    results, laptop = render_blocks(report)
    text = update_readme(text, results)
    start = text.find(LAPTOP_START)
    end = text.find(LAPTOP_END)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"README must contain {LAPTOP_START} and {LAPTOP_END} in order")
    return text[: start + len(LAPTOP_START)] + "\n" + laptop + text[end:]


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if README.md is stale")
    args = parser.parse_args(argv)

    report = load_report(REPORT) if REPORT.is_file() else None
    current = README.read_text(encoding="utf-8")
    try:
        rendered = render_readme(current, report)
    except ValueError as exc:
        print(f"render_readme: {exc}", file=sys.stderr)
        return 2

    if not args.check:
        if rendered != current:
            README.write_text(rendered, encoding="utf-8")
            print(f"Rewrote {README.relative_to(REPO)}")
        else:
            print("README is up to date")
        return 0

    status = 0
    if rendered != current:
        print(
            "README.md is out of date: run `uv run python scripts/render_readme.py`",
            file=sys.stderr,
        )
        status = 1
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
