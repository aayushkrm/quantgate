"""Report building and rendering.

Writes results/report.json (schema_version "1") and renders Markdown, a static HTML page and
the README results block. Every number comes from the QuantReport passed in; nothing is typed
by hand. Rows of metrics show the 95% CI for quality metrics, and the min-max over repeats for
perf metrics.
"""

import html
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantgate.gate import HIDDEN_MISSING_PREFIX, gate_label
from quantgate.numfmt import fixed
from quantgate.schema import GateDecision, MetricValue, QuantReport, VariantResult

QUALITY_METRICS: tuple[str, ...] = (
    "kl_mean",
    "top1_agree",
    "ppl",
    "arc_acc",
    "gsm8k_acc",
    "refusal_rate",
    "benign_refusal_rate",
)
PERF_METRICS: tuple[str, ...] = ("decode_tok_s", "prefill_tok_s", "peak_rss_mb", "file_mb")
METRIC_TITLES: dict[str, str] = {
    "kl_mean": "KL vs fp16 (nats/token)",
    "top1_agree": "Top-1 agree (%)",
    "ppl": "Perplexity",
    "arc_acc": "ARC-Easy (%)",
    "gsm8k_acc": "GSM8K (%)",
    "refusal_rate": "Refusal, harmful (%)",
    "benign_refusal_rate": "Refusal, benign (%)",
    "decode_tok_s": "Decode tok/s",
    "prefill_tok_s": "Prefill tok/s",
    "peak_rss_mb": "Peak RSS (MiB)",
    "file_mb": "File (MiB)",
}
README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"
LIMITATIONS: tuple[str, ...] = (
    "Results describe the model family named in the report, not LLMs in general.",
    "Latency and memory are CPU-only and describe the machine recorded in the environment block.",
    "Sample sizes are finite. A null result means the study did not detect a difference, "
    "not that none exists.",
    "The refusal classifier is a regex list and has not been validated by a human.",
)


@dataclass(frozen=True)
class Table:
    """Headers and string cells, shared by the Markdown and HTML renderers."""

    headers: list[str]
    rows: list[list[str]]


def build_report(
    model_family: str,
    reference: str,
    results: list[VariantResult],
    decisions: list[GateDecision],
    recommended: str | None,
    env_info: dict[str, Any],
) -> QuantReport:
    """Assemble a QuantReport from its parts."""
    return QuantReport(
        model_family=model_family,
        reference=reference,
        variants=results,
        decisions=decisions,
        recommended=recommended,
        env_info=env_info,
    )


def load_report(path: Path) -> QuantReport:
    """Read and validate a report JSON.

    Raises pydantic.ValidationError (a ValueError) on a bad schema, including a wrong
    schema_version, and OSError if the file cannot be read.
    """
    return QuantReport.model_validate_json(path.read_text(encoding="utf-8"))


def write_report(report: QuantReport, path: Path) -> None:
    """Write the report as indented JSON, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _decimals(metric: str) -> int:
    """Decimal places used when showing a metric."""
    if metric == "kl_mean":
        return 4
    if metric == "ppl":
        return 2
    return 1


def _fmt(metric: str, value: float) -> str:
    """Format one number for display, rounding ties the same way on every Python version."""
    return fixed(value, _decimals(metric))


MIN_N_FOR_CI: int = 20  # below this, a bootstrap or Wilson interval is not shown


def _cell(metric: str, mv: MetricValue | None) -> str:
    """One table cell: value, bracketed interval (if any) and n.

    Intervals are not shown for n below MIN_N_FOR_CI. A percentile bootstrap over a handful of
    units is not a meaningful interval, and the cell says so instead of printing one. The JSON
    keeps the values.
    """
    if mv is None:
        return "n/a"
    text = _fmt(metric, mv.value)
    if mv.n < MIN_N_FOR_CI:
        return f"{text} (n={mv.n}, CI not shown below n={MIN_N_FOR_CI})"
    if mv.ci_low is not None and mv.ci_high is not None:
        text += f" [{_fmt(metric, mv.ci_low)}, {_fmt(metric, mv.ci_high)}]"
    return f"{text} (n={mv.n})"


def _range_cell(metric: str, mv: MetricValue | None) -> str:
    """One perf cell: value, then the min-max over repeats (not a CI), then n. Never hidden by the
    CI rule, because a range over five repeats is what the perf table reports."""
    if mv is None:
        return "n/a"
    text = _fmt(metric, mv.value)
    if mv.ci_low is not None and mv.ci_high is not None:
        text += f" [{_fmt(metric, mv.ci_low)}, {_fmt(metric, mv.ci_high)}]"
    return f"{text} (n={mv.n})"


def _present_metrics(report: QuantReport, group: Sequence[str], other: Sequence[str]) -> list[str]:
    """Metrics of one table: the group's known metrics present in any variant, in canonical order,
    then metrics that belong to neither group, sorted. The other group's metrics are excluded."""
    seen: set[str] = set()
    for vr in report.variants:
        seen.update(vr.metrics)
    ordered = [m for m in group if m in seen]
    extra = sorted(seen - set(group) - set(other))
    return ordered + extra


def _title(metric: str) -> str:
    """Display title of a metric."""
    return METRIC_TITLES.get(metric, metric)


def _quality_table(report: QuantReport) -> Table:
    """Per-variant quality metrics, cells with 95% CI and n."""
    metrics = _present_metrics(report, QUALITY_METRICS, PERF_METRICS)
    headers = ["Variant", *(_title(m) for m in metrics)]
    rows = [
        [vr.variant.name, *(_cell(m, vr.metrics.get(m)) for m in metrics)] for vr in report.variants
    ]
    return Table(headers, rows)


def _perf_table(report: QuantReport) -> Table | None:
    """Per-variant perf and size metrics, cells with min-max range and n. None if absent."""
    metrics = _present_metrics(report, PERF_METRICS, QUALITY_METRICS)
    if not metrics:
        return None
    headers = ["Variant", *(_title(m) for m in metrics)]
    rows = [
        [vr.variant.name, *(_range_cell(m, vr.metrics.get(m)) for m in metrics)]
        for vr in report.variants
    ]
    return Table(headers, rows)


GATE_NOTE = (
    "Gate is PASS only when every blocking rule passes and the variant is not a hidden-danger "
    "row. A hidden-danger row fails the gate even when every rule passes. A point violation whose "
    "interval still includes the threshold is labelled not distinguishable at this n. The "
    "intervals are paired bootstrap (ARC, GSM8K drops), Wilson bounds (refusal drops) or the "
    "metric's own 95% CI (abs rules)."
)


def _judgement_cell(d: GateDecision) -> str:
    """Per failing rule: its verdict, basis and 95% interval, from the significance record."""
    parts: list[str] = []
    for label in d.failed_rules:
        if label.startswith("hidden_danger"):
            continue
        sig = d.significance.get(label.replace(" [missing]", ""), {})
        verdict = sig.get("verdict", "not recorded")
        basis = sig.get("basis", "")
        interval = sig.get("interval")
        text = f"{label}: {verdict}"
        if basis and basis != "none":
            text += f" ({basis}"
            if interval:
                text += f", [{_fmt_interval(interval)}]"
            text += ")"
        parts.append(text)
    return "; ".join(parts) if parts else "none"


def _fmt_interval(interval: list[float]) -> str:
    """Two numbers of a 95% interval, to two decimals, for the judgement column."""
    return f"{fixed(interval[0], 2)}, {fixed(interval[1], 2)}"


def _hidden_cell(d: GateDecision) -> str:
    """Hidden-danger cell: YES, no, or 'not evaluated' when a needed metric was absent."""
    if any(r.startswith(HIDDEN_MISSING_PREFIX) for r in d.failed_rules):
        return "NOT EVALUATED (a metric it needs is missing)"
    if d.hidden_danger:
        return "YES (quality within band, refusal drop over threshold)"
    return "no"


def _gate_table(report: QuantReport) -> Table:
    """Gate verdict per non-reference variant, with the failing rules and the hidden-danger flag
    in separate columns, so that a hidden-danger failure is visible as such."""
    headers = [
        "Variant",
        "Gate",
        "Rules that fail",
        "Judgement (verdict, basis, 95% interval)",
        "Hidden danger",
    ]
    rows = [
        [
            d.variant,
            gate_label(d),
            ", ".join(d.failed_rules) if d.failed_rules else "none",
            _judgement_cell(d),
            _hidden_cell(d),
        ]
        for d in report.decisions
    ]
    return Table(headers, rows)


def _hidden_danger_names(report: QuantReport) -> list[str]:
    """Variants flagged as hidden danger, in decision order."""
    return [d.variant for d in report.decisions if d.hidden_danger]


def _verdict_line(report: QuantReport) -> str:
    """One-line headline: how many candidates pass, and whether any hidden-danger row exists."""
    total = len(report.decisions)
    passing = sum(1 for d in report.decisions if d.passed)
    flagged = _hidden_danger_names(report)
    if flagged:
        danger = f"Hidden-danger rows: {', '.join(flagged)}."
    else:
        danger = "No hidden-danger row under this policy."
    return f"Gate: {passing} of {total} candidate variants pass. {danger}"


def _recommendation_line(report: QuantReport) -> str:
    """Recommended variant, or a statement that none passes."""
    if report.recommended is None:
        return "Recommended variant: none. No candidate passes every rule."
    return f"Recommended variant: {report.recommended} (smallest file that passes every rule)."


def _env_lines(report: QuantReport) -> list[str]:
    """Environment disclosure as bullet lines, from env_info keys that are present."""
    env = report.env_info
    lines: list[str] = []
    if env.get("synthetic") is True:
        lines.append(f"- **SYNTHETIC:** {env.get('label', 'not a measurement')}")
    for key, label in (
        ("git_sha", "Git SHA"),
        ("git_dirty", "Uncommitted changes"),
        ("utc_time", "UTC time"),
        ("cpu_model", "CPU"),
        ("cpu_flags", "CPU flags"),
        ("cpu_cores_logical", "Logical cores"),
        ("cpu_cores_physical", "Physical cores"),
        ("ram_bytes", "RAM (bytes)"),
    ):
        if key in env and env[key] is not None:
            lines.append(f"- {label}: {env[key]}")
    os_info = env.get("os")
    if isinstance(os_info, dict):
        lines.append(f"- OS: {os_info.get('platform', os_info)}")
    py = env.get("python")
    if isinstance(py, dict):
        lines.append(f"- Python: {py.get('version')}")
    packages = env.get("packages")
    if isinstance(packages, dict):
        present = {k: v for k, v in packages.items() if v is not None}
        if present:
            lines.append("- Packages: " + ", ".join(f"{k} {v}" for k, v in present.items()))
    run = env.get("run")
    if isinstance(run, dict):
        if "n_scale" in run:
            lines.append(f"- Sample scale: {run['n_scale']} (1.0 = pre-registered n)")
        if "suite" in run:
            lines.append(f"- Suite settings: {run['suite']}")
    return lines


def _markdown_table(table: Table) -> list[str]:
    """Render a Table as GitHub-flavoured Markdown lines."""

    def esc(text: str) -> str:
        return text.replace("|", "\\|")

    lines = ["| " + " | ".join(esc(h) for h in table.headers) + " |"]
    lines.append("|" + "|".join("---" for _ in table.headers) + "|")
    for row in table.rows:
        lines.append("| " + " | ".join(esc(c) for c in row) + " |")
    return lines


def render_markdown(report: QuantReport) -> str:
    """Full Markdown report: quality and perf tables with CIs, gate verdicts, hidden-danger rows,
    recommendation and the environment disclosure."""
    out: list[str] = [f"# quantgate report: {report.model_family}", ""]
    if report.env_info.get("synthetic") is True:
        label = report.env_info.get("label", "not a measurement")
        out += [f"> **SYNTHETIC.** {label}", ""]
    out += [
        f"Reference: {report.reference}. {_verdict_line(report)}",
        "",
        _recommendation_line(report),
        "",
        "## Quality metrics",
        "",
        "Rates are percentages. Brackets hold the 95% CI, n is the sample size.",
        "",
        *_markdown_table(_quality_table(report)),
        "",
    ]
    perf = _perf_table(report)
    if perf is not None:
        out += [
            "## Speed and size",
            "",
            "Brackets hold the min and max over repeats, not a CI. CPU-only.",
            "",
            *_markdown_table(perf),
            "",
        ]
    out += ["## Gate verdict", "", GATE_NOTE, "", *_markdown_table(_gate_table(report)), ""]
    flagged = _hidden_danger_names(report)
    out += ["## Hidden-danger rows", ""]
    if flagged:
        out += [
            f"- {name}: every quality metric within the policy band of the reference, and the "
            "refusal rate drops by at least the policy threshold. This row FAILS the gate."
            for name in flagged
        ]
    else:
        out += ["No hidden-danger row under this policy. This is a null result for this sample."]
    out += ["", "## Environment", ""]
    env = _env_lines(report)
    out += env if env else ["No environment record in this report."]
    out += ["", "## Limitations", ""]
    out += [f"- {line}" for line in LIMITATIONS]
    return "\n".join(out) + "\n"


def render_readme_block(report: QuantReport, env_link: str = "results/env.json") -> str:
    """Compact Markdown placed between the README results markers. Numbers come from the report.

    Ends with a link to the environment record.
    """
    lines: list[str] = []
    if report.env_info.get("synthetic") is True:
        lines += ["> **SYNTHETIC.** Not a measurement.", ""]
    lines += [
        f"Reference: {report.reference}. {_recommendation_line(report)}",
        "",
        _verdict_line(report),
        "",
    ]
    compact = Table(
        ["Variant", "Verdict", "KL (nats/token)", "Top-1 agree (%)", "ARC-Easy (%)"]
        + [_title("refusal_rate"), _title("decode_tok_s")],
        [],
    )
    decisions = {d.variant: d for d in report.decisions}
    for vr in report.variants:
        name = vr.variant.name
        if name == report.reference:
            verdict = "reference"
        else:
            d = decisions.get(name)
            verdict = "n/a" if d is None else gate_label(d)
        m = vr.metrics
        compact.rows.append(
            [
                name,
                verdict,
                _cell("kl_mean", m.get("kl_mean")),
                _cell("top1_agree", m.get("top1_agree")),
                _cell("arc_acc", m.get("arc_acc")),
                _cell("refusal_rate", m.get("refusal_rate")),
                _range_cell("decode_tok_s", m.get("decode_tok_s")),
            ]
        )
    lines += _markdown_table(compact)
    lines += [
        "",
        f"Environment record: [env.json]({env_link}). Brackets hold 95% intervals for rates and "
        "KL. For decode speed they hold the min and max over the repeats, not an interval.",
    ]
    return "\n".join(lines) + "\n"


def update_readme(text: str, block: str) -> str:
    """Replace the text between the results markers with block.

    Raises ValueError if the markers are missing or out of order.
    """
    start = text.find(README_START)
    end = text.find(README_END)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"README must contain {README_START} and {README_END} in order")
    head = text[: start + len(README_START)]
    return f"{head}\n{block}{text[end:]}"


def _html_table(table: Table) -> str:
    """Render a Table as an HTML table, escaping every cell."""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in table.headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>" for row in table.rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


_CSS = """
:root { --fg: #1d1d1f; --bg: #ffffff; --muted: #5b5b61; --line: #d6d6db; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --fg: #ececf1; --bg: #16161a; --muted: #a3a3ad; --line: #3a3a42;
  }
}
:root[data-theme="dark"] { --fg: #ececf1; --bg: #16161a; --muted: #a3a3ad; --line: #3a3a42; }
body { color: var(--fg); background: var(--bg); font-family: system-ui, sans-serif;
  margin: 0 auto; padding: 16px; max-width: 1100px; line-height: 1.45; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem;
  display: block; overflow-x: auto; }
th, td { border-bottom: 1px solid var(--line); padding: 6px 8px;
  text-align: left; vertical-align: top; }
.banner { border: 2px solid #d55e00; padding: 8px 12px; border-radius: 6px; }
.muted { color: var(--muted); }
img { max-width: 100%; height: auto; }
"""


def render_html(report: QuantReport, plot_paths: Sequence[str] = ()) -> str:
    """Static HTML page with the same tables as the Markdown report and optional plot images.

    plot_paths are relative URLs to image files written next to the page.
    """
    parts: list[str] = [f"<h1>quantgate report: {html.escape(report.model_family)}</h1>"]
    if report.env_info.get("synthetic") is True:
        label = html.escape(str(report.env_info.get("label", "not a measurement")))
        parts.append(f'<p class="banner"><strong>SYNTHETIC.</strong> {label}</p>')
    parts.append(
        f"<p>Reference: {html.escape(report.reference)}. {html.escape(_verdict_line(report))}</p>"
    )
    parts.append(f"<p><strong>{html.escape(_recommendation_line(report))}</strong></p>")
    for src in plot_paths:
        parts.append(f'<p><img src="{html.escape(src)}" alt="Pareto plot"></p>')
    parts.append("<h2>Quality metrics</h2>")
    parts.append(
        '<p class="muted">Rates are percentages. Brackets hold the 95% CI, '
        "n is the sample size.</p>"
    )
    parts.append(_html_table(_quality_table(report)))
    perf = _perf_table(report)
    if perf is not None:
        parts.append("<h2>Speed and size</h2>")
        parts.append('<p class="muted">Brackets hold the min and max over repeats. CPU-only.</p>')
        parts.append(_html_table(perf))
    parts.append("<h2>Gate verdict</h2>")
    parts.append(f'<p class="muted">{html.escape(GATE_NOTE)}</p>')
    parts.append(_html_table(_gate_table(report)))
    flagged = _hidden_danger_names(report)
    parts.append("<h2>Hidden-danger rows</h2>")
    if flagged:
        items = "".join(f"<li>{html.escape(name)}</li>" for name in flagged)
        parts.append(f"<ul>{items}</ul>")
    else:
        parts.append(
            "<p>No hidden-danger row under this policy. This is a null result for this sample.</p>"
        )
    parts.append("<h2>Environment</h2>")
    env = _env_lines(report)
    if env:
        items = "".join(f"<li>{html.escape(line.removeprefix('- '))}</li>" for line in env)
        parts.append(f"<ul>{items}</ul>")
    else:
        parts.append("<p>No environment record in this report.</p>")
    parts.append('<p><a href="env.json">Environment record (env.json)</a></p>')
    parts.append("<h2>Limitations</h2><ul>")
    parts.extend(f"<li>{html.escape(line)}</li>" for line in LIMITATIONS)
    parts.append("</ul>")
    body = "\n".join(parts)
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>quantgate report</title>\n<style>{_CSS}</style>\n</head>\n"
        f"<body>\n{body}\n</body>\n</html>\n"
    )


def write_site(report: QuantReport, out_dir: Path, plot_paths: Sequence[Path]) -> list[Path]:
    """Write report.json, report.md and index.html into out_dir. Returns the written paths.

    plot_paths must already live under out_dir; the page links them by relative path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_plots = [os.path.relpath(p, out_dir) for p in plot_paths]
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    html_path = out_dir / "index.html"
    write_report(report, json_path)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    html_path.write_text(render_html(report, rel_plots), encoding="utf-8")
    return [json_path, md_path, html_path]
