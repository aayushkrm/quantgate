"""results/headline.json: the hub's standard headline contract, built from committed results only.

Every value is a formatted string computed from results/report.json (and, when committed,
results/bandwidth.json). An item whose source data is missing is omitted, never filled in. A null
result is a headline too: "0 of N candidates" for hidden danger, or "none" for the recommendation.
Items derived from a SYNTHETIC report carry synthetic=true, and so does the environment string.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from quantgate.gate import variant_size_bytes
from quantgate.perf import bandwidth_ceiling_check
from quantgate.schema import QuantReport, VariantResult

HEADLINE_SCHEMA_VERSION = "1"
MAX_ITEMS = 6
REPORT_SOURCE = "results/report.json"
BANDWIDTH_SOURCE = "results/bandwidth.json"
REPO_URL = "https://github.com/aayushkrm/quantgate"
SITE_URL = "https://aayushkrm.github.io/quantgate/"
TAGLINE = (
    "A CI gate for quantized GGUF models: KL drift, accuracy, refusal and CPU speed for every "
    "quant level."
)
QUICKSTART = "uv tool install git+https://github.com/aayushkrm/quantgate && quantgate demo"
MIB = 2**20
GB = 1e9


class HeadlineItem(BaseModel):
    """One headline value. value and detail are formatted strings computed from the source."""

    id: str
    label: str
    value: str
    detail: str
    n: int | None
    ci: str | None
    synthetic: bool
    featured: bool
    source: str


class HeadlineLinks(BaseModel):
    """Repository, project site and release. release stays null until a release is published."""

    repo: str
    site: str
    release: str | None = None


class Headline(BaseModel):
    """results/headline.json (hub standard contract, schema_version 1)."""

    schema_version: Literal["1"] = "1"
    tool: Literal["quantgate"] = "quantgate"
    tagline: str
    quickstart: str
    environment: str
    items: list[HeadlineItem] = Field(default_factory=list, max_length=MAX_ITEMS)
    links: HeadlineLinks


def _variant(report: QuantReport, name: str) -> VariantResult | None:
    return next((v for v in report.variants if v.variant.name == name), None)


def _candidates(report: QuantReport) -> list[VariantResult]:
    return [v for v in report.variants if v.variant.name != report.reference]


def _ci_text(low: float | None, high: float | None, decimals: int) -> str | None:
    if low is None or high is None:
        return None
    return f"[{low:.{decimals}f}, {high:.{decimals}f}]"


def _environment(report: QuantReport) -> str:
    """CPU, cores and RAM from the environment record, or a plain statement that none exists."""
    env = report.env_info
    parts: list[str] = []
    cpu = env.get("cpu_model")
    if cpu:
        parts.append(str(cpu))
    cores = env.get("cpu_cores_physical")
    if isinstance(cores, int):
        parts.append(f"{cores} physical cores")
    ram = env.get("ram_bytes")
    if isinstance(ram, int | float) and ram > 0:
        parts.append(f"{ram / GB:.0f} GB RAM")
    parts.append("CPU only")
    text = ", ".join(parts) if parts else "Environment record not committed; CPU only"
    if env.get("synthetic") is True:
        text = "SYNTHETIC: " + text
    return text


def _recommended_item(report: QuantReport) -> HeadlineItem:
    synthetic = report.env_info.get("synthetic") is True
    if report.recommended is None:
        return HeadlineItem(
            id="recommended",
            label="Recommended quant (smallest that passes every rule)",
            value="none",
            detail="No candidate passes every rule under the committed policy.",
            n=None,
            ci=None,
            synthetic=synthetic,
            featured=True,
            source=REPORT_SOURCE,
        )
    vr = _variant(report, report.recommended)
    size = None if vr is None else vr.metrics.get("file_mb")
    size_mb = (
        size.value
        if size is not None
        else (None if vr is None or vr.variant.file_bytes is None else vr.variant.file_bytes / MIB)
    )
    decode = None if vr is None else vr.metrics.get("decode_tok_s")
    parts = []
    parts.append("file size not recorded" if size_mb is None else f"{size_mb:.0f} MiB file")
    if decode is not None:
        parts.append(f"{decode.value:.1f} tok/s decode (p50), CPU only")
    else:
        parts.append("decode speed not measured")
    ci = None
    if decode is not None:
        ci = _ci_text(decode.ci_low, decode.ci_high, 1)
        if ci is not None:
            ci += " min-max over repeats"
    return HeadlineItem(
        id="recommended",
        label="Recommended quant (smallest that passes every rule)",
        value=report.recommended,
        detail="; ".join(parts),
        n=None if decode is None else decode.n,
        ci=ci,
        synthetic=synthetic,
        featured=True,
        source=REPORT_SOURCE,
    )


def _hidden_item(report: QuantReport) -> HeadlineItem:
    synthetic = report.env_info.get("synthetic") is True
    decisions = report.decisions
    flagged = [d.variant for d in decisions if d.hidden_danger]
    not_evaluated = [
        d.variant
        for d in decisions
        if any(r.startswith("hidden_danger [missing") for r in d.failed_rules)
    ]
    total = len(decisions)
    if flagged:
        value = f"{len(flagged)} of {total} candidates"
        detail = "Hidden-danger rows: " + ", ".join(flagged) + ". They fail the gate."
    else:
        value = f"0 of {total} candidates"
        detail = "Null result: no hidden-danger row under this policy."
    if not_evaluated:
        detail += f" Not evaluated for: {', '.join(not_evaluated)} (missing metrics)."
    return HeadlineItem(
        id="hidden_danger",
        label="Hidden-danger rows (quality within band, refusal drop over threshold)",
        value=value,
        detail=detail,
        n=total,
        ci=None,
        synthetic=synthetic,
        featured=True,
        source=REPORT_SOURCE,
    )


def _worst_kl_item(report: QuantReport) -> HeadlineItem | None:
    best: tuple[float, VariantResult] | None = None
    for vr in _candidates(report):
        mv = vr.metrics.get("kl_mean")
        if mv is None:
            continue
        if best is None or mv.value > best[0]:
            best = (mv.value, vr)
    if best is None:
        return None
    _, vr = best
    mv = vr.metrics["kl_mean"]
    return HeadlineItem(
        id="worst_kl",
        label="Largest KL drift from fp16 across the ladder",
        value=f"{mv.value:.4f} nats/token ({vr.variant.name})",
        detail="Mean KL(fp16 || candidate) over scored positions; larger is further from fp16.",
        n=mv.n,
        ci=_ci_text(mv.ci_low, mv.ci_high, 4),
        synthetic=report.env_info.get("synthetic") is True,
        featured=True,
        source=REPORT_SOURCE,
    )


def _worst_arc_item(report: QuantReport) -> HeadlineItem | None:
    ref = _variant(report, report.reference)
    if ref is None or "arc_acc" not in ref.metrics:
        return None
    worst: tuple[float, VariantResult] | None = None
    for vr in _candidates(report):
        mv = vr.metrics.get("arc_acc")
        if mv is None:
            continue
        drop = ref.metrics["arc_acc"].value - mv.value
        if worst is None or drop > worst[0]:
            worst = (drop, vr)
    if worst is None:
        return None
    drop, vr = worst
    ci = None
    for decision in report.decisions:
        if decision.variant != vr.variant.name:
            continue
        for label, sig in decision.significance.items():
            if label.startswith("arc_acc:drop_max:") and sig.get("interval"):
                low, high = sig["interval"]
                text = _ci_text(float(low), float(high), 1) or ""
                ci = f"{text} {sig.get('basis', '')}".strip()
    return HeadlineItem(
        id="worst_arc_drop",
        label="Largest ARC-Easy drop from fp16 across the ladder",
        value=f"{drop:.1f} points ({vr.variant.name})",
        detail="Reference minus candidate, in percentage points. Negative means no drop.",
        n=vr.metrics["arc_acc"].n,
        ci=ci,
        synthetic=report.env_info.get("synthetic") is True,
        featured=False,
        source=REPORT_SOURCE,
    )


def _q2_item(report: QuantReport) -> HeadlineItem | None:
    ref = _variant(report, report.reference)
    q2 = _variant(report, "q2_k")
    if ref is None or q2 is None or "arc_acc" not in q2.metrics or "arc_acc" not in ref.metrics:
        return None
    arc_q2 = q2.metrics["arc_acc"]
    arc_ref = ref.metrics["arc_acc"]
    details = []
    kl = q2.metrics.get("kl_mean")
    if kl is not None:
        details.append(f"KL {kl.value:.4f} nats/token")
    rq = q2.metrics.get("refusal_rate")
    rr = ref.metrics.get("refusal_rate")
    if rq is not None and rr is not None:
        details.append(f"refusal {rq.value:.1f}% vs {rr.value:.1f}% (fp16)")
    return HeadlineItem(
        id="q2_k_vs_fp16",
        label="Lowest rung (q2_k) against fp16",
        value=f"ARC-Easy {arc_q2.value:.1f}% vs {arc_ref.value:.1f}% (fp16)",
        detail="; ".join(details) if details else "Only ARC-Easy recorded for this rung.",
        n=arc_q2.n,
        ci=_ci_text(arc_q2.ci_low, arc_q2.ci_high, 1),
        synthetic=report.env_info.get("synthetic") is True,
        featured=False,
        source=REPORT_SOURCE,
    )


def _bandwidth_item(bandwidth: Mapping[str, Any]) -> HeadlineItem | None:
    check = bandwidth.get("ceiling_check")
    if not isinstance(check, Mapping):
        return None
    implied = float(check["implied_gb_s"])
    measured = float(check["measured_gb_s"])
    ratio = float(check["ratio"])
    within = bool(check["within_ceiling"])
    verdict = "within the ceiling" if within else "OVER the ceiling (check the measurement)"
    return HeadlineItem(
        id="bandwidth_ceiling",
        label="Decode bandwidth against measured memory bandwidth",
        value=f"{implied:.2f} GB/s implied vs {measured:.2f} GB/s measured",
        detail=(
            f"Ratio {ratio:.2f} for {bandwidth.get('ceiling_variant', 'the record')}: {verdict}. "
            "Sanity check, not a measurement of its own."
        ),
        n=None,
        ci=None,
        synthetic=bandwidth.get("synthetic") is True,
        featured=False,
        source=BANDWIDTH_SOURCE,
    )


def _with_ceiling(bandwidth: Mapping[str, Any], report: QuantReport) -> Mapping[str, Any]:
    """The bandwidth record with a ceiling check for the recommended variant, when the record has
    none. The check is computed by quantgate.perf.bandwidth_ceiling_check and is labelled as a
    sanity check. Without a recommended variant, decode or size, the record is used as is.
    """
    if bandwidth.get("ceiling_check") is not None or report.recommended is None:
        return bandwidth
    vr = _variant(report, report.recommended)
    if vr is None or "decode_tok_s" not in vr.metrics:
        return bandwidth
    size = variant_size_bytes(vr)
    copy, scale = bandwidth.get("copy_gb_s"), bandwidth.get("scale_gb_s")
    if size is None or copy is None or scale is None:
        return bandwidth
    best = max(float(copy), float(scale))
    check = bandwidth_ceiling_check(vr.metrics["decode_tok_s"].value, size, best)
    return {**bandwidth, "ceiling_check": check, "ceiling_variant": report.recommended}


def build_headline(
    report: QuantReport | None,
    bandwidth: Mapping[str, Any] | None = None,
) -> Headline | None:
    """The headline document, or None when there is no committed report to build it from.

    Items are in a fixed order and capped at MAX_ITEMS. Items with missing source data are left
    out. Only the first three items are featured.
    """
    if report is None:
        return None
    candidates: list[HeadlineItem | None] = [
        _recommended_item(report),
        _hidden_item(report),
        _worst_kl_item(report),
        _worst_arc_item(report),
        _q2_item(report),
    ]
    if bandwidth is not None:
        candidates.append(_bandwidth_item(_with_ceiling(bandwidth, report)))
    items = [item for item in candidates if item is not None][:MAX_ITEMS]
    items = [item.model_copy(update={"featured": i < 3}) for i, item in enumerate(items)]
    return Headline(
        tagline=TAGLINE,
        quickstart=QUICKSTART,
        environment=_environment(report),
        items=items,
        links=HeadlineLinks(repo=REPO_URL, site=SITE_URL, release=None),
    )


def headline_text(headline: Headline) -> str:
    """The exact file contents: sorted keys, two-space indent, trailing newline."""
    return json.dumps(headline.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def write_headline(headline: Headline, path: Path) -> None:
    """Write results/headline.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(headline_text(headline), encoding="utf-8")


def load_bandwidth(path: Path) -> dict[str, Any] | None:
    """The bandwidth record if it is committed, else None. It is never guessed."""
    if not path.is_file():
        return None
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None
