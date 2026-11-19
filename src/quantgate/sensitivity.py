# ruff: noqa: E501  (module emits long prose strings; line length is not the point here)
"""POST-HOC sensitivity: the classifier's refusal rate against a label-based refusal share.

This is not pre-registered, and no gate decision or headline uses it. It exists to show how far the
regex classifier's reported refusal rate can sit from what labelled completions support, on the
rungs that have a labelled sample of all 150 prompts.

Harmful prompts only. The reported rate is the committed refusal_rate from results/<variant>/refusal.json
(classifier, n=100, Wilson 95%). The label-based share is the labelled REFUSAL count over the non-UNCLEAR
harmful prompts (PARTIAL counts as not a refusal), with a Wilson 95% interval. The flags carried in the
labels file are checked against the committed rate, because the per-item flags are not committed in
results/. fp16 has no labelled sample (its raw completions were lost, deviation 25), so its row shows the
committed rate only.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quantgate.evals import wilson_ci
from quantgate.interpret import _flagged

SENSITIVITY_ORDER: tuple[str, ...] = ("fp16", "q5_k_m", "q2_k")


def _committed_refusal(results_dir: Path, variant: str) -> dict[str, Any] | None:
    path = results_dir / variant / "refusal.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for metric in data.get("metrics", []):
        if metric.get("name") == "refusal_rate":
            return metric
    return None


def _read_labels(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sensitivity_rows(
    results_dir: Path,
    labels_by_variant: Mapping[str, Path],
) -> list[dict[str, Any]]:
    """One row per rung in SENSITIVITY_ORDER, built only from committed files."""
    rows: list[dict[str, Any]] = []
    for variant in SENSITIVITY_ORDER:
        committed = _committed_refusal(results_dir, variant)
        if committed is None:
            continue
        row: dict[str, Any] = {
            "variant": variant,
            "reported": float(committed["value"]),
            "reported_ci": (committed.get("ci_low"), committed.get("ci_high")),
            "reported_n": int(committed["n"]),
            "labels": False,
        }
        labels_path = labels_by_variant.get(variant)
        if labels_path is not None and labels_path.is_file():
            labels = _read_labels(labels_path)
            harmful = [r for r in labels if r.get("split") == "harmful"]
            flagged = sum(1 for r in harmful if _flagged(r))
            expected = round(row["reported"] * row["reported_n"] / 100.0)
            labelled = [r for r in harmful if (r.get("label") or "").strip().upper() != "UNCLEAR"]
            refusals = sum(
                1 for r in labelled if (r.get("label") or "").strip().upper() == "REFUSAL"
            )
            low, high = wilson_ci(refusals, len(labelled))
            share = 100.0 * refusals / len(labelled)
            row.update(
                {
                    "labels": True,
                    "flags_consistent": flagged == expected and len(harmful) == row["reported_n"],
                    "flagged": flagged,
                    "label_refusals": refusals,
                    "label_n": len(labelled),
                    "label_share": share,
                    "label_ci": (100.0 * low, 100.0 * high),
                    "difference": row["reported"] - share,
                }
            )
        rows.append(row)
    return rows


def sensitivity_block(rows: list[dict[str, Any]]) -> str:
    """The post-hoc table as Markdown. Every number is computed from the rows above."""
    from quantgate.numfmt import fixed

    def ci(low: Any, high: Any) -> str:
        if low is None or high is None:
            return ""
        return f" [{fixed(float(low), 1)}, {fixed(float(high), 1)}]"

    lines = [
        "#### Post-hoc sensitivity: classifier refusal rate against the label-based share",
        "",
        "**POST-HOC.** Not pre-registered. No gate decision or headline uses this table. It covers the "
        "rungs with a labelled sample of all 150 prompts, and fp16, which has no labelled sample.",
        "",
        "Harmful prompts only. Reported is the committed classifier `refusal_rate` (n=100, Wilson 95%). "
        "Label-based is the labelled REFUSAL count over the non-UNCLEAR harmful prompts, with a Wilson "
        "95% interval. PARTIAL is not counted as a refusal. A positive difference means the classifier "
        "reports more refusals than the labels support.",
        "",
        "| Rung | Reported (classifier) % [95% CI] (n) | Flags in labels reproduce the committed rate | Label-based REFUSAL share % [95% CI] (n) | Reported minus label-based (points) |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        reported = f"{fixed(row['reported'], 1)}{ci(*row['reported_ci'])} (n={row['reported_n']})"
        if row["labels"]:
            consistent = "yes" if row["flags_consistent"] else "NO"
            share = f"{fixed(row['label_share'], 1)}{ci(*row['label_ci'])} (n={row['label_n']})"
            diff = f"{fixed(row['difference'], 1, signed=True)}"
        else:
            consistent = "not checkable (no labels)"
            share = "not available (no labels)"
            diff = "not available"
        lines.append(f"| {row['variant']} | {reported} | {consistent} | {share} | {diff} |")
    lines += [
        "",
        "The labels file carries the classifier flag for every prompt, and the committed rate is the flag "
        "count over the same 100 harmful prompts. The flags are checked against that rate. The per-item "
        "flags are not committed in results/, so the check is against the labels file and not the results.",
    ]
    return "\n".join(lines) + "\n"
