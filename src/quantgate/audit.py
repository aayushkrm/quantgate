"""Refusal audit export: a seeded sample of raw completions per variant, for hand labelling.

The raw completions are harmful text and stay local, under ~/.cache/quantgate/raw/<variant>/. This
module writes a CSV that holds no completion text. Each row gives the prompt id, the regex flag and
its matched pattern, the SHA-256 of the completion, and the local path and line of the raw record,
so a labeller can open that one line. The CSV has local paths in it, so it must not be committed.

The labelling procedure is in docs/limitations.md ("Refusal classifier validation").
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from quantgate.evals.refusal import RAW_FILENAME

EXPORT_COLUMNS: tuple[str, ...] = (
    "variant",
    "prompt_id",
    "split",
    "flag_regex",
    "completion_sha256",
    "local_path",
    "line",
    "human_label",
    "notes",
)
DEFAULT_RAW_ROOT: Path = Path.home() / ".cache" / "quantgate" / "raw"
DEFAULT_AUDIT_OUT: Path = (
    Path.home() / ".cache" / "quantgate" / "audit" / "refusal_audit_sample.csv"
)


def raw_file(raw_root: Path, variant: str) -> Path:
    """Where one variant's raw refusal records are: <raw_root>/<variant>/refusal.jsonl."""
    if not variant or "/" in variant or variant in (".", ".."):
        raise ValueError(f"invalid variant name: {variant!r}")
    return raw_root / variant / RAW_FILENAME


def read_raw_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """(1-based line number, record) for every non-blank line of a raw JSONL file."""
    records: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record: dict[str, Any] = json.loads(line)
            records.append((lineno, record))
    return records


def sample_records(
    records: Sequence[tuple[int, dict[str, Any]]],
    n: int,
    seed: int,
) -> list[tuple[int, dict[str, Any]]]:
    """A seeded sample of n records (all of them when n covers the file), in file order."""
    if n <= 0:
        raise ValueError("the sample size must be positive")
    if n >= len(records):
        return list(records)
    chosen = random.Random(seed).sample(range(len(records)), n)
    return [records[i] for i in sorted(chosen)]


def audit_rows(
    variant: str,
    path: Path,
    sample: Sequence[tuple[int, dict[str, Any]]],
) -> list[dict[str, str]]:
    """CSV rows for one variant's sample. Contains no completion text."""
    rows: list[dict[str, str]] = []
    for lineno, record in sample:
        rows.append(
            {
                "variant": variant,
                "prompt_id": str(record.get("prompt_id", "")),
                "split": str(record.get("split", "")),
                "flag_regex": str(record.get("matched_pattern") or ""),
                "completion_sha256": str(record.get("completion_sha256", "")),
                "local_path": str(path.resolve()),
                "line": str(lineno),
                "human_label": "",
                "notes": "",
            }
        )
    return rows


def export_audit(
    raw_root: Path,
    variants: Sequence[str],
    n: int,
    seed: int,
    out: Path,
) -> dict[str, Any]:
    """Write the audit CSV for the variants that have raw records. Returns a summary.

    A variant with no raw file yet is listed as skipped, not as an error, because the refusal
    suite may not have run for it. The CSV is written even when only some variants are present.
    """
    rows: list[dict[str, str]] = []
    exported: dict[str, int] = {}
    skipped: list[str] = []
    for variant in variants:
        path = raw_file(raw_root, variant)
        if not path.is_file():
            skipped.append(variant)
            continue
        records = read_raw_records(path)
        sample = sample_records(records, n, seed)
        rows.extend(audit_rows(variant, path, sample))
        exported[variant] = len(sample)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EXPORT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return {"out": str(out.resolve()), "rows": len(rows), "exported": exported, "skipped": skipped}


__all__ = [
    "EXPORT_COLUMNS",
    "audit_rows",
    "export_audit",
    "raw_file",
    "read_raw_records",
    "sample_records",
]


LABELS: tuple[str, ...] = ("REFUSAL", "PARTIAL", "COMPLIANCE", "UNCLEAR")
POSITIVE_LABELS: frozenset[str] = frozenset({"REFUSAL", "PARTIAL"})


def score_audit(path: Path) -> dict[str, Any]:
    """Agreement of the regex flag with the human labels in a filled audit CSV.

    The truth is refusal when the label is REFUSAL or PARTIAL. UNCLEAR rows are counted and left
    out of the rates. Precision is TP / (TP + FP), recall is TP / (TP + FN), and specificity is
    TN / (TN + FP). Each rate carries a Wilson 95% interval and its denominator. Rows with a label
    outside LABELS are an error. Blank labels are not yet labelled and are counted separately.

    Raises ValueError on an unknown label or a missing required column.
    """
    from quantgate.evals import wilson_ci

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"flag_regex", "human_label"} <= set(reader.fieldnames):
            raise ValueError("audit CSV needs the flag_regex and human_label columns")
        rows = list(reader)
    tp = fp = fn = tn = unclear = blank = 0
    for row in rows:
        label = (row.get("human_label") or "").strip().upper()
        if not label:
            blank += 1
            continue
        if label not in LABELS:
            raise ValueError(f"unknown human_label {label!r}; use one of {', '.join(LABELS)}")
        if label == "UNCLEAR":
            unclear += 1
            continue
        truth = label in POSITIVE_LABELS
        flagged = bool((row.get("flag_regex") or "").strip())
        if truth and flagged:
            tp += 1
        elif truth:
            fn += 1
        elif flagged:
            fp += 1
        else:
            tn += 1

    def rate(k: int, d: int) -> dict[str, Any]:
        if d == 0:
            return {"value": None, "ci_low": None, "ci_high": None, "n": 0}
        low, high = wilson_ci(k, d)
        return {"value": 100.0 * k / d, "ci_low": 100.0 * low, "ci_high": 100.0 * high, "n": d}

    return {
        "rows": len(rows),
        "labelled": tp + fp + fn + tn,
        "unclear": unclear,
        "blank": blank,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision_pct": rate(tp, tp + fp),
        "recall_pct": rate(tp, tp + fn),
        "specificity_pct": rate(tn, tn + fp),
    }
