"""Post-hoc sensitivity table: committed-rate checks and label-based shares, on tiny synthetic
files in tmp_path. Nothing here reads the committed results or labels."""

import csv
import json
from pathlib import Path

from quantgate.sensitivity import sensitivity_block, sensitivity_rows


def _results(root: Path, variant: str, value: float, n: int) -> None:
    folder = root / variant
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": variant,
        "metrics": [
            {"name": "refusal_rate", "value": value, "ci_low": 40.0, "ci_high": 60.0, "n": n}
        ],
    }
    (folder / "refusal.json").write_text(json.dumps(payload), encoding="utf-8")


def _labels(path: Path, rows: list[tuple[str, str, str, str]]) -> Path:
    """Rows of (split, flagged, label, prompt)."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "prompt_id", "split", "regex_flagged", "label", "labeller"])
        for split, flagged, label, prompt in rows:
            writer.writerow(["q2_k", prompt, split, flagged, label, "test"])
    return path


def test_consistent_flags_and_label_share(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _results(results, "q2_k", value=75.0, n=4)
    labels = _labels(
        tmp_path / "labels.csv",
        [
            ("harmful", "true", "REFUSAL", "harm-1"),
            ("harmful", "true", "COMPLIANCE", "harm-2"),
            ("harmful", "true", "REFUSAL", "harm-3"),
            ("harmful", "false", "UNCLEAR", "harm-4"),
            ("benign", "false", "COMPLIANCE", "benign-1"),
        ],
    )
    rows = sensitivity_rows(results, {"q2_k": labels})
    (row,) = rows
    assert row["flags_consistent"] is True  # 3 of 4 harmful flagged equals round(75 * 4 / 100)
    assert row["label_n"] == 3 and row["label_refusals"] == 2


def test_mismatched_flags_are_reported_as_inconsistent(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _results(results, "q2_k", value=100.0, n=2)
    labels = _labels(
        tmp_path / "labels.csv",
        [("harmful", "false", "REFUSAL", "harm-1"), ("harmful", "true", "REFUSAL", "harm-2")],
    )
    (row,) = sensitivity_rows(results, {"q2_k": labels})
    assert row["flags_consistent"] is False


def test_fp16_row_has_no_label_reference(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _results(results, "fp16", value=72.0, n=100)
    rows = sensitivity_rows(results, {})
    (row,) = rows
    assert row["labels"] is False
    text = sensitivity_block(rows)
    assert "not available (no labels)" in text
    assert "POST-HOC" in text and "Not pre-registered" in text
