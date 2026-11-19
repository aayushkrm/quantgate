"""Refusal audit export: seeded samples per variant, no completion text in the CSV, paths that
point at the right raw line. All raw files are synthetic and live in tmp_path."""

import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quantgate.audit import (
    EXPORT_COLUMNS,
    audit_rows,
    export_audit,
    raw_file,
    read_raw_records,
    sample_records,
)
from quantgate.cli import app

SECRET = "SYNTHETIC-COMPLETION-TEXT-MUST-NOT-APPEAR"


def _write_raw(root: Path, variant: str, n: int) -> Path:
    path = raw_file(root, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for i in range(n):
            record = {
                "prompt_id": f"harm-{i:04d}",
                "split": "harmful" if i % 2 == 0 else "benign",
                "completion": f"{SECRET} {variant} {i}",
                "flagged": i % 3 == 0,
                "matched_pattern": "I can't" if i % 3 == 0 else None,
                "completion_sha256": f"{i:064x}",
            }
            handle.write(json.dumps(record) + "\n")
    return path


def test_sample_is_seeded_and_keeps_file_order() -> None:
    records = [(i + 1, {"prompt_id": f"p{i}"}) for i in range(30)]
    a = sample_records(records, 10, seed=0)
    assert a == sample_records(records, 10, seed=0)
    assert a != sample_records(records, 10, seed=1)
    lines = [lineno for lineno, _ in a]
    assert lines == sorted(lines) and len(a) == 10


def test_sample_covers_the_whole_file_when_n_is_large() -> None:
    records = [(1, {"prompt_id": "a"}), (2, {"prompt_id": "b"})]
    assert sample_records(records, 50, seed=0) == records
    with pytest.raises(ValueError, match="positive"):
        sample_records(records, 0, seed=0)


def test_export_writes_n_per_variant_and_no_completion_text(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _write_raw(raw_root, "q4_k_m", 40)
    _write_raw(raw_root, "q8_0", 40)
    out = tmp_path / "audit.csv"
    summary = export_audit(raw_root, ["q4_k_m", "q8_0"], n=12, seed=0, out=out)

    text = out.read_text(encoding="utf-8")
    assert SECRET not in text, "completion text must never reach the audit CSV"
    rows = list(csv.DictReader(text.splitlines()))
    assert tuple(rows[0].keys()) == EXPORT_COLUMNS
    assert summary["exported"] == {"q4_k_m": 12, "q8_0": 12}
    assert len(rows) == 24
    assert {r["variant"] for r in rows} == {"q4_k_m", "q8_0"}
    assert all(r["human_label"] == "" and r["notes"] == "" for r in rows)


def test_each_line_pointer_opens_the_matching_raw_record(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    path = _write_raw(raw_root, "q4_k_m", 20)
    out = tmp_path / "audit.csv"
    export_audit(raw_root, ["q4_k_m"], n=5, seed=3, out=out)
    lines = path.read_text(encoding="utf-8").splitlines()
    for row in csv.DictReader(out.read_text(encoding="utf-8").splitlines()):
        assert row["local_path"] == str(path.resolve())
        record = json.loads(lines[int(row["line"]) - 1])
        assert record["prompt_id"] == row["prompt_id"]
        assert record["completion_sha256"] == row["completion_sha256"]
        assert row["flag_regex"] == (record["matched_pattern"] or "")


def test_missing_variant_is_skipped_not_an_error(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _write_raw(raw_root, "q4_k_m", 5)
    summary = export_audit(raw_root, ["q4_k_m", "q2_k"], n=3, seed=0, out=tmp_path / "a.csv")
    assert summary["skipped"] == ["q2_k"]
    assert summary["exported"] == {"q4_k_m": 3}


def test_read_raw_records_numbers_lines_from_one(tmp_path: Path) -> None:
    path = _write_raw(tmp_path / "raw", "q4_k_m", 3)
    numbered = read_raw_records(path)
    assert [n for n, _ in numbered] == [1, 2, 3]


def test_audit_rows_have_the_template_columns(tmp_path: Path) -> None:
    path = _write_raw(tmp_path / "raw", "q4_k_m", 2)
    rows = audit_rows("q4_k_m", path, read_raw_records(path))
    assert set(rows[0]) == set(EXPORT_COLUMNS)


def test_cli_audit_export_end_to_end(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _write_raw(raw_root, "q4_k_m", 9)
    out = tmp_path / "audit.csv"
    result = CliRunner().invoke(
        app,
        [
            "audit",
            "export",
            "--only",
            "q4_k_m",
            "--n",
            "4",
            "--seed",
            "1",
            "--raw-root",
            str(raw_root),
            "--out",
            str(out),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["rows"] == 4
    assert SECRET not in out.read_text(encoding="utf-8")


def test_cli_audit_export_rejects_unknown_variant(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "audit",
            "export",
            "--only",
            "nope",
            "--raw-root",
            str(tmp_path),
            "--out",
            str(tmp_path / "a.csv"),
        ],
    )
    assert result.exit_code == 2
    assert "unknown variant" in result.output


def test_score_audit_confusion_and_rates(tmp_path: Path) -> None:
    from quantgate.audit import score_audit

    path = tmp_path / "labelled.csv"
    rows = [
        ("I can't help", "REFUSAL"),  # TP
        ("I can't help", "COMPLIANCE"),  # FP
        ("", "REFUSAL"),  # FN
        ("", "COMPLIANCE"),  # TN
        ("", "UNCLEAR"),  # excluded
        ("", ""),  # blank
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["flag_regex", "human_label"])
        writer.writeheader()
        for flag, label in rows:
            writer.writerow({"flag_regex": flag, "human_label": label})
    summary = score_audit(path)
    assert summary["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert summary["unclear"] == 1 and summary["blank"] == 1 and summary["labelled"] == 4
    assert summary["precision_pct"]["value"] == 50.0
    assert summary["recall_pct"]["value"] == 50.0
    assert summary["precision_pct"]["n"] == 2


def test_score_audit_rejects_unknown_labels(tmp_path: Path) -> None:
    from quantgate.audit import score_audit

    path = tmp_path / "bad.csv"
    path.write_text("flag_regex,human_label\n,MAYBE\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown label"):
        score_audit(path)


def test_cli_audit_score_prints_rates(tmp_path: Path) -> None:
    path = tmp_path / "labelled.csv"
    path.write_text("flag_regex,human_label\nI can't,REFUSAL\n,COMPLIANCE\n", encoding="utf-8")
    result = CliRunner().invoke(app, ["audit", "score", str(path)])
    assert result.exit_code == 0, result.output
    assert "precision_pct: 100.0" in result.output
