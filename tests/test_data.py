"""Subset selection, committed ID files and loader plumbing (no network: loaders are patched)."""

import json
from pathlib import Path
from typing import Any

import pytest

from quantgate import data
from quantgate.data import (
    SPECS,
    SUBSET_SEED,
    arc_row_to_item,
    load_arc_easy,
    load_gsm8k,
    load_subset,
    load_subset_ids,
    make_subset,
    pick_rows,
    row_ids,
    select_subset,
    wikitext_passage_pool,
    write_subset,
)

ALLOWED_KEYS = {
    "dataset",
    "config",
    "split",
    "seed",
    "n",
    "selection",
    "pool_size",
    "min_chars",
    "ids",
}


def test_select_subset_is_deterministic_and_sorted() -> None:
    pool = [f"item{i:04d}" for i in range(500)]
    first = select_subset(pool, 30, seed=7)
    assert first == select_subset(pool, 30, seed=7)
    assert first == sorted(first)
    assert len(set(first)) == 30
    assert set(first) <= set(pool)


def test_select_subset_ignores_input_order() -> None:
    pool = [f"id{i}" for i in range(100)]
    shuffled = list(reversed(pool))
    assert select_subset(pool, 10, seed=3) == select_subset(shuffled, 10, seed=3)


def test_select_subset_depends_on_seed() -> None:
    pool = [f"id{i}" for i in range(1000)]
    assert select_subset(pool, 50, seed=1) != select_subset(pool, 50, seed=2)


def test_select_subset_full_pool_and_bounds() -> None:
    pool = ["a", "b", "c"]
    assert select_subset(pool, 3, seed=0) == ["a", "b", "c"]
    with pytest.raises(ValueError):
        select_subset(pool, 4, seed=0)
    with pytest.raises(ValueError):
        select_subset(pool, -1, seed=0)


def test_select_subset_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError):
        select_subset(["a", "a", "b"], 1, seed=0)


def test_write_then_load_subset_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    write_subset(
        path,
        dataset="x/y",
        split="test",
        seed=5,
        ids=["b", "a", "c"],
        config="cfg",
        notes="note",
        extra={"pool_size": 9},
    )
    subset = load_subset(path)
    assert subset.dataset == "x/y"
    assert subset.config == "cfg"
    assert subset.split == "test"
    assert subset.seed == 5
    assert subset.ids == ["a", "b", "c"]
    assert subset.meta == {"selection": "note", "pool_size": 9}
    assert load_subset_ids(path) == ["a", "b", "c"]


def test_load_subset_rejects_inconsistent_n(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"dataset": "d", "config": None, "split": "t", "seed": 0, "n": 3, "ids": ["a"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_subset(path)


def test_load_subset_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "dup.json"
    path.write_text(
        json.dumps(
            {"dataset": "d", "config": None, "split": "t", "seed": 0, "n": 2, "ids": ["a", "a"]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_subset(path)


@pytest.mark.parametrize("name", sorted(SPECS))
def test_committed_subset_files_match_specs(name: str) -> None:
    spec = SPECS[name]
    path = data.SUBSETS_DIR / spec.filename
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert set(raw) <= ALLOWED_KEYS, "only IDs and selection metadata may be committed"
    assert raw["dataset"] == spec.dataset
    assert raw["config"] == spec.config
    assert raw["split"] == spec.split
    assert raw["seed"] == SUBSET_SEED
    ids = raw["ids"]
    assert len(ids) == spec.n == raw["n"]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)
    # Item IDs are short identifiers, never text.
    assert all(isinstance(i, str) and len(i) <= 64 for i in ids)


def test_row_ids_uses_id_field_or_index() -> None:
    rows: list[dict[str, Any]] = [{"id": "abc", "q": 1}, {"q": 2}, {"id": 7}]
    assert row_ids(rows) == ["abc", "1", "7"]


def test_pick_rows_keeps_requested_order_and_rejects_gaps() -> None:
    rows: list[dict[str, Any]] = [{"question": "q0"}, {"question": "q1"}, {"question": "q2"}]
    picked = pick_rows(rows, ["2", "0"])
    assert [r["question"] for r in picked] == ["q2", "q0"]
    with pytest.raises(ValueError):
        pick_rows(rows, ["9"])
    with pytest.raises(ValueError):
        pick_rows([{"id": "a"}, {"id": "a"}], ["a"])


def test_arc_row_to_item_maps_fields() -> None:
    row = {
        "id": "Q1",
        "question": " Which is a plant? ",
        "choices": {"text": ["rock", "oak"], "label": ["1", "2"]},
        "answerKey": "2",
    }
    item = arc_row_to_item(row)
    assert item.item_id == "Q1"
    assert item.question == " Which is a plant? "
    assert item.choices == ["rock", "oak"]
    assert item.labels == ["1", "2"]
    assert item.answer_key == "2"


def test_wikitext_pool_skips_headings_blanks_and_short_lines() -> None:
    long_a = "a" * 30
    lines = [
        " = Title = \n",
        "\n",
        " " + long_a + " \n",
        " short \n",
        " = = Section = = \n",
        " " + "b" * 40 + "\n",
    ]
    pool = wikitext_passage_pool(lines, min_chars=30)
    assert pool == [(2, long_a), (5, "b" * 40)]


def _fake_rows(dataset: str, config: str | None, split: str) -> list[dict[str, Any]]:
    """Synthetic stand-ins for the three Hub splits, shaped like the real columns."""
    if dataset == "allenai/ai2_arc":
        return [
            {
                "id": f"arc{i}",
                "question": f"question {i}",
                "choices": {"text": ["w", "x", "y", "z"], "label": ["A", "B", "C", "D"]},
                "answerKey": "B",
            }
            for i in range(40)
        ]
    if dataset == "openai/gsm8k":
        return [{"question": f"problem {i}", "answer": f"steps\n#### {i}"} for i in range(40)]
    if dataset == "Salesforce/wikitext":
        return [
            {"text": " " + ("word " * 300) + f"{i}\n"} if i % 2 else {"text": " = h = \n"}
            for i in range(40)
        ]
    raise AssertionError(dataset)


def test_load_helpers_read_committed_ids_via_patched_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data, "_download_rows", _fake_rows)
    write_subset(
        tmp_path / "arc_easy_test.json",
        "allenai/ai2_arc",
        "test",
        1,
        ["arc3", "arc0"],
        config="ARC-Easy",
    )
    write_subset(tmp_path / "gsm8k_test.json", "openai/gsm8k", "test", 1, ["5", "2"], config="main")
    write_subset(
        tmp_path / "wikitext2_test_passages.json",
        "Salesforce/wikitext",
        "test",
        1,
        ["3", "1"],
        config="wikitext-2-raw-v1",
        extra={"min_chars": 1200},
    )

    items = load_arc_easy(tmp_path)
    assert [i.item_id for i in items] == ["arc0", "arc3"]
    assert items[0].answer_key == "B"

    pairs = load_gsm8k(tmp_path)
    assert pairs == [("problem 2", "steps\n#### 2"), ("problem 5", "steps\n#### 5")]

    passages = data.load_wikitext_passages(tmp_path)
    assert len(passages) == 2
    assert passages[0].startswith("word word")
    assert passages[0].endswith("1")


def test_make_subset_is_reproducible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "_download_rows", _fake_rows)
    spec = data.SubsetSpec(
        name="gsm8k",
        dataset="openai/gsm8k",
        config="main",
        split="test",
        n=10,
        filename="gsm8k_test.json",
    )
    first = make_subset(spec, subsets_dir=tmp_path / "a", seed=SUBSET_SEED).read_text()
    second = make_subset(spec, subsets_dir=tmp_path / "b", seed=SUBSET_SEED).read_text()
    assert first == second
    assert load_subset_ids(tmp_path / "a" / "gsm8k_test.json") == sorted(
        load_subset_ids(tmp_path / "a" / "gsm8k_test.json")
    )
    assert len(load_subset_ids(tmp_path / "a" / "gsm8k_test.json")) == 10


def test_make_subset_wikitext_records_min_chars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data, "_download_rows", _fake_rows)
    spec = data.SubsetSpec(
        name="wikitext2",
        dataset="Salesforce/wikitext",
        config="wikitext-2-raw-v1",
        split="test",
        n=5,
        filename="wikitext2_test_passages.json",
    )
    path = make_subset(spec, subsets_dir=tmp_path, seed=0)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["min_chars"] == data.WIKITEXT_MIN_CHARS
    assert raw["pool_size"] == 20
    assert all(int(i) % 2 == 1 for i in raw["ids"])
