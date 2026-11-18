"""Fixed evaluation subsets.

Only item IDs and selection metadata are committed, in data/subsets/*.json. Item text is loaded
at runtime through the Hugging Face ``datasets`` library (DECISION.md §2.2), so no dataset text
is stored in the repo. Selection is seeded with SUBSET_SEED and reproducible:
``scripts/make_subsets.py`` writes the files once, and the ``load_*`` functions read the
committed IDs back.

Selection rules:
- ARC-Easy (allenai/ai2_arc, ARC-Easy, test): row ``id`` strings, n=300.
- GSM8K (openai/gsm8k, main, test): row index as a string, n=100.
- WikiText-2 (Salesforce/wikitext, wikitext-2-raw-v1, test): paragraph rows (not headings, not
  blank) of at least WIKITEXT_MIN_CHARS characters. Each such row is one passage, identified by
  its row index, n=50. The eval truncates each passage to ctx tokens.
"""

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quantgate.evals.mcq import McqItem

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
SUBSETS_DIR: Path = REPO_ROOT / "data" / "subsets"

# Fixed seed for the committed selections. Changing it changes the committed IDs, so any change
# goes in docs/deviations.md.
SUBSET_SEED: int = 2026

# A WikiText paragraph this long is about 256 tokens with the Qwen tokenizer (about 4.5 chars
# per token). Shorter passages still run, but then they contribute fewer positions.
WIKITEXT_MIN_CHARS: int = 1200


@dataclass(frozen=True)
class SubsetSpec:
    """Where a subset comes from and how many items it holds."""

    name: str
    dataset: str
    config: str | None
    split: str
    n: int
    filename: str


SPECS: dict[str, SubsetSpec] = {
    "arc_easy": SubsetSpec(
        name="arc_easy",
        dataset="allenai/ai2_arc",
        config="ARC-Easy",
        split="test",
        n=300,
        filename="arc_easy_test.json",
    ),
    "gsm8k": SubsetSpec(
        name="gsm8k",
        dataset="openai/gsm8k",
        config="main",
        split="test",
        n=100,
        filename="gsm8k_test.json",
    ),
    "wikitext2": SubsetSpec(
        name="wikitext2",
        dataset="Salesforce/wikitext",
        config="wikitext-2-raw-v1",
        split="test",
        n=50,
        filename="wikitext2_test_passages.json",
    ),
}


@dataclass(frozen=True)
class Subset:
    """A committed subset: where its text comes from and which item IDs were selected."""

    dataset: str
    config: str | None
    split: str
    seed: int
    ids: list[str]
    meta: dict[str, Any] = field(default_factory=dict)


def select_subset(item_ids: list[str], n: int, seed: int) -> list[str]:
    """Deterministic seeded selection of n IDs, returned in sorted order.

    The IDs are sorted before sampling, so the result does not depend on the input order.
    Raises ValueError if n exceeds the number of available IDs, or if IDs repeat.
    """
    ids = sorted(item_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("item IDs must be unique")
    if n < 0 or n > len(ids):
        raise ValueError(f"cannot select n={n} from {len(ids)} IDs")
    chosen = random.Random(seed).sample(ids, n)
    return sorted(chosen)


def load_subset(path: Path) -> Subset:
    """Read a committed subset file (data/subsets/<name>.json)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = data["ids"]
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise ValueError(f"{path}: 'ids' must be a list of strings")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate IDs")
    if data["n"] != len(ids):
        raise ValueError(f"{path}: n={data['n']} does not match {len(ids)} IDs")
    known = {"dataset", "config", "split", "seed", "n", "ids"}
    meta = {k: v for k, v in data.items() if k not in known}
    return Subset(
        dataset=str(data["dataset"]),
        config=None if data["config"] is None else str(data["config"]),
        split=str(data["split"]),
        seed=int(data["seed"]),
        ids=list(ids),
        meta=meta,
    )


def load_subset_ids(path: Path) -> list[str]:
    """Read a committed subset file (data/subsets/<name>.json) and return its item IDs."""
    return load_subset(path).ids


def write_subset(
    path: Path,
    dataset: str,
    split: str,
    seed: int,
    ids: list[str],
    *,
    config: str | None = None,
    notes: str = "",
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Write a subset file with dataset, config, split, seed, selection notes and sorted IDs."""
    payload: dict[str, Any] = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "seed": seed,
        "n": len(ids),
        "selection": notes,
    }
    if extra:
        payload.update(extra)
    payload["ids"] = sorted(ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def row_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Item ID of each row: its ``id`` field if it has one, otherwise its index as a string."""
    return [str(row["id"]) if "id" in row else str(i) for i, row in enumerate(rows)]


def pick_rows(rows: Sequence[Mapping[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    """Return the rows named by ids, in the order of ids.

    Raises ValueError on duplicate row IDs and on subset IDs missing from the rows.
    """
    index: dict[str, dict[str, Any]] = {}
    for rid, row in zip(row_ids(rows), rows, strict=True):
        if rid in index:
            raise ValueError(f"duplicate row id {rid!r}")
        index[rid] = dict(row)
    missing = [i for i in ids if i not in index]
    if missing:
        raise ValueError(f"{len(missing)} subset IDs not in the dataset, e.g. {missing[:3]}")
    return [index[i] for i in ids]


def _download_rows(dataset: str, config: str | None, split: str) -> list[dict[str, Any]]:
    """Load one split from the Hub (network or local HF cache)."""
    from datasets import load_dataset

    ds = load_dataset(dataset, name=config, split=split)
    return [dict(row) for row in ds]


def load_items(
    dataset: str,
    config: str | None,
    split: str,
    ids: list[str],
) -> list[dict[str, Any]]:
    """Load the rows named by ids from a Hugging Face dataset. Needs network or a local cache."""
    return pick_rows(_download_rows(dataset, config, split), ids)


def arc_row_to_item(row: Mapping[str, Any]) -> McqItem:
    """Convert one allenai/ai2_arc row into an McqItem."""
    choices = row["choices"]
    return McqItem(
        item_id=str(row["id"]),
        question=str(row["question"]),
        choices=[str(t) for t in choices["text"]],
        labels=[str(lab) for lab in choices["label"]],
        answer_key=str(row["answerKey"]),
    )


def wikitext_passage_pool(lines: Sequence[str], min_chars: int) -> list[tuple[int, str]]:
    """Paragraph lines of at least min_chars characters, as (row index, stripped text) pairs.

    A heading line starts with '=' after optional spaces. Blank lines are skipped.
    """
    pool: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        text = line.strip()
        if not text or text.startswith("="):
            continue
        if len(text) >= min_chars:
            pool.append((i, text))
    return pool


def make_subset(spec: SubsetSpec, subsets_dir: Path = SUBSETS_DIR, seed: int = SUBSET_SEED) -> Path:
    """Select the subset described by spec from the Hub split and write its ID file.

    Needs network. Used by scripts/make_subsets.py, which is run once.
    """
    if spec.name == "wikitext2":
        lines = [str(row["text"]) for row in _download_rows(spec.dataset, spec.config, spec.split)]
        pool_ids = [str(i) for i, _ in wikitext_passage_pool(lines, WIKITEXT_MIN_CHARS)]
        notes = (
            "uniform sample without replacement from the sorted row indices of paragraph "
            f"lines with >= {WIKITEXT_MIN_CHARS} characters (headings and blanks excluded)"
        )
        extra: dict[str, Any] = {"min_chars": WIKITEXT_MIN_CHARS, "pool_size": len(pool_ids)}
    else:
        rows = _download_rows(spec.dataset, spec.config, spec.split)
        pool_ids = row_ids(rows)
        notes = "uniform sample without replacement from the sorted item IDs"
        extra = {"pool_size": len(pool_ids)}
    ids = select_subset(pool_ids, spec.n, seed)
    path = subsets_dir / spec.filename
    write_subset(
        path,
        dataset=spec.dataset,
        split=spec.split,
        seed=seed,
        ids=ids,
        config=spec.config,
        notes=notes,
        extra=extra,
    )
    return path


def load_arc_easy(subsets_dir: Path = SUBSETS_DIR) -> list[McqItem]:
    """The committed ARC-Easy subset as McqItems (text loaded from the Hub)."""
    subset = load_subset(subsets_dir / SPECS["arc_easy"].filename)
    rows = load_items(subset.dataset, subset.config, subset.split, subset.ids)
    return [arc_row_to_item(row) for row in rows]


def load_gsm8k(subsets_dir: Path = SUBSETS_DIR) -> list[tuple[str, str]]:
    """The committed GSM8K subset as (question, answer field) pairs, in subset order.

    The answer field is the full reference solution; its final number follows '####'.
    """
    subset = load_subset(subsets_dir / SPECS["gsm8k"].filename)
    rows = load_items(subset.dataset, subset.config, subset.split, subset.ids)
    return [(str(row["question"]), str(row["answer"])) for row in rows]


def load_wikitext_passages(subsets_dir: Path = SUBSETS_DIR) -> list[str]:
    """The committed WikiText-2 passages, in subset order (text loaded from the Hub).

    Raises ValueError if a committed ID is no longer a valid passage row, which happens when
    the dataset revision changes.
    """
    subset = load_subset(subsets_dir / SPECS["wikitext2"].filename)
    min_chars = int(subset.meta.get("min_chars", WIKITEXT_MIN_CHARS))
    lines = [
        str(row["text"]) for row in _download_rows(subset.dataset, subset.config, subset.split)
    ]
    pool = {str(i): text for i, text in wikitext_passage_pool(lines, min_chars)}
    missing = [i for i in subset.ids if i not in pool]
    if missing:
        raise ValueError(
            f"{len(missing)} passage IDs are not valid passage rows, e.g. {missing[:3]}"
        )
    return [pool[i] for i in subset.ids]
