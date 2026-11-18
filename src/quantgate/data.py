"""Fixed evaluation subsets.

Only item IDs are committed, in data/subsets/*.json, so the selection is reproducible. Item
text is loaded at runtime from the Hugging Face datasets named in DECISION.md §2.2.
"""

from pathlib import Path
from typing import Any


def select_subset(item_ids: list[str], n: int, seed: int) -> list[str]:
    """Deterministic seeded selection of n IDs, returned in sorted order.

    Raises ValueError if n exceeds the number of available IDs.
    """
    raise NotImplementedError


def load_subset_ids(path: Path) -> list[str]:
    """Read a committed subset file (data/subsets/<name>.json) and return its item IDs."""
    raise NotImplementedError


def write_subset(path: Path, dataset: str, split: str, seed: int, ids: list[str]) -> None:
    """Write a subset file with dataset, split, seed and IDs. Used once, when the subset is made."""
    raise NotImplementedError


def load_items(
    dataset: str,
    config: str | None,
    split: str,
    ids: list[str],
) -> list[dict[str, Any]]:
    """Load the rows named by ids from a Hugging Face dataset. Needs network or a local cache."""
    raise NotImplementedError
