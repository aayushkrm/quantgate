"""Refusal behaviour: refusal rate on harmful prompts and over-refusal on benign prompts.

Harmful prompts: walledai/AdvBench (fallback JailbreakBench/JBB-Behaviors). Benign control:
tatsu-lab/alpaca. The classifier is a documented regex list. Raw completions stay local, under
the gitignored raw directory. Only flags, hashes and matched patterns are committed.
The classifier has not been validated by a human yet (see audit/).
"""

from pathlib import Path

from quantgate.backend import Backend
from quantgate.schema import MetricValue

# Published refusal patterns. Empty until the pattern list is written and reviewed.
REFUSAL_PATTERNS: tuple[str, ...] = ()


def matched_pattern(text: str) -> str | None:
    """The first REFUSAL_PATTERNS entry that matches the text, or None."""
    raise NotImplementedError


def is_refusal(text: str) -> bool:
    """True if any REFUSAL_PATTERNS entry matches the text."""
    raise NotImplementedError


def run_refusal(
    backend: Backend,
    harmful: list[str],
    benign: list[str],
    max_tokens: int,
    seed: int,
    raw_dir: Path | None = None,
) -> list[MetricValue]:
    """Return refusal_rate (harmful) and benign_refusal_rate (over-refusal) with CIs.

    If raw_dir is given, raw completions are written there as JSONL. That directory must be
    gitignored.
    """
    raise NotImplementedError
