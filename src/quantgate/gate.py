"""Gate: pass/fail rules relative to the reference variant, hidden-danger flags, recommendation.

Rules come from configs/gate.yaml (defaults are pre-registered in docs/prereg.md before any
model run). Hidden danger is flagged when every quality metric is within band_pts of the
reference, but the refusal rate drops by at least refusal_drop_pts ([03 §4.1]).
"""

from pathlib import Path

from quantgate.schema import GateDecision, GatePolicy, VariantResult


def load_policy(path: Path) -> GatePolicy:
    """Parse and validate a gate policy YAML file."""
    raise NotImplementedError


def evaluate(
    results: list[VariantResult],
    reference: str,
    policy: GatePolicy,
) -> list[GateDecision]:
    """One GateDecision per non-reference variant. Raises ValueError if reference is absent."""
    raise NotImplementedError


def recommend(
    decisions: list[GateDecision],
    results: list[VariantResult],
) -> str | None:
    """Name of the smallest file (by file_bytes) that passes every rule, or None if none does."""
    raise NotImplementedError
