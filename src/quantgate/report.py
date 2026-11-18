"""Report building and rendering.

Writes results/report.json (schema_version "1") and renders Markdown for README result blocks
between the <!-- results:start --> and <!-- results:end --> markers. Numbers come only from
committed results, never typed by hand.
"""

from pathlib import Path
from typing import Any

from quantgate.schema import GateDecision, QuantReport, VariantResult


def build_report(
    model_family: str,
    reference: str,
    results: list[VariantResult],
    decisions: list[GateDecision],
    recommended: str | None,
    env_info: dict[str, Any],
) -> QuantReport:
    """Assemble a QuantReport from its parts."""
    raise NotImplementedError


def load_report(path: Path) -> QuantReport:
    """Read and validate a report JSON. Raises pydantic.ValidationError on a bad schema_version."""
    raise NotImplementedError


def write_report(report: QuantReport, path: Path) -> None:
    """Write the report as indented JSON, creating parent directories as needed."""
    raise NotImplementedError


def render_markdown(report: QuantReport) -> str:
    """Per-variant table with n and 95% CIs, gate decisions and hidden-danger rows."""
    raise NotImplementedError


def render_results_block(report: QuantReport) -> str:
    """Markdown placed between the README results markers. Ends with the env.json link."""
    raise NotImplementedError
