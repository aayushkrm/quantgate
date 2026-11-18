"""Pareto plots (file size vs ARC accuracy, decode tok/s vs KL, refusal vs size) as PNG and SVG.

Dominated points are labelled. Plots are written next to the results they summarise.
"""

from pathlib import Path


def pareto_plot(
    points: list[tuple[str, float, float]],
    x_label: str,
    y_label: str,
    out_stem: Path,
    lower_x_better: bool = True,
    lower_y_better: bool = True,
) -> list[Path]:
    """Scatter of (label, x, y) with the Pareto front drawn. Writes out_stem.png and .svg.

    Returns the written paths.
    """
    raise NotImplementedError
