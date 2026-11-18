"""Pareto plots (file size vs ARC accuracy, decode tok/s vs KL, refusal vs size) as PNG and SVG.

Dominated points are labelled. Plots are written next to the results they summarise. The figures
are built on matplotlib's Figure with the Agg canvas, so no GUI backend or pyplot state is used.
Colours are the Okabe-Ito colour-blind-safe pair, and points also differ in marker shape and fill,
so the plots do not depend on colour alone. Backgrounds are white with dark text, which reads in
light and dark viewers.
"""

import math
from collections.abc import Sequence
from pathlib import Path

from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from quantgate.gate import variant_size_bytes
from quantgate.schema import QuantReport

FRONT_COLOUR = "#0072B2"  # Okabe-Ito blue
DOMINATED_COLOUR = "#D55E00"  # Okabe-Ito vermillion
TEXT_COLOUR = "#1d1d1f"
GRID_COLOUR = "#d6d6db"
MIB = 2**20


def pareto_front(
    points: Sequence[tuple[float, float]],
    lower_x_better: bool = True,
    lower_y_better: bool = True,
) -> list[bool]:
    """For each point, True if no other point is at least as good on both axes and better on one.

    'Better' follows the direction flags. Identical points do not dominate each other.

    Raises ValueError if any coordinate is not finite.
    """
    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"non-finite point ({x}, {y})")
    # Convert to minimisation on both axes.
    coords = [(x if lower_x_better else -x, y if lower_y_better else -y) for x, y in points]
    flags: list[bool] = []
    for i, (xi, yi) in enumerate(coords):
        dominated = False
        for j, (xj, yj) in enumerate(coords):
            if i == j:
                continue
            if xj <= xi and yj <= yi and (xj < xi or yj < yi):
                dominated = True
                break
        flags.append(not dominated)
    return flags


def _direction_label(label: str, lower_better: bool) -> str:
    """Append the direction of 'better' to an axis label."""
    return f"{label} ({'lower' if lower_better else 'higher'} is better)"


def pareto_plot(
    points: list[tuple[str, float, float]],
    x_label: str,
    y_label: str,
    out_stem: Path,
    lower_x_better: bool = True,
    lower_y_better: bool = True,
) -> list[Path]:
    """Scatter of (label, x, y) with the Pareto front drawn. Writes out_stem.png and .svg.

    Front points are filled circles joined by a dashed line. Dominated points are hollow squares.
    Every point is labelled with its name.

    Returns the written paths. Raises ValueError if points is empty or a coordinate is not finite.
    """
    if not points:
        raise ValueError("pareto_plot needs at least one point")
    flags = pareto_front([(x, y) for _, x, y in points], lower_x_better, lower_y_better)

    fig = Figure(figsize=(7.5, 5.0), dpi=150, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.set_facecolor("white")
    ax.set_xlabel(_direction_label(x_label, lower_x_better), color=TEXT_COLOUR)
    ax.set_ylabel(_direction_label(y_label, lower_y_better), color=TEXT_COLOUR)
    ax.tick_params(colors=TEXT_COLOUR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOUR)
    ax.grid(True, color=GRID_COLOUR, linewidth=0.6)

    front = sorted(
        ((x, y) for (_, x, y), on in zip(points, flags, strict=True) if on),
    )
    if len(front) >= 2:
        ax.plot(
            [p[0] for p in front],
            [p[1] for p in front],
            linestyle="--",
            color=FRONT_COLOUR,
            linewidth=1.2,
            label="Pareto front",
        )

    front_pts = [(x, y) for (_, x, y), on in zip(points, flags, strict=True) if on]
    dom_pts = [(x, y) for (_, x, y), on in zip(points, flags, strict=True) if not on]
    if front_pts:
        ax.scatter(
            [p[0] for p in front_pts],
            [p[1] for p in front_pts],
            marker="o",
            s=60,
            color=FRONT_COLOUR,
            zorder=3,
            label="On the front",
        )
    if dom_pts:
        ax.scatter(
            [p[0] for p in dom_pts],
            [p[1] for p in dom_pts],
            marker="s",
            s=60,
            facecolors="white",
            edgecolors=DOMINATED_COLOUR,
            linewidths=1.8,
            zorder=3,
            label="Dominated",
        )
    for (name, x, y), on in zip(points, flags, strict=True):
        suffix = "" if on else " (dominated)"
        ax.annotate(
            f"{name}{suffix}",
            (x, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color=TEXT_COLOUR,
        )
    ax.legend(loc="best", fontsize=8, frameon=True, facecolor="white", edgecolor=GRID_COLOUR)
    fig.tight_layout()

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in (".png", ".svg"):
        path = out_stem.parent / (out_stem.name + suffix)
        # Keep SVG text as text (not glyph paths), so labels stay readable and searchable.
        with rc_context({"svg.fonttype": "none"}):
            fig.savefig(path, facecolor="white")
        written.append(path)
    return written


def _size_mib(report: QuantReport) -> dict[str, float]:
    """File size in MiB per variant name, for variants with a known size."""
    sizes: dict[str, float] = {}
    for vr in report.variants:
        size = variant_size_bytes(vr)
        if size is not None:
            sizes[vr.variant.name] = size / MIB
    return sizes


def write_report_plots(report: QuantReport, out_dir: Path) -> list[Path]:
    """Write the three Pareto plots that the report's metrics support.

    A plot is skipped when none of its variants has both coordinates. Returns all written paths.
    """
    sizes = _size_mib(report)
    specs: list[tuple[str, str, str, list[tuple[str, float, float]], bool, bool]] = []

    arc_points = [
        (vr.variant.name, sizes[vr.variant.name], vr.metrics["arc_acc"].value)
        for vr in report.variants
        if vr.variant.name in sizes and "arc_acc" in vr.metrics
    ]
    specs.append(
        ("pareto_size_vs_arc", "File size (MiB)", "ARC-Easy accuracy (%)", arc_points, True, False)
    )

    speed_points = [
        (vr.variant.name, vr.metrics["kl_mean"].value, vr.metrics["decode_tok_s"].value)
        for vr in report.variants
        if "kl_mean" in vr.metrics and "decode_tok_s" in vr.metrics
    ]
    specs.append(
        (
            "pareto_decode_vs_kl",
            "KL vs fp16 (nats/token)",
            "Decode tok/s (CPU)",
            speed_points,
            True,
            False,
        )
    )

    refusal_points = [
        (vr.variant.name, sizes[vr.variant.name], vr.metrics["refusal_rate"].value)
        for vr in report.variants
        if vr.variant.name in sizes and "refusal_rate" in vr.metrics
    ]
    specs.append(
        (
            "pareto_refusal_vs_size",
            "File size (MiB)",
            "Refusal rate, harmful (%)",
            refusal_points,
            True,
            False,
        )
    )

    written: list[Path] = []
    for stem, x_label, y_label, points, lower_x, lower_y in specs:
        if not points:
            continue
        written += pareto_plot(
            points, x_label, y_label, out_dir / stem, lower_x_better=lower_x, lower_y_better=lower_y
        )
    return written
