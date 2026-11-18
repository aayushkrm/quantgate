"""Pareto front logic and plot files. Plots are built from SYNTHETIC fixtures; no display needed."""

from pathlib import Path

import pytest

from quantgate.plots import pareto_front, pareto_plot, write_report_plots
from quantgate.report import load_report

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_pareto_front_lower_is_better_on_both_axes() -> None:
    points = [(1.0, 1.0), (2.0, 2.0), (0.5, 3.0), (3.0, 0.5)]
    # (1,1) dominates (2,2). (0.5,3) and (3,0.5) are not dominated by anything.
    assert pareto_front(points, True, True) == [True, False, True, True]


def test_pareto_front_higher_is_better_on_y() -> None:
    points = [(1.0, 10.0), (2.0, 5.0), (3.0, 1.0)]
    # Lower x and higher y are better. (1,10) beats both others on both axes.
    assert pareto_front(points, lower_x_better=True, lower_y_better=False) == [True, False, False]


def test_pareto_front_flags_a_point_dominated_on_both_axes_with_direction() -> None:
    points = [(1.0, 10.0), (2.0, 5.0)]
    # Lower x and higher y better: (1,10) beats (2,5) on both, so (2,5) is dominated.
    assert pareto_front(points, lower_x_better=True, lower_y_better=False) == [True, False]


def test_identical_points_do_not_dominate_each_other() -> None:
    assert pareto_front([(1.0, 1.0), (1.0, 1.0)]) == [True, True]


def test_pareto_front_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        pareto_front([(float("nan"), 1.0)])


def test_pareto_plot_writes_non_empty_png_and_svg(tmp_path: Path) -> None:
    stem = tmp_path / "plots" / "demo_pareto"
    written = pareto_plot(
        [("fp16", 900.0, 60.0), ("q8_0", 500.0, 59.0), ("q2_k", 200.0, 40.0)],
        x_label="File size (MiB)",
        y_label="ARC-Easy (%)",
        out_stem=stem,
        lower_x_better=True,
        lower_y_better=False,
    )
    assert [p.name for p in written] == ["demo_pareto.png", "demo_pareto.svg"]
    png, svg = written
    assert png.read_bytes().startswith(PNG_MAGIC)
    assert png.stat().st_size > 1000
    svg_text = svg.read_text(encoding="utf-8")
    assert "<svg" in svg_text
    assert "dominated" in svg_text or "fp16" in svg_text


def test_pareto_plot_labels_dominated_points(tmp_path: Path) -> None:
    written = pareto_plot(
        [("best", 1.0, 90.0), ("worse", 2.0, 10.0)],
        "size",
        "acc",
        tmp_path / "labelled",
        lower_x_better=True,
        lower_y_better=False,
    )
    svg = written[1].read_text(encoding="utf-8")
    assert "worse (dominated)" in svg
    assert "best (dominated)" not in svg


def test_pareto_plot_rejects_empty_points(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one point"):
        pareto_plot([], "x", "y", tmp_path / "empty")


def test_pareto_plot_single_point_has_no_front_line(tmp_path: Path) -> None:
    written = pareto_plot([("only", 1.0, 1.0)], "x", "y", tmp_path / "one")
    assert all(p.stat().st_size > 0 for p in written)


def test_write_report_plots_from_fixture(tmp_path: Path) -> None:
    report = load_report(FIXTURES / "report_mixed.json")
    written = write_report_plots(report, tmp_path)
    names = {p.name for p in written}
    assert names == {
        "pareto_size_vs_arc.png",
        "pareto_size_vs_arc.svg",
        "pareto_decode_vs_kl.png",
        "pareto_decode_vs_kl.svg",
        "pareto_refusal_vs_size.png",
        "pareto_refusal_vs_size.svg",
    }
    for path in written:
        assert path.exists()
        assert path.stat().st_size > 0


def test_write_report_plots_skips_plot_without_inputs(tmp_path: Path) -> None:
    report = load_report(FIXTURES / "report_mixed.json")
    stripped = report.model_copy(
        update={
            "variants": [
                vr.model_copy(update={"metrics": {"kl_mean": vr.metrics["kl_mean"]}})
                for vr in report.variants
            ]
        }
    )
    written = write_report_plots(stripped, tmp_path)
    assert written == []
