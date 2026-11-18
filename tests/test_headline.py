"""results/headline.json: built from SYNTHETIC fixtures only. Each value is checked against the
fixture data it came from, and an item whose source is missing must be absent."""

import json
from pathlib import Path

import pytest

from quantgate.headline import (
    BANDWIDTH_SOURCE,
    MAX_ITEMS,
    REPORT_SOURCE,
    Headline,
    build_headline,
    headline_text,
    load_bandwidth,
    write_headline,
)
from quantgate.report import load_report

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SCHEMA_FILE = Path(__file__).resolve().parents[1] / "schemas" / "headline.schema.json"


@pytest.fixture
def mixed():  # type: ignore[no-untyped-def]
    return load_report(FIXTURES / "report_mixed.json")


@pytest.fixture
def hidden():  # type: ignore[no-untyped-def]
    return load_report(FIXTURES / "report_hidden_danger.json")


@pytest.fixture
def all_pass():  # type: ignore[no-untyped-def]
    return load_report(FIXTURES / "report_all_pass.json")


def _by_id(headline: Headline) -> dict[str, object]:
    return {item.id: item for item in headline.items}


def test_no_report_means_no_headline() -> None:
    assert build_headline(None) is None


def test_contract_fields_and_limits(mixed) -> None:  # type: ignore[no-untyped-def]
    h = build_headline(mixed)
    assert h is not None
    assert h.schema_version == "1" and h.tool == "quantgate"
    assert h.links.repo == "https://github.com/aayushkrm/quantgate"
    assert h.links.site == "https://aayushkrm.github.io/quantgate/"
    assert h.links.release is None
    assert 1 <= len(h.items) <= MAX_ITEMS
    assert [item.featured for item in h.items] == [i < 3 for i in range(len(h.items))]


def test_every_item_is_synthetic_and_labelled(mixed) -> None:  # type: ignore[no-untyped-def]
    h = build_headline(mixed)
    assert h is not None
    assert all(item.synthetic for item in h.items)
    assert h.environment.startswith("SYNTHETIC")
    assert all(item.source == REPORT_SOURCE for item in h.items)


def test_recommended_item_matches_the_report(mixed) -> None:  # type: ignore[no-untyped-def]
    h = build_headline(mixed)
    assert h is not None
    rec = _by_id(h)["recommended"]
    assert rec.value == mixed.recommended
    decode = next(v for v in mixed.variants if v.variant.name == mixed.recommended)
    assert f"{decode.metrics['decode_tok_s'].value:.1f} tok/s" in rec.detail


def test_worst_kl_is_the_largest_candidate_value(mixed) -> None:  # type: ignore[no-untyped-def]
    h = build_headline(mixed)
    assert h is not None
    worst = max(
        (v for v in mixed.variants if v.variant.name != mixed.reference),
        key=lambda v: v.metrics["kl_mean"].value,
    )
    item = _by_id(h)["worst_kl"]
    assert item.value == f"{worst.metrics['kl_mean'].value:.4f} nats/token ({worst.variant.name})"


def test_hidden_danger_counts_rows_and_states_null_results(
    hidden,  # type: ignore[no-untyped-def]
    all_pass,  # type: ignore[no-untyped-def]
) -> None:
    flagged = [d.variant for d in hidden.decisions if d.hidden_danger]
    h = build_headline(hidden)
    assert h is not None
    item = _by_id(h)["hidden_danger"]
    assert item.value == f"{len(flagged)} of {len(hidden.decisions)} candidates"
    assert flagged and all(name in item.detail for name in flagged)

    null = build_headline(all_pass)
    assert null is not None
    null_item = _by_id(null)["hidden_danger"]
    assert null_item.value == f"0 of {len(all_pass.decisions)} candidates"
    assert "Null result" in null_item.detail


def test_q2_item_is_omitted_when_the_rung_is_absent(mixed) -> None:  # type: ignore[no-untyped-def]
    without = mixed.model_copy(
        update={"variants": [v for v in mixed.variants if v.variant.name != "q2_k"]}
    )
    assert "q2_k_vs_fp16" in _by_id(build_headline(mixed))
    assert "q2_k_vs_fp16" not in _by_id(build_headline(without))


def test_bandwidth_item_only_when_the_record_exists(mixed) -> None:  # type: ignore[no-untyped-def]
    assert "bandwidth_ceiling" not in _by_id(build_headline(mixed))
    record = {
        "copy_gb_s": 14.0,
        "ceiling_check": {
            "implied_gb_s": 1.1,
            "measured_gb_s": 14.0,
            "ratio": 0.08,
            "within_ceiling": True,
            "model_gb": 0.49,
        },
        "synthetic": True,
    }
    h = build_headline(mixed, bandwidth=record)
    assert h is not None
    item = _by_id(h)["bandwidth_ceiling"]
    assert item.source == BANDWIDTH_SOURCE
    assert item.value == "1.10 GB/s implied vs 14.00 GB/s measured"
    assert item.synthetic is True
    assert len(h.items) <= MAX_ITEMS


def test_bandwidth_loader_returns_none_when_absent(tmp_path: Path) -> None:
    assert load_bandwidth(tmp_path / "bandwidth.json") is None


def test_no_value_is_empty_or_missing(mixed, hidden, all_pass) -> None:  # type: ignore[no-untyped-def]
    for report in (mixed, hidden, all_pass):
        h = build_headline(report)
        assert h is not None
        for item in h.items:
            for text in (item.value, item.detail, item.label):
                assert text and "None" not in text and "nan" not in text


def test_text_round_trips_through_the_model(mixed, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    h = build_headline(mixed)
    assert h is not None
    path = tmp_path / "headline.json"
    write_headline(h, path)
    loaded = Headline.model_validate(json.loads(path.read_text(encoding="utf-8")))
    assert loaded == h
    assert headline_text(h) == path.read_text(encoding="utf-8")


def test_same_report_gives_the_same_text(mixed) -> None:  # type: ignore[no-untyped-def]
    assert headline_text(build_headline(mixed)) == headline_text(build_headline(mixed))


def test_committed_schema_equals_generated() -> None:
    generated = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **Headline.model_json_schema(),
    }
    assert SCHEMA_FILE.read_text(encoding="utf-8") == (
        json.dumps(generated, indent=2, sort_keys=True) + "\n"
    )


def _load_render_script():  # type: ignore[no-untyped-def]
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "render_readme.py"
    spec = importlib.util.spec_from_file_location("render_readme", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_script_writes_checks_and_removes_headline(  # type: ignore[no-untyped-def]
    mixed, tmp_path: Path, monkeypatch
) -> None:
    render = _load_render_script()
    target = tmp_path / "headline.json"
    monkeypatch.setattr(render, "HEADLINE", target)

    h = build_headline(mixed)
    assert render._write_or_check_headline(h, check=True) == 1  # missing file is stale
    assert render._write_or_check_headline(h, check=False) == 0  # writes it
    assert render._write_or_check_headline(h, check=True) == 0  # now current

    target.write_text("{}\n", encoding="utf-8")
    assert render._write_or_check_headline(h, check=True) == 1  # edited file is stale

    # No committed report: a headline file must not exist, and nothing is invented.
    assert render._write_or_check_headline(None, check=True) == 1
    assert render._write_or_check_headline(None, check=False) == 0
    assert not target.exists()
