"""Schema round-trip and protocol checks for the frozen interfaces (src/quantgate/schema.py)."""

import typing

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from quantgate import schema
from quantgate.schema import (
    ArcConfig,
    Backend,
    GateDecision,
    GatePolicy,
    GateRule,
    GenResult,
    GsmConfig,
    HiddenDangerRule,
    KLConfig,
    MetricValue,
    PerfConfig,
    QuantReport,
    RefusalConfig,
    SuiteConfig,
    Variant,
    VariantResult,
)


def _variant(name: str = "q4_k_m", is_reference: bool = False) -> Variant:
    return Variant(
        name=name,
        repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename=f"qwen2.5-0.5b-instruct-{name}.gguf",
        sha256=None,
        file_bytes=None,
        is_reference=is_reference,
    )


def _suite() -> SuiteConfig:
    return SuiteConfig(
        kl=KLConfig(n_passages=50, ctx=256),
        arc=ArcConfig(n=300),
        gsm8k=GsmConfig(n=100, max_tokens=256),
        refusal=RefusalConfig(n_harmful=100, n_benign=50, max_tokens=64),
        perf=PerfConfig(prompt_tokens=512, gen_tokens=128, warmup=2, repeats=5, threads=4),
        seed=0,
    )


def _policy() -> GatePolicy:
    return GatePolicy(
        rules=[
            GateRule(metric="kl_mean", kind="abs_max", threshold=0.10),
            GateRule(metric="top1_agree", kind="abs_min", threshold=90.0),
            GateRule(metric="arc_acc", kind="drop_max", threshold=2.0),
            GateRule(metric="refusal_rate", kind="drop_max", threshold=10.0),
        ],
        hidden_danger=HiddenDangerRule(quality_metrics=["arc_acc", "gsm8k_acc"]),
    )


def _report() -> QuantReport:
    ref = _variant("fp16", is_reference=True)
    cand = _variant("q4_k_m")
    metric = MetricValue(name="arc_acc", value=60.0, ci_low=55.0, ci_high=65.0, n=300)
    return QuantReport(
        model_family="qwen2.5-0.5b-instruct",
        reference="fp16",
        variants=[
            VariantResult(variant=ref, metrics={"arc_acc": metric}, raw_path=None),
            VariantResult(variant=cand, metrics={"arc_acc": metric}, raw_path="raw/q4.jsonl"),
        ],
        decisions=[
            GateDecision(variant="q4_k_m", passed=True, failed_rules=[], hidden_danger=False)
        ],
        recommended="q4_k_m",
        env_info={"git_sha": "0" * 40, "cpu_flags": {"avx2": True, "avx512": False}},
    )


def _gen_result() -> GenResult:
    return GenResult(
        text="42",
        n_prompt_tokens=12,
        n_gen_tokens=3,
        prefill_s=0.25,
        decode_s=0.5,
    )


SAMPLES: dict[type[BaseModel], BaseModel] = {
    schema.Variant: _variant(),
    schema.KLConfig: KLConfig(n_passages=50, ctx=256),
    schema.ArcConfig: ArcConfig(n=300),
    schema.GsmConfig: GsmConfig(n=100, max_tokens=256),
    schema.RefusalConfig: RefusalConfig(n_harmful=100, n_benign=50, max_tokens=64),
    schema.PerfConfig: PerfConfig(
        prompt_tokens=512, gen_tokens=128, warmup=2, repeats=5, threads=4
    ),
    schema.SuiteConfig: _suite(),
    schema.MetricValue: MetricValue(name="kl_mean", value=0.02, ci_low=0.01, ci_high=0.03, n=50),
    schema.VariantResult: VariantResult(variant=_variant(), metrics={}, raw_path=None),
    schema.GateRule: GateRule(metric="kl_mean", kind="abs_max", threshold=0.10),
    schema.HiddenDangerRule: HiddenDangerRule(quality_metrics=["arc_acc"]),
    schema.GatePolicy: _policy(),
    schema.GateDecision: GateDecision(
        variant="q2_k", passed=False, failed_rules=["kl_mean"], hidden_danger=True
    ),
    schema.QuantReport: _report(),
    schema.GenResult: _gen_result(),
}


def test_every_public_model_is_covered_by_a_sample() -> None:
    """Guard: a new BaseModel added to schema.py must get a round-trip sample here."""
    declared = {
        obj
        for obj in vars(schema).values()
        if isinstance(obj, type)
        and issubclass(obj, BaseModel)
        and obj.__module__ == schema.__name__
    }
    assert declared == set(SAMPLES)


@pytest.mark.parametrize("model_type", list(SAMPLES), ids=lambda t: t.__name__)
def test_json_round_trip(model_type: type[BaseModel]) -> None:
    original = SAMPLES[model_type]
    restored = model_type.model_validate_json(original.model_dump_json())
    assert restored == original
    assert type(restored) is model_type


def test_quant_report_schema_version_is_two() -> None:
    assert _report().schema_version == "2"
    assert '"schema_version":"2"' in _report().model_dump_json()


def test_quant_report_rejects_other_schema_versions() -> None:
    data = _report().model_dump(mode="json")
    data["schema_version"] = "9"
    with pytest.raises(ValidationError):
        QuantReport.model_validate(data)


def test_gate_rule_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        GateRule.model_validate({"metric": "kl_mean", "kind": "abs_between", "threshold": 1.0})


def test_hidden_danger_defaults_match_plan() -> None:
    rule = HiddenDangerRule(quality_metrics=["arc_acc"])
    assert rule.band_pts == 2.0
    assert rule.refusal_drop_pts == 10.0


def test_backend_is_a_typing_protocol() -> None:
    assert typing.Protocol in Backend.__bases__
    assert getattr(Backend, "_is_protocol", False) is True
    for method in ("tokenize", "logprobs", "score_continuations", "generate", "close"):
        assert callable(getattr(Backend, method))


def test_logprobs_annotation_is_ndarray() -> None:
    hints = typing.get_type_hints(Backend.logprobs)
    assert hints["return"] is np.ndarray
