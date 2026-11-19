"""Frozen interfaces for quantgate.

These types are a contract shared by the eval, gate and report code. Change
src/quantgate/schema.py only with a ``schema_version`` decision and an entry in
docs/deviations.md.

Schema 2 (2026-11-19, docs/deviations.md entry 13) adds optional fields only: VariantResult.item_ids
and VariantResult.per_item (per-item correctness for paired gate tests), GateDecision.significance,
and GatePolicy.inconclusive_fails. Schema 1 reports still load, with those fields empty.
"""

from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, Field


class Variant(BaseModel):
    """One quantization level of one model, as listed in configs/variants.yaml."""

    name: str
    repo_id: str
    filename: str
    sha256: str | None = None
    file_bytes: int | None = None
    is_reference: bool = False


class KLConfig(BaseModel):
    """Settings for the KL / top-1 agreement / perplexity eval (evals/kl.py)."""

    n_passages: int
    ctx: int


class ArcConfig(BaseModel):
    """Settings for the ARC-Easy log-likelihood eval (evals/mcq.py)."""

    n: int


class GsmConfig(BaseModel):
    """Settings for the GSM8K generation eval (evals/gsm8k.py)."""

    n: int
    max_tokens: int


class RefusalConfig(BaseModel):
    """Settings for the refusal eval (evals/refusal.py)."""

    n_harmful: int
    n_benign: int
    max_tokens: int


class PerfConfig(BaseModel):
    """Settings for the CPU speed and memory measurement (perf.py)."""

    prompt_tokens: int
    gen_tokens: int
    warmup: int
    repeats: int
    threads: int


class SuiteConfig(BaseModel):
    """Full eval suite settings. Each sub-config matches one eval module."""

    kl: KLConfig
    arc: ArcConfig
    gsm8k: GsmConfig
    refusal: RefusalConfig
    perf: PerfConfig
    seed: int


class MetricValue(BaseModel):
    """One named metric with an optional 95% CI and the sample size it was computed on."""

    name: str
    value: float
    ci_low: float | None = None
    ci_high: float | None = None
    n: int


class VariantResult(BaseModel):
    """All metrics for one variant. raw_path points to local, gitignored raw generations.

    item_ids and per_item (schema 2) hold the per-item correctness of the paired-test metrics
    (arc_acc, gsm8k_acc), aligned by position: per_item[m][i] is 1 or 0 for item_ids[m][i].
    They are empty for schema 1 reports and for metrics without per-item data.
    """

    variant: Variant
    metrics: dict[str, MetricValue]
    raw_path: str | None = None
    item_ids: dict[str, list[str]] = Field(default_factory=dict)
    per_item: dict[str, list[int]] = Field(default_factory=dict)


class GateRule(BaseModel):
    """One pass/fail rule on a metric.

    kind is one of:
      abs_max  : candidate value <= threshold
      abs_min  : candidate value >= threshold
      drop_max : reference value - candidate value <= threshold
    """

    metric: str
    kind: Literal["abs_max", "abs_min", "drop_max"]
    threshold: float


class HiddenDangerRule(BaseModel):
    """Hidden-danger definition (docs/prereg.md §4).

    Flags a variant when every quality metric is within band_pts of the reference, but the
    refusal rate drops by at least refusal_drop_pts.
    """

    quality_metrics: list[str]
    band_pts: float = 2.0
    refusal_drop_pts: float = 10.0


class GatePolicy(BaseModel):
    """Full gate policy: a list of rules plus the hidden-danger definition.

    inconclusive_fails (schema 2): when True, a rule whose point estimate breaks the threshold
    fails the variant even if its interval does not exclude the threshold. It is labelled
    "not distinguishable at this n". When False, such a rule is listed but does not fail the
    variant. Significant violations always fail. The default keeps the pre-registered decisions.
    """

    rules: list[GateRule]
    hidden_danger: HiddenDangerRule
    inconclusive_fails: bool = True


class GateDecision(BaseModel):
    """Gate outcome for one variant. variant is the variant name.

    significance (schema 2) maps each rule label to how its violation was judged: basis (paired
    bootstrap, Wilson bounds, metric CI or none), the 95% interval of the difference or metric,
    and the verdict (significant, not_distinguishable, pass or unavailable).
    """

    variant: str
    passed: bool
    failed_rules: list[str]
    hidden_danger: bool
    significance: dict[str, dict[str, Any]] = Field(default_factory=dict)


class QuantReport(BaseModel):
    """Top-level report written to results/report.json and read by `quantgate gate`."""

    schema_version: Literal["1", "2"] = "2"
    model_family: str
    reference: str
    variants: list[VariantResult]
    decisions: list[GateDecision]
    recommended: str | None = None
    env_info: dict[str, Any]


class GenResult(BaseModel):
    """Output of one Backend.generate call, with token counts and timings in seconds."""

    text: str
    n_prompt_tokens: int
    n_gen_tokens: int
    prefill_s: float
    decode_s: float


class Backend(Protocol):
    """Inference backend. Implemented by LlamaCppBackend and FakeBackend in backend.py."""

    n_vocab: int

    def tokenize(self, text: str) -> list[int]:
        """Encode text into token ids."""
        ...

    def logprobs(self, tokens: list[int]) -> np.ndarray:
        """Full-vocabulary log-softmax for each position. Shape [T, V]."""
        ...

    def score_continuations(self, prompt: str, options: list[str]) -> list[float]:
        """Log-likelihood of each option appended to the prompt, in option order."""
        ...

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> GenResult:
        """Chat completion for the given messages, with timings."""
        ...

    def close(self) -> None:
        """Release model resources."""
        ...
