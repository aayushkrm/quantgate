---
name: quantgate
description: Work on the quantgate repo, a CI gate for quantized GGUF models. Use when adding or changing evals, the gate policy, the variant registry, reports or the CLI, or when asked to run the quant ladder or check a quant report.
---

# quantgate

## What it does

Runs one fixed suite over each quant of a model and gates the result against the fp16 reference:
KL drift and top-1 agreement, ARC-Easy accuracy, GSM8K accuracy, refusal rate on harmful and benign
prompts, and CPU decode and prefill speed with peak RSS. Output is `results/report.json`
(`schema_version` "1"), which `quantgate gate` checks against `configs/gate.yaml`.

## Status

Scaffold. The CLI commands and most modules are stubs that raise `NotImplementedError`.
Check `src/quantgate/` before assuming that a function works.

## Commands (planned)

| Command | Purpose | Exit codes |
|---|---|---|
| `quantgate fetch -c configs/variants.yaml` | Download GGUFs to `~/models`, verify sha256 | 0 ok, 2 error |
| `quantgate eval -c configs/variants.yaml --suite kl,arc,gsm8k,refusal,perf --out results/` | Run the suite | 0 ok, 2 error |
| `quantgate gate results/report.json --policy configs/gate.yaml` | Apply the policy | 0 pass, 1 check failed, 2 error |
| `quantgate report results/` | Render Markdown, plots and site | 0 ok, 2 error |
| `quantgate demo` | Scripted FakeBackend run, labelled SYNTHETIC | 0 ok |

Every command takes `--json`.

## Rules when working here

1. The schema in `src/quantgate/schema.py` is frozen. Request changes, do not make them.
2. Never type a measured number into the README or docs. Use `make report`.
3. Model runs follow `docs/prereg.md`. Commit that file before the first run. Log every deviation in
   `docs/deviations.md`.
4. Perf runs go last, on an idle machine on AC power. Do not run two heavy jobs at once.
5. Tests must pass with `uv run pytest -m "not model"` and no network access.
6. Refusal completions are raw, sensitive text. Keep them in gitignored raw directories. Commit only
   flags, hashes and matched patterns.

## Gate semantics

- `abs_max`: candidate value must be at most the threshold.
- `abs_min`: candidate value must be at least the threshold.
- `drop_max`: reference value minus candidate value must be at most the threshold.
- Hidden danger: every quality metric in `hidden_danger.quality_metrics` is within `band_pts` of the
  reference, and the refusal rate drops by at least `refusal_drop_pts`.
- Recommendation: the smallest file (by bytes) that passes every rule, or none.
