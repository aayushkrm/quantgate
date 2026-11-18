---
name: quantgate
description: Work on the quantgate repo, a CI gate for quantized GGUF models. Use when adding or changing evals, the gate policy, the variant registry, reports or the CLI, when asked to run the quant ladder, or when asked to check a quant report or read a gate result.
---

# quantgate

## What it does

Runs one fixed suite over each quant of Qwen2.5-0.5B-Instruct and gates it against the fp16
reference: KL drift and top-1 agreement, ARC-Easy and GSM8K accuracy, refusal rate on harmful and
benign prompts, and CPU decode and prefill speed with peak RSS. The output is
`results/report.json` (`schema_version` "1"). `quantgate gate` checks it against
`configs/gate.yaml` and recommends the smallest file that passes.

## When to run what

- Reading or checking an existing report: `quantgate gate` and `quantgate report` only. Both are
  read-only apart from the output directory. Do this freely.
- Running `quantgate eval`, or any code that loads a GGUF file: only on an explicit request from
  the user. Runs are compute-queued, must follow `docs/prereg.md`, and perf runs must go last on an
  idle machine on AC power. Never start one to "check that it works".
- `quantgate demo` is safe at any time: scripted FakeBackend, no downloads, under 30 s.

## Commands

| Command | Purpose | Exit codes |
|---|---|---|
| `quantgate fetch -c configs/variants.yaml [--models-dir ~/models]` | Download GGUFs, verify sha256 and size. Prints the observed sha256 for variants whose registry hash is null. | 0 ok, 2 error |
| `quantgate eval -c configs/variants.yaml --suite kl,arc,gsm8k,refusal,perf --out results/` | Run the suite, one variant at a time. Options: `--only q4_k_m,q2_k` (the reference always runs), `--n-scale 0.5` (shrink every sample count, recorded in `env.json`), `-p configs/gate.yaml` | 0 ok, 2 error |
| `quantgate gate results/report.json -p configs/gate.yaml` | Apply the policy to a report. Also accepts the results directory. | 0 all pass, 1 any fail or hidden danger, 2 error |
| `quantgate report results/ --out-dir site/ [--readme README.md]` | Write `index.html`, `report.md`, `report.json`, Pareto plots (PNG and SVG) and `env.json`. `--readme` splices the block between `<!-- results:start -->` and `<!-- results:end -->`. | 0 ok, 2 error |
| `quantgate demo` | Full pipeline on FakeBackend. Output is labelled SYNTHETIC and writes no files. | 0 ok, 2 on error |

Every command takes `--json`. With `--json`, errors are a JSON object with `ok: false`, `error`
and `exit_code`. Without it, errors are one line on stderr. Tracebacks are never shown for
expected failures.

## Eval layout and resumability

`eval` writes, under `--out`:

- `<variant>/<suite>.json`: metrics for one variant and suite, plus provenance (refusal prompt
  sets, including any fallback source).
- `<variant>/variant.json`: the assembled `VariantResult`.
- `<variant>/raw/`: refusal completions. Gitignored. Never commit it.
- `env.json`: environment record from `scripts/envinfo.py`, plus the run settings (`run.suite`,
  `run.n_scale`, `run.suites`, `run.variants`, `run.policy`).
- `report.json`: the `QuantReport`, with decisions from the policy given by `-p`.

Re-running skips every `<variant>/<suite>.json` that exists. To redo a suite, delete its file.
Re-running after a crash resumes at the first missing file.

## Rules when working here

1. `src/quantgate/schema.py` is frozen. If it needs a change, say so and stop. Do not edit it.
2. Never type a measured number into the README or docs. Numbers come from `quantgate report`,
   which renders committed JSON. `render_readme_block` writes the README block.
3. Pre-register before running. `docs/prereg.md` must be committed before the first model run.
   Every change to n, prompt, threshold or model goes in `docs/deviations.md` with a reason.
   `--n-scale` below 1 is a deviation.
4. Tests must pass with `pytest -m "not model"` and no network. Fixtures in `tests/fixtures/` are
   SYNTHETIC and say so in `env_info`. Do not put real measurements in fixtures.
5. Refusal completions are raw, sensitive text. They stay under a `raw/` directory. Commit only
   flags, hashes and matched patterns.
6. Units: rates and accuracies are percentages (0 to 100). `kl_mean` is nats per token. Drop
   thresholds are percentage points. `file_mb` and `peak_rss_mb` are MiB (2**20 bytes).
7. Scope: latency and memory are CPU-only and describe the recorded machine. Results describe
   Qwen2.5-0.5B-Instruct, not LLMs in general.

## Gate semantics

Implemented in `src/quantgate/gate.py`. Rule text ids look like `kl_mean:abs_max:0.1`.

- `abs_max`: candidate value must be at most the threshold.
- `abs_min`: candidate value must be at least the threshold.
- `drop_max`: reference value minus candidate value must be at most the threshold. A candidate
  above the reference always passes. This is the same for every metric, including
  `refusal_rate`. To gate a rise in `benign_refusal_rate`, use `abs_max` with an absolute threshold.
- A rule whose metric is missing fails closed. It is reported with `[missing]`.
- Hidden danger: every metric in `hidden_danger.quality_metrics` is within `band_pts` of the
  reference (inclusive), and the `refusal_rate` drop is at least `refusal_drop_pts`. A flagged
  variant fails the gate even when no rule fails. With the default policy the refusal rule has the
  same threshold, so a hidden-danger row also appears in `failed_rules`.
- Each violated rule also gets a verdict from its interval (`significance` in the decision): a
  paired bootstrap over per-item correctness for ARC-Easy and GSM8K drops, the Wilson bounds for
  refusal drops, and the metric's 95% CI for abs rules. The verdicts are `significant` and
  `not_distinguishable`. A not-distinguishable violation fails the variant when
  `inconclusive_fails` is true (the default in `configs/gate.yaml`). A significant one always
  fails.
- `passed` means every blocking rule passes and there is no hidden danger.
- Recommendation: the smallest passing variant by file bytes (ties by name). Hidden-danger rows
  and variants with no known size are skipped. The reference is never recommended. If nothing
  passes, the recommendation is none.

## Plots and site

`src/quantgate/plots.py` writes three Pareto plots: file size against ARC-Easy, decode tok/s
against KL, and refusal against file size. Dominated points are labelled "(dominated)". Colours
are Okabe-Ito, and marker shape also differs, so colour is not the only cue. Backgrounds are white
with dark text. SVG text stays as text.

## Where things are

- Gate and recommendation: `src/quantgate/gate.py`
- Refusal audit export and scoring: `src/quantgate/audit.py`
- Headline for the hub: `src/quantgate/headline.py` (schema `schemas/headline.schema.json`)
- Report JSON, Markdown, HTML, README block: `src/quantgate/report.py`
- Plots: `src/quantgate/plots.py`
- CLI and eval orchestration: `src/quantgate/cli.py`
- JSON schema, generated from the model: `schemas/quant_report.schema.json`. A test fails if it
  differs from `QuantReport.model_json_schema()`. Regenerate it only after a schema change is
  approved.
- Default policy: `configs/gate.yaml`. Variant registry and the `suite:` section:
  `configs/variants.yaml`.
- GitHub Action: `action.yml` (composite, runs `quantgate gate` on a committed report)

## Checks before you finish

```bash
.venv/bin/ruff check <files>
.venv/bin/ruff format --check <files>
.venv/bin/mypy src/quantgate/<files>
.venv/bin/python -m pytest tests -q -m "not model"
```

Use `make lint`, `make type` and `make test` for a whole-repo check. Do not run `uv sync` in a
shared checkout.
