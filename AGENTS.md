# AGENTS.md

Guide for AI coding agents working in this repository. `CLAUDE.md` imports this file.

## What this repo is

`quantgate` is a CI gate for quantized GGUF models. It runs a fixed eval suite (KL drift, ARC-Easy,
GSM8K, refusal, CPU speed) over the Qwen2.5-0.5B-Instruct quant ladder and fails when a quant is
worse than the fp16 reference under the policy in `configs/gate.yaml`.

Status: the eval pipeline, gate, report and CLI are implemented and tested. The pre-registered
quant ladder has been run, and its results are committed in `results/`. Synthetic and demo output
is labelled SYNTHETIC and is not a measurement.

## Setup and commands

```bash
uv venv -p 3.11 .venv
uv sync --extra dev          # add --extra llama only on a machine that can build llama-cpp-python
make lint                    # ruff check + ruff format --check
make type                    # mypy on src/
make test                    # pytest -m "not model" (no model files, no network)
make fmt                     # ruff format + ruff check --fix
```

Other targets: `make demo`, `make report`, `make readme`, `make check-results`, `make site` and
`make release`. The targets run through `uv run` with `UV_NO_SYNC=1`, so uv does not resync the
venv and remove llama-cpp-python.

## Layout

```
src/quantgate/
  schema.py        FROZEN pydantic models and the Backend protocol (see invariants)
  backend.py       LlamaCppBackend (llama-cpp-python) and FakeBackend (tests)
  models.py        variant registry, fetch, sha256 verification
  data.py          fixed subsets (item IDs only, committed under data/subsets/)
  evals/           kl.py, mcq.py (ARC-Easy), gsm8k.py, refusal.py
  perf.py          CPU speed and RSS, one fresh subprocess per variant
  gate.py          rules, hidden-danger flag, recommendation
  report.py        QuantReport JSON and Markdown rendering
  plots.py         Pareto plots
  cli.py           typer app: fetch, eval, gate, report, demo
configs/           variants.yaml (model ladder), gate.yaml (default policy)
data/subsets/      committed item IDs only
results/           committed JSON results and env.json (never hand-edited)
docs/              design, prereg, deviations, limitations
audit/             labelled refusal samples (LLM labels) and the blank human audit template
scripts/envinfo.py thin wrapper; the environment record lives in quantgate/envinfo.py
tests/             pytest; model tests carry @pytest.mark.model
```

## Invariants

1. **`src/quantgate/schema.py` is frozen.** Change it only with a `schema_version` decision and an
   entry in `docs/deviations.md`.
2. **Results are generated, never typed.** Numbers in README and docs come only from `make report`
   rendering committed JSON in `results/`. Never write a measured number by hand.
3. **Pre-register before running.** `docs/prereg.md` is committed before any model run. Every
   change to n, prompt, threshold or model goes in `docs/deviations.md` with a reason.
4. **Raw generations stay local.** Refusal completions and per-token dumps go to gitignored
   directories. Only flags, hashes and matched patterns are committed.
5. **No model weights in the repo.** Weights live in `~/models`. The cache lives in
   `~/.cache/quantgate`, or in `$QUANTGATE_CACHE_DIR` when that is set. Tests run with both `HOME`
   and the cache redirected to a temporary directory (`tests/conftest.py`). Do not write a test
   that uses the real cache, and do not remove that fixture (`docs/deviations.md`, entry 25).
6. **Tests do not need the network or models.** Mark anything that needs a GGUF file with
   `@pytest.mark.model`. CI runs `-m "not model"`.
7. **Scope labels.** Latency and memory numbers are CPU-only and describe the measured machine.
   Results describe Qwen2.5-0.5B-Instruct, not LLMs in general. Synthetic output is labelled SYNTHETIC.
8. **Banned words** in docs and READMEs unless a measured result supports them: "production-ready",
   "state-of-the-art", "SOTA", "blazing", "first", "novel", "robust", "enterprise".

## Conventions

- Python 3.11+, `src/` layout, ruff line length 100, mypy with `disallow_untyped_defs`.
- Every CLI command takes `--json`. Exit codes: 0 pass, 1 check failed, 2 usage or runtime error.
- Commits: one logical change each.
- No hostnames or usernames in any committed output.
