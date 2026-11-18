# Design

This file records the decisions behind the code and the reasons for them. The plan is in
`DECISION.md` §2.2. Experiments are pre-registered in `docs/prereg.md`.

## Decisions

- **One model family.** Qwen2.5-0.5B-Instruct, the official GGUF ladder from one publisher
  (Apache-2.0). Reference is fp16. The ladder is fixed in `configs/variants.yaml`.
- **Inference with llama-cpp-python 0.3.37**, CPU only, `n_gpu_layers=0`, `n_threads=4`. An earlier
  spot check by the orchestrator on this 4-core laptop found 4 threads faster than 8 for decode.
  That check is not in `results/` and is not a measurement reported here.
- **KL over the full vocabulary.** Per position, KL(P_ref || P_cand) in nats, pooled over all
  positions. Top-1 agreement and perplexity are companions. Each has a percentile bootstrap CI
  over passages. Reference and candidate are computed on the same token IDs.
- **Quality backend: `n_ctx=512`, `n_batch=512`, `logits_all=True`.** The logits buffer grows
  as `n_ctx * n_vocab * 4` bytes. At 2048 it was about 1.2 GB. Each passage is one batch, so
  the batch shape depends only on its length.
- **Perf backend: `logits_all=False`, `n_ctx=1024`.** Perf measures ordinary generation. A full
  vocabulary projection on every prefill token would distort prefill tok/s.
- **One subprocess per perf measurement, run last.** Memory and warm-up are not shared between
  variants. Peak RSS comes from `resource.getrusage` inside the child. Power and thermal state are
  recorded before and after.
- **Refusal classifier is a published regex list.** The patterns are in
  `src/quantgate/evals/refusal.py`. Completions are written only to a gitignored `raw/`
  directory, and only flags, hashes and the matched pattern are committed. The classifier has
  not been validated by a human. Hand validation uses `quantgate audit export` and `audit score`,
  described in `docs/limitations.md`. Known false positives and negatives are listed in the module
  docstring.
- **Wilson 95% intervals** for every rate (`quantgate.evals.wilson_ci`, shared by all evals).
- **Gate operates on point estimates.** A rule is a threshold on a metric, or a drop from the
  reference. Hidden danger is a separate flag that fails the gate even when every rule passes.
  The rules are not significance tests.
- **Lock file for hashes.** `quantgate fetch` writes `configs/variants.lock.json` with the
  observed sha256 and size. The registry YAML keeps its null pins and its comments. `eval`
  verifies against the lock and records any check it could not run in `env.json`.
- **Results are generated.** The README results block and the Pages site are rendered from
  `results/report.json`. `scripts/render_readme.py --check` fails CI when the README is stale.
- **Packaging.** The default policy, the registry and the subsets are bundled into the wheel
  (`quantgate/_data/`), so the installed `quantgate demo` runs from any directory. In a checkout
  they are read in place.
- **Test seams.** `QUANTGATE_PERF_BACKEND` swaps the perf backend only when `QUANTGATE_TESTING=1`.
  Tests use `FakeBackend` and never need a GGUF file or the network.

## Measurement and environment

- Every result carries `env.json` from `quantgate.envinfo`: git SHA, dirty flag, UTC time,
  CPU model and flags, cores, RAM, OS, Python and package versions, and the run settings. No host
  name or user name is recorded.
- Latency and memory numbers describe the recorded machine and nothing else.

## Where things live

- `src/quantgate/schema.py`: frozen interfaces. Changes need approval and a `schema_version` bump.
- `src/quantgate/cli.py`: the typer app. Every command takes `--json`.
- `results/`: committed JSON only. Raw completions are gitignored.
- `data/subsets/`: committed item IDs. Text is loaded at run time from the Hub.
