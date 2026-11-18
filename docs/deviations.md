# Deviations

Every change to n, prompt, threshold, model or procedure goes here, with a reason and a date.
Entries are in date order. Changes made before the initial experiment are listed too, so that the
reader can see what the pre-registration in `docs/prereg.md` was written against.

## Log

### 2026-11-19: before any model run

1. **Pre-registration committed before any run.** `docs/prereg.md` (commit `e9605b9`) was committed
   before any model run, including the smoke runs below. Its token counts come from a
   vocab-only tokenizer load and are facts about the inputs, not results.

2. **Quality backend context: 2048 to 512, with n_batch = n_ctx.** `LlamaCppBackend` with
   `logits_all=True` reserved about 1.2 GB for an `n_ctx=2048` logits buffer. The KL passages are
   256 tokens and the GSM8K prompt plus 256 generated tokens stays under 512 (checked in
   `docs/prereg.md` §1). `n_batch` equals `n_ctx`, so every input is one batch and the batch shape
   depends only on the input length. Reference and candidate get identical input lengths, enforced
   by the tokenizer check in `run_kl`. This changes memory only; no metric definition changed.

3. **Perf backend: `logits_all=False`, `n_ctx=1024`.** An earlier version of the perf child used
   `logits_all=True`. That computes the vocabulary projection for every prefill token and would
   distort prefill tok/s. Perf measures ordinary generation, so it now uses `logits_all=False`.
   Made before any perf run.

4. **The perf backend seam is testing-only.** `QUANTGATE_PERF_BACKEND` is honoured only with
   `QUANTGATE_TESTING=1`. Otherwise it is ignored, with a note on stderr.

5. **Peak RSS has one contract.** The perf child emits `peak_rss_bytes` at the top level of its
   JSON. The parent adds it to the metrics and raises if it is absent. Before this, the key was
   present only because the child also placed it in `metrics`.

6. **Sidecar lock file.** `quantgate fetch` writes `configs/variants.lock.json` (sha256, size,
   repo_id, filename; no local paths). The registry YAML keeps its null pins and its comments.
   `eval` verifies against the lock. A check that cannot run is recorded in `env.json`
   under `run.unverified`.

7. **Hidden-danger verdict made explicit.** A variant flagged hidden danger fails the gate even
   when every rule passes. Gate labels now say which of the two causes applies. This changes the
   rendering only, not the decision logic in `src/quantgate/gate.py`.

8. **Packaging.** The environment collector moved into `quantgate.envinfo`, and
   `scripts/envinfo.py` is a thin wrapper. The default gate policy, the variant registry and the
   subsets are bundled into the wheel (`quantgate/_data/`), so `quantgate demo` runs from any
   directory. The file contents are unchanged.

9. **Smoke runs are not experiments.** Before E1, the pipeline is run on two variants with
   `--n-scale 0.03`. This is below 1 and so is a reduction. The outputs go to the scratchpad, are
   not in `results/`, and are not committed. They are used only to check that the pipeline works.
   Their numbers inform no threshold or n.

### Open items

- **E5 (KV-cache, 2K context):** needs concatenated WikiText paragraphs. The loader does not
  produce them yet. E5 is dropped unless that is implemented, and the drop is logged here.
- **Harmful prompts:** the JBB harmful split has exactly 100 items, so the "100 sample" is the
  whole split. This is a fact of the dataset, not a deviation.
