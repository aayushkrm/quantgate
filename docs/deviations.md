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

10. **Benign control changed from Alpaca to Dolly-15k (licence).** `tatsu-lab/alpaca` is
   CC-BY-NC-4.0, which is non-commercial. Its card and metadata were checked on 2026-11-19. The
   owner's decision was to replace it. The replacement is `databricks/databricks-dolly-15k`,
   CC-BY-SA-3.0, checked against the card and metadata. The benign set is the seeded sample of
   50 rows (seed 0) with an empty `context` and a non-blank `instruction`. There are 10,544
   eligible rows. Ids are `dolly-<index>`. Made after the pre-registration commit and before any
   result file exists. `docs/prereg.md` §1 is revised to match. The smoke run that was already in
   progress used the Alpaca set (2 benign prompts at `--n-scale 0.03`). It is a smoke run, not an
   experiment, its output is not committed, and no full run uses Alpaca.

11. **Refusal classifier: patterns 7 and 8 removed (validity).** The appropriateness pattern
    ("This is not appropriate") and the ethics predicate ("is illegal", "is unethical") flagged
    compliant answers that carried a disclaimer. A quantized model that complies with a warning
    would then count as refusing, and a real drop in refusal could be hidden. Both were removed.
    The classifier has 8 patterns. Tests cover the former false positives. This is a change to the
    pre-registered classifier, made before any result file exists, and `docs/prereg.md` §2 is
    updated to match. Validation against hand-labelled completions is still pending, and the
    README says so.

12. **Perf procedure: full-length decode, n-1 gaps, prompt token count (measurement).** The
    perf smoke on q4_k_m (2026-11-19, AC power) stopped at 27 of the 128 requested tokens, so
    the decode rate rested on a short generation. Perf now uses `generate_fixed_length`, which
    suppresses the end-of-sequence tokens with a logit bias of -100. The perf output records
    `gen_fixed_length`. Decode tok/s is computed over the n-1 gaps between the initial and the final
    token, not over n tokens (review Q7). The 512-token budget in `docs/prereg.md` §3 applies to
    the text before the chat template. The templated prompt measured 541 tokens (p50). The
    prefill rate uses the counted tokens, so it stays correct. The smoke numbers are not results,
    and no full run has started with this procedure.

13. **Schema 2 and CI-aware gate labels (review Q2).** `schemas/`, `src/quantgate/schema.py`:
    additive optional fields only. `VariantResult.item_ids` and `VariantResult.per_item` (per-item
    correctness for ARC and GSM8K), `GateDecision.significance`, and `GatePolicy.inconclusive_fails`
    (default true). The report schema version is now "2". Schema 1 reports still load. The change
    is needed because the gate cannot judge a drop without per-item data. Pre-registered decisions
    do not change under the default. Labels and the significance record are new. The judgement
    rules are in `docs/prereg.md` §4. The point rules still decide `failed_rules`.

14. **Guards on resume, power and hidden danger (review Q3, Q4, Q5).** A per-suite result carries a
    fingerprint: method version, quantgate version, scaled suite settings, seed, n-scale, context,
    the model's sha256 and size, and a digest of the subsets. A mismatch is refused, and the file is
    not overwritten. env.json refuses a different suite, n-scale or policy in the same directory,
    and it keeps a run history. Perf results record pmset before and after. On macOS a variant on
    battery or with a thermal or performance warning writes no result, as the rerun rule requires.
    The hidden-danger check fails closed when a metric it needs is missing. These guards change
    only what the tool refuses to do. No measured quantity changes.

15. **GSM8K budget 256 to 384, and generation backends split from the KL backend (decided by
    the orchestrator, before any result file).** The initial perf and GSM8K smoke showed that all
    three sampled completions reached 256 tokens and were cut off, so the last-number extraction
    scored truncated text. The budget is raised to 384 in `configs/variants.yaml` and
    `docs/prereg.md` §2. The generation suites (gsm8k, refusal, perf) load the backend with
    `logits_all=False` and `n_ctx=1024`, which keeps memory small. The KL backend keeps
    `logits_all=True` with `n_ctx=512` (passage 256 plus margin). ARC is the exception: its scoring
    (`score_continuations`) needs logits, so ARC runs on the KL backend. Its prompts are at most
    132 tokens, so they fit. The suite files record a count of truncated GSM8K generations.
    The method version moves to 3 (fingerprints change), and the smoke check was 1 variant at
    0.03 scale: truncated GSM8K generations fell from 3 of 3 at 256 to 1 of 3 at 384, and decode
    p50 stayed within run-to-run noise (27.1 to 28.7 tok/s). Accuracy on that sample is still 0 of 3,
    so the budget change does not by itself fix GSM8K accuracy.

16. **Report aggregation bug, found after the perf run (fixed, then regenerated).** A perf-only
    `eval` invocation rebuilt `results/report.json` from the perf files alone. Every quality metric
    then read as missing, and the gate failed closed with no recommendation. The run history in
    `results/env.json` shows the two invocations. The merge now reads every suite file on disk,
    whatever the invocation ran (commit `312137c`, with regression tests for both orders).
    `results/report.json` was regenerated from the per-variant files with
    `quantgate report results --rebuild`. That step ran no model. It is a regeneration, not a
    re-run: the per-variant files were merged again. A later review recomputed the aggregates and
    the gate decisions from those files and found them consistent with the report. The env block of
    the regenerated report carries the SHA of the latest invocation, and the quality invocation's SHA
    (`b799df3`) stays in `run_history`.

17. **Refusal audit covers three variants, not nine.** The AI-assisted labelling sampled 40 prompts
    from each of fp16, q4_k_m and q2_k (120 rows). It was not run on the other six rungs. The refusal
    rates for those six have no label-based validation. The 120 rows are 40 distinct prompts, not
    120 independent items. The validation block states this. The decision rule was written for
    items, and it is reported on both readings (`docs/limitations.md`).

18. **Refusal validation was done by an AI assistant, not a human.** The orchestrator labelled the
    sample, blind to the regex flag by its own record. This is not a human audit. A human audit
    needs `audit/refusal_audit_template.csv` to be filled in by the author. The README and
    limitations say which one was done.

19. **Scorer aligned with the committed labels file (tool fix, same numbers).** `quantgate audit
    score` now reads the `regex_flagged` and `label` columns that the labels file uses, and it
    reports distinct prompts and labelled rows per variant. Before this, the committed score JSON
    could not be reproduced by the scorer on the committed file. It was regenerated with the fixed
    tool. The counts match the orchestrator's figures: 67 true positives, 1 false negative, 51 true
    negatives, no false positives, 1 UNCLEAR.

20. **Bandwidth ratios above 1.0 were flagged, not repeated.** The E4 check (decode tok/s times file
    size, against the measured copy or scale bandwidth) gives a ratio above 1.0 for fp16, q8_0 and
    q6_k. The protocol (DECISION section 6) says such a run is repeated before it is reported. The
    coordinator asked for no re-runs, so the result is reported with the ratios and flagged. The
    bandwidth figure is a single measurement (5 repeats, best kept).

21. **Perf settings as actually run.** Every variant ran alone on AC power (`env.json`
    `power_at_start` shows AC). Each run used 2 warm-up runs and 5 repeats, 4 threads, a 512-token
    text budget that counts 541 tokens with the chat template, and 128 generated tokens with the
    end-of-sequence tokens suppressed (`gen_fixed_length`, entry 12). Brackets in the speed table are
    min and max over the five repeats, not confidence intervals.

22. **The quality ladder ran with the GSM8K budget of 384 and the generation backend split (entry 15).**
    The suite files record `max_tokens` 384 in their diagnostics. Each candidate's own GSM8K file
    counts its truncated generations.

23. **Provenance of the committed results (disclosed, not repaired).** The quality invocation ran at
    commit `b799df3` and the perf invocation at commit `9e8be88`. Both are in `run_history` in
    `results/env.json`. The record keeps a dirty flag for the latest invocation only, and that flag
    is `true` for `9e8be88`, so the perf results came from a working tree with uncommitted changes.
    The quality invocation's dirty state is not recorded, and it is not claimed here. The
    environment record has no diff-digest field, so no digest of the uncommitted changes exists for
    these runs. Commits after `b799df3` changed the report aggregation and the audit scorer, as
    entries 16 and 19 record. The README's provenance section states this. A re-run would give
    comparable numbers, but it would not reproduce the same working tree.

24. **Documentation and renderer corrections found in pre-publication review.** The README and the
    laptop block labelled decode min to max over repeats as a 95% interval. They now say
    "min to max over 5 repeats", and the headline's `ci` field carries the same text, with its
    schema description saying the field is not always an interval. The gate header in
    `configs/gate.yaml` said the thresholds were not pre-registered. Its comment now refers to
    `docs/prereg.md`, and no value changed. The README's hidden-danger citation now gives the
    paper's title and author. The README no longer says the paper excluded sub-1B models, which was
    not checked. The README says which numbers cannot be reproduced from committed files. The
    refusal "blind" statement is marked as a self-report. The bandwidth note says that copy and
    scale count read and write traffic, while a decode pass is read-mostly. The ratio above 1.0 for
    fp16, q8_0 and q6_k is flagged and not explained.

### Open items

- **E5 (KV-cache, 2K context):** needs concatenated WikiText paragraphs. The loader does not
  produce them yet. E5 is dropped unless that is implemented, and the drop is logged here.
- **WikiText licence discrepancy (not a deviation).** The dataset metadata lists CC-BY-SA-3.0 and
  GFDL. The dataset card text says CC-BY-SA-4.0. The card text is cited in the README and the
  discrepancy is flagged in `docs/limitations.md`.
- **Harmful prompts:** the JBB harmful split has exactly 100 items, so the "100 sample" is the
  whole split. This is a fact of the dataset, not a deviation.
