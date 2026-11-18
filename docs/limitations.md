# Limitations

Written before results exist. Each item stays in the final report, including any null result.

## What this does not show

- **One model family.** Results describe Qwen2.5-0.5B-Instruct. They do not describe LLMs in
  general, other sizes, or other families. A 1.5B arm is a stretch goal and is not pre-registered.
- **One hardware target.** Latency and memory were measured, if at all, on an Intel i7-4770HQ
  laptop (Haswell, 4 cores, 8 threads, 16 GB), macOS x86_64, CPU only. They make no claim about
  other machines. CI runs on Linux and checks no speed numbers.
- **Power.** The gate rules use point estimates. At n=100 the refusal interval is about ±10 points,
  so only large drops in refusal can be distinguished from noise. At n=300 the ARC interval near
  60% is about ±5.5 points, so a 2-point ARC drop is inside the interval. A pass does not show that
  a quant is safe. It shows that no large effect was seen in this sample.
- **Refusal classifier.** It is a regex list (10 patterns, see `src/quantgate/evals/refusal.py`).
  It has known false positives (capability disclaimers, some ethics statements in benign text) and
  false negatives (soft deflections, refusals without the listed verbs, refusals after the opening
  64 tokens). It has not been checked against human labels. `audit/refusal_audit_template.csv`
  is the template for that check, and it is empty.
- **Benign set.** Fifty instruction-only Alpaca prompts. Over-refusal measured on them is a
  sample, not a rate for any population of users.
- **Harmful set.** The JailbreakBench harmful split (100 behaviours) is the whole split. The
  AdvBench fallback is used only if JailbreakBench cannot be loaded. The run records which source
  was used.
- **Context.** Passages are 256 tokens and generations are at most 256 tokens. Long-context
  behaviour under quantization is not studied.
- **Quantization method.** The GGUF files come from the publisher. Some rungs may be
  self-quantized from fp16 if the publisher lacks them. Those are labelled "self-quantized, no
  imatrix". No imatrix sweep is run.
- **Reference.** Everything is measured against fp16. The reference's own accuracy is not
  compared with any published figure.
- **Literature values.** The hidden-danger pattern is reported by arXiv:2606.10154. Values from
  that paper are not reproduced here. Any such value is cited as reported.
- **No GPU, no OpenVINO or ONNX backend.** The Haswell laptop has no macOS x86 wheels for those
  stacks, so this study uses llama.cpp on CPU only.
- **Pre-registration deviations.** Changes made after the pre-registration are listed in
  `docs/deviations.md`. A deviation can affect how the numbers should be read.
