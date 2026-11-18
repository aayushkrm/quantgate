# quantgate

*A CI gate for quantized GGUF models: it measures KL drift, task accuracy, refusal behaviour and CPU speed for every quant level, and fails the build when a quant is quietly worse.*

> **Status: scaffold.** The package, schema and CLI are in place as stubs. No experiment has run and there are no results yet.

## Results

<!-- results:start -->
_No results yet. This block is rendered from `results/` by `make report` once the pre-registered experiments have run._
<!-- results:end -->

## Quickstart

TODO: `quantgate demo` (runs on a scripted fake backend, output labelled SYNTHETIC). Not implemented yet.

## Install

```bash
uv tool install git+https://github.com/aayushkrm/quantgate
```

TODO: attach a wheel to a GitHub Release. The llama.cpp backend is the optional `llama` extra (`llama-cpp-python==0.3.37`).

## What it measures

TODO: KL drift against the fp16 reference, ARC-Easy and GSM8K accuracy, refusal rate on harmful and benign prompts, and CPU decode and prefill tok/s with peak RSS.

## Method and pre-registration

TODO: link `docs/design.md` and `docs/prereg.md`. Thresholds in `configs/gate.yaml` are defaults that are not yet pre-registered.

## Limitations and what this does not show

TODO: see `docs/limitations.md`. Planned scope: one model family (Qwen2.5-0.5B-Instruct), developed and measured on an Intel i7-4770HQ laptop (macOS x86_64, CPU only), CI on Linux.

## Licences

TODO: model and dataset licences, verified at build time (Qwen2.5-0.5B-Instruct GGUF: Apache-2.0).

## How this was built

This repository was built with AI coding assistants (Claude Code). The design decisions and experiment specifications were written by the author. The code and documents were drafted with AI assistance. The author's final review of the repository is pending, and no human has yet validated the refusal classifier.

## Licence of this code

MIT. See `LICENSE`.
