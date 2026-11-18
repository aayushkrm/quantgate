# Design

Status: stub. Records design decisions and the reasons for them. Each entry links to its plan
section in DECISION.md. Written before results exist.

## Decisions

- TODO: Qwen2.5-0.5B-Instruct GGUF ladder as the only model family.
- TODO: llama-cpp-python for inference, with a FakeBackend for tests.
- TODO: KL over the full vocabulary, with top-1 agreement and perplexity as companions.
- TODO: refusal classifier is a published regex list. It has not been validated by a human.
- TODO: perf runs last, one fresh subprocess per variant.
