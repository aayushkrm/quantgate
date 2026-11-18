"""Inference backends.

LlamaCppBackend wraps llama-cpp-python (optional ``llama`` extra). FakeBackend is a
deterministic toy backend used by all tests.

Notes from research/03 §2.2 (llama-cpp-python 0.3.37):
- ``type_k`` / ``type_v`` are ggml type enum integers, not strings. Use the exported
  constants, e.g. ``llama_cpp.GGML_TYPE_Q8_0`` (8), ``GGML_TYPE_Q4_0`` (2), ``GGML_TYPE_Q4_K`` (12).
- KL needs ``logits_all=True``.
"""

from pathlib import Path

import numpy as np

from quantgate.schema import Backend, GenResult


class LlamaCppBackend:
    """llama-cpp-python backend for one GGUF file, CPU only (GGML_METAL off)."""

    n_vocab: int

    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 2048,
        n_threads: int = 4,
        seed: int = 0,
        type_k: int | None = None,
        type_v: int | None = None,
    ) -> None:
        """Load the model. type_k and type_v are ggml type ints (E5 KV-cache study only)."""
        raise NotImplementedError

    def tokenize(self, text: str) -> list[int]:
        """Encode text into token ids."""
        raise NotImplementedError

    def logprobs(self, tokens: list[int]) -> np.ndarray:
        """Full-vocabulary log-softmax for each position. Shape [T, V]."""
        raise NotImplementedError

    def score_continuations(self, prompt: str, options: list[str]) -> list[float]:
        """Log-likelihood of each option appended to the prompt, in option order."""
        raise NotImplementedError

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> GenResult:
        """Chat completion with prefill and decode timings."""
        raise NotImplementedError

    def close(self) -> None:
        """Free the llama.cpp context and model."""
        raise NotImplementedError


class FakeBackend:
    """Deterministic toy backend for tests. A controllable perturbation shifts its distributions."""

    n_vocab: int

    def __init__(self, n_vocab: int = 32, seed: int = 0, perturbation: float = 0.0) -> None:
        """Build a fixed toy model. perturbation = 0 gives the reference distributions."""
        raise NotImplementedError

    def tokenize(self, text: str) -> list[int]:
        """Encode text into token ids."""
        raise NotImplementedError

    def logprobs(self, tokens: list[int]) -> np.ndarray:
        """Full-vocabulary log-softmax for each position. Shape [T, V]."""
        raise NotImplementedError

    def score_continuations(self, prompt: str, options: list[str]) -> list[float]:
        """Log-likelihood of each option appended to the prompt, in option order."""
        raise NotImplementedError

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> GenResult:
        """Chat completion with prefill and decode timings."""
        raise NotImplementedError

    def close(self) -> None:
        """No-op release for API parity with LlamaCppBackend."""
        raise NotImplementedError


def _check_protocol_conformance() -> None:
    """Never called. mypy checks that both backends satisfy the frozen Backend protocol."""
    _fake: Backend = FakeBackend()
    _llama: Backend = LlamaCppBackend(Path("unused.gguf"))
    del _fake, _llama
