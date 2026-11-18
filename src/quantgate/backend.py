"""Inference backends.

LlamaCppBackend wraps llama-cpp-python (optional ``llama`` extra). FakeBackend is a
deterministic toy backend used by all tests.

Notes from research/03 §2.2 (llama-cpp-python 0.3.37):
- ``type_k`` / ``type_v`` are ggml type enum integers, not strings. Use the exported
  constants, e.g. ``llama_cpp.GGML_TYPE_Q8_0`` (8), ``GGML_TYPE_Q4_0`` (2), ``GGML_TYPE_Q4_K`` (12).
- KL needs ``logits_all=True``.

Conventions shared by both backends:
- ``logprobs(tokens)[i]`` is the full-vocabulary log-softmax of the distribution that predicts
  token ``i + 1`` given ``tokens[: i + 1]``. Row ``T - 1`` predicts the token after the input.
- ``score_continuations`` sums ``logprobs(prompt + option)`` over the option tokens only.
- Tokenization never adds a BOS token (Qwen2.5 GGUFs do not use one).
"""

import hashlib
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from quantgate.schema import Backend, GenResult


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax of a float array, returned as a new float32 array.

    The row maximum is subtracted before exp (numerically stable). The normaliser is summed
    in float64. Peak memory is about twice the size of the input, which matters for [T, V]
    with V near 151k.
    """
    out = np.array(logits, dtype=np.float32, copy=True)
    out -= out.max(axis=-1, keepdims=True)
    expd = np.exp(out)
    log_norm = np.log(expd.sum(axis=-1, keepdims=True, dtype=np.float64))
    del expd
    out -= log_norm.astype(np.float32)
    return out


class LlamaCppBackend:
    """llama-cpp-python backend for one GGUF file, CPU only (GGML_METAL off, n_gpu_layers=0).

    Measured on this laptop (Haswell, 4 cores): n_threads=4 is faster than 8 for decode.
    ``llm.eval`` with ``logits_all=True`` writes one float32 row of size V per position into
    ``llm.scores``, which is allocated as (n_ctx, V) at load time. For V = 151936 that is
    about 0.6 MB per n_ctx row (about 1.2 GB at n_ctx=2048, mostly untouched virtual memory
    until used). A [256, V] float32 logprobs result is about 155 MB, and log-softmax peaks
    at about twice that.
    """

    n_vocab: int

    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 512,
        n_threads: int = 4,
        seed: int = 0,
        type_k: int | None = None,
        type_v: int | None = None,
        logits_all: bool = True,
        flash_attn: bool = False,
        n_batch: int | None = None,
    ) -> None:
        """Load the model. type_k and type_v are ggml type ints (E5 KV-cache study only).

        logits_all must be True for logprobs and score_continuations. generate works either
        way. flash_attn is only needed for some quantized V-cache types.

        n_batch defaults to n_ctx, so any input up to n_ctx is evaluated as one batch. The batch
        shape then depends only on the input length, which keeps reference and candidate
        comparable (Q-A measured about 1.3e-4 nats of noise from batch shape alone).
        n_ctx defaults to 512: the KL passages are 256 tokens and the GSM8K prompt plus 256
        generated tokens fits. A larger n_ctx costs n_ctx * n_vocab * 4 bytes of logits buffer
        when logits_all=True (about 1.2 GB at 2048).

        Raises FileNotFoundError if the GGUF is missing and RuntimeError if llama-cpp-python
        is not installed (install the ``llama`` extra).
        """
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"GGUF model not found: {path} (run `quantgate fetch`)")
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed; install the 'llama' extra"
            ) from exc

        self._llm: Any = Llama(
            model_path=str(path),
            n_ctx=n_ctx,
            n_batch=n_ctx if n_batch is None else n_batch,
            n_threads=n_threads,
            n_threads_batch=n_threads,
            n_gpu_layers=0,
            seed=seed,
            logits_all=logits_all,
            type_k=type_k,
            type_v=type_v,
            flash_attn=flash_attn,
            verbose=False,
        )
        self._n_ctx: int = int(self._llm.n_ctx())
        self._logits_all = logits_all
        self.n_vocab = int(self._llm.n_vocab())

    def _require(self) -> Any:
        """Return the live Llama object, or raise if close() was already called."""
        if self._llm is None:
            raise RuntimeError("LlamaCppBackend is closed")
        return self._llm

    def tokenize(self, text: str) -> list[int]:
        """Encode text into token ids (no BOS, special tokens not parsed)."""
        llm = self._require()
        return list(llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))

    def logprobs(self, tokens: list[int]) -> np.ndarray:
        """Full-vocabulary log-softmax for each input position. Shape [T, V], float32.

        Runs one full forward pass from an empty KV cache (llm.reset then llm.eval), so the
        result depends only on ``tokens``. Requires logits_all=True.
        """
        llm = self._require()
        if not self._logits_all:
            raise RuntimeError("logprobs needs a backend created with logits_all=True")
        t = len(tokens)
        if t == 0:
            return np.empty((0, self.n_vocab), dtype=np.float32)
        if t > self._n_ctx:
            raise ValueError(f"{t} tokens exceed n_ctx={self._n_ctx}")
        llm.reset()
        llm.eval(tokens)
        return _log_softmax(llm.scores[:t])

    def score_continuations(self, prompt: str, options: list[str]) -> list[float]:
        """Log-likelihood of each option appended to the prompt, in option order.

        The prompt is evaluated once. Each option is then evaluated on top of the cached
        prompt by resetting ``llm.n_tokens`` to the prompt length, so the prompt is not
        recomputed per option. Options are tokenized separately from the prompt.
        """
        llm = self._require()
        if not self._logits_all:
            raise RuntimeError("score_continuations needs a backend created with logits_all=True")
        p = self.tokenize(prompt)
        if not p:
            raise ValueError("prompt tokenizes to zero tokens")
        if len(p) >= self._n_ctx:
            raise ValueError(f"prompt of {len(p)} tokens leaves no room in n_ctx={self._n_ctx}")
        llm.reset()
        llm.eval(p)
        n_p = len(p)
        scores: list[float] = []
        for option in options:
            o = self.tokenize(option)
            if not o:
                raise ValueError(f"option {option!r} tokenizes to zero tokens")
            if n_p + len(o) > self._n_ctx:
                raise ValueError(f"prompt plus option exceed n_ctx={self._n_ctx}")
            llm.n_tokens = n_p  # drop the previous option's KV entries; prompt stays cached
            llm.eval(o)
            # Row n_p - 1 predicts o[0]; row n_p + j - 1 predicts o[j].
            rows = _log_softmax(llm.scores[n_p - 1 : n_p + len(o) - 1])
            scores.append(float(sum(float(rows[j, tok]) for j, tok in enumerate(o))))
        return scores

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> GenResult:
        """Chat completion with the model's chat template, with prefill and decode timings.

        Timings come from a streamed call. prefill_s is the time from the call to the first
        streamed chunk (prompt evaluation plus sampling the first token). decode_s is the
        time from the first chunk to the last one (the remaining tokens). n_prompt_tokens is
        read from llm.n_tokens at the first chunk, when only the prompt has been evaluated.
        n_gen_tokens counts content chunks with no finish_reason, one per sampled token.
        temperature=0 is greedy decoding. Verified against llm.n_tokens in the model smoke test.
        """
        llm = self._require()
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        llm.reset()
        t0 = time.perf_counter()
        first_t: float | None = None
        last_t = t0
        n_prompt = 0
        n_gen = 0
        pieces: list[str] = []
        stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            stream=True,
        )
        for chunk in stream:
            now = time.perf_counter()
            if first_t is None:
                first_t = now
                n_prompt = int(llm.n_tokens)
            last_t = now
            choice = chunk["choices"][0]
            delta = choice.get("delta") or {}
            if "content" in delta and choice.get("finish_reason") is None:
                n_gen += 1
            content = delta.get("content")
            if content:
                pieces.append(content)
        if first_t is None:
            first_t = last_t
        return GenResult(
            text="".join(pieces),
            n_prompt_tokens=n_prompt,
            n_gen_tokens=n_gen,
            prefill_s=first_t - t0,
            decode_s=last_t - first_t,
        )

    def close(self) -> None:
        """Free the llama.cpp context and model. Calling close twice is a no-op."""
        if self._llm is not None:
            self._llm.close()
            self._llm = None


def _seeded_rng(digest: bytes, tag: bytes) -> np.random.Generator:
    """A numpy Generator seeded from a digest and a tag (keeps streams independent)."""
    seed = int.from_bytes(hashlib.sha256(digest + tag).digest()[:8], "little")
    return np.random.default_rng(seed)


class FakeBackend:
    """Deterministic toy backend for tests. A controllable perturbation shifts its distributions.

    Logits at each position come from a seeded RNG keyed on the whole token prefix, so the
    output is causal and reproducible. With perturbation=0 two instances with the same seed
    give bit-identical logprobs. perturbation>0 adds seeded noise to the logits, so KL against
    the reference is positive.

    generate is scripted: a prompt whose last user message contains one of ``refuse_on``
    (case-insensitive substring) gets a refusal. A message equal to a key in ``script`` gets
    that reply. Everything else gets a compliance reply. Timings are fixed synthetic values
    (SYNTHETIC, not measured).
    """

    n_vocab: int

    def __init__(
        self,
        n_vocab: int = 32,
        seed: int = 0,
        perturbation: float = 0.0,
        *,
        refuse_on: Iterable[str] = (),
        script: Mapping[str, str] | None = None,
    ) -> None:
        """Build a fixed toy model. perturbation = 0 gives the reference distributions."""
        if n_vocab < 2:
            raise ValueError("n_vocab must be at least 2")
        if perturbation < 0:
            raise ValueError("perturbation must be >= 0")
        self.n_vocab = n_vocab
        self.seed = seed
        self.perturbation = float(perturbation)
        self._refuse_on = tuple(s.lower() for s in refuse_on)
        self._script = dict(script or {})
        self._base_key = f"quantgate-fake|{seed}".encode()

    def _check_token(self, tok: int) -> None:
        if not 0 <= tok < self.n_vocab:
            raise ValueError(f"token {tok} outside vocabulary of size {self.n_vocab}")

    def tokenize(self, text: str) -> list[int]:
        """Encode text as one token per UTF-8 byte, modulo n_vocab."""
        return [b % self.n_vocab for b in text.encode("utf-8")]

    def logprobs(self, tokens: list[int]) -> np.ndarray:
        """Full-vocabulary log-softmax for each input position. Shape [T, V], float32."""
        v = self.n_vocab
        out = np.empty((len(tokens), v), dtype=np.float32)
        prefix = hashlib.sha256(self._base_key)
        for i, tok in enumerate(tokens):
            self._check_token(int(tok))
            prefix.update(int(tok).to_bytes(4, "little"))
            digest = prefix.digest()
            logits = _seeded_rng(digest, b"base").standard_normal(v) * 2.0
            if self.perturbation > 0.0:
                logits = logits + self.perturbation * _seeded_rng(digest, b"pert").standard_normal(
                    v
                )
            out[i] = _log_softmax(logits)
        return out

    def score_continuations(self, prompt: str, options: list[str]) -> list[float]:
        """Log-likelihood of each option appended to the prompt, in option order."""
        p = self.tokenize(prompt)
        if not p:
            raise ValueError("prompt tokenizes to zero tokens")
        scores: list[float] = []
        for option in options:
            o = self.tokenize(option)
            if not o:
                raise ValueError(f"option {option!r} tokenizes to zero tokens")
            lp = self.logprobs(p + o)
            n_p = len(p)
            scores.append(float(sum(float(lp[n_p + j - 1, tok]) for j, tok in enumerate(o))))
        return scores

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> GenResult:
        """Scripted chat completion. Timings are SYNTHETIC: 1 us per prompt or generated token."""
        del temperature, seed  # the scripted reply does not sample
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        user_texts = [m["content"] for m in messages if m.get("role") == "user"]
        last_user = user_texts[-1] if user_texts else (messages[-1]["content"] if messages else "")
        prompt_text = "\n".join(m["content"] for m in messages)
        reply = self._scripted_reply(last_user)
        reply = reply.encode("utf-8")[:max_tokens].decode("utf-8", errors="ignore")
        n_prompt = len(self.tokenize(prompt_text))
        n_gen = len(self.tokenize(reply))
        return GenResult(
            text=reply,
            n_prompt_tokens=n_prompt,
            n_gen_tokens=n_gen,
            prefill_s=1e-6 * n_prompt,
            decode_s=1e-6 * n_gen,
        )

    def _scripted_reply(self, last_user: str) -> str:
        """Pick the scripted reply for one user message."""
        if last_user in self._script:
            return self._script[last_user]
        lowered = last_user.lower()
        if any(pattern in lowered for pattern in self._refuse_on):
            return "I can't help with that request."
        return "Sure, here is the answer."

    def close(self) -> None:
        """No-op release for API parity with LlamaCppBackend."""


def _check_protocol_conformance() -> None:
    """Never called. mypy checks that both backends satisfy the frozen Backend protocol."""
    _fake: Backend = FakeBackend()
    _llama: Backend = LlamaCppBackend(Path("unused.gguf"))
    del _fake, _llama
