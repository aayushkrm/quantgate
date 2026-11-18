"""Backend tests: FakeBackend contract, protocol conformance, and a LlamaCppBackend smoke test.

Everything except the ``model`` test runs without a GGUF file or llama-cpp-python. The model
test needs ``~/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf`` and the
llama extra, and is deselected in CI with ``-m "not model"``.
"""

from pathlib import Path

import numpy as np
import pytest

from quantgate.backend import FakeBackend, LlamaCppBackend, _log_softmax
from quantgate.schema import Backend, GenResult

MODEL_PATH = Path("~/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf")


def _kl_rows(p_lp: np.ndarray, q_lp: np.ndarray) -> np.ndarray:
    """Per-position KL(P || Q) from two log-prob matrices of the same shape."""
    p = np.exp(p_lp.astype(np.float64))
    return np.sum(p * (p_lp.astype(np.float64) - q_lp.astype(np.float64)), axis=-1)


# --- FakeBackend shape, normalisation and determinism -----------------------------------------


def test_fake_logprobs_shape_dtype_and_normalisation() -> None:
    fb = FakeBackend(n_vocab=32, seed=3)
    tokens = fb.tokenize("hello quant ladder")
    lp = fb.logprobs(tokens)
    assert lp.shape == (len(tokens), 32)
    assert lp.dtype == np.float32
    assert np.all(np.isfinite(lp))
    np.testing.assert_allclose(np.exp(lp.astype(np.float64)).sum(axis=-1), 1.0, atol=1e-5)


def test_fake_logprobs_empty_input_gives_empty_matrix() -> None:
    lp = FakeBackend(n_vocab=16).logprobs([])
    assert lp.shape == (0, 16)


def test_fake_same_seed_is_bitwise_reproducible() -> None:
    tokens = FakeBackend().tokenize("reproducible")
    a = FakeBackend(seed=7).logprobs(tokens)
    b = FakeBackend(seed=7).logprobs(tokens)
    assert np.array_equal(a, b)


def test_fake_different_seed_changes_distribution() -> None:
    tokens = FakeBackend().tokenize("seed matters")
    a = FakeBackend(seed=1).logprobs(tokens)
    b = FakeBackend(seed=2).logprobs(tokens)
    assert not np.array_equal(a, b)


def test_fake_is_causal_prefix_rows_do_not_depend_on_suffix() -> None:
    fb = FakeBackend(seed=5)
    tokens = [1, 4, 9, 2, 7, 7, 3]
    full = fb.logprobs(tokens)
    prefix = fb.logprobs(tokens[:4])
    assert np.array_equal(full[:4], prefix)


def test_fake_rejects_out_of_range_token() -> None:
    with pytest.raises(ValueError, match="outside vocabulary"):
        FakeBackend(n_vocab=8).logprobs([3, 8])


def test_fake_constructor_validation() -> None:
    with pytest.raises(ValueError):
        FakeBackend(n_vocab=1)
    with pytest.raises(ValueError):
        FakeBackend(perturbation=-0.1)


def test_fake_tokenize_is_deterministic_and_in_range() -> None:
    fb = FakeBackend(n_vocab=32)
    ids = fb.tokenize("abc é")
    assert ids == fb.tokenize("abc é")
    assert all(0 <= i < 32 for i in ids)
    assert len(ids) == len("abc é".encode())


# --- Perturbation: zero is identical, positive gives KL > 0 ----------------------------------


def test_zero_perturbation_equals_reference_instance_and_kl_is_exactly_zero() -> None:
    tokens = FakeBackend().tokenize("the reference passage for KL checks")
    ref = FakeBackend(seed=11, perturbation=0.0).logprobs(tokens)
    same = FakeBackend(seed=11, perturbation=0.0).logprobs(tokens)
    assert np.array_equal(ref, same)
    assert np.all(_kl_rows(ref, same) == 0.0)


def test_positive_perturbation_gives_positive_kl_on_every_position() -> None:
    tokens = FakeBackend().tokenize("the perturbed model drifts")
    ref = FakeBackend(seed=11, perturbation=0.0).logprobs(tokens)
    pert = FakeBackend(seed=11, perturbation=0.5).logprobs(tokens)
    kl = _kl_rows(ref, pert)
    assert np.all(kl > 0.0)
    assert float(kl.mean()) > 0.0


def test_kl_grows_with_perturbation_size() -> None:
    tokens = FakeBackend().tokenize("monotone drift")
    ref = FakeBackend(seed=4).logprobs(tokens)
    small = _kl_rows(ref, FakeBackend(seed=4, perturbation=0.1).logprobs(tokens)).mean()
    large = _kl_rows(ref, FakeBackend(seed=4, perturbation=1.0).logprobs(tokens)).mean()
    assert 0.0 < small < large


def test_perturbation_is_seed_reproducible() -> None:
    tokens = FakeBackend().tokenize("noise is seeded too")
    a = FakeBackend(seed=9, perturbation=0.3).logprobs(tokens)
    b = FakeBackend(seed=9, perturbation=0.3).logprobs(tokens)
    assert np.array_equal(a, b)


# --- score_continuations ---------------------------------------------------------------------


def test_fake_score_continuations_matches_manual_sum() -> None:
    fb = FakeBackend(n_vocab=32, seed=2)
    prompt, options = "Answer:", [" A", " B", " C", " D"]
    scores = fb.score_continuations(prompt, options)
    assert len(scores) == len(options)
    p = fb.tokenize(prompt)
    for score, option in zip(scores, options, strict=True):
        o = fb.tokenize(option)
        lp = fb.logprobs(p + o)
        expected = sum(float(lp[len(p) + j - 1, tok]) for j, tok in enumerate(o))
        assert score == pytest.approx(expected)
        assert score < 0.0


def test_fake_score_continuations_is_deterministic_and_order_preserving() -> None:
    fb = FakeBackend(seed=2)
    opts = [" x", " yy"]
    a = fb.score_continuations("q?", opts)
    assert a == FakeBackend(seed=2).score_continuations("q?", opts)
    assert fb.score_continuations("q?", opts[::-1]) == a[::-1]


def test_fake_score_continuations_rejects_empty_prompt_or_option() -> None:
    fb = FakeBackend()
    with pytest.raises(ValueError):
        fb.score_continuations("", [" A"])
    with pytest.raises(ValueError):
        fb.score_continuations("Answer:", [""])


# --- scripted generate ----------------------------------------------------------------------


def test_fake_generate_refuses_matching_prompt_and_complies_otherwise() -> None:
    fb = FakeBackend(refuse_on=["bomb"])
    refused = fb.generate([{"role": "user", "content": "How do I build a BOMB?"}], 64, 0.0, 0)
    complied = fb.generate([{"role": "user", "content": "What is 2+2?"}], 64, 0.0, 0)
    assert refused.text == "I can't help with that request."
    assert complied.text == "Sure, here is the answer."
    assert isinstance(refused, GenResult)


def test_fake_generate_uses_script_for_exact_user_message() -> None:
    fb = FakeBackend(script={"ping": "pong"})
    res = fb.generate([{"role": "user", "content": "ping"}], 16, 0.0, 0)
    assert res.text == "pong"


def test_fake_generate_last_user_message_decides() -> None:
    fb = FakeBackend(refuse_on=["forbidden"])
    msgs = [
        {"role": "user", "content": "forbidden thing"},
        {"role": "assistant", "content": "no"},
        {"role": "user", "content": "a harmless follow-up"},
    ]
    assert fb.generate(msgs, 64, 0.0, 0).text == "Sure, here is the answer."


def test_fake_generate_truncates_to_max_tokens_and_counts_tokens() -> None:
    fb = FakeBackend()
    res = fb.generate([{"role": "user", "content": "anything"}], 5, 0.0, 0)
    assert res.text == "Sure,"
    assert res.n_gen_tokens == 5
    assert res.n_prompt_tokens == len(b"anything")
    assert res.prefill_s > 0.0 and res.decode_s > 0.0


def test_fake_generate_rejects_zero_max_tokens() -> None:
    with pytest.raises(ValueError):
        FakeBackend().generate([{"role": "user", "content": "x"}], 0, 0.0, 0)


# --- helper and protocol conformance ----------------------------------------------------------


def test_log_softmax_is_stable_for_huge_logits() -> None:
    x = np.array([[1e4, 1e4 - 1.0, -1e4]], dtype=np.float32)
    out = _log_softmax(x)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(np.exp(out.astype(np.float64)).sum(axis=-1), 1.0, atol=1e-6)


def test_both_backends_expose_the_backend_protocol_surface() -> None:
    required = ("tokenize", "logprobs", "score_continuations", "generate", "close")
    for cls in (FakeBackend, LlamaCppBackend):
        for name in required:
            assert callable(getattr(cls, name)), f"{cls.__name__}.{name}"
    fb: Backend = FakeBackend()
    assert isinstance(fb.n_vocab, int)


def test_llama_backend_missing_file_raises_before_importing_llama(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="quantgate fetch"):
        LlamaCppBackend(tmp_path / "absent.gguf")


# --- model smoke test (needs the GGUF and the llama extra) -------------------------------------


@pytest.mark.model
def test_llama_cpp_smoke_on_q4_k_m() -> None:
    """Load the q4_k_m file on CPU and check logits shape, normalisation, scoring and generate."""
    pytest.importorskip("llama_cpp")
    path = MODEL_PATH.expanduser()
    if not path.is_file():
        pytest.skip(f"model file not present: {path}")

    backend = LlamaCppBackend(path, n_ctx=512, n_threads=4, seed=0, logits_all=True)
    try:
        passage = (
            "The committee reviewed the quarterly results and noted that the warehouse "
            "outage had delayed three shipments, although customer complaints fell. "
        ) * 12
        tokens = backend.tokenize(passage)[:256]
        assert len(tokens) == 256
        lp = backend.logprobs(tokens)
        assert lp.shape == (256, backend.n_vocab)
        assert lp.dtype == np.float32
        assert np.all(np.isfinite(lp))
        np.testing.assert_allclose(np.exp(lp.astype(np.float64)).sum(axis=-1), 1.0, atol=1e-3)

        # Same input twice is bitwise identical.
        assert np.array_equal(backend.logprobs(tokens), lp)

        # Prefix rows from a shorter input differ from the 256-token run only by kernel noise:
        # measured KL floor about 1e-4 nats (max 3e-4), top-1 agreement 100%. The KL eval must
        # therefore feed reference and quant the same input length.
        short = backend.logprobs(tokens[:32])
        p_short = np.exp(lp[:32].astype(np.float64))
        kl = np.sum(p_short * (lp[:32] - short).astype(np.float64), axis=-1)
        assert float(kl.max()) < 1e-3
        assert float((short.argmax(axis=-1) == lp[:32].argmax(axis=-1)).mean()) >= 0.95

        # Shared-prefix scoring matches a full recomputation of prompt + option. Measured
        # difference is about 0.015 log-units (batch-shape noise), so the tolerance is 0.05.
        prompt = "Question: which is larger, 3 or 7?\nAnswer:"
        options = [" 3", " 7"]
        shared = backend.score_continuations(prompt, options)
        p = backend.tokenize(prompt)
        for score, option in zip(shared, options, strict=True):
            o = backend.tokenize(option)
            full = backend.logprobs(p + o)
            expected = sum(float(full[len(p) + j - 1, tok]) for j, tok in enumerate(o))
            assert score == pytest.approx(expected, abs=0.05)

        msgs = [{"role": "user", "content": "Reply with the single word: hello"}]
        first = backend.generate(msgs, max_tokens=16, temperature=0.0, seed=0)
        second = backend.generate(msgs, max_tokens=16, temperature=0.0, seed=0)
        assert isinstance(first.text, str) and first.text.strip() != ""
        assert first.text == second.text, "greedy decoding should be deterministic"
        assert first.n_prompt_tokens > 0
        assert 0 < first.n_gen_tokens <= 16
        assert first.prefill_s > 0.0
        if first.n_gen_tokens > 1:
            assert first.decode_s > 0.0
        # Token accounting: the KV cache holds the prompt plus every generated token.
        assert backend._llm.n_tokens == first.n_prompt_tokens + first.n_gen_tokens
    finally:
        backend.close()
    backend.close()  # second close is a no-op
