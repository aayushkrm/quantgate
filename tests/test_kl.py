"""KL drift, top-1 agreement, perplexity and the reference cache, on a synthetic bigram LM."""

from pathlib import Path

import numpy as np
import pytest

from quantgate.evals.kl import RefCache, per_token_kl, run_kl, top1_agreement
from quantgate.schema import MetricValue

N_VOCAB = 40
PASSAGES = [
    "the quick brown fox jumps over the lazy dog",
    "quantized models drift a little from the reference",
    "log probabilities are summed in float64",
    "a short one",
]
CTX = 16


def _log_softmax(x: np.ndarray) -> np.ndarray:
    """Reference log-softmax for the tests, float64."""
    m = x.max(axis=-1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True)))


class FakeLM:
    """Bigram toy LM: the next-token logits are one row of a fixed table, so log-probs are exact.

    noise > 0 perturbs the table, which gives a candidate that drifts from the reference.
    Tokens are characters mod n_vocab, so tokenization is deterministic.
    """

    def __init__(self, n_vocab: int = N_VOCAB, noise: float = 0.0, table_seed: int = 0) -> None:
        self.n_vocab = n_vocab
        table = np.random.default_rng(table_seed).normal(0.0, 2.0, size=(n_vocab, n_vocab))
        if noise > 0:
            table = table + noise * np.random.default_rng(99).normal(size=table.shape)
        self.table = table

    def tokenize(self, text: str) -> list[int]:
        """Characters mod n_vocab."""
        return [ord(c) % self.n_vocab for c in text]

    def logprobs(self, tokens: list[int]) -> np.ndarray:
        """Log-softmax rows of the table at each token, float32 [T, V]."""
        return _log_softmax(self.table[np.asarray(tokens, dtype=np.intp)]).astype(np.float32)


class UniformLM(FakeLM):
    """Every position is uniform over the vocabulary, so perplexity equals n_vocab."""

    def logprobs(self, tokens: list[int]) -> np.ndarray:
        """Uniform log-probabilities, float32 [T, V]."""
        return np.full((len(tokens), self.n_vocab), -np.log(self.n_vocab), dtype=np.float32)


class OtherTokenizerLM(FakeLM):
    """Same table but a different vocabulary mapping, so token ids differ from FakeLM."""

    def tokenize(self, text: str) -> list[int]:
        """Characters mod (n_vocab - 1), which disagrees with FakeLM on most characters."""
        return [ord(c) % (self.n_vocab - 1) for c in text]


def _metric(metrics: list[MetricValue], name: str) -> MetricValue:
    """Pick one MetricValue by name from the run_kl output."""
    return next(m for m in metrics if m.name == name)


def test_per_token_kl_identical_is_zero() -> None:
    rng = np.random.default_rng(0)
    lp = _log_softmax(rng.normal(size=(20, 50)))
    kl = per_token_kl(lp, lp.copy())
    assert kl.shape == (20,)
    assert float(np.max(np.abs(kl))) < 1e-9


def test_per_token_kl_matches_closed_form() -> None:
    ref = np.log(np.array([[0.5, 0.5]]))
    cand = np.log(np.array([[0.25, 0.75]]))
    expected = 0.5 * np.log(0.5 / 0.25) + 0.5 * np.log(0.5 / 0.75)
    assert per_token_kl(ref, cand)[0] == pytest.approx(expected, abs=1e-12)


def test_per_token_kl_positive_under_perturbation_and_nonnegative() -> None:
    rng = np.random.default_rng(1)
    ref = _log_softmax(rng.normal(size=(30, 60)))
    cand = _log_softmax(ref + 0.3 * rng.normal(size=ref.shape))
    kl = per_token_kl(ref, cand)
    assert np.all(kl >= 0.0)
    assert float(kl.mean()) > 1e-3


def test_per_token_kl_handles_zero_probability_reference_entries() -> None:
    ref = np.array([[-np.inf, 0.0]])
    cand = np.log(np.array([[0.5, 0.5]]))
    assert per_token_kl(ref, cand)[0] == pytest.approx(np.log(2.0), abs=1e-12)


def test_per_token_kl_chunked_matches_single_chunk() -> None:
    rng = np.random.default_rng(3)
    ref = _log_softmax(rng.normal(size=(150, 30)))
    cand = _log_softmax(ref + 0.2 * rng.normal(size=ref.shape))
    chunked = per_token_kl(ref, cand)
    per_row = np.array([per_token_kl(ref[i : i + 1], cand[i : i + 1])[0] for i in range(150)])
    assert np.allclose(chunked, per_row, atol=1e-12)


def test_per_token_kl_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        per_token_kl(np.zeros((2, 3)), np.zeros((2, 4)))
    with pytest.raises(ValueError):
        per_token_kl(np.zeros((0, 3)), np.zeros((0, 3)))


def test_top1_agreement_identical_is_all_true_and_flip_is_detected() -> None:
    rng = np.random.default_rng(2)
    ref = _log_softmax(rng.normal(size=(10, 8)))
    same = top1_agreement(ref, ref)
    assert same.dtype == bool
    assert bool(same.all())

    cand = ref.copy()
    cand[3] = ref[3][::-1]  # reversing the row moves the argmax unless it sits in the middle
    agree = top1_agreement(ref, cand)
    expected = np.argmax(ref, axis=-1) == np.argmax(cand, axis=-1)
    assert np.array_equal(agree, expected)
    assert not bool(agree[3]) or int(np.argmax(ref[3])) == int(np.argmax(cand[3]))
    assert 0.0 <= float(agree.mean()) <= 1.0


def test_run_kl_identical_backends_give_zero_kl_and_full_agreement() -> None:
    metrics = run_kl(FakeLM(), FakeLM(), PASSAGES, CTX, n_boot=50, seed=0)
    assert [m.name for m in metrics] == ["kl_mean", "top1_agree", "ppl"]
    kl = _metric(metrics, "kl_mean")
    assert kl.value < 1e-9
    assert kl.ci_low is not None and kl.ci_high is not None
    assert abs(kl.ci_low) < 1e-9 and abs(kl.ci_high) < 1e-9
    assert _metric(metrics, "top1_agree").value == pytest.approx(100.0)
    assert kl.n == len(PASSAGES)


def test_run_kl_perturbed_candidate_drifts_with_bounded_agreement() -> None:
    metrics = run_kl(FakeLM(), FakeLM(noise=1.0), PASSAGES, CTX, n_boot=200, seed=0)
    kl = _metric(metrics, "kl_mean")
    top1 = _metric(metrics, "top1_agree")
    assert kl.value > 0.01
    assert kl.ci_low is not None and kl.ci_high is not None
    assert kl.ci_low <= kl.value <= kl.ci_high
    assert 0.0 <= top1.value < 100.0
    assert top1.ci_low is not None and top1.ci_high is not None
    assert 0.0 <= top1.ci_low <= top1.ci_high <= 100.0


def test_ppl_of_uniform_model_equals_vocab_size() -> None:
    metrics = run_kl(UniformLM(), UniformLM(), PASSAGES, CTX, n_boot=0, seed=0)
    ppl = _metric(metrics, "ppl")
    assert ppl.value == pytest.approx(N_VOCAB, rel=1e-6)
    assert ppl.ci_low is None and ppl.ci_high is None


def test_bootstrap_is_seeded() -> None:
    first = run_kl(FakeLM(), FakeLM(noise=0.8), PASSAGES, CTX, n_boot=100, seed=11)
    second = run_kl(FakeLM(), FakeLM(noise=0.8), PASSAGES, CTX, n_boot=100, seed=11)
    assert [m.model_dump() for m in first] == [m.model_dump() for m in second]


def test_missing_reference_without_cache_is_rejected() -> None:
    with pytest.raises(ValueError):
        run_kl(None, FakeLM(), PASSAGES, CTX)


def test_tokenizer_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        run_kl(FakeLM(), OtherTokenizerLM(), PASSAGES, CTX, n_boot=0)


def test_ctx_and_passage_checks() -> None:
    with pytest.raises(ValueError):
        run_kl(FakeLM(), FakeLM(), PASSAGES, 1)
    with pytest.raises(ValueError):
        run_kl(FakeLM(), FakeLM(), [], CTX)


def test_ref_cache_identical_gives_zero_kl(tmp_path: Path) -> None:
    cache = RefCache("fp16", root=tmp_path)
    metrics = run_kl(FakeLM(), FakeLM(), PASSAGES, CTX, ref_cache=cache, n_boot=0)
    assert _metric(metrics, "kl_mean").value < 1e-9
    assert _metric(metrics, "top1_agree").value == pytest.approx(100.0)


def test_ref_cache_stores_float16_and_tokens(tmp_path: Path) -> None:
    cache = RefCache("fp16", root=tmp_path)
    run_kl(FakeLM(), FakeLM(), PASSAGES, CTX, ref_cache=cache, n_boot=0)
    assert cache.dir == tmp_path / "fp16"
    assert all(cache.has(i) for i in range(len(PASSAGES)))
    tokens, lp = cache.load(0)
    assert lp.dtype == np.float16
    assert lp.shape == (len(tokens), N_VOCAB)
    assert len(tokens) == min(CTX, len(PASSAGES[0]))


def test_ref_cache_close_to_direct_computation(tmp_path: Path) -> None:
    direct = run_kl(FakeLM(), FakeLM(noise=1.0), PASSAGES, CTX, n_boot=0)
    cache = RefCache("fp16", root=tmp_path)
    cached = run_kl(FakeLM(), FakeLM(noise=1.0), PASSAGES, CTX, ref_cache=cache, n_boot=0)
    kl_direct = _metric(direct, "kl_mean").value
    kl_cached = _metric(cached, "kl_mean").value
    assert kl_direct > 0.01
    assert abs(kl_cached - kl_direct) < 2e-3


def test_cache_only_mode_needs_no_reference(tmp_path: Path) -> None:
    cache = RefCache("fp16", root=tmp_path)
    with_ref = run_kl(FakeLM(), FakeLM(noise=0.5), PASSAGES, CTX, ref_cache=cache, n_boot=0)
    cache_only = run_kl(None, FakeLM(noise=0.5), PASSAGES, CTX, ref_cache=cache, n_boot=0)
    assert [m.model_dump() for m in with_ref] == [m.model_dump() for m in cache_only]


def test_incomplete_cache_without_reference_is_rejected(tmp_path: Path) -> None:
    cache = RefCache("fp16", root=tmp_path)
    cache.bind(CTX, PASSAGES)
    with pytest.raises(ValueError):
        run_kl(None, FakeLM(), PASSAGES, CTX, ref_cache=cache)


def test_cache_built_for_other_passages_is_refused(tmp_path: Path) -> None:
    cache = RefCache("fp16", root=tmp_path)
    run_kl(FakeLM(), FakeLM(), PASSAGES, CTX, ref_cache=cache, n_boot=0)
    with pytest.raises(ValueError):
        run_kl(FakeLM(), FakeLM(), PASSAGES[:2], CTX, ref_cache=cache, n_boot=0)
    with pytest.raises(ValueError):
        run_kl(FakeLM(), FakeLM(), PASSAGES, CTX + 1, ref_cache=cache, n_boot=0)


def test_cache_clear_removes_directory(tmp_path: Path) -> None:
    cache = RefCache("q4_k_m", root=tmp_path)
    run_kl(FakeLM(), FakeLM(), PASSAGES, CTX, ref_cache=cache, n_boot=0)
    assert cache.dir.is_dir()
    cache.clear()
    assert not cache.dir.exists()
