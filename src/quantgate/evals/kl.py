"""KL drift, top-1 agreement and perplexity against the reference variant.

For a passage cut to ctx tokens (T positions, each with a full-vocabulary distribution):
- KL(P_ref || P_cand) at each position, in nats: sum_v p_ref(v) (log p_ref(v) - log p_cand(v)).
  kl_mean pools every position of every passage.
- top1_agree: percent of positions where the argmax of the reference equals the argmax of the
  candidate (0-100).
- ppl: exp of the mean negative log-likelihood of tokens[1:] under the candidate, pooled over
  passages.

Each metric has a 95% percentile bootstrap CI over passages (n_boot resamples, seeded). The
MetricValue n is the number of passages, which is the unit of resampling.

Reference log-probs come from one of two modes, chosen by the caller:
1. Both backends loaded together (ref_cache=None). The math runs on the backend output as is.
2. A RefCache (ref_cache given). The reference log-probs of each passage are written once as
   float16 .npy under ~/.cache/quantgate/ref/<variant>/, about 4 GB for 50 passages x 256 tokens
   x 151936 entries. After that only the candidate has to be loaded. Because float16 rounds the
   reference, the candidate is rounded the same way before the comparison. Identical models then
   give KL exactly 0. The cached and uncached KL differ only by the rounding error, which the
   tests bound.

All KL arithmetic is float64, with log-sum-exp normalization, so each row is a proper
log-distribution before the sums.
"""

import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quantgate.schema import Backend, MetricValue

DEFAULT_CACHE_ROOT: Path = Path.home() / ".cache" / "quantgate" / "ref"
DEFAULT_N_BOOT: int = 1000

_CHUNK_ROWS = 64
_FP16_FLOOR = -6.0e4  # float16 max magnitude is 65504; clip before the cast
_CI_PERCENTILES = (2.5, 97.5)


def _log_normalize(x: np.ndarray) -> np.ndarray:
    """Log-softmax over the last axis, computed in float64 with the max-shift (log-sum-exp)."""
    m = np.max(x, axis=-1, keepdims=True)
    lse = m + np.log(np.sum(np.exp(x - m), axis=-1, keepdims=True))
    return x - lse


def _fp16_roundtrip(logprobs: np.ndarray) -> np.ndarray:
    """Log-probs rounded to float16 and returned as float32, the same rounding the cache applies."""
    clipped = np.maximum(np.asarray(logprobs, dtype=np.float32), _FP16_FLOOR)
    return clipped.astype(np.float16).astype(np.float32)


def _check_pair(ref: np.ndarray, cand: np.ndarray) -> None:
    """Raise ValueError unless both arrays are [T, V] with the same shape."""
    if ref.ndim != 2 or ref.shape != cand.shape:
        raise ValueError(f"expected two [T, V] arrays of equal shape, got {ref.shape} {cand.shape}")
    if ref.shape[0] == 0:
        raise ValueError("no positions to compare")


def per_token_kl(ref_logprobs: np.ndarray, cand_logprobs: np.ndarray) -> np.ndarray:
    """KL(P_ref || P_cand) per position, in nats. Inputs are log-softmax [T, V]. Output [T].

    Rows are processed in chunks, so the float64 temporaries stay small. Tiny negative values
    from float round-off are clamped to 0; by Gibbs' inequality the true value is >= 0.
    """
    ref = np.asarray(ref_logprobs)
    cand = np.asarray(cand_logprobs)
    _check_pair(ref, cand)
    n_rows = ref.shape[0]
    out = np.empty(n_rows, dtype=np.float64)
    for start in range(0, n_rows, _CHUNK_ROWS):
        rows = slice(start, start + _CHUNK_ROWS)
        r = _log_normalize(ref[rows].astype(np.float64))
        c = _log_normalize(cand[rows].astype(np.float64))
        p = np.exp(r)
        with np.errstate(invalid="ignore", over="ignore"):
            terms = np.where(p > 0.0, p * (r - c), 0.0)
        out[rows] = terms.sum(axis=-1)
    return np.maximum(out, 0.0)


def top1_agreement(ref_logprobs: np.ndarray, cand_logprobs: np.ndarray) -> np.ndarray:
    """Per-position boolean: argmax of the reference equals argmax of the candidate. Output [T]."""
    ref = np.asarray(ref_logprobs)
    cand = np.asarray(cand_logprobs)
    _check_pair(ref, cand)
    return np.argmax(ref, axis=-1) == np.argmax(cand, axis=-1)


def _sha256(text: str) -> str:
    """Hex SHA-256 of a passage's UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RefCache:
    """Float16 on-disk cache of reference log-probs for one reference variant.

    Layout: <root>/<variant>/manifest.json, and per passage index i, pNNNN.npy (float16 log-probs
    [T, V]) with pNNNN.tokens.npy (int64 token ids). The manifest records ctx and the SHA-256 of
    each passage, so a cache built for other passages or another ctx is refused.
    """

    def __init__(self, variant: str, root: Path | None = None) -> None:
        """Cache for one reference variant. Files go under root (default ~/.cache/quantgate/ref)."""
        self.variant = variant
        base = DEFAULT_CACHE_ROOT if root is None else Path(root)
        self.dir: Path = base / variant

    def _paths(self, idx: int) -> tuple[Path, Path]:
        """Log-prob path and token path for passage index idx."""
        stem = f"p{idx:04d}"
        return self.dir / f"{stem}.npy", self.dir / f"{stem}.tokens.npy"

    def has(self, idx: int) -> bool:
        """True if both files for passage idx exist."""
        lp_path, tok_path = self._paths(idx)
        return lp_path.is_file() and tok_path.is_file()

    def save(self, idx: int, tokens: Sequence[int], logprobs: np.ndarray) -> None:
        """Write passage idx: token ids and float16 log-probs of shape [len(tokens), V]."""
        if logprobs.ndim != 2 or logprobs.shape[0] != len(tokens):
            raise ValueError("logprobs must be [len(tokens), V]")
        self.dir.mkdir(parents=True, exist_ok=True)
        lp_path, tok_path = self._paths(idx)
        np.save(tok_path, np.asarray(tokens, dtype=np.int64))
        np.save(
            lp_path,
            np.maximum(np.asarray(logprobs, dtype=np.float32), _FP16_FLOOR).astype(np.float16),
        )

    def load(self, idx: int) -> tuple[list[int], np.ndarray]:
        """Token ids and float16 log-probs for passage idx. Raises FileNotFoundError if absent."""
        lp_path, tok_path = self._paths(idx)
        tokens = [int(t) for t in np.load(tok_path)]
        return tokens, np.load(lp_path)

    def bind(self, ctx: int, passages: Sequence[str]) -> None:
        """Write the manifest, or verify an existing one matches ctx and these passages.

        Raises ValueError on mismatch. Clear the cache to rebuild it for other passages.
        """
        meta = {
            "variant": self.variant,
            "ctx": ctx,
            "passage_sha256": [_sha256(p) for p in passages],
        }
        path = self.dir / "manifest.json"
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != meta:
                raise ValueError(
                    f"reference cache {self.dir} was built for other passages or ctx; "
                    "clear it first"
                )
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def clear(self) -> None:
        """Delete this variant's cache directory (about 4 GB after a full run)."""
        shutil.rmtree(self.dir, ignore_errors=True)


def build_ref_cache(
    reference: Backend,
    passages: Sequence[str],
    ctx: int,
    cache: RefCache,
) -> None:
    """Fill the cache with the reference log-probs of every passage not yet cached.

    Passages are cut to their first ctx tokens. Existing entries are kept, so the function can
    be rerun after an interruption.
    """
    cache.bind(ctx, passages)
    for idx, text in enumerate(passages):
        if cache.has(idx):
            continue
        tokens = reference.tokenize(text)[:ctx]
        if not tokens:
            raise ValueError(f"passage {idx} tokenizes to nothing")
        cache.save(idx, tokens, reference.logprobs(tokens))


@dataclass
class _PassageStats:
    """Per-passage sums, kept so the bootstrap can resample passages."""

    kl_sum: np.ndarray
    n_pos: np.ndarray
    top1_sum: np.ndarray
    nll_sum: np.ndarray
    n_nll: np.ndarray

    @classmethod
    def empty(cls, n: int) -> "_PassageStats":
        """Zeroed stats for n passages."""
        return cls(
            kl_sum=np.zeros(n, dtype=np.float64),
            n_pos=np.zeros(n, dtype=np.float64),
            top1_sum=np.zeros(n, dtype=np.float64),
            nll_sum=np.zeros(n, dtype=np.float64),
            n_nll=np.zeros(n, dtype=np.float64),
        )


def _ci(draws: np.ndarray) -> tuple[float, float]:
    """Percentile 95% interval of bootstrap draws."""
    low, high = np.percentile(draws, _CI_PERCENTILES)
    return float(low), float(high)


def run_kl(
    reference: Backend | None,
    candidate: Backend,
    passages: Sequence[str],
    ctx: int,
    *,
    ref_cache: RefCache | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> list[MetricValue]:
    """Return kl_mean (nats/token), top1_agree (percent) and ppl for the candidate.

    Passages are processed one at a time, so memory stays at about two [ctx, V] arrays. With a
    ref_cache, a missing reference is allowed once the cache is complete. If reference is given
    together with a cache, missing entries are computed and saved first.

    Raises ValueError on tokenizer mismatch between reference and candidate, an empty passage,
    a cache built for other passages, or a missing reference with an incomplete cache.
    """
    if ctx < 2:
        raise ValueError("ctx must be at least 2 tokens")
    if not passages:
        raise ValueError("no passages")
    if ref_cache is not None:
        if reference is not None:
            build_ref_cache(reference, passages, ctx, ref_cache)
        else:
            ref_cache.bind(ctx, passages)
            missing = [i for i in range(len(passages)) if not ref_cache.has(i)]
            if missing:
                raise ValueError(f"reference cache incomplete: {len(missing)} passages missing")
    elif reference is None:
        raise ValueError("a reference backend or a complete RefCache is required")

    n = len(passages)
    stats = _PassageStats.empty(n)
    for idx, text in enumerate(passages):
        cand_tokens = candidate.tokenize(text)[:ctx]
        if not cand_tokens:
            raise ValueError(f"passage {idx} tokenizes to nothing")
        if ref_cache is not None:
            ref_tokens, ref_cached = ref_cache.load(idx)
            if ref_tokens != cand_tokens:
                raise ValueError(
                    f"tokenizer mismatch on passage {idx}; reference and candidate differ"
                )
            ref_lp = ref_cached.astype(np.float32)
            cand_lp = _fp16_roundtrip(candidate.logprobs(cand_tokens))
        else:
            assert reference is not None  # guaranteed by the checks above
            ref_tokens = reference.tokenize(text)[:ctx]
            if ref_tokens != cand_tokens:
                raise ValueError(
                    f"tokenizer mismatch on passage {idx}; reference and candidate differ"
                )
            ref_lp = np.asarray(reference.logprobs(ref_tokens))
            cand_lp = np.asarray(candidate.logprobs(cand_tokens))

        t = len(cand_tokens)
        kl = per_token_kl(ref_lp, cand_lp)
        top1 = top1_agreement(ref_lp, cand_lp)
        stats.kl_sum[idx] = float(kl.sum())
        stats.n_pos[idx] = t
        stats.top1_sum[idx] = float(top1.sum())
        if t >= 2:
            targets = np.asarray(cand_tokens[1:], dtype=np.intp)
            picked = cand_lp[np.arange(t - 1), targets].astype(np.float64)
            stats.nll_sum[idx] = float(-picked.sum())
            stats.n_nll[idx] = t - 1

    if stats.n_nll.sum() == 0:
        raise ValueError("no passage has at least 2 tokens, so perplexity is undefined")

    kl_mean = float(stats.kl_sum.sum()) / float(stats.n_pos.sum())
    top1_pct = 100.0 * float(stats.top1_sum.sum()) / float(stats.n_pos.sum())
    ppl = float(np.exp(float(stats.nll_sum.sum()) / float(stats.n_nll.sum())))

    kl_ci: tuple[float | None, float | None] = (None, None)
    top1_ci: tuple[float | None, float | None] = (None, None)
    ppl_ci: tuple[float | None, float | None] = (None, None)
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, n, size=(n_boot, n))
        with np.errstate(invalid="ignore", divide="ignore"):
            kl_b = stats.kl_sum[draws].sum(axis=1) / stats.n_pos[draws].sum(axis=1)
            top1_b = 100.0 * stats.top1_sum[draws].sum(axis=1) / stats.n_pos[draws].sum(axis=1)
            ppl_b = np.exp(stats.nll_sum[draws].sum(axis=1) / stats.n_nll[draws].sum(axis=1))
        kl_ci = _ci(kl_b)
        top1_ci = _ci(top1_b)
        ppl_ci = _ci(ppl_b)

    return [
        MetricValue(name="kl_mean", value=kl_mean, ci_low=kl_ci[0], ci_high=kl_ci[1], n=n),
        MetricValue(name="top1_agree", value=top1_pct, ci_low=top1_ci[0], ci_high=top1_ci[1], n=n),
        MetricValue(name="ppl", value=ppl, ci_low=ppl_ci[0], ci_high=ppl_ci[1], n=n),
    ]
