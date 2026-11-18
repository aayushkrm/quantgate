"""KL drift, top-1 agreement and perplexity against the reference variant.

KL is computed per token over the full vocabulary, KL(P_ref || P_q), in nats, on fixed
passages from wikitext-2-raw-v1 (test split). Reference log-probs may be cached as float16
.npy files under ~/.cache/quantgate, or both models may be loaded together.
"""

import numpy as np

from quantgate.backend import Backend
from quantgate.schema import MetricValue


def per_token_kl(ref_logprobs: np.ndarray, cand_logprobs: np.ndarray) -> np.ndarray:
    """KL(P_ref || P_cand) per position, in nats. Inputs are log-softmax [T, V]. Output [T]."""
    raise NotImplementedError


def top1_agreement(ref_logprobs: np.ndarray, cand_logprobs: np.ndarray) -> np.ndarray:
    """Per-position boolean: argmax of the reference equals argmax of the candidate. Output [T]."""
    raise NotImplementedError


def run_kl(
    reference: Backend,
    candidate: Backend,
    passages: list[str],
    ctx: int,
) -> list[MetricValue]:
    """Return kl_mean, top1_agree and ppl for the candidate over all passages.

    Each passage is truncated to ctx tokens. Metrics carry a bootstrap CI over passages.
    """
    raise NotImplementedError
