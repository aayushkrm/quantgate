"""Quality evals: KL drift (kl), ARC-Easy log-likelihood accuracy (mcq), GSM8K (gsm8k) and
refusal behaviour (refusal). Shared statistics live here."""


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% interval for a proportion. Returns (low, high) as fractions in [0, 1].

    Raises ValueError if n is 0.
    """
    raise NotImplementedError
