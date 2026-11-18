"""Quality evals: KL drift (kl), ARC-Easy log-likelihood accuracy (mcq), GSM8K (gsm8k) and
refusal behaviour (refusal). Shared statistics live here."""

import math


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% interval for a proportion. Returns (low, high) as fractions in [0, 1].

    Raises ValueError if n is 0 or successes is outside [0, n].
    """
    if n <= 0:
        raise ValueError("wilson_ci needs n > 0")
    if not 0 <= successes <= n:
        raise ValueError(f"successes={successes} is outside [0, {n}]")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    low = 0.0 if successes == 0 else max(0.0, center - half)
    high = 1.0 if successes == n else min(1.0, center + half)
    return low, high
