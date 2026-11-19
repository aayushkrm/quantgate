"""Platform-stable number formatting (copied from passk/src/passk/numfmt.py).

Exact fractions such as 0.1375 sit on a rounding tie. Float summation order differs between
Python versions and platforms (Python 3.12 sums floats more accurately), so ``f"{x:.3f}"`` can
print 0.137 on one machine and 0.138 on another. ``fixed`` removes that noise first and then
rounds half away from zero, so the same value prints the same text everywhere.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal


def fixed(value: float, digits: int, *, signed: bool = False) -> str:
    """Format ``value`` with ``digits`` decimals, rounding ties half away from zero.

    Args:
        value: Number to format.
        digits: Number of decimals.
        signed: Prefix non-negative numbers with ``+``.

    Returns:
        The formatted text.
    """
    quantum = Decimal(1).scaleb(-digits)
    exact = Decimal(repr(round(float(value), 9))).quantize(quantum, rounding=ROUND_HALF_UP)
    text = f"{exact:f}"
    if exact == 0 and text.startswith("-"):
        text = text[1:]
    return f"+{text}" if signed and not text.startswith("-") else text
