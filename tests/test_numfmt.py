"""Tie-stable rounding: exact fractions on a rounding tie print the same text on every Python."""

from quantgate.numfmt import fixed


def test_ties_round_half_away_from_zero() -> None:
    assert fixed(0.1375, 3) == "0.138"
    assert fixed(2.5, 0) == "3"
    assert fixed(-2.5, 0) == "-3"
    assert fixed(1.2345, 3) == "1.235"


def test_signed_and_negative_zero() -> None:
    assert fixed(1.0, 1, signed=True) == "+1.0"
    assert fixed(-1.0, 1, signed=True) == "-1.0"
    assert fixed(-0.0, 1) == "0.0"
