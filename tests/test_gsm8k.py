"""GSM8K answer extraction, gold parsing, and the generation loop with a scripted backend."""

from dataclasses import dataclass, field

import pytest

from quantgate.evals import wilson_ci
from quantgate.evals.gsm8k import (
    build_messages,
    extract_last_number,
    gold_answer,
    is_correct,
    run_gsm8k,
    score_gsm8k,
)
from quantgate.schema import GenResult


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The answer is 42.", "42"),
        ("So we get 1,234 apples.", "1234"),
        ("Total: $1,250.50", "1250.50"),
        ("The final answer is 7", "7"),
        ("x = -3", "-3"),
        ("I have 5-3=2 left", "2"),
        ("We need 3-4 more, so the result is 12.", "12"),
        ("answer: 0.5", "0.5"),
        ("no numbers here", None),
        ("", None),
        ("1,2,3", "3"),
        ("ten and 20 and 30 total: 60", "60"),
        ("$5", "5"),
        ("-$5", "-5"),
        ("the answer is 18.00.", "18.00"),
        ("Final: 1,000,000.", "1000000"),
        ("step 2: 4 cups", "4"),
        ("\\boxed{9}", "9"),
        ("12 and then 3-", "3"),
        ("It costs $1,200.", "1200"),
    ],
)
def test_extract_last_number(text: str, expected: str | None) -> None:
    assert extract_last_number(text) == expected


def test_gold_answer_reads_number_after_marker() -> None:
    assert gold_answer("Natalia sold 48 clips.\n#### 72") == "72"
    assert gold_answer("steps\n#### 1,200") == "1200"
    assert gold_answer("steps\n#### -4") == "-4"


def test_gold_answer_rejects_missing_marker_or_number() -> None:
    with pytest.raises(ValueError):
        gold_answer("no marker here 72")
    with pytest.raises(ValueError):
        gold_answer("steps\n#### ")


def test_is_correct_compares_last_number_as_decimal() -> None:
    assert is_correct("so the total is 72.0", "72")
    assert is_correct("72 is wrong, the answer is 72", "72")
    assert not is_correct("72 vs 70", "72")
    assert not is_correct("no number", "72")
    assert is_correct("The answer is $1,200.", "1200")


def test_is_correct_rejects_non_numeric_gold() -> None:
    with pytest.raises(ValueError):
        is_correct("5", "five")


def test_build_messages_is_a_single_user_turn_with_the_problem() -> None:
    messages = build_messages("  How many apples?  ")
    assert [m["role"] for m in messages] == ["user"]
    assert "Think step by step" in messages[0]["content"]
    assert messages[0]["content"].endswith("Problem: How many apples?")


@dataclass
class ScriptedGenBackend:
    """Backend stub for generate. It returns scripted texts in order and records each call."""

    texts: list[str]
    calls: list[dict[str, object]] = field(default_factory=list)
    n_vocab: int = 4

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> GenResult:
        """Return the next scripted text and record the call arguments."""
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "seed": seed,
            }
        )
        text = self.texts[len(self.calls) - 1]
        return GenResult(text=text, n_prompt_tokens=10, n_gen_tokens=5, prefill_s=0.0, decode_s=0.0)

    def tokenize(self, text: str) -> list[int]:
        """Unused in these tests."""
        raise NotImplementedError

    def logprobs(self, tokens: list[int]) -> object:
        """Unused in these tests."""
        raise NotImplementedError

    def score_continuations(self, prompt: str, options: list[str]) -> list[float]:
        """Unused in these tests."""
        raise NotImplementedError

    def close(self) -> None:
        """Nothing to release."""


PAIRS = [
    ("Tom has 3 boxes with 4 pens. How many pens?", "3 times 4 is 12.\n#### 12"),
    ("Ann buys 5 apples at $2 each.", "5 * 2 = 10\n#### 10"),
    ("A train goes 60 km in 2 hours. Speed?", "60 / 2 = 30\n#### 30"),
    ("Bob has 7 coins and finds 2 more.", "7 + 2 = 9\n#### 9"),
]


def test_run_gsm8k_scores_last_numbers_against_gold() -> None:
    backend = ScriptedGenBackend(
        texts=["So 3*4 = 12. The answer is 12.", "5 * 2 = 10 dollars.", "Speed is 25.", "9"]
    )
    metric = run_gsm8k(backend, PAIRS, max_tokens=256, seed=0)
    assert metric.name == "gsm8k_acc"
    assert metric.n == 4
    assert metric.value == pytest.approx(75.0)
    low, high = wilson_ci(3, 4)
    assert metric.ci_low == pytest.approx(100.0 * low)
    assert metric.ci_high == pytest.approx(100.0 * high)


def test_score_gsm8k_returns_per_item_correctness_in_order() -> None:
    backend = ScriptedGenBackend(texts=["12", "wrong 11", "30", "9 coins"])
    result = score_gsm8k(backend, PAIRS, max_tokens=256, seed=0)
    assert result.correct == [True, False, True, True]
    assert result.metric.value == pytest.approx(75.0)


def test_generation_is_greedy_zero_shot_with_given_budget_and_seed() -> None:
    backend = ScriptedGenBackend(texts=["12", "10", "30", "9"])
    score_gsm8k(backend, PAIRS, max_tokens=256, seed=7)
    assert len(backend.calls) == len(PAIRS)
    for call, (question, _answer) in zip(backend.calls, PAIRS, strict=True):
        assert call["max_tokens"] == 256
        assert call["temperature"] == 0.0
        assert call["seed"] == 7
        messages = call["messages"]
        assert isinstance(messages, list)
        assert question in messages[0]["content"]


def test_gold_comes_from_the_answer_field_not_the_question() -> None:
    backend = ScriptedGenBackend(texts=["The answer is 99."])
    result = score_gsm8k(
        backend, [("Only 99 is in the question.", "work\n#### 5")], max_tokens=64, seed=0
    )
    assert result.correct == [False]


def test_score_gsm8k_rejects_empty_items_and_bad_budget() -> None:
    backend = ScriptedGenBackend(texts=[])
    with pytest.raises(ValueError):
        score_gsm8k(backend, [], max_tokens=64, seed=0)
    with pytest.raises(ValueError):
        score_gsm8k(backend, PAIRS, max_tokens=0, seed=0)
