"""ARC-Easy scoring, prompt rendering, answer mapping, and the Wilson interval."""

from collections.abc import Callable

import pytest

from quantgate.evals import wilson_ci
from quantgate.evals.mcq import (
    McqItem,
    answer_index,
    build_prompt,
    option_strings,
    predict,
    run_mcq,
    score_mcq,
)


class ScriptedScorer:
    """Backend stub for score_continuations. It returns scripted scores per prompt, and the
    prompts it saw are recorded so the tests can check what was scored."""

    n_vocab = 4

    def __init__(self, score_fn: Callable[[str, list[str]], list[float]]) -> None:
        self.score_fn = score_fn
        self.prompts: list[str] = []
        self.options: list[list[str]] = []

    def score_continuations(self, prompt: str, options: list[str]) -> list[float]:
        """Record the call and return the scripted scores."""
        self.prompts.append(prompt)
        self.options.append(options)
        return self.score_fn(prompt, options)

    def tokenize(self, text: str) -> list[int]:
        """Unused in these tests."""
        raise NotImplementedError

    def logprobs(self, tokens: list[int]) -> object:
        """Unused in these tests."""
        raise NotImplementedError

    def generate(
        self, messages: list[dict[str, str]], max_tokens: int, temperature: float, seed: int
    ) -> object:
        """Unused in these tests."""
        raise NotImplementedError

    def close(self) -> None:
        """Nothing to release."""


def _item(answer: str = "B", labels: list[str] | None = None, n: int = 4) -> McqItem:
    """A four-choice item with the given answer key."""
    return McqItem(
        item_id="Q1",
        question="Which organism makes its own food?",
        choices=[f"choice {i}" for i in range(n)],
        labels=labels if labels is not None else [chr(65 + i) for i in range(n)],
        answer_key=answer,
    )


def test_build_prompt_lists_letters_and_ends_with_answer_marker() -> None:
    prompt = build_prompt(_item())
    assert prompt.endswith("Answer:")
    for letter in "ABCD":
        assert f"\n{letter}. choice " in prompt
    assert "Question: Which organism makes its own food?" in prompt


def test_build_prompt_rejects_bad_choice_counts() -> None:
    with pytest.raises(ValueError):
        build_prompt(_item(n=0))
    with pytest.raises(ValueError):
        build_prompt(_item(n=9))


def test_option_strings_are_leading_space_letters() -> None:
    assert option_strings(3) == [" A", " B", " C"]


def test_answer_index_maps_dataset_labels_to_positions() -> None:
    item = _item(answer="3", labels=["1", "2", "3", "4"])
    assert answer_index(item) == 2
    with pytest.raises(ValueError):
        answer_index(_item(answer="E"))


def test_predict_picks_highest_log_likelihood_option() -> None:
    scorer = ScriptedScorer(lambda _p, _o: [-3.0, -0.5, -2.0, -4.0])
    assert predict(scorer, _item()) == 1
    assert scorer.options[0] == [" A", " B", " C", " D"]
    assert scorer.prompts[0] == build_prompt(_item())


def test_predict_ties_go_to_the_earlier_choice() -> None:
    scorer = ScriptedScorer(lambda _p, _o: [-1.0, -1.0, -5.0, -5.0])
    assert predict(scorer, _item()) == 0


def test_predict_rejects_wrong_score_count() -> None:
    scorer = ScriptedScorer(lambda _p, _o: [-1.0, -2.0])
    with pytest.raises(ValueError):
        predict(scorer, _item())


def test_run_mcq_accuracy_percent_with_wilson_interval() -> None:
    # Ten items; the scorer always picks letter B, so the gold answers decide the accuracy.
    answers = ["B", "B", "B", "B", "B", "B", "B", "A", "C", "D"]
    items = [_item(answer=a) for a in answers]
    scorer = ScriptedScorer(lambda _p, _o: [-9.0, -0.1, -9.0, -9.0])
    result = score_mcq(scorer, items)
    assert result.correct == [a == "B" for a in answers]
    metric = result.metric
    assert metric.name == "arc_acc"
    assert metric.n == 10
    assert metric.value == pytest.approx(70.0)
    low, high = wilson_ci(7, 10)
    assert metric.ci_low == pytest.approx(100.0 * low)
    assert metric.ci_high == pytest.approx(100.0 * high)
    assert run_mcq(scorer, items) == metric


def test_run_mcq_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        run_mcq(ScriptedScorer(lambda _p, _o: []), [])


def test_wilson_ci_known_values() -> None:
    low, high = wilson_ci(0, 10)
    assert low == pytest.approx(0.0, abs=1e-12)
    assert high == pytest.approx(0.2775, abs=5e-4)

    low, high = wilson_ci(5, 10)
    assert low == pytest.approx(0.2366, abs=5e-4)
    assert high == pytest.approx(0.7634, abs=5e-4)

    low, high = wilson_ci(10, 10)
    assert low == pytest.approx(0.7225, abs=5e-4)
    assert high == pytest.approx(1.0, abs=1e-12)

    low, high = wilson_ci(150, 300)
    assert low == pytest.approx(0.4438, abs=5e-4)
    assert high == pytest.approx(0.5562, abs=5e-4)


def test_wilson_ci_contains_point_estimate_and_stays_in_unit_interval() -> None:
    for n in (1, 2, 7, 50, 300):
        for k in range(n + 1):
            low, high = wilson_ci(k, n)
            assert 0.0 <= low <= k / n <= high <= 1.0


def test_wilson_ci_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError):
        wilson_ci(0, 0)
    with pytest.raises(ValueError):
        wilson_ci(11, 10)
    with pytest.raises(ValueError):
        wilson_ci(-1, 10)
