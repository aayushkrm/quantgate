"""ARC-Easy multiple-choice accuracy by log-likelihood of the answer letter.

Dataset: allenai/ai2_arc, config ARC-Easy, test split. The prompt lists the choices under fixed
letters A, B, C, ... in the order given by the dataset, and ends with "Answer:". Each letter
(" A", " B", ...) is scored as a continuation of that prompt, and the highest log-likelihood wins.
The letters are positions, not the dataset's own labels, so label formats such as "1".."4" map to
the same letters. Accuracy is a percent with a Wilson 95% interval.
"""

from dataclasses import dataclass

import numpy as np

from quantgate.evals import wilson_ci
from quantgate.schema import Backend, MetricValue

LETTERS: str = "ABCDEFGH"
PROMPT_INTRO: str = (
    "The following is a multiple choice science question. Choose the correct answer.\n\n"
)
MAX_CHOICES: int = len(LETTERS)


@dataclass(frozen=True)
class McqItem:
    """One multiple-choice question."""

    item_id: str
    question: str
    choices: list[str]
    labels: list[str]
    answer_key: str


@dataclass(frozen=True)
class McqResult:
    """Accuracy metric plus per-item correctness, in item order, for paired comparisons."""

    metric: MetricValue
    correct: list[bool]


def _check_choices(item: McqItem) -> None:
    """Raise ValueError if the item has no choices or more than MAX_CHOICES."""
    if not 1 <= len(item.choices) <= MAX_CHOICES:
        raise ValueError(f"item {item.item_id} has {len(item.choices)} choices")


def build_prompt(item: McqItem) -> str:
    """Render the question and lettered choices, ending with 'Answer:'."""
    _check_choices(item)
    lines = [f"Question: {item.question.strip()}"]
    for i, choice in enumerate(item.choices):
        lines.append(f"{LETTERS[i]}. {choice.strip()}")
    return PROMPT_INTRO + "\n".join(lines) + "\nAnswer:"


def option_strings(n_choices: int) -> list[str]:
    """Continuations scored after the prompt: ' A', ' B', ... for n_choices choices."""
    return [f" {LETTERS[i]}" for i in range(n_choices)]


def answer_index(item: McqItem) -> int:
    """Index into item.choices of the gold answer. Raises ValueError if answer_key is absent."""
    if item.answer_key not in item.labels:
        raise ValueError(f"answer key {item.answer_key!r} not in labels {item.labels}")
    return item.labels.index(item.answer_key)


def predict(backend: Backend, item: McqItem) -> int:
    """Index of the choice whose letter has the highest log-likelihood after the prompt.

    Ties go to the earlier choice (np.argmax order).
    """
    _check_choices(item)
    options = option_strings(len(item.choices))
    scores = backend.score_continuations(build_prompt(item), options)
    if len(scores) != len(options):
        raise ValueError(f"backend returned {len(scores)} scores for {len(options)} options")
    return int(np.argmax(np.asarray(scores, dtype=np.float64)))


def score_mcq(backend: Backend, items: list[McqItem]) -> McqResult:
    """Accuracy over items, with per-item correctness. Raises ValueError on an empty list."""
    if not items:
        raise ValueError("no items to score")
    correct = [predict(backend, item) == answer_index(item) for item in items]
    n = len(correct)
    k = sum(correct)
    low, high = wilson_ci(k, n)
    metric = MetricValue(
        name="arc_acc",
        value=100.0 * k / n,
        ci_low=100.0 * low,
        ci_high=100.0 * high,
        n=n,
    )
    return McqResult(metric=metric, correct=correct)


def run_mcq(backend: Backend, items: list[McqItem]) -> MetricValue:
    """Accuracy over items as MetricValue(name='arc_acc'), percent, with a Wilson CI and n."""
    return score_mcq(backend, items).metric
