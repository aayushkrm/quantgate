"""ARC-Easy multiple-choice accuracy by log-likelihood of the answer letter.

Dataset: allenai/ai2_arc, config ARC-Easy, test split. Accuracy is reported with a Wilson 95%
interval.
"""

from dataclasses import dataclass

from quantgate.backend import Backend
from quantgate.schema import MetricValue


@dataclass(frozen=True)
class McqItem:
    """One multiple-choice question."""

    item_id: str
    question: str
    choices: list[str]
    labels: list[str]
    answer_key: str


def build_prompt(item: McqItem) -> str:
    """Render the question and lettered choices, ending with 'Answer:'."""
    raise NotImplementedError


def answer_index(item: McqItem) -> int:
    """Index into item.choices of the gold answer. Raises ValueError if answer_key is absent."""
    raise NotImplementedError


def predict(backend: Backend, item: McqItem) -> int:
    """Index of the choice whose letter has the highest log-likelihood after the prompt."""
    raise NotImplementedError


def run_mcq(backend: Backend, items: list[McqItem]) -> MetricValue:
    """Accuracy over items as MetricValue(name='arc_acc'), with a Wilson CI and n."""
    raise NotImplementedError
