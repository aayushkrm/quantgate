"""GSM8K generation accuracy.

Dataset: openai/gsm8k, config main, test split. Zero-shot chat prompt asking for step-by-step
reasoning, greedy decoding, max_tokens from the suite config, and last-number extraction.
"""

from quantgate.backend import Backend
from quantgate.schema import MetricValue


def gold_answer(answer_field: str) -> str:
    """The final number after '####' in a GSM8K reference solution."""
    raise NotImplementedError


def extract_last_number(text: str) -> str | None:
    """The last number in a generation, with commas and a trailing '.' removed. None if absent."""
    raise NotImplementedError


def is_correct(generated: str, gold: str) -> bool:
    """True if the extracted last number equals gold, compared numerically."""
    raise NotImplementedError


def run_gsm8k(
    backend: Backend,
    items: list[tuple[str, str]],
    max_tokens: int,
    seed: int,
) -> MetricValue:
    """Accuracy over (question, gold_answer) pairs as MetricValue(name='gsm8k_acc'), with a CI."""
    raise NotImplementedError
