"""GSM8K generation accuracy.

Dataset: openai/gsm8k, config main, test split. Zero-shot chat prompt asking for step-by-step
reasoning, greedy decoding (temperature 0), max_tokens from the suite config, and last-number
extraction from the generation. The gold answer is the number after '####' in the reference
solution. Accuracy is a percent with a Wilson 95% interval. Only per-item correctness is kept;
completions are not stored here.
"""

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from quantgate.evals import wilson_ci
from quantgate.schema import Backend, MetricValue

GSM8K_INSTRUCTION: str = (
    "Solve the following math word problem. Think step by step and show your reasoning. "
    "Finish with the final answer as a single number."
)

# Optional minus sign, optional dollar sign, then 1,234.5-style or plain numbers.
_NUMBER_RE = re.compile(r"(?P<sign>-)?\$?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class GsmResult:
    """Accuracy metric plus per-item correctness, in item order, for paired comparisons.

    n_truncated counts generations that reached max_tokens. Those are cut off, so their last
    number is not an answer. The count is a diagnostic and is stored in the suite file.
    """

    metric: MetricValue
    correct: list[bool]
    n_truncated: int = 0


def _glued_to_previous(text: str, pos: int) -> bool:
    """True if the character before pos ends a word or number, so a '-' there is a separator."""
    if pos == 0:
        return False
    prev = text[pos - 1]
    return prev.isalnum() or prev in "._)"


def extract_last_number(text: str) -> str | None:
    """The last number in a generation, normalized: commas and '$' removed, trailing '.' dropped.

    A leading '-' counts as a sign only when it does not follow a word or number, so '5-3' ends
    in '3'. Returns None when the text has no number.
    """
    last: str | None = None
    for match in _NUMBER_RE.finditer(text):
        num = match.group("num").replace(",", "")
        negative = match.group("sign") is not None and not _glued_to_previous(text, match.start())
        last = f"-{num}" if negative else num
    return last


def gold_answer(answer_field: str) -> str:
    """The final number after '####' in a GSM8K reference solution, normalized.

    Raises ValueError if there is no '####' marker or no number after it.
    """
    if "####" not in answer_field:
        raise ValueError("reference solution has no '####' marker")
    gold = extract_last_number(answer_field.rsplit("####", 1)[1])
    if gold is None:
        raise ValueError("no number after '####'")
    return gold


def is_correct(generated: str, gold: str) -> bool:
    """True if the last number in generated equals gold, compared as decimals.

    A generation with no number is incorrect. Decimal comparison makes '72.0' equal '72'.
    """
    extracted = extract_last_number(generated)
    if extracted is None:
        return False
    try:
        return Decimal(extracted) == Decimal(gold)
    except InvalidOperation as exc:
        raise ValueError(f"gold answer {gold!r} is not a number") from exc


def build_messages(question: str) -> list[dict[str, str]]:
    """Single user turn: the fixed instruction followed by the problem."""
    return [{"role": "user", "content": f"{GSM8K_INSTRUCTION}\n\nProblem: {question.strip()}"}]


def score_gsm8k(
    backend: Backend,
    items: list[tuple[str, str]],
    max_tokens: int,
    seed: int,
) -> GsmResult:
    """Accuracy over (question, answer field) pairs, with per-item correctness in item order.

    Raises ValueError on an empty list or a non-positive max_tokens.
    """
    if not items:
        raise ValueError("no items to score")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    correct: list[bool] = []
    n_truncated = 0
    for question, answer_field in items:
        gold = gold_answer(answer_field)
        result = backend.generate(
            build_messages(question),
            max_tokens=max_tokens,
            temperature=0.0,
            seed=seed,
        )
        if result.n_gen_tokens >= max_tokens:
            n_truncated += 1
        correct.append(is_correct(result.text, gold))
    n = len(correct)
    k = sum(correct)
    low, high = wilson_ci(k, n)
    metric = MetricValue(
        name="gsm8k_acc",
        value=100.0 * k / n,
        ci_low=100.0 * low,
        ci_high=100.0 * high,
        n=n,
    )
    return GsmResult(metric=metric, correct=correct, n_truncated=n_truncated)


def run_gsm8k(
    backend: Backend,
    items: list[tuple[str, str]],
    max_tokens: int,
    seed: int,
) -> MetricValue:
    """Accuracy over (question, gold_answer) pairs as MetricValue(name='gsm8k_acc'), with a CI.

    The second element of each pair is the raw GSM8K answer field (the full solution); its
    number after '####' is the gold answer. Percent, with a Wilson 95% interval and n.
    """
    return score_gsm8k(backend, items, max_tokens, seed).metric
