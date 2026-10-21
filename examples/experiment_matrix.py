"""Deterministic trial matrix and the exact judge request text."""

import hashlib
import random
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from pathlib import Path
from typing import TypedDict

from experiment_cases import CASES, Case


class Variant(StrEnum):
    SHORT = "short"
    VERBOSE = "verbose"


class Order(StrEnum):
    CORRECT_FIRST = "correct_first"
    WRONG_FIRST = "wrong_first"


class RequestRow(TypedDict):
    id: str
    case_id: str
    domain: str
    variant: str
    trial: int
    order: str
    prompt_version: str
    prompt_sha256: str
    request_sha256: str
    judge_input: str


PROMPT_VERSION = "correctness-v2"
PROMPT = (Path(__file__).parent / "judge_prompt.txt").read_text(encoding="utf-8")
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
ORDER_SEED = 20261021
TRIAL_NUMBERS = (1, 2)
MAX_COMPLETION_TOKENS = 256


def trial_id(case_id: str, variant: str, trial: int, order: str) -> str:
    return f"{case_id}:{variant}:{trial}:{order}"


@dataclass(frozen=True, slots=True)
class Trial:
    case: Case
    variant: Variant
    number: int
    order: Order

    def request_row(self) -> RequestRow:
        erroneous = self.case.wrong
        if self.variant is Variant.VERBOSE:
            erroneous += " " + self.case.wrong_explanation
        answers = (self.case.correct, erroneous)
        if self.order is Order.WRONG_FIRST:
            answers = (erroneous, self.case.correct)
        judge_input = PROMPT.format(
            question=self.case.question,
            answer_a=answers[0],
            answer_b=answers[1],
        )
        return {
            "id": trial_id(self.case.id, self.variant.value, self.number, self.order.value),
            "case_id": self.case.id,
            "domain": self.case.domain.value,
            "variant": self.variant.value,
            "trial": self.number,
            "order": self.order.value,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": PROMPT_SHA256,
            "request_sha256": hashlib.sha256(judge_input.encode("utf-8")).hexdigest(),
            "judge_input": judge_input,
        }


def request_rows() -> Iterator[RequestRow]:
    """Yield the frozen 144 rows in the preregistered shuffled order."""
    trials = (
        Trial(case, variant, number, order)
        for case, variant, number, order in product(CASES, Variant, TRIAL_NUMBERS, Order)
    )
    rows = [trial.request_row() for trial in trials]
    random.Random(ORDER_SEED).shuffle(rows)
    yield from rows


def request_payload(row: RequestRow, model: str) -> dict[str, object]:
    """Construct only the text and parameters sent to the judge."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": row["judge_input"]}],
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "reasoning_effort": "none",
        "response_format": {"type": "json_object"},
    }
