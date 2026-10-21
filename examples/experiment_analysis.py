"""Offline validation and metrics for synthetic controls and recorded responses."""

import json
from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Literal, TypedDict

from experiment_matrix import PROMPT_SHA256, PROMPT_VERSION, Order, RequestRow, request_rows, trial_id


class Winner(StrEnum):
    A = "A"
    B = "B"
    TIE = "tie"


Choice = Literal["correct", "wrong", "tie"]


class ControlSummary(TypedDict):
    rows: int
    valid: int
    wrong_wins: int
    ties: int
    invalid: int
    paired: int
    flips: int


class GroupCounts(TypedDict):
    valid: int
    wrong_wins: int
    ties: int


class RecordedSummary(TypedDict):
    model: str
    attempts: int
    valid: int
    wrong_wins: int
    ties: int
    invalid_output: int
    refusals: int
    timeouts: int
    api_errors: int
    position_flips: int
    position_pairs: int
    repeat_disagreements: int
    repeat_pairs: int
    groups: dict[str, GroupCounts]


def _expected_rows() -> dict[str, RequestRow]:
    return {row["id"]: row for row in request_rows()}


def _winner(value: object) -> Winner | None:
    if not isinstance(value, str):
        return None
    try:
        return Winner(value)
    except ValueError:
        return None


def _choice(winner: Winner, order: str) -> Choice:
    if winner is Winner.TIE:
        return "tie"
    wrong_won = (order == Order.CORRECT_FIRST and winner is Winner.B) or (
        order == Order.WRONG_FIRST and winner is Winner.A
    )
    return "wrong" if wrong_won else "correct"


def _position_pairs(expected: Mapping[str, RequestRow], picked: Mapping[str, Choice]) -> tuple[int, int]:
    pairs = flips = 0
    for row in expected.values():
        if row["order"] != Order.CORRECT_FIRST or row["id"] not in picked:
            continue
        peer = trial_id(row["case_id"], row["variant"], row["trial"], Order.WRONG_FIRST)
        if peer in picked:
            pairs += 1
            flips += picked[row["id"]] != picked[peer]
    return pairs, flips


def evaluate(observations: Iterable[Mapping[str, object]], require_complete: bool) -> ControlSummary:
    """Score verdicts by candidate identity, including A/B reversals."""
    expected = _expected_rows()
    seen: set[str] = set()
    models: set[str] = set()
    picked: dict[str, Choice] = {}
    invalid = ties = wrong_wins = 0
    for observation in observations:
        key = observation.get("id")
        if not isinstance(key, str) or key not in expected or key in seen:
            raise ValueError(f"unknown or duplicate trial ID: {key}")
        model = observation.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError(f"missing model for {key}")
        seen.add(key)
        models.add(model)
        winner = _winner(observation.get("winner"))
        if winner is None:
            invalid += 1
            continue
        choice = _choice(winner, expected[key]["order"])
        picked[key] = choice
        ties += choice == "tie"
        wrong_wins += choice == "wrong"
    if len(models) > 1:
        raise ValueError("mixed model IDs")
    if require_complete and seen != expected.keys():
        raise ValueError(f"incomplete: {len(seen)}/{len(expected)}")
    pairs, flips = _position_pairs(expected, picked)
    return {
        "rows": len(seen),
        "valid": len(picked),
        "wrong_wins": wrong_wins,
        "ties": ties,
        "invalid": invalid,
        "paired": pairs,
        "flips": flips,
    }


def parse_response(raw_response: object) -> Winner | None:
    """Accept only a complete JSON verdict with a nonempty reason."""
    if not isinstance(raw_response, str):
        return None
    try:
        value: object = json.loads(raw_response)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or set(value) != {"winner", "reason"}:
        return None
    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip():
        return None
    return _winner(value["winner"])


def _checked_record(
    record: Mapping[str, object], expected: Mapping[str, RequestRow], seen: set[str]
) -> tuple[str, RequestRow, str, str]:
    key = record.get("id")
    if not isinstance(key, str) or key not in expected or key in seen:
        raise ValueError(f"unknown or duplicate trial ID: {key}")
    row = expected[key]
    for field, value in (
        ("request_sha256", row["request_sha256"]),
        ("prompt_sha256", PROMPT_SHA256),
        ("prompt_version", PROMPT_VERSION),
        ("judge_input", row["judge_input"]),
    ):
        if record.get(field) != value:
            raise ValueError(f"{field} mismatch: {key}")
    model = record.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"missing model: {key}")
    status = record.get("status")
    if status not in ("ok", "refusal", "timeout", "api_error"):
        raise ValueError(f"unknown status: {key}")
    assert isinstance(status, str)
    return key, row, model, status


def _group_counts(verdicts: Mapping[str, Winner | None], expected: Mapping[str, RequestRow]) -> dict[str, GroupCounts]:
    groups: dict[str, GroupCounts] = {
        name: {"valid": 0, "wrong_wins": 0, "ties": 0} for name in ("python", "http", "arithmetic", "short", "verbose")
    }
    for key, winner in verdicts.items():
        if winner is None:
            continue
        row = expected[key]
        choice = _choice(winner, row["order"])
        for name in (row["domain"], row["variant"]):
            group = groups[name]
            group["valid"] += 1
            group["wrong_wins"] += choice == "wrong"
            group["ties"] += choice == "tie"
    return groups


def _repeat_pairs(verdicts: Mapping[str, Winner | None], expected: Mapping[str, RequestRow]) -> tuple[int, int]:
    pairs = disagreements = 0
    for row in expected.values():
        if row["trial"] != 1 or verdicts.get(row["id"]) is None:
            continue
        peer = trial_id(row["case_id"], row["variant"], 2, row["order"])
        if verdicts.get(peer) is not None:
            pairs += 1
            disagreements += verdicts[row["id"]] != verdicts[peer]
    return pairs, disagreements


def analyze_recorded(records: Iterable[Mapping[str, object]]) -> RecordedSummary:
    """Validate the full frozen matrix and aggregate its raw JSON responses."""
    expected = _expected_rows()
    seen: set[str] = set()
    models: set[str] = set()
    statuses = {name: 0 for name in ("invalid_output", "refusal", "timeout", "api_error")}
    verdicts: dict[str, Winner | None] = {}
    observations: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("record must be a JSON object")
        key, _, model, status = _checked_record(record, expected, seen)
        seen.add(key)
        models.add(model)
        winner = parse_response(record.get("raw_response")) if status == "ok" else None
        if status == "ok" and winner is None:
            statuses["invalid_output"] += 1
        elif status != "ok":
            statuses[status] += 1
        verdicts[key] = winner
        observations.append({"id": key, "model": model, "winner": winner})
    if len(models) != 1:
        raise ValueError("mixed or missing model IDs")
    if seen != expected.keys():
        raise ValueError(f"incomplete: {len(seen)}/{len(expected)}")
    base = evaluate(observations, require_complete=True)
    repeat_pairs, disagreements = _repeat_pairs(verdicts, expected)
    return {
        "model": next(iter(models)),
        "attempts": base["rows"],
        "valid": base["valid"],
        "wrong_wins": base["wrong_wins"],
        "ties": base["ties"],
        "invalid_output": statuses["invalid_output"],
        "refusals": statuses["refusal"],
        "timeouts": statuses["timeout"],
        "api_errors": statuses["api_error"],
        "position_flips": base["flips"],
        "position_pairs": base["paired"],
        "repeat_disagreements": disagreements,
        "repeat_pairs": repeat_pairs,
        "groups": _group_counts(verdicts, expected),
    }
