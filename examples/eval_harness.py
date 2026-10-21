"""Offline command line entry point; this module never calls the model."""

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

# Isolated Python (-I) does not add the script directory to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_analysis import analyze_recorded, evaluate  # noqa: E402
from experiment_cases import CASES, check_gold  # noqa: E402
from experiment_matrix import request_rows  # noqa: E402


def fixture() -> list[dict[str, object]]:
    """Deliberately incorrect verdicts for an analyzer control, not model data."""
    verdicts = (
        ("math-01:short:1:correct_first", "B"),
        ("math-01:short:1:wrong_first", "B"),
        ("http-01:short:1:correct_first", "B"),
        ("http-01:short:1:wrong_first", "A"),
    )
    return [{"id": key, "model": "synthetic-control-NOT-LLM", "winner": winner} for key, winner in verdicts]


def _expect_rejection(observations: Sequence[Mapping[str, object]], complete: bool) -> None:
    try:
        evaluate(observations, require_complete=complete)
    except ValueError:
        return
    raise AssertionError("invalid observations accepted")


def self_test() -> None:
    check_gold()
    assert len(list(request_rows())) == 144
    assert evaluate(fixture(), False) == {
        "rows": 4,
        "valid": 4,
        "wrong_wins": 3,
        "ties": 0,
        "invalid": 0,
        "paired": 2,
        "flips": 1,
    }
    first = fixture()[0]
    assert evaluate([{**first, "winner": "tie"}], False)["ties"] == 1
    assert evaluate([{**first, "winner": "nonsense"}], False)["invalid"] == 1
    _expect_rejection([first, first], complete=False)
    _expect_rejection([{"id": "missing", "model": "x", "winner": "A"}], complete=False)
    _expect_rejection(fixture(), complete=True)
    print("self_tests=8 passed")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value: object = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("record must be a JSON object")
        records.append(value)
    return records


def main(args: Sequence[str]) -> None:
    if list(args) == ["--self-test"]:
        self_test()
        return
    check_gold()
    if list(args) == ["--emit-requests"]:
        for row in request_rows():
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))
        return
    if len(args) == 2 and args[0] == "--analyze-recorded":
        recorded = analyze_recorded(_read_jsonl(Path(args[1])))
        print(json.dumps(recorded, ensure_ascii=False, sort_keys=True))
        return
    if args:
        raise SystemExit("usage: eval_harness.py [--self-test|--emit-requests|--analyze-recorded PATH]")
    control = evaluate(fixture(), require_complete=False)
    print(f"gold_cases={len(CASES)}")
    print(f"planned_trials={sum(1 for _ in request_rows())}")
    print("llm_calls=0")
    print("verdicts=synthetic_control_not_llm")
    print(f"control_wrong_wins={control['wrong_wins']}/{control['valid']}")
    print(f"control_order_flips={control['flips']}/{control['paired']}")


if __name__ == "__main__":
    main(sys.argv[1:])
