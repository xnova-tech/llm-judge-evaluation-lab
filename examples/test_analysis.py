"""Regression checks for the frozen requests and offline analyzer."""

import hashlib
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_harness import fixture  # noqa: E402
from experiment_analysis import analyze_recorded, evaluate, parse_response  # noqa: E402
from experiment_cases import check_gold  # noqa: E402
from experiment_matrix import (  # noqa: E402
    PROMPT_SHA256,
    PROMPT_VERSION,
    RequestRow,
    request_payload,
    request_rows,
)

FROZEN_MATRIX_SHA256 = "7603ee046f912aff60acf463ffe48dbeb6f68757378c15568a70e82117cbfd5e"


def complete_control() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for request in request_rows():
        winner = "A" if request["order"] == "correct_first" else "B"
        records.append(
            {
                "id": request["id"],
                "model": "synthetic-control-NOT-LLM",
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": PROMPT_SHA256,
                "request_sha256": request["request_sha256"],
                "judge_input": request["judge_input"],
                "status": "ok",
                "raw_response": json.dumps({"winner": winner, "reason": "Independent gold control."}),
            }
        )
    return records


class AnalysisTests(unittest.TestCase):
    def test_gold_and_frozen_matrix_have_no_label_leak(self) -> None:
        check_gold()
        rows = list(request_rows())
        self.assertEqual(len(rows), 144)
        self.assertEqual(len({row["id"] for row in rows}), 144)
        self.assertEqual({row["domain"] for row in rows}, {"python", "http", "arithmetic"})
        encoded = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
        self.assertEqual(hashlib.sha256(encoded.encode("utf-8")).hexdigest(), FROZEN_MATRIX_SHA256)
        for row in rows:
            self.assertNotIn("gold_label", row["judge_input"])
            self.assertNotIn("correct_first", row["judge_input"])
            self.assertNotIn(row["case_id"], row["judge_input"])

    def test_live_payload_sends_only_judge_text_and_frozen_parameters(self) -> None:
        row: RequestRow = next(request_rows())
        payload = request_payload(row, "test-model")
        self.assertEqual(
            set(payload), {"model", "messages", "max_completion_tokens", "reasoning_effort", "response_format"}
        )
        self.assertEqual(payload["messages"], [{"role": "user", "content": row["judge_input"]}])
        self.assertEqual(payload["max_completion_tokens"], 256)
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_complete_correct_control(self) -> None:
        result = analyze_recorded(complete_control())
        self.assertEqual((result["attempts"], result["valid"], result["wrong_wins"]), (144, 144, 0))
        self.assertEqual((result["position_flips"], result["position_pairs"]), (0, 72))
        self.assertEqual((result["repeat_disagreements"], result["repeat_pairs"]), (0, 72))
        self.assertEqual([result["groups"][name]["valid"] for name in ("python", "http", "arithmetic")], [48, 48, 48])
        self.assertEqual([result["groups"][name]["valid"] for name in ("short", "verbose")], [72, 72])

    def test_wrong_winner_and_reversal_use_candidate_identity(self) -> None:
        records = complete_control()
        target = next(row for row in records if row["id"] == "py-01:short:1:correct_first")
        target["raw_response"] = '{"winner":"B","reason":"Deliberately wrong control."}'
        result = analyze_recorded(records)
        self.assertEqual(result["wrong_wins"], 1)
        self.assertEqual(result["position_flips"], 1)
        self.assertEqual(result["repeat_disagreements"], 1)
        self.assertEqual(evaluate(fixture(), False)["wrong_wins"], 3)

    def test_tie_is_valid_but_not_a_wrong_win(self) -> None:
        records = complete_control()
        records[0]["raw_response"] = '{"winner":"tie","reason":"Control tie."}'
        result = analyze_recorded(records)
        self.assertEqual((result["valid"], result["ties"], result["wrong_wins"]), (144, 1, 0))

    def test_invalid_json_and_schema_do_not_enter_valid_denominator(self) -> None:
        for bad in (
            "not json",
            '{"winner":"A"}',
            '{"winner":"C","reason":"x"}',
            '{"winner":"A","reason":""}',
            '{"winner":"A","reason":"x","extra":1}',
        ):
            with self.subTest(bad=bad):
                self.assertIsNone(parse_response(bad))
        records = complete_control()
        records[0]["raw_response"] = "not json"
        result = analyze_recorded(records)
        self.assertEqual((result["valid"], result["invalid_output"]), (143, 1))
        self.assertEqual(result["position_pairs"], 71)

    def test_refusal_timeout_and_api_error_stay_separate(self) -> None:
        records = complete_control()
        for record, status in zip(records[:3], ("refusal", "timeout", "api_error"), strict=True):
            record["status"] = status
            record["raw_response"] = ""
        result = analyze_recorded(records)
        self.assertEqual(
            (result["valid"], result["refusals"], result["timeouts"], result["api_errors"]), (141, 1, 1, 1)
        )
        self.assertEqual(result["invalid_output"], 0)

    def test_incomplete_duplicate_unknown_and_mixed_model_rejected(self) -> None:
        records = complete_control()
        corruptions = (
            records[:-1],
            records + [records[0]],
            [{**records[0], "id": "unknown"}] + records[1:],
            [{**records[0], "model": "other-model"}] + records[1:],
        )
        for corrupted in corruptions:
            with self.assertRaises(ValueError):
                analyze_recorded(corrupted)

    def test_stale_prompt_or_request_rejected(self) -> None:
        records = complete_control()
        for changed in (
            {"prompt_version": "old"},
            {"prompt_sha256": "0" * 64},
            {"request_sha256": "0" * 64},
            {"judge_input": "changed"},
        ):
            with self.assertRaises(ValueError):
                analyze_recorded([{**records[0], **changed}] + records[1:])


if __name__ == "__main__":
    unittest.main()
