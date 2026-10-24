"""Offline negative controls. Do not invoke any model or network."""

import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest
from generation_audit import audit
from v2_harness import Record, analyze, generation_rows, load_cases, request_rows, synthetic_control


class TestStressV2:
    def test_frozen_matrix_and_gold(self) -> None:
        cases = load_cases()
        assert len(cases) == 24
        rows = request_rows()
        assert len(rows) == 192
        assert len({r["id"] for r in rows}) == 192
        assert rows == request_rows()
        assert len(generation_rows()) == 24
        for row in rows:
            for secret in ("gold_note", "correct_first", "wrong_first", "oracle", "case_id"):
                assert secret not in row["judge_input"]

    def test_zero_errors_synthetic_is_not_llm(self) -> None:
        r = analyze(synthetic_control())
        assert (r["attempts"], r["valid"], r["correct_wins"], r["wrong_wins"]) == (192, 192, 192, 0)
        assert (r["position_pairs"], r["repeat_pairs"]) == (96, 96)

    def test_wrong_answer_is_detected(self) -> None:
        data = synthetic_control()
        rec = next(x for x in data if x["id"] == "code-01:short:1:correct_first")
        rec["raw_response"] = '{"winner":"B","reason":"negative"}'
        r = analyze(data)
        assert (r["wrong_wins"], r["position_flips"], r["repeat_disagreements"]) == (1, 1, 1)

    def test_tie_and_invalid_and_refusal(self) -> None:
        data = synthetic_control()
        data[0]["raw_response"] = '{"winner":"tie","reason":"indeterminate"}'
        data[1]["raw_response"] = '{"winner":"A"}'
        data[2]["status"] = "refusal"
        r = analyze(data)
        assert (r["ties"], r["invalid_output"], r["refusals"], r["valid"]) == (1, 1, 1, 190)

    def test_duplicate_and_missing_are_rejected(self) -> None:
        data = synthetic_control()
        for altered in (data[:-1], data + [data[0]]):
            with pytest.raises(ValueError):
                analyze(altered)

    def test_tampered_hashes_are_rejected(self) -> None:
        for field in ("dataset_sha256", "request_sha256", "prompt_sha256", "judge_input"):
            data = synthetic_control()
            data[0][field] = "tampered"
            with pytest.raises(ValueError):
                analyze(data)

    def test_generation_trace_integrity_not_truth(self) -> None:
        records = [
            {
                **r,
                "model": "SYNTHETIC-CONTROL-NOT-LLM",
                "status": "ok",
                "raw_response": json.dumps({"answer": "synthetic", "evidence": "test only", "confidence": "low"}),
            }
            for r in generation_rows()
        ]
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "generation.jsonl"
            f.write_text("\n".join(json.dumps(x) for x in records) + "\n", encoding="utf-8")
            result = audit(f)
            assert result["attempts"] == 24
            assert "NOT VERIFIED" in result["truth_quality"]
            records[0]["request_sha256"] = "tampered"
            f.write_text("\n".join(json.dumps(x) for x in records) + "\n", encoding="utf-8")
            with pytest.raises(ValueError):
                audit(f)

    def test_reused_response_id_is_rejected(self) -> None:
        data = synthetic_control()
        data[0]["response_id"] = "duplicate"
        data[1]["response_id"] = "duplicate"
        with pytest.raises(ValueError):
            analyze(data)

    def test_live_entry_points_with_offline_adapter(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Exercise both recorder paths with a fake transport and no API call."""
        monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
        import run_generate_v2
        import run_live  # type: ignore[import-not-found]  # Dynamic sibling CLI path.
        import run_live_v2

        calls = 0

        def fake_send(row: Record, config: Any) -> Record:
            nonlocal calls
            calls += 1
            if "order" in row:
                winner = "A" if row["order"] == "correct_first" else "B"
                raw = {"winner": winner, "reason": "offline adapter control"}
            else:
                raw = {"answer": "synthetic", "evidence": "offline adapter control", "confidence": "low"}
            return {
                "status": "ok",
                "model": config.model,
                "response_id": f"fake-{calls}",
                "raw_response": json.dumps(raw),
            }

        config = {
            "JUDGE_API_URL": "https://example.invalid/v1",
            "JUDGE_API_KEY": "offline-only",
            "JUDGE_MODEL": "SYNTHETIC-CONTROL-NOT-LLM",
            "JUDGE_V2_APPROVED_BUDGET_USD": "5",
            "JUDGE_V2_INPUT_USD_PER_M": "2",
            "JUDGE_V2_OUTPUT_USD_PER_M": "12",
        }
        for key, value in config.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setattr(run_live, "send", fake_send)
        generation = tmp_path / "generation.jsonl"
        judge = tmp_path / "judge.jsonl"
        monkeypatch.setattr(sys, "argv", ["run_generate_v2.py", "--output", str(generation)])
        with redirect_stdout(io.StringIO()):
            run_generate_v2.main()
        monkeypatch.setattr(sys, "argv", ["run_live_v2.py", "--output", str(judge)])
        with redirect_stdout(io.StringIO()):
            run_live_v2.main()
        assert calls == 216
        assert audit(generation)["statuses"]["ok"] == 24
        assert analyze([json.loads(line) for line in judge.read_text().splitlines()])["correct_wins"] == 192
