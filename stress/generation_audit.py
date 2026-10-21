# ruff: noqa: T201  # CLI commands intentionally print records and summaries
"""Review-bound generation inventory: validates trace integrity, never invents truth labels."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_harness import generation_rows  # noqa: E402 - isolated entry point needs its local module


def audit(path):  # noqa: C901 - sequential checks preserve the record-integrity contract
    rows = [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    expected = {r["id"]: r for r in generation_rows()}
    seen = set()
    models = set()
    response_ids = set()
    statuses = {"ok": 0, "invalid_output": 0, "refusal": 0, "timeout": 0, "api_error": 0}
    for rec in rows:
        key = rec.get("id")
        if key not in expected or key in seen:
            raise ValueError("unknown/duplicate ID " + str(key))
        seen.add(key)
        for f in ("dataset_sha256", "prompt_sha256", "prompt_version", "request_sha256", "judge_input"):
            if rec.get(f) != expected[key][f]:
                raise ValueError("tampered " + f + " for " + key)
        model = rec.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("missing model")
        models.add(model)
        status = rec.get("status")
        if status not in ("ok", "refusal", "timeout", "api_error"):
            raise ValueError("invalid status")
        if status == "ok":
            response_id = rec.get("response_id")
            if response_id is not None:
                if not isinstance(response_id, str) or response_id in response_ids:
                    raise ValueError("duplicate/malformed response ID")
                response_ids.add(response_id)
            try:
                parsed = json.loads(rec.get("raw_response", ""))
            except (TypeError, ValueError):
                parsed = None
            if (
                not isinstance(parsed, dict)
                or set(parsed) != {"answer", "evidence", "confidence"}
                or not isinstance(parsed.get("answer"), str)
                or not parsed["answer"].strip()
                or not isinstance(parsed.get("evidence"), str)
                or not parsed["evidence"].strip()
                or parsed.get("confidence") not in ("low", "medium", "high")
            ):
                statuses["invalid_output"] += 1
            else:
                statuses["ok"] += 1
        else:
            statuses[status] += 1
    if len(models) != 1:
        raise ValueError("mixed models")
    if seen != set(expected):
        raise ValueError("incomplete generation matrix: " + str(len(seen)) + "/24")
    return {
        "phase": "stress-v3-generation",
        "attempts": len(rows),
        "statuses": statuses,
        "truth_quality": "NOT VERIFIED: case-level factual adjudication required for all 24 responses",
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: generation_audit.py FILE")
    print(json.dumps(audit(sys.argv[1]), ensure_ascii=False, sort_keys=True))
