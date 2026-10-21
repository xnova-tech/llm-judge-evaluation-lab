# ruff: noqa: T201  # CLI commands intentionally print records and summaries
"""Separate opt-in 24-request generation arm. No automatic correctness claims."""

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTICLE = HERE.parent
sys.path.insert(0, str(HERE))
from run_live_v2 import bound_usd  # noqa: E402 - required with -I
from v2_harness import generation_rows, request_rows  # noqa: E402 - required with -I


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    rows = generation_rows()
    if args.plan:
        print(
            json.dumps(
                {"phase": "stress-v3-generation", "calls": len(rows), "factual_review_required": True},
                sort_keys=True,
            )
        )
        return
    if args.output is None or args.output.exists():
        p.error("new --output required; do not overwrite or retry")
    for env in (
        "JUDGE_API_URL",
        "JUDGE_API_KEY",
        "JUDGE_MODEL",
        "JUDGE_V2_APPROVED_BUDGET_USD",
        "JUDGE_V2_INPUT_USD_PER_M",
        "JUDGE_V2_OUTPUT_USD_PER_M",
    ):
        if not os.getenv(env):
            p.error("missing environment variable: " + env)
    if not os.environ["JUDGE_API_URL"].startswith("https://"):
        p.error("HTTPS required")
    input_rate = float(os.environ["JUDGE_V2_INPUT_USD_PER_M"])
    output_rate = float(os.environ["JUDGE_V2_OUTPUT_USD_PER_M"])
    reserve = bound_usd(rows, input_rate, output_rate) + bound_usd(request_rows(), input_rate, output_rate)
    budget = float(os.environ["JUDGE_V2_APPROVED_BUDGET_USD"])
    if min(input_rate, output_rate, budget) <= 0 or reserve > budget:
        p.error("approved budget insufficient for 2x combined token proxy reserve")
    sys.path.insert(0, str(ARTICLE / "examples"))
    from run_live import MAX_COMPLETION_TOKENS, ApiConfig, send

    config = ApiConfig(
        url=os.environ["JUDGE_API_URL"],
        model=os.environ["JUDGE_MODEL"],
        key=os.environ["JUDGE_API_KEY"],
        ca_bundle=os.environ.get("JUDGE_CA_BUNDLE"),
    )
    print(
        json.dumps(
            {
                "calls": len(rows),
                "model": config.model,
                "input_output_rates_usd_per_m": [input_rate, output_rate],
                "2x_combined_study_reserve_usd": round(reserve, 4),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as dst:
        for idx, row in enumerate(rows, 1):
            resp = send(row, config)
            record = {
                **row,
                "model": resp.get("model", config.model),
                "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
                "requested_model": config.model,
                "phase": "stress-v3-generation",
                "api_style": "chat_completions",
                "reasoning_effort": "none",
                "max_completion_tokens": MAX_COMPLETION_TOKENS,
                "response_format": "json_object",
                **resp,
            }
            dst.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            dst.flush()
            print(f"completed={idx}/{len(rows)} status={record['status']}", flush=True)
            if record["status"] == "api_error" and record.get("http_status") in (401, 403, 404, 429):
                raise SystemExit("stop on provider access/limit error, no retry")


if __name__ == "__main__":
    main()
