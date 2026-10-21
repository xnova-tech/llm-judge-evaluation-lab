# ruff: noqa: T201  # CLI commands intentionally print records and summaries
"""Opt-in live recorder for the independent stress-v2 cohort. NEVER used by article verify."""

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTICLE = HERE.parent
sys.path.insert(0, str(HERE))
MAX_COMPLETION_TOKENS = 256
from v2_harness import SEED, generation_rows, request_rows  # noqa: E402 - required with -I


def bound_usd(rows, input_rate, output_rate):
    # Conservative preflight upper-bound proxy, not a guaranteed vendor invoice.
    input_proxy = sum(len(r["judge_input"].encode("utf-8")) + 200 for r in rows)
    output_limit = len(rows) * MAX_COMPLETION_TOKENS
    return 2 * (input_proxy * input_rate + output_limit * output_rate) / 1_000_000


def main():
    p = argparse.ArgumentParser(description="No retries, no overwrite; opt-in spend approval required")
    p.add_argument("--plan", action="store_true", help="Print matrix size only; no API calls")
    p.add_argument("--output", type=Path, help="Fresh JSONL path for 192 live requests")
    args = p.parse_args()
    rows = request_rows()
    if args.plan:
        print(
            json.dumps(
                {"phase": "stress-v3", "calls": len(rows), "seed": SEED, "model": "must_be_set_at_run"}, sort_keys=True
            )
        )
        return
    if args.output is None:
        p.error("--output required; --plan is the only zero-cost mode")
    if args.output.exists():
        p.error("output exists; no overwrite or implicit retry")
    for env in (
        "JUDGE_API_URL",
        "JUDGE_API_KEY",
        "JUDGE_MODEL",
        "JUDGE_V2_APPROVED_BUDGET_USD",
        "JUDGE_V2_INPUT_USD_PER_M",
        "JUDGE_V2_OUTPUT_USD_PER_M",
    ):
        if not os.getenv(env):
            p.error("missing required environment variable: " + env)
    url = os.environ["JUDGE_API_URL"]
    if not url.startswith("https://"):
        p.error("HTTPS required")
    model = os.environ["JUDGE_MODEL"]
    rates = (float(os.environ["JUDGE_V2_INPUT_USD_PER_M"]), float(os.environ["JUDGE_V2_OUTPUT_USD_PER_M"]))
    budget = float(os.environ["JUDGE_V2_APPROVED_BUDGET_USD"])
    reserve = bound_usd(rows, *rates) + bound_usd(generation_rows(), *rates)
    if min(*rates, budget) <= 0 or reserve > budget:
        p.error(f"unapproved or insufficient spend cap; 2x proxy reserve={reserve:.4f} USD")
    print(
        json.dumps(
            {
                "calls": len(rows),
                "model": model,
                "input_output_rates_usd_per_m": rates,
                "2x_combined_study_reserve_usd": round(reserve, 4),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ARTICLE / "examples"))
    from run_live import ApiConfig, send  # Reuse the existing adapter only after explicit run approval.

    config = ApiConfig(
        url=url, model=model, key=os.environ["JUDGE_API_KEY"], ca_bundle=os.environ.get("JUDGE_CA_BUNDLE")
    )
    with args.output.open("x", encoding="utf-8") as dst:
        for index, row in enumerate(rows, 1):
            result = send(row, config)
            rec = {
                **row,
                "requested_model": model,
                "model": result.get("model", model),
                "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
                "api_style": "chat_completions",
                "reasoning_effort": "none",
                "max_completion_tokens": MAX_COMPLETION_TOKENS,
                "response_format": "json_object",
                **result,
            }
            dst.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            dst.flush()
            print(f"completed={index}/{len(rows)} status={rec['status']}", flush=True)
            if rec["status"] == "api_error" and rec.get("http_status") in (401, 403, 404, 429):
                raise SystemExit("provider failure; stop without retry or reinterpretation")


if __name__ == "__main__":
    main()
