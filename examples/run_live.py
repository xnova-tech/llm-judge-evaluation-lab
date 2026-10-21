"""Optional live recorder; offline verification never executes this module."""

import argparse
import datetime as dt
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NotRequired, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_matrix import (  # noqa: E402
    MAX_COMPLETION_TOKENS,
    ORDER_SEED,
    RequestRow,
    request_payload,
    request_rows,
)

TIMEOUT_SECONDS = 60
USD_PER_M_INPUT = 2.0
USD_PER_M_OUTPUT = 12.0
MAX_BUDGET_USD = 20.0
PRICE_MULTIPLIER_RESERVE = 10.0


@dataclass(frozen=True, slots=True)
class ApiConfig:
    url: str
    model: str
    key: str = field(repr=False)
    ca_bundle: str | None = None

    @classmethod
    def from_environment(cls) -> "ApiConfig":
        return cls(
            url=os.environ["JUDGE_API_URL"],
            model=os.environ["JUDGE_MODEL"],
            key=os.environ["JUDGE_API_KEY"],
            ca_bundle=os.environ.get("JUDGE_CA_BUNDLE"),
        )


class ApiResult(TypedDict):
    status: str
    raw_response: str
    model: NotRequired[str]
    response_id: NotRequired[object]
    created: NotRequired[object]
    finish_reason: NotRequired[object]
    usage: NotRequired[object]
    refusal: NotRequired[object]
    http_status: NotRequired[int]
    transport_error_type: NotRequired[str]
    malformed_api_response: NotRequired[bool]


def upper_cost_usd(rows: Sequence[RequestRow]) -> float:
    """Estimate a conservative reserve from UTF-8 bytes and output limits."""
    input_bound = sum(len(row["judge_input"].encode("utf-8")) + 200 for row in rows)
    output_bound = len(rows) * MAX_COMPLETION_TOKENS
    return PRICE_MULTIPLIER_RESERVE * (input_bound * USD_PER_M_INPUT + output_bound * USD_PER_M_OUTPUT) / 1_000_000


def _api_response(body: object) -> ApiResult:
    if not isinstance(body, dict):
        return {"status": "api_error", "raw_response": "", "malformed_api_response": True}
    try:
        choice = body["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        refusal = message.get("refusal")
        model = body["model"]
        if not isinstance(content, str) or not isinstance(model, str):
            raise TypeError("unexpected API field")
        return {
            "status": "refusal" if refusal else "ok",
            "raw_response": content,
            "model": model,
            "response_id": body.get("id"),
            "created": body.get("created"),
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage"),
            "refusal": refusal,
        }
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return {"status": "api_error", "raw_response": "", "malformed_api_response": True}


def send(row: RequestRow, config: ApiConfig) -> ApiResult:
    payload = request_payload(row, config.model)
    request = urllib.request.Request(
        config.url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + config.key, "Content-Type": "application/json"},
        method="POST",
    )
    context = ssl.create_default_context(cafile=config.ca_bundle)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS, context=context) as response:
            body: object = json.load(response)
    except urllib.error.HTTPError as error:
        return {"status": "api_error", "http_status": error.code, "raw_response": ""}
    except (TimeoutError, urllib.error.URLError) as error:
        status = "timeout" if isinstance(error, TimeoutError) else "api_error"
        return {"status": status, "transport_error_type": type(error).__name__, "raw_response": ""}
    return _api_response(body)


def _record(row: RequestRow, result: ApiResult, model: str) -> dict[str, object]:
    record: dict[str, object] = dict(row)
    record.update(
        {
            "requested_model": model,
            "model": result.get("model", model),
            "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
            "api_style": "chat_completions",
            "reasoning_effort": "none",
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "response_format": "json_object",
        }
    )
    record.update(result)
    return record


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=144)
    args = parser.parse_args(argv)
    if not (1 <= args.limit <= 144):
        parser.error("--limit must be between 1 and 144")
    if args.output.exists():
        parser.error("output already exists; no implicit overwrite or retry")
    config = ApiConfig.from_environment()
    rows = list(request_rows())[: args.limit]
    reserve = upper_cost_usd(rows)
    if reserve > MAX_BUDGET_USD:
        parser.error("token reserve exceeds the approved budget")
    print(
        json.dumps(
            {
                "planned_calls": len(rows),
                "max_completion_tokens": MAX_COMPLETION_TOKENS,
                "order_seed": ORDER_SEED,
                "official_price_10x_reserve_usd": round(reserve, 4),
            }
        ),
        flush=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for index, row in enumerate(rows, 1):
            record = _record(row, send(row, config), config.model)
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            print(f"completed={index}/{len(rows)} status={record['status']}", flush=True)
            if record["status"] == "api_error" and record.get("http_status") in (401, 403, 404):
                raise SystemExit("authentication or model access failed; stopped without retry")


if __name__ == "__main__":
    main()
