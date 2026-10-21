"""Optional live recorder. Offline verification never imports or executes this file."""
import argparse
import datetime as dt
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_harness import ORDER_SEED, request_rows

MAX_COMPLETION_TOKENS = 256
TIMEOUT_SECONDS = 60
USD_PER_M_INPUT = 2.0
USD_PER_M_OUTPUT = 12.0
MAX_BUDGET_USD = 20.0
PRICE_MULTIPLIER_RESERVE = 10.0


def upper_cost_usd(rows):
    """Conservative token reserve: UTF-8 bytes bound text token count."""
    input_bound = sum(len(row['judge_input'].encode('utf-8')) + 200 for row in rows)
    output_bound = len(rows) * MAX_COMPLETION_TOKENS
    return PRICE_MULTIPLIER_RESERVE * (
        input_bound * USD_PER_M_INPUT + output_bound * USD_PER_M_OUTPUT
    ) / 1_000_000


def send(row, api_url, api_key, model):
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': row['judge_input']}],
        'max_completion_tokens': MAX_COMPLETION_TOKENS,
        'reasoning_effort': 'none',
        'response_format': {'type': 'json_object'},
    }
    request = urllib.request.Request(
        api_url.rstrip('/') + '/chat/completions',
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json'},
        method='POST',
    )
    ca_bundle = os.environ.get('JUDGE_CA_BUNDLE')
    context = ssl.create_default_context(cafile=ca_bundle) if ca_bundle else ssl.create_default_context()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS, context=context) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        return {'status': 'api_error', 'http_status': error.code, 'raw_response': ''}
    except (TimeoutError, urllib.error.URLError) as error:
        return {'status': 'timeout' if isinstance(error, TimeoutError) else 'api_error',
                'transport_error_type': type(error).__name__, 'raw_response': ''}
    try:
        choice = body['choices'][0]
        message = choice['message']
        content = message.get('content') or ''
        refusal = message.get('refusal')
        return {
            'status': 'refusal' if refusal else 'ok',
            'raw_response': content,
            'model': body['model'],
            'response_id': body.get('id'),
            'created': body.get('created'),
            'finish_reason': choice.get('finish_reason'),
            'usage': body.get('usage'),
            'refusal': refusal,
        }
    except (KeyError, IndexError, TypeError, ValueError):
        return {'status': 'api_error', 'raw_response': '', 'malformed_api_response': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=144)
    args = parser.parse_args()
    if not (1 <= args.limit <= 144):
        parser.error('--limit must be between 1 and 144')
    if args.output.exists():
        parser.error('output already exists; no implicit overwrite or retry')
    api_url = os.environ['JUDGE_API_URL']
    api_key = os.environ['JUDGE_API_KEY']
    model = os.environ['JUDGE_MODEL']
    rows = list(request_rows())[:args.limit]
    reserve = upper_cost_usd(rows)
    if reserve > MAX_BUDGET_USD:
        parser.error('token reserve exceeds the approved budget')
    print(json.dumps({'planned_calls': len(rows), 'max_completion_tokens': MAX_COMPLETION_TOKENS,
                      'order_seed': ORDER_SEED, 'official_price_10x_reserve_usd': round(reserve, 4)}),
          flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        for index, row in enumerate(rows, 1):
            result = send(row, api_url, api_key, model)
            record = {
                **row,
                'requested_model': model,
                'model': result.pop('model', model),
                'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'api_style': 'chat_completions',
                'reasoning_effort': 'none',
                'max_completion_tokens': MAX_COMPLETION_TOKENS,
                'response_format': 'json_object',
                **result,
            }
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
            output.flush()
            print(f"completed={index}/{len(rows)} status={record['status']}", flush=True)
            if record['status'] == 'api_error' and record.get('http_status') in (401, 403, 404):
                raise SystemExit('authentication or model access failed; stopped without retry')


if __name__ == '__main__':
    main()
