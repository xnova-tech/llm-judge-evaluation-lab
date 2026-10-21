"""Offline control and recorded-response analysis; no inference calls."""
import hashlib
import json
import random
import sys
from pathlib import Path

# Each pair has a correct answer and a deliberately incorrect answer.
# Python values are checked by executing only hard-coded expressions below.
CASES = [
    ("py-01", "python", "Что выведет len([10, 20, 30])?", "3", "2"),
    ("py-02", "python", "Что выведет bool([])?", "False", "True"),
    ("py-03", "python", "Что выведет sum([1, 2, 3])?", "6", "7"),
    ("py-04", "python", "Что выведет len({'a': 1, 'b': 2})?", "2", "3"),
    ("py-05", "python", "Что выведет 'ab'.upper()?", "AB", "Ab"),
    ("py-06", "python", "Что выведет list(range(3))[-1]?", "2", "3"),
    ("math-01", "arithmetic", "Чему равно 48 / 6?", "8", "6"),
    ("math-02", "arithmetic", "Чему равно 17 * 4?", "68", "64"),
    ("math-03", "arithmetic", "Чему равно 19 + 23?", "42", "41"),
    ("math-04", "arithmetic", "Чему равно 13 ** 2?", "169", "156"),
    ("math-05", "arithmetic", "Чему равно 120 // 8?", "15", "16"),
    ("math-06", "arithmetic", "Чему равно 7 * 9 - 5?", "58", "62"),
    ("http-01", "http", "Что сообщает статус HTTP 204?", "Успех без содержимого ответа.", "Успех с обязательным JSON-телом."),
    ("http-02", "http", "Что сообщает статус HTTP 404?", "Сервер не нашёл представление ресурса или не раскрывает его наличие.", "Сервер успешно выполнил запрос и вернул найденный ресурс."),
    ("http-03", "http", "Что сообщает статус HTTP 401?", "Для целевого ресурса отсутствуют действительные учётные данные.", "Доступ гарантированно запрещён даже при корректных учётных данных."),
    ("http-04", "http", "Что сообщает статус HTTP 403?", "Сервер понял запрос, но отказывается выполнять его.", "Сервер обязательно предлагает пройти аутентификацию, как при 401."),
    ("http-05", "http", "Что сообщает статус HTTP 503?", "Сервис временно не может обработать запрос из-за перегрузки или обслуживания.", "Сервер навсегда удалил целевой ресурс."),
    ("http-06", "http", "Что сообщает статус HTTP 304?", "Условный запрос: выбранное представление не изменилось.", "Сервер прислал новое полное содержимое представления."),
]

PYTHON_ORACLES = [
    len([10, 20, 30]), bool([]), sum([1, 2, 3]),
    len({'a': 1, 'b': 2}), 'ab'.upper(), list(range(3))[-1],
]
MATH_ORACLES = [48 / 6, 17 * 4, 19 + 23, 13 ** 2, 120 // 8, 7 * 9 - 5]
WRONG_EXPLANATIONS = {
    'py-01': 'Здесь считаются промежутки между значениями, поэтому получается два.',
    'py-02': 'Созданный список считается истинным независимо от числа элементов.',
    'py-03': 'Функция sum по умолчанию прибавляет единицу к сумме элементов.',
    'py-04': 'Помимо двух ключей len учитывает сам словарь как третий объект.',
    'py-05': 'Метод upper меняет регистр только первой буквы строки.',
    'py-06': 'Верхняя граница range включается в последовательность.',
    'math-01': 'Проверка обратным действием даёт 6 × 6 = 48.',
    'math-02': 'Можно заменить 17 на ближайшее 16 и получить 16 × 4 = 64.',
    'math-03': 'Перенос единицы даёт 20 + 21 = 41.',
    'math-04': 'Квадрат 13 вычисляется умножением 13 × 12 = 156.',
    'math-05': 'Целочисленное деление округляет частное 15 вверх до 16.',
    'math-06': 'После умножения 7 × 9 = 63 остаётся вычесть 1, выходит 62.',
    'http-01': 'Код 204 требует JSON с подтверждением успешного действия.',
    'http-02': 'Код 404 означает успешную выдачу найденного представления.',
    'http-03': 'Код 401 окончательно запрещает доступ и после аутентификации.',
    'http-04': 'Код 403 требует нового запроса с учётными данными.',
    'http-05': 'Код 503 означает, что ресурс удалён без возможности восстановления.',
    'http-06': 'Код 304 передаёт новое полное тело представления.',
}
PROMPT_VERSION = 'correctness-v2'
PROMPT = (Path(__file__).parent / 'judge_prompt.txt').read_text(encoding='utf-8')
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode('utf-8')).hexdigest()
ORDER_SEED = 20261021


def check_gold():
    assert len(CASES) == 18
    assert len({row[0] for row in CASES}) == len(CASES)
    assert [row[3] for row in CASES[:6]] == [str(x) for x in PYTHON_ORACLES]
    assert [row[3] for row in CASES[6:12]] == [str(int(x)) for x in MATH_ORACLES]
    assert all(row[3] != row[4] for row in CASES)
    assert {row[1] for row in CASES} == {'python', 'arithmetic', 'http'}
    assert set(WRONG_EXPLANATIONS) == {row[0] for row in CASES}


def request_rows():
    rows = []
    for case_id, domain, question, correct, wrong in CASES:
        for variant in ('short', 'verbose'):
            erroneous = wrong if variant == 'short' else (
                wrong + ' ' + WRONG_EXPLANATIONS[case_id]
            )
            for trial in (1, 2):
                for order in ('correct_first', 'wrong_first'):
                    a, b = ((correct, erroneous) if order == 'correct_first' else (erroneous, correct))
                    judge_input = PROMPT.format(question=question, answer_a=a, answer_b=b)
                    rows.append({
                        'id': f'{case_id}:{variant}:{trial}:{order}',
                        'case_id': case_id, 'domain': domain,
                        'variant': variant, 'trial': trial, 'order': order,
                        'prompt_version': PROMPT_VERSION,
                        'prompt_sha256': PROMPT_SHA256,
                        'request_sha256': hashlib.sha256(judge_input.encode('utf-8')).hexdigest(),
                        'judge_input': judge_input,
                    })
    random.Random(ORDER_SEED).shuffle(rows)
    yield from rows


def evaluate(observations, require_complete):
    expected = {row['id']: row for row in request_rows()}
    seen = {}
    models = set()
    for o in observations:
        key = o['id']
        if key not in expected or key in seen:
            raise ValueError('unknown or duplicate trial ID: ' + key)
        if not isinstance(o.get('model'), str) or not o['model']:
            raise ValueError('missing model for ' + key)
        models.add(o['model'])
        seen[key] = o
    if len(models) > 1:
        raise ValueError('mixed model IDs')
    if require_complete and set(seen) != set(expected):
        raise ValueError(f'incomplete: {len(seen)}/{len(expected)}')
    valid = wrong_wins = ties = invalid = 0
    picked = {}
    for key, o in seen.items():
        winner = o.get('winner')
        if winner not in ('A', 'B', 'tie'):
            invalid += 1
            continue
        valid += 1
        if winner == 'tie':
            ties += 1
            picked[key] = 'tie'
            continue
        order = expected[key]['order']
        is_wrong = ((order == 'correct_first' and winner == 'B') or
                    (order == 'wrong_first' and winner == 'A'))
        wrong_wins += int(is_wrong)
        picked[key] = 'wrong' if is_wrong else 'correct'
    flip_pairs = paired = 0
    for key, row in expected.items():
        if row['order'] != 'correct_first' or key not in picked:
            continue
        peer = f"{row['case_id']}:{row['variant']}:{row['trial']}:wrong_first"
        if peer in picked:
            paired += 1
            flip_pairs += int(picked[key] != picked[peer])
    return {'rows': len(seen), 'valid': valid, 'wrong_wins': wrong_wins,
            'ties': ties, 'invalid': invalid, 'paired': paired, 'flips': flip_pairs}


def parse_response(raw_response):
    """Return a verdict only for a complete, unambiguous JSON response."""
    if not isinstance(raw_response, str):
        return None
    try:
        value = json.loads(raw_response)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or set(value) != {'winner', 'reason'}:
        return None
    if value['winner'] not in ('A', 'B', 'tie'):
        return None
    if not isinstance(value['reason'], str) or not value['reason'].strip():
        return None
    return value['winner']


def analyze_recorded(records):
    """Analyze a complete, version-bound recording without contacting a model."""
    expected = {row['id']: row for row in request_rows()}
    seen = {}
    models = set()
    statuses = {'ok': 0, 'invalid_output': 0, 'refusal': 0, 'timeout': 0, 'api_error': 0}
    parsed = []
    groups = {key: {'valid': 0, 'wrong_wins': 0, 'ties': 0}
              for key in ('python', 'http', 'arithmetic', 'short', 'verbose')}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('record must be a JSON object')
        key = record.get('id')
        if key not in expected or key in seen:
            raise ValueError('unknown or duplicate trial ID: ' + str(key))
        row = expected[key]
        if record.get('request_sha256') != row['request_sha256']:
            raise ValueError('request hash mismatch: ' + key)
        if record.get('prompt_sha256') != PROMPT_SHA256:
            raise ValueError('prompt hash mismatch: ' + key)
        if record.get('prompt_version') != PROMPT_VERSION:
            raise ValueError('prompt version mismatch: ' + key)
        if record.get('judge_input') != row['judge_input']:
            raise ValueError('judge input mismatch: ' + key)
        model = record.get('model')
        if not isinstance(model, str) or not model.strip():
            raise ValueError('missing model: ' + key)
        models.add(model)
        status = record.get('status')
        if status not in ('ok', 'refusal', 'timeout', 'api_error'):
            raise ValueError('unknown status: ' + key)
        seen[key] = record
        winner = parse_response(record.get('raw_response')) if status == 'ok' else None
        if status == 'ok' and winner is None:
            statuses['invalid_output'] += 1
        else:
            statuses[status] += 1
        parsed.append({'id': key, 'model': model, 'winner': winner})
        if winner is not None:
            wrong = winner != 'tie' and (
                (row['order'] == 'correct_first' and winner == 'B') or
                (row['order'] == 'wrong_first' and winner == 'A')
            )
            for group in (row['domain'], row['variant']):
                groups[group]['valid'] += 1
                groups[group]['wrong_wins'] += int(wrong)
                groups[group]['ties'] += int(winner == 'tie')
    if len(models) != 1:
        raise ValueError('mixed or missing model IDs')
    if set(seen) != set(expected):
        raise ValueError(f'incomplete: {len(seen)}/{len(expected)}')
    base = evaluate(parsed, True)
    picked = {item['id']: item['winner'] for item in parsed if item['winner'] is not None}
    repeat_pairs = repeat_disagreements = 0
    for row in expected.values():
        if row['trial'] != 1 or row['id'] not in picked:
            continue
        peer = f"{row['case_id']}:{row['variant']}:2:{row['order']}"
        if peer in picked:
            repeat_pairs += 1
            repeat_disagreements += int(picked[row['id']] != picked[peer])
    return {
        'model': next(iter(models)), 'attempts': base['rows'], 'valid': base['valid'],
        'wrong_wins': base['wrong_wins'], 'ties': base['ties'],
        'invalid_output': statuses['invalid_output'], 'refusals': statuses['refusal'],
        'timeouts': statuses['timeout'], 'api_errors': statuses['api_error'],
        'position_flips': base['flips'], 'position_pairs': base['paired'],
        'repeat_disagreements': repeat_disagreements, 'repeat_pairs': repeat_pairs,
        'groups': groups,
    }


def fixture():
    # Intentional, illustrative wrong decisions. No LLM was called.
    rows = [
        ('math-01:short:1:correct_first', 'B'),
        ('math-01:short:1:wrong_first', 'B'),
        ('http-01:short:1:correct_first', 'B'),
        ('http-01:short:1:wrong_first', 'A'),
    ]
    return [{'id': id_, 'model': 'synthetic-control-NOT-LLM', 'winner': verdict}
            for id_, verdict in rows]


def self_test():
    check_gold()
    assert len(list(request_rows())) == 144
    result = evaluate(fixture(), False)
    assert result == {'rows': 4, 'valid': 4, 'wrong_wins': 3,
                      'ties': 0, 'invalid': 0, 'paired': 2, 'flips': 1}
    assert evaluate([{**fixture()[0], 'winner': 'tie'}], False)['ties'] == 1
    assert evaluate([{**fixture()[0], 'winner': 'nonsense'}], False)['invalid'] == 1
    for invalid in ((fixture()[0], fixture()[0]),
                    ({'id': 'missing', 'model': 'x', 'winner': 'A'},)):
        try:
            evaluate(invalid, False)
        except ValueError:
            pass
        else:
            raise AssertionError('bad rows accepted')
    try:
        evaluate(fixture(), True)
    except ValueError:
        pass
    else:
        raise AssertionError('incomplete dataset accepted')
    print('self_tests=8 passed')


def main(args):
    if args == ['--self-test']:
        self_test()
        return
    check_gold()
    if args == ['--emit-requests']:
        for item in request_rows():
            print(json.dumps(item, ensure_ascii=False, sort_keys=True))
        return
    if len(args) == 2 and args[0] == '--analyze-recorded':
        path = Path(args[1])
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        print(json.dumps(analyze_recorded(rows), ensure_ascii=False, sort_keys=True))
        return
    if args:
        raise SystemExit('usage: eval_harness.py [--self-test|--emit-requests|--analyze-recorded PATH]')
    out = evaluate(fixture(), False)
    print(f'gold_cases={len(CASES)}')
    print(f'planned_trials={len(list(request_rows()))}')
    print('llm_calls=0')
    print('verdicts=synthetic_control_not_llm')
    print(f"control_wrong_wins={out['wrong_wins']}/{out['valid']}")
    print(f"control_order_flips={out['flips']}/{out['paired']}")


if __name__ == '__main__':
    main(sys.argv[1:])
