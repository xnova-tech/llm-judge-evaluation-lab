"""Negative controls for the offline judge-recording contract."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_harness import (
    PROMPT_SHA256, PROMPT_VERSION, analyze_recorded, check_gold, evaluate, fixture,
    parse_response, request_rows,
)


def complete_control():
    rows = []
    for request in request_rows():
        winner = 'A' if request['order'] == 'correct_first' else 'B'
        rows.append({
            'id': request['id'], 'model': 'synthetic-control-NOT-LLM',
            'prompt_version': PROMPT_VERSION,
            'prompt_sha256': PROMPT_SHA256,
            'request_sha256': request['request_sha256'],
            'judge_input': request['judge_input'],
            'status': 'ok',
            'raw_response': json.dumps({'winner': winner, 'reason': 'Independent gold control.'}),
        })
    return rows


class AnalysisTests(unittest.TestCase):
    def test_gold_and_matrix_have_no_prompt_label_leak(self):
        check_gold()
        rows = list(request_rows())
        self.assertEqual(len(rows), 144)
        self.assertEqual(len({row['id'] for row in rows}), 144)
        self.assertEqual({row['domain'] for row in rows}, {'python', 'http', 'arithmetic'})
        for row in rows:
            self.assertNotIn('gold_label', row['judge_input'])
            self.assertNotIn('correct_first', row['judge_input'])
            self.assertNotIn(row['case_id'], row['judge_input'])
        self.assertEqual(rows, list(request_rows()))

    def test_complete_correct_control(self):
        result = analyze_recorded(complete_control())
        self.assertEqual((result['attempts'], result['valid'], result['wrong_wins']), (144, 144, 0))
        self.assertEqual((result['position_flips'], result['position_pairs']), (0, 72))
        self.assertEqual((result['repeat_disagreements'], result['repeat_pairs']), (0, 72))
        self.assertEqual([result['groups'][name]['valid'] for name in ('python', 'http', 'arithmetic')], [48, 48, 48])
        self.assertEqual([result['groups'][name]['valid'] for name in ('short', 'verbose')], [72, 72])

    def test_wrong_winner_and_reversal_use_candidate_identity(self):
        rows = complete_control()
        target = next(row for row in rows if row['id'] == 'py-01:short:1:correct_first')
        target['raw_response'] = '{"winner":"B","reason":"Deliberately wrong control."}'
        result = analyze_recorded(rows)
        self.assertEqual(result['wrong_wins'], 1)
        self.assertEqual(result['position_flips'], 1)
        self.assertEqual(result['repeat_disagreements'], 1)
        self.assertEqual(evaluate(fixture(), False)['wrong_wins'], 3)

    def test_tie_is_valid_but_not_a_wrong_win(self):
        rows = complete_control()
        rows[0]['raw_response'] = '{"winner":"tie","reason":"Control tie."}'
        result = analyze_recorded(rows)
        self.assertEqual((result['valid'], result['ties'], result['wrong_wins']), (144, 1, 0))

    def test_invalid_json_and_schema_do_not_enter_valid_denominator(self):
        for bad in ('not json', '{"winner":"A"}', '{"winner":"C","reason":"x"}',
                    '{"winner":"A","reason":""}', '{"winner":"A","reason":"x","extra":1}'):
            with self.subTest(bad=bad):
                self.assertIsNone(parse_response(bad))
        rows = complete_control()
        rows[0]['raw_response'] = 'not json'
        result = analyze_recorded(rows)
        self.assertEqual((result['valid'], result['invalid_output']), (143, 1))
        self.assertEqual(result['position_pairs'], 71)

    def test_refusal_timeout_and_api_error_stay_separate(self):
        rows = complete_control()
        for record, status in zip(rows[:3], ('refusal', 'timeout', 'api_error')):
            record['status'] = status
            record['raw_response'] = ''
        result = analyze_recorded(rows)
        self.assertEqual((result['valid'], result['refusals'], result['timeouts'], result['api_errors']), (141, 1, 1, 1))
        self.assertEqual(result['invalid_output'], 0)

    def test_incomplete_duplicate_unknown_and_mixed_model_rejected(self):
        rows = complete_control()
        for bad in (rows[:-1], rows + [rows[0]], [{**rows[0], 'id': 'unknown'}] + rows[1:],
                    [{**rows[0], 'model': 'other-model'}] + rows[1:]):
            with self.assertRaises(ValueError):
                analyze_recorded(bad)

    def test_stale_prompt_or_request_rejected(self):
        rows = complete_control()
        for changed in ({'prompt_version': 'old'}, {'prompt_sha256': '0' * 64},
                        {'request_sha256': '0' * 64}, {'judge_input': 'changed'}):
            with self.assertRaises(ValueError):
                analyze_recorded([{**rows[0], **changed}] + rows[1:])


if __name__ == '__main__':
    unittest.main()
