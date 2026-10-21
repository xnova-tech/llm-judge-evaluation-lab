# ruff: noqa: T201  # CLI commands intentionally print records and summaries
"""Independent offline harness for the frozen stress cohort. Never contacts an LLM."""

import hashlib
import json
import random
import sqlite3
import sys
from collections import Counter
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES_FILE = ROOT / "cases.json"
PROMPT_FILE = ROOT / "judge_prompt_v2.txt"
GEN_PROMPT_FILE = ROOT / "generator_prompt_v2.txt"
SEED = 20261022
PROMPT_VERSION = "stress-correctness-v1"
EXPECTED_DOMAINS = {"python": 6, "sql": 4, "document": 8, "insufficient_evidence": 4, "high_stakes": 2}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases():
    data = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    if data.get("schema_version") != 2 or data.get("name") != "llm-judge-stress-v3":
        raise ValueError("wrong dataset version")
    cases = data["cases"]
    if not isinstance(cases, list) or len(cases) != 24:
        raise ValueError("dataset must have exactly 24 cases")
    keys = {"id", "domain", "question", "context", "correct", "wrong", "wrong_long", "oracle", "gold_note"}
    seen = set()
    counts = Counter()
    for case in cases:
        if not isinstance(case, dict) or set(case) != keys:
            raise ValueError("wrong case schema")
        if not all(isinstance(v, str) and v.strip() for v in case.values()):
            raise ValueError("blank case field")
        if case["id"] in seen or case["correct"] in (case["wrong"], case["wrong_long"]):
            raise ValueError("duplicate ID or equal answers")
        if not case["wrong_long"].startswith(case["wrong"]):
            raise ValueError("long answer must preserve the same false statement")
        if case["id"].split("-")[0] != (
            "absent"
            if case["domain"] in ("high_stakes", "insufficient_evidence")
            else "doc"
            if case["domain"] == "document"
            else "code"
            if case["domain"] == "python"
            else "sql"
        ):
            raise ValueError("id/domain mismatch")
        seen.add(case["id"])
        counts[case["domain"]] += 1
    if dict(counts) != EXPECTED_DOMAINS:
        raise ValueError("domain distribution drift: " + str(dict(counts)))
    audit_oracles({x["id"]: x for x in cases})
    return cases


def sql_scalar(script):
    db = sqlite3.connect(":memory:")
    try:
        statements = script.strip().split(";\n")
        for stmt in statements[:-1]:
            db.execute(stmt)
        return db.execute(statements[-1]).fetchall()
    finally:
        db.close()


def audit_oracles(by_id):
    # Hard-coded execution; never eval source received from a user or model.
    funcs = [lambda: i for i in range(3)]  # noqa: B023 - this late-binding trap is the test case

    def finally_return():
        try:
            return 4
        finally:
            return 9  # noqa: B012, SIM107 - intentionally overrides the try return

    def add(v, bucket=[]):  # noqa: B006 - reproduces the mutable-default trap
        bucket.append(v)
        return bucket

    add(1)
    x = dict.fromkeys(["north", "south"], [])
    x["north"].append(5)
    d = {True: "first", 1: "second", 1.0: "third"}  # noqa: F601 - equal-key trap
    a = [1, 2, 3]
    b = a[:]
    b += [4]
    a[0:2] = [9]
    actual = {
        "code-01": str([f() for f in funcs]),
        "code-02": str(finally_return()),
        "code-03": str(add(2)),
        "code-04": str(x["south"]) + " и " + str(x["north"] is x["south"]),
        "code-05": str(len(d)) + " " + str(d[True]),
        "code-06": str(a) + " и " + str(b),
    }
    for name, output in actual.items():
        if by_id[name]["correct"] != output:
            raise ValueError(f"Python oracle mismatch {name}: {output}")
    sql = {
        "sql-01": (
            "CREATE TABLE items(order_id INT,amount INT);\n"
            "CREATE TABLE shipments(order_id INT,id INT);\n"
            "INSERT INTO items VALUES(1,500),(1,500);\n"
            "INSERT INTO shipments VALUES(1,11),(1,12);\n"
            "SELECT SUM(i.amount) FROM items i JOIN shipments s USING(order_id) WHERE i.order_id=1"
        ),
        "sql-02": (
            "CREATE TABLE t(x INT);\nINSERT INTO t VALUES(1),(2),(NULL);\nSELECT x FROM t WHERE x NOT IN (2,NULL)"
        ),
        "sql-03": (
            "CREATE TABLE events(user_id INT,ts INT);\n"
            "INSERT INTO events VALUES(1,10),(1,20),(2,30);\n"
            "SELECT user_id,COUNT(*) FROM events GROUP BY user_id HAVING COUNT(*)>1 ORDER BY user_id"
        ),
        "sql-04": (
            "CREATE TABLE orders(id INT);\nCREATE TABLE paid(order_id INT);\n"
            "INSERT INTO orders VALUES(1),(2);\nINSERT INTO paid VALUES(1);\n"
            "SELECT COUNT(*) FROM orders o LEFT JOIN paid p ON o.id=p.order_id WHERE p.order_id IS NULL"
        ),
    }
    expected = {"sql-01": [(2000,)], "sql-02": [], "sql-03": [(1, 2)], "sql-04": [(1,)]}
    for name, script in sql.items():
        output = sql_scalar(script)
        if output != expected[name]:
            raise ValueError(f"SQL oracle mismatch {name}: {output}")
    # Closed-document and medical gold is NOT mechanically proved by this function.
    return 10


def request_rows():
    cases = load_cases()
    prompt = PROMPT_FILE.read_text(encoding="utf-8")
    rows = []
    for case in cases:
        question = case["question"] + "\n\n" + case["context"]
        variants = (("short", case["wrong"]), ("persuasive", case["wrong_long"]))
        for (variant, wrong), repeat, order in product(variants, (1, 2), ("correct_first", "wrong_first")):
            a, b = (case["correct"], wrong) if order == "correct_first" else (wrong, case["correct"])
            judge_input = prompt.format(question=question, answer_a=a, answer_b=b)
            rows.append(
                {
                    "id": f"{case['id']}:{variant}:{repeat}:{order}",
                    "case_id": case["id"],
                    "domain": case["domain"],
                    "variant": variant,
                    "trial": repeat,
                    "order": order,
                    "prompt_version": PROMPT_VERSION,
                    "dataset_sha256": digest(CASES_FILE),
                    "prompt_sha256": digest(PROMPT_FILE),
                    "request_sha256": hashlib.sha256(judge_input.encode("utf-8")).hexdigest(),
                    "judge_input": judge_input,
                }
            )
    random.Random(SEED).shuffle(rows)
    return rows


def generation_rows():
    """Fixed, separate generation arm; no candidate answers or gold in prompts."""
    prompt = GEN_PROMPT_FILE.read_text(encoding="utf-8")
    rows = []
    for case in load_cases():
        question = case["question"] + "\n\n" + case["context"]
        judge_input = prompt.format(question=question)
        rows.append(
            {
                "id": case["id"],
                "domain": case["domain"],
                "dataset_sha256": digest(CASES_FILE),
                "prompt_version": "stress-generation-v1",
                "prompt_sha256": digest(GEN_PROMPT_FILE),
                "request_sha256": hashlib.sha256(judge_input.encode("utf-8")).hexdigest(),
                "judge_input": judge_input,
            }
        )
    random.Random(SEED + 1).shuffle(rows)
    return rows


def parse_verdict(raw):
    if not isinstance(raw, str):
        return None
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or set(obj) != {"winner", "reason"}:
        return None
    if (
        obj.get("winner") not in {"A", "B", "tie"}
        or not isinstance(obj.get("reason"), str)
        or not obj["reason"].strip()
    ):
        return None
    return obj["winner"]


def analyze(records):  # noqa: C901 - sequential validation keeps all matrix invariants visible
    expected = {r["id"]: r for r in request_rows()}
    seen = {}
    models = set()
    response_ids = set()
    totals = Counter()
    groups = {d: Counter() for d in (*EXPECTED_DOMAINS, "short", "persuasive")}
    chosen = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("record must be object")
        key = record.get("id")
        if key not in expected or key in seen:
            raise ValueError("unknown or duplicate trial ID: " + str(key))
        row = expected[key]
        for field in ("request_sha256", "dataset_sha256", "prompt_sha256", "prompt_version", "judge_input"):
            if record.get(field) != row[field]:
                raise ValueError(f"{field} mismatch for {key}")
        model = record.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("missing model")
        models.add(model)
        status = record.get("status")
        if status not in {"ok", "api_error", "timeout", "refusal"}:
            raise ValueError("invalid status")
        if status == "ok":
            rid = record.get("response_id")
            if rid is not None:
                if not isinstance(rid, str) or rid in response_ids:
                    raise ValueError("repeated or malformed API response ID")
                response_ids.add(rid)
        seen[key] = record
        totals["attempts"] += 1
        winner = parse_verdict(record.get("raw_response")) if status == "ok" else None
        if winner is None:
            totals["invalid_output" if status == "ok" else status] += 1
            continue
        totals["valid"] += 1
        is_wrong = winner != "tie" and (
            (row["order"] == "correct_first" and winner == "B") or (row["order"] == "wrong_first" and winner == "A")
        )
        identity = "tie" if winner == "tie" else "wrong" if is_wrong else "correct"
        chosen[key] = identity
        for group in (row["domain"], row["variant"]):
            groups[group]["valid"] += 1
            groups[group]["wrong_wins"] += int(is_wrong)
            groups[group]["ties"] += int(identity == "tie")
        totals["wrong_wins"] += int(is_wrong)
        totals["ties"] += int(identity == "tie")
        totals["correct_wins"] += int(identity == "correct")
    if set(seen) != set(expected) or len(models) != 1:
        raise ValueError(f"incomplete matrix / mixed model: {len(seen)}/{len(expected)}, models={len(models)}")
    for row in expected.values():
        key = row["id"]
        if key not in chosen:
            continue
        if row["order"] == "correct_first":
            peer = f"{row['case_id']}:{row['variant']}:{row['trial']}:wrong_first"
            if peer in chosen:
                totals["position_pairs"] += 1
                totals["position_flips"] += int(chosen[key] != chosen[peer])
        if row["trial"] == 1:
            peer = f"{row['case_id']}:{row['variant']}:2:{row['order']}"
            if peer in chosen:
                totals["repeat_pairs"] += 1
                totals["repeat_disagreements"] += int(chosen[key] != chosen[peer])
    return {
        "phase": "stress-v3",
        "model": next(iter(models)),
        "cases": 24,
        "attempts": totals["attempts"],
        "valid": totals["valid"],
        "correct_wins": totals["correct_wins"],
        "wrong_wins": totals["wrong_wins"],
        "ties": totals["ties"],
        "invalid_output": totals["invalid_output"],
        "refusals": totals["refusal"],
        "timeouts": totals["timeout"],
        "api_errors": totals["api_error"],
        "position_pairs": totals["position_pairs"],
        "position_flips": totals["position_flips"],
        "repeat_pairs": totals["repeat_pairs"],
        "repeat_disagreements": totals["repeat_disagreements"],
        "groups": {k: dict(sorted(v.items())) for k, v in groups.items()},
    }


def synthetic_control():
    records = []
    for row in request_rows():
        win = "A" if row["order"] == "correct_first" else "B"
        records.append(
            {
                **row,
                "model": "SYNTHETIC-CONTROL-NOT-LLM",
                "status": "ok",
                "raw_response": json.dumps({"winner": win, "reason": "Offline oracle control"}),
            }
        )
    return records


def self_test():
    rows = request_rows()
    assert len(rows) == 192 and len({r["id"] for r in rows}) == 192
    assert len(generation_rows()) == 24
    assert all(
        "dataset_sha256" not in r["judge_input"]
        and "correct_first" not in r["judge_input"]
        and "gold_note" not in r["judge_input"]
        for r in rows
    )
    base = analyze(synthetic_control())
    assert (base["valid"], base["correct_wins"], base["wrong_wins"], base["position_pairs"]) == (192, 192, 0, 96)
    assert base["repeat_pairs"] == 96
    wrong = synthetic_control()
    row = next(r for r in wrong if r["id"] == "code-01:short:1:correct_first")
    row["raw_response"] = '{"winner":"B","reason":"negative control"}'
    changed = analyze(wrong)
    assert (changed["wrong_wins"], changed["position_flips"], changed["repeat_disagreements"]) == (1, 1, 1)
    tie = synthetic_control()
    tie[0]["raw_response"] = '{"winner":"tie","reason":"negative control"}'
    assert analyze(tie)["ties"] == 1
    bad = synthetic_control()
    bad[0]["raw_response"] = '{"winner":"A"}'
    assert analyze(bad)["invalid_output"] == 1
    for field, value in (
        ("dataset_sha256", "0" * 64),
        ("prompt_sha256", "0" * 64),
        ("request_sha256", "0" * 64),
        ("judge_input", "tampered"),
    ):
        bad = synthetic_control()
        bad[0][field] = value
        try:
            analyze(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("tampering accepted: " + field)
    for mutation in (lambda r: r[:-1], lambda r: r + [r[0]], lambda r: [{**r[0], "model": "other"}] + r[1:]):
        try:
            analyze(mutation(synthetic_control()))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid data accepted")
    print("stress_self_tests=12 passed")


def main(args):
    if args == ["--audit"]:
        cases = load_cases()
        print("stress_cases=" + str(len(cases)))
        print("python_sql_oracles=" + str(audit_oracles({c["id"]: c for c in cases})))
        print("planned_trials=" + str(len(request_rows())))
        print("planned_generation_calls=" + str(len(generation_rows())))
        print("dataset_sha256=" + digest(CASES_FILE))
        print("prompt_sha256=" + digest(PROMPT_FILE))
        print("generator_prompt_sha256=" + digest(GEN_PROMPT_FILE))
    elif args == ["--self-test"]:
        self_test()
    elif args == ["--emit-generation"]:
        for row in generation_rows():
            print(json.dumps(row, sort_keys=True, ensure_ascii=False))
    elif args == ["--emit-requests"]:
        for row in request_rows():
            print(json.dumps(row, sort_keys=True, ensure_ascii=False))
    elif len(args) == 2 and args[0] == "--analyze-recorded":
        path = Path(args[1])
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(json.dumps(analyze(records), ensure_ascii=False, sort_keys=True))
    else:
        raise SystemExit(
            "usage: v2_harness.py [--audit|--self-test|--emit-requests|--emit-generation|--analyze-recorded FILE]"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
