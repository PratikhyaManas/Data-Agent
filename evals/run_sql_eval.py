"""
SQL Analyst eval: execution-accuracy grading.

For each case, runs a hand-written golden SQL query and the agent's
generated SQL against the same seeded DB, then compares result rows
(not query text - many valid SQL formulations answer the same question).

Usage: python -m evals.run_sql_eval
Requires: ANTHROPIC_API_KEY set, and `python feed_db.py` already run.
"""
import json
import os
import sys
import time

from Models.schema import AgentSchema
from agents.sql_analyst import sql_analyst
from evals.metrics import rows_equal, accuracy
from project_bootstrap import ensure_repo_root
from utils.database import DatabaseUtil

ensure_repo_root()

CASES_PATH = os.path.join(os.path.dirname(__file__), "datasets", "sql_cases.jsonl")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "sql_eval_results.json")


def load_cases(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval():
    db = DatabaseUtil()
    cases = load_cases(CASES_PATH)
    results = []

    for case in cases:
        start = time.time()
        record = {"id": case["id"], "question": case["question"], "golden_sql": case["golden_sql"]}
        try:
            golden_rows = db.run_query(case["golden_sql"], row_limit=1000)
        except Exception as e:
            record.update(passed=False, error=f"Golden SQL itself failed: {e}")
            results.append(record)
            continue

        try:
            out = sql_analyst.invoke(AgentSchema(user_question=case["question"]))
            generated_sql = out["generated_sql_query"] if isinstance(out, dict) else out.generated_sql_query
            judge_verdict = out["judge_verdict"] if isinstance(out, dict) else out.judge_verdict
            cost_level = out["cost_level"] if isinstance(out, dict) else out.cost_level
            record["generated_sql"] = generated_sql
            record["judge_verdict"] = judge_verdict
            record["cost_level"] = cost_level

            if not DatabaseUtil.is_query_safe(generated_sql):
                record.update(passed=False, error="Generated SQL failed the safety check.")
            else:
                actual_rows = db.run_query(generated_sql, row_limit=1000)
                record["passed"] = rows_equal(golden_rows, actual_rows)
                if not record["passed"]:
                    record["golden_rows"] = golden_rows
                    record["actual_rows"] = actual_rows
        except Exception as e:
            record.update(passed=False, error=str(e))

        record["duration_sec"] = round(time.time() - start, 2)
        results.append(record)

    return results


def main():
    results = run_eval()
    score = accuracy(results)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({"accuracy": score, "results": results}, f, indent=2, default=str)

    print(f"\nSQL Analyst eval: {sum(r['passed'] for r in results)}/{len(results)} passed ({score:.0%})\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['id']}: {r['question']}")
        if not r["passed"]:
            print(f"         error: {r.get('error', 'row mismatch')}")

    return 0 if score >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
