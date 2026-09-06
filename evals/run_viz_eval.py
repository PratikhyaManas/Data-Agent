"""
Visualization Agent eval: chart-type appropriateness.

Runs the full visualization_agent graph (including its own internal
judge/retry loop) and checks whether the final chart type lands in the
set of chart types a human would consider reasonable for that request.
This is a looser check than SQL execution-accuracy by necessity - "is
this a good chart" has more valid answers than "is this the right data."

Usage: python -m evals.run_viz_eval
Requires: ANTHROPIC_API_KEY set.
"""
import json
import os
import sys
import time

from Models.schema import VizAgentSchema
from agents.visualization_agent import visualization_agent
from evals.metrics import accuracy
from project_bootstrap import ensure_repo_root

ensure_repo_root()

CASES_PATH = os.path.join(os.path.dirname(__file__), "datasets", "viz_cases.jsonl")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "viz_eval_results.json")


def load_cases(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval():
    cases = load_cases(CASES_PATH)
    results = []

    for case in cases:
        start = time.time()
        record = {
            "id": case["id"],
            "request": case["request"],
            "acceptable_chart_types": case["acceptable_chart_types"],
        }
        try:
            out = visualization_agent.invoke(
                VizAgentSchema(user_request=case["request"], data_source=case["data_source"])
            )
            chart_type = out["chart_type"] if isinstance(out, dict) else out.chart_type
            judge_verdict = out["judge_verdict"] if isinstance(out, dict) else out.judge_verdict
            record["chart_type"] = chart_type
            record["judge_verdict"] = judge_verdict
            record["passed"] = chart_type in case["acceptable_chart_types"]
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

    print(f"\nVisualization eval: {sum(r['passed'] for r in results)}/{len(results)} passed ({score:.0%})\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['id']}: {r['request']!r} -> got {r.get('chart_type', 'ERROR')}, acceptable: {r['acceptable_chart_types']}")

    return 0 if score >= 0.75 else 1


if __name__ == "__main__":
    sys.exit(main())
