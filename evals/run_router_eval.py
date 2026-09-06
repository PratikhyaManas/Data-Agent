"""
Router eval: classification accuracy.

Calls the router node in isolation (not the full data_agent graph) so a
misrouted case doesn't also spend tokens running a downstream agent.

Usage: python -m evals.run_router_eval
Requires: ANTHROPIC_API_KEY set.
"""
import json
import os
import sys
import time

from langchain_core.messages import HumanMessage
from Models.schema import DataAgentSchema
from agents.data_agent import route_node
from evals.metrics import accuracy
from project_bootstrap import ensure_repo_root

ensure_repo_root()

CASES_PATH = os.path.join(os.path.dirname(__file__), "datasets", "router_cases.jsonl")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "router_eval_results.json")


def load_cases(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval():
    cases = load_cases(CASES_PATH)
    results = []

    for case in cases:
        start = time.time()
        record = {"id": case["id"], "question": case["question"], "expected_route": case["expected_route"]}
        try:
            state = route_node(DataAgentSchema(messages=[HumanMessage(content=case["question"])]))
            actual_route = state.route_response
            record["actual_route"] = actual_route
            record["passed"] = actual_route == case["expected_route"]
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

    print(f"\nRouter eval: {sum(r['passed'] for r in results)}/{len(results)} passed ({score:.0%})\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['id']}: {r['question']!r} -> expected {r['expected_route']}, got {r.get('actual_route', 'ERROR')}")

    return 0 if score >= 0.85 else 1


if __name__ == "__main__":
    sys.exit(main())
