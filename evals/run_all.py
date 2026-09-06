"""
Runs the full eval suite (router, SQL, visualization) and prints a
combined summary. Exits non-zero if any suite falls below its threshold,
so this can gate a CI job.

Usage: python -m evals.run_all
Requires: ANTHROPIC_API_KEY set, and `python feed_db.py` already run
(the SQL eval needs the seeded demo DB).
"""
import sys

from evals import run_router_eval, run_sql_eval, run_viz_eval
from evals.metrics import accuracy
from project_bootstrap import ensure_repo_root

ensure_repo_root()


def main():
    print("=" * 60)
    print("Running eval suite")
    print("=" * 60)

    router_results = run_router_eval.run_eval()
    router_score = accuracy(router_results)

    sql_results = run_sql_eval.run_eval()
    sql_score = accuracy(sql_results)

    viz_results = run_viz_eval.run_eval()
    viz_score = accuracy(viz_results)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Router:        {sum(r['passed'] for r in router_results)}/{len(router_results)}  ({router_score:.0%})")
    print(f"  SQL Analyst:   {sum(r['passed'] for r in sql_results)}/{len(sql_results)}  ({sql_score:.0%})")
    print(f"  Visualization: {sum(r['passed'] for r in viz_results)}/{len(viz_results)}  ({viz_score:.0%})")

    thresholds = {"router": 0.85, "sql": 0.80, "viz": 0.75}
    failed = []
    if router_score < thresholds["router"]:
        failed.append(f"router ({router_score:.0%} < {thresholds['router']:.0%})")
    if sql_score < thresholds["sql"]:
        failed.append(f"sql ({sql_score:.0%} < {thresholds['sql']:.0%})")
    if viz_score < thresholds["viz"]:
        failed.append(f"viz ({viz_score:.0%} < {thresholds['viz']:.0%})")

    if failed:
        print(f"\nFAILED thresholds: {', '.join(failed)}")
        return 1

    print("\nAll suites above threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
