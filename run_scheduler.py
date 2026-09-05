"""
CLI for recurring ETL jobs (utils/scheduler.py). Run this from cron, a
GitHub Actions schedule, Windows Task Scheduler, or `--loop` for a simple
long-running poll - whatever fits your deployment, since a persistent
schedule the file-backed store doesn't care who calls it or how often.

Usage:
  python run_scheduler.py add "Extract data from https://pokeapi.co/api/v2/pokemon and save as CSV" --interval hourly --name pokeapi_sync
  python run_scheduler.py list
  python run_scheduler.py disable pokeapi_sync
  python run_scheduler.py enable pokeapi_sync
  python run_scheduler.py remove pokeapi_sync
  python run_scheduler.py run              # run whatever's due right now, once, then exit
  python run_scheduler.py run --loop 60     # keep polling for due jobs every 60s until Ctrl+C
"""
import argparse
import time

from utils.scheduler import add_job, remove_job, set_job_enabled, list_jobs, run_due_jobs


def _etl_runner(request: str) -> str:
    """Lazily imported so `run_scheduler.py list` etc. don't need API access configured."""
    from langchain_core.messages import HumanMessage
    from Models.schema import ETLAgentSchema
    from agents.etl_analyst import etl_analyst

    result = etl_analyst.invoke(ETLAgentSchema(messages=[HumanMessage(content=request)]))
    msgs = result["messages"] if isinstance(result, dict) else result.messages
    return msgs[-1].content


def cmd_add(args):
    job_id = add_job(args.request, args.interval, name=args.name)
    print(f"Scheduled job '{job_id}' every {args.interval}: {args.request!r}")


def cmd_remove(args):
    print("Removed." if remove_job(args.job_id) else f"No job named '{args.job_id}'.")


def cmd_enable(args):
    print("Enabled." if set_job_enabled(args.job_id, True) else f"No job named '{args.job_id}'.")


def cmd_disable(args):
    print("Disabled." if set_job_enabled(args.job_id, False) else f"No job named '{args.job_id}'.")


def cmd_list(args):
    jobs = list_jobs()
    if not jobs:
        print("No scheduled jobs.")
        return
    for job_id, job in jobs.items():
        status = "enabled" if job.get("enabled", True) else "disabled"
        print(f"[{status}] {job_id}: every {job['interval_seconds']}s - {job['request']!r}")
        print(f"    last_run_at={job.get('last_run_at')} next_run_at={job.get('next_run_at')}")


def cmd_run(args):
    while True:
        summaries = run_due_jobs(_etl_runner)
        if summaries:
            for s in summaries:
                if "error" in s:
                    print(f"[{s['id']}] FAILED: {s['error']}")
                else:
                    print(f"[{s['id']}] OK: {s['result']}")
        elif not args.loop:
            print("No jobs due.")
        if not args.loop:
            break
        time.sleep(args.loop)


def main():
    parser = argparse.ArgumentParser(description="Recurring ETL job scheduler.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Register a new recurring ETL job.")
    p_add.add_argument("request", help="Natural-language ETL instruction.")
    p_add.add_argument("--interval", required=True, help="'hourly' | 'daily' | 'weekly' | seconds.")
    p_add.add_argument("--name", default=None, help="Optional job id (auto-generated if omitted).")
    p_add.set_defaults(func=cmd_add)

    p_remove = sub.add_parser("remove", help="Remove a scheduled job.")
    p_remove.add_argument("job_id")
    p_remove.set_defaults(func=cmd_remove)

    p_enable = sub.add_parser("enable", help="Re-enable a disabled job.")
    p_enable.add_argument("job_id")
    p_enable.set_defaults(func=cmd_enable)

    p_disable = sub.add_parser("disable", help="Disable a job without removing it.")
    p_disable.add_argument("job_id")
    p_disable.set_defaults(func=cmd_disable)

    p_list = sub.add_parser("list", help="List all scheduled jobs.")
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="Run whatever jobs are due right now.")
    p_run.add_argument("--loop", type=int, default=0, metavar="SECONDS", help="Keep polling every N seconds instead of running once.")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
