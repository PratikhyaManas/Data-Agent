"""
Scheduler for recurring ETL runs. File-backed (data/etl_schedule.json), so
jobs survive CLI/process restarts, mirroring the same atomic-write +
cross-platform-lock pattern used by utils/query_cache.py.

Jobs are defined as a plain-language ETL request plus an interval; this
module only owns *when* a job is due and *persisting* that state - it
doesn't invoke the ETL agent itself (see run_scheduler.py for the runner
that wires this up to agents.etl_analyst.etl_analyst). Keeping the "what's
due" logic free of any LLM dependency keeps it fully unit-testable without
an API key, consistent with the rest of tests/.
"""
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from utils.file_lock import locked, atomic_write_json

SCHEDULE_PATH = "data/etl_schedule.json"

# Friendly presets so callers don't have to compute seconds by hand.
INTERVAL_PRESETS = {
    "hourly": 60 * 60,
    "daily": 24 * 60 * 60,
    "weekly": 7 * 24 * 60 * 60,
}


def resolve_interval_seconds(interval: Any) -> int:
    """Accepts either a preset name ('hourly'/'daily'/'weekly') or a raw
    number of seconds (int or numeric string, e.g. from the CLI's argparse
    strings). Raises ValueError on anything else."""
    if isinstance(interval, str) and interval in INTERVAL_PRESETS:
        return INTERVAL_PRESETS[interval]
    try:
        seconds = int(interval)
    except (TypeError, ValueError):
        raise ValueError(
            f"Unknown interval '{interval}'. Use one of {list(INTERVAL_PRESETS)} or a number of seconds."
        )
    if seconds <= 0:
        raise ValueError("interval_seconds must be positive.")
    return seconds


def _load_schedule() -> Dict[str, Any]:
    import json
    import os

    if not os.path.exists(SCHEDULE_PATH):
        return {}
    try:
        with open(SCHEDULE_PATH) as f:
            return json.load(f)
    except (ValueError, OSError):
        return {}


def _save_schedule(schedule: Dict[str, Any]) -> bool:
    return atomic_write_json(SCHEDULE_PATH, schedule)


def add_job(request: str, interval: Any, name: Optional[str] = None, enabled: bool = True) -> str:
    """
    Registers a recurring ETL job. `request` is the natural-language ETL
    instruction (the same text you'd type at the CLI), `interval` is a
    preset name or a number of seconds. Returns the job id.
    """
    interval_seconds = resolve_interval_seconds(interval)
    job_id = name or uuid.uuid4().hex[:8]
    now = time.time()

    with locked(SCHEDULE_PATH):
        schedule = _load_schedule()
        schedule[job_id] = {
            "request": request,
            "interval_seconds": interval_seconds,
            "enabled": enabled,
            "created_at": now,
            "last_run_at": None,
            "next_run_at": now,  # due immediately on first check
            "last_result": None,
        }
        _save_schedule(schedule)
    return job_id


def remove_job(job_id: str) -> bool:
    with locked(SCHEDULE_PATH):
        schedule = _load_schedule()
        if job_id not in schedule:
            return False
        del schedule[job_id]
        _save_schedule(schedule)
    return True


def set_job_enabled(job_id: str, enabled: bool) -> bool:
    with locked(SCHEDULE_PATH):
        schedule = _load_schedule()
        if job_id not in schedule:
            return False
        schedule[job_id]["enabled"] = enabled
        _save_schedule(schedule)
    return True


def list_jobs() -> Dict[str, Any]:
    return _load_schedule()


def due_jobs(now: Optional[float] = None) -> List[Dict[str, Any]]:
    """Returns [{"id": ..., **job_fields}, ...] for every enabled job whose next_run_at has passed."""
    now = time.time() if now is None else now
    schedule = _load_schedule()
    return [
        {"id": job_id, **job}
        for job_id, job in schedule.items()
        if job.get("enabled", True) and job.get("next_run_at", 0) <= now
    ]


def run_due_jobs(
    etl_runner: Callable[[str], str],
    now: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Runs `etl_runner(request)` for every due job, then reschedules it
    (last_run_at = now, next_run_at = now + interval_seconds) and persists
    the result summary. `etl_runner` is injected so this is testable
    without a real LLM/agent call - the production entry point
    (run_scheduler.py) passes a runner backed by agents.etl_analyst.

    Returns a list of {"id", "request", "result" | "error"} run summaries.
    """
    from utils.audit import log_event  # local import - avoids a circular import at module load

    now = time.time() if now is None else now
    run_summaries = []

    with locked(SCHEDULE_PATH):
        schedule = _load_schedule()
        for job_id, job in schedule.items():
            if not job.get("enabled", True) or job.get("next_run_at", 0) > now:
                continue

            summary = {"id": job_id, "request": job["request"]}
            try:
                result = etl_runner(job["request"])
                summary["result"] = result
                job["last_result"] = {"status": "ok", "output": str(result)[:500]}
            except Exception as e:
                summary["error"] = str(e)
                job["last_result"] = {"status": "error", "output": str(e)[:500]}

            job["last_run_at"] = now
            job["next_run_at"] = now + job["interval_seconds"]
            run_summaries.append(summary)

            log_event(
                "scheduled_etl_run",
                job_id=job_id,
                request=job["request"],
                status=job["last_result"]["status"],
                next_run_at=job["next_run_at"],
            )

        if run_summaries:
            _save_schedule(schedule)

    return run_summaries
