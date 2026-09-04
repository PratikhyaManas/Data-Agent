"""
Audit/Logging utility. Every router decision and sub-agent run gets
appended to data/audit_log.jsonl so you can answer "who ran what,
when, and what SQL actually executed" after the fact.

Logging is a side effect, not the primary task - a full disk, a
permission error, or a bad path should degrade this to a no-op (with a
stderr warning) rather than crash whatever agent step called it. The
log file itself is also rotated so it doesn't grow forever on a
long-running deployment.
"""
import json
import os
import sys
from collections import deque
from datetime import datetime, timezone

LOG_PATH = "data/audit_log.jsonl"
MAX_LOG_SIZE_BYTES = int(os.getenv("AUDIT_LOG_MAX_BYTES", str(10 * 1024 * 1024)))  # 10 MB
ROTATED_SUFFIX = ".1"


def _rotate_if_needed(path: str) -> None:
    """Single-generation rotation: audit_log.jsonl -> audit_log.jsonl.1, start fresh."""
    try:
        if os.path.exists(path) and os.path.getsize(path) >= MAX_LOG_SIZE_BYTES:
            rotated_path = path + ROTATED_SUFFIX
            if os.path.exists(rotated_path):
                os.remove(rotated_path)
            os.rename(path, rotated_path)
    except OSError:
        pass  # rotation is best-effort; a failure here shouldn't block logging either


def log_event(event_type: str, **fields) -> None:
    """
    event_type: 'route' | 'sql_run' | 'etl_run' | 'viz_run' | 'clarify' | ...
    fields: arbitrary JSON-serializable details (question, sql, is_safe, etc.)

    Best-effort: never raises. A logging failure prints a warning to
    stderr and moves on rather than taking down the agent step that
    triggered it.
    """
    try:
        os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
        _rotate_if_needed(LOG_PATH)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **fields,
        }
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError as e:
        print(f"[audit] warning: failed to write audit log entry ({event_type}): {e}", file=sys.stderr)


def read_recent(n: int = 20) -> list:
    """
    Returns the last n entries without loading the whole file into memory -
    important once the file is rotation-sized (up to MAX_LOG_SIZE_BYTES).
    """
    if not os.path.exists(LOG_PATH):
        return []
    tail: deque = deque(maxlen=n)
    try:
        with open(LOG_PATH) as f:
            for line in f:
                tail.append(line)
    except OSError:
        return []
    return [json.loads(line) for line in tail]
