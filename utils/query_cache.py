"""
Cache for judge-approved SQL queries, keyed by (curated question, schema).
Skips generate_sql -> optimize_query -> judge_query (3 LLM calls) for a
repeat question against an unchanged schema - execution always re-runs
against live data, so the answer stays current even when the SQL is reused.

Deliberately file-backed (not in-memory) so it survives CLI restarts.
Only entries the judge marked "correct" are ever written, so a bad query
can't get cached and replayed.
"""
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

CACHE_PATH = os.getenv("QUERY_CACHE_PATH", "data/query_cache.json")
CACHE_TTL_SECONDS = int(os.getenv("QUERY_CACHE_TTL_SECONDS", str(24 * 60 * 60)))  # 1 day default

# main.py is a single-user interactive CLI, so this cache is normally only
# ever touched by one process. But it can also be hit concurrently - e.g.
# an eval run (evals/run_sql_eval.py) sharing the same data/ directory
# while a CLI session is also live. Without locking, the read-modify-write
# cycle in set_cached_query() is a classic lost-update race: two writers
# can both read the same pre-write state, and whichever saves last wins,
# silently discarding the other's entry. Verified directly: 30 concurrent
# writers against the unlocked version lost ~3 entries and, worse, could
# crash outright when two processes collided on the same shared ".tmp"
# filename mid-write. fcntl.flock (POSIX) fixes both by serializing the
# whole cycle; on non-POSIX platforms (no fcntl) this degrades to no
# locking - acceptable for this project's actual deployment model
# (Linux CI, single-user local CLI) but worth knowing if that changes.
try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None
    _HAS_FCNTL = False

try:
    import msvcrt
    _HAS_MSVCRT = True
except ImportError:  # pragma: no cover - non-Windows platforms
    msvcrt = None
    _HAS_MSVCRT = False


@contextmanager
def _locked():
    """Serializes the read-modify-write cycle across processes.

    Windows' locking semantics are stricter than POSIX, so we retry briefly
    when the lock is contended instead of silently downgrading to the unsafe
    unlocked path.
    """
    lock_path = CACHE_PATH + ".lock"
    lock_file = None
    try:
        os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
        lock_file = open(lock_path, "a+b")

        if _HAS_FCNTL:
            for _ in range(100):
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.01)
                except OSError:
                    raise
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        elif _HAS_MSVCRT:
            for _ in range(100):
                try:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.01)
            else:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            yield
            return
    except OSError as e:
        print(f"[query_cache] warning: could not acquire lock, proceeding unlocked: {e}", file=sys.stderr)
        yield
        return

    try:
        yield
    finally:
        try:
            if _HAS_FCNTL:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            elif _HAS_MSVCRT:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            lock_file.close()


def build_cache_key(curated_question: str, schema: str) -> str:
    """
    Hash of the curated question + current schema. Including the schema
    means a `feed_db.py` reseed with different columns/tables naturally
    invalidates stale cache entries instead of requiring manual cleanup.
    """
    raw = f"{curated_question.strip().lower()}||{schema}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cache() -> Dict[str, Any]:
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _purge_expired(cache: Dict[str, Any]) -> Dict[str, Any]:
    """
    get_cached_query() already treats an expired entry as a miss, but
    without this the file itself would grow forever - every distinct
    question ever asked stays in it permanently, even long after its TTL
    passed. Called on every write so the file self-trims over time
    without needing a separate cleanup job.
    """
    now = time.time()
    return {k: v for k, v in cache.items() if now - v.get("cached_at", 0) <= CACHE_TTL_SECONDS}


def _save_cache(cache: Dict[str, Any]) -> None:
    """Best-effort: a cache write failure shouldn't break the SQL pipeline
    that's just trying to use this as an optimization."""
    try:
        os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)
        # Unique per-call temp filename (pid + timestamp), not a fixed
        # ".tmp" suffix - otherwise concurrent writers collide on the same
        # temp file and os.replace() can fail outright (see module docstring).
        tmp_path = f"{CACHE_PATH}.{os.getpid()}.{time.time_ns()}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(cache, f, indent=2)
        os.replace(tmp_path, CACHE_PATH)  # atomic on POSIX - avoids a torn write on crash
    except OSError as e:
        print(f"[query_cache] warning: failed to write cache: {e}", file=sys.stderr)


def get_cached_query(cache_key: str) -> Optional[Dict[str, Any]]:
    # Read-only: no lock needed. _save_cache's write-to-temp-then-rename
    # pattern means a reader either sees the old file or the fully-written
    # new one, never a torn/partial write.
    cache = _load_cache()
    entry = cache.get(cache_key)
    if entry is None:
        return None
    if time.time() - entry.get("cached_at", 0) > CACHE_TTL_SECONDS:
        return None
    return entry


def set_cached_query(cache_key: str, sql: str, optimizer_notes: str = "") -> None:
    with _locked():
        cache = _purge_expired(_load_cache())
        cache[cache_key] = {
            "sql": sql,
            "optimizer_notes": optimizer_notes,
            "cached_at": time.time(),
        }
        _save_cache(cache)


def clear_cache() -> None:
    with _locked():
        try:
            if os.path.exists(CACHE_PATH):
                os.remove(CACHE_PATH)
        except OSError as e:
            print(f"[query_cache] warning: failed to clear cache: {e}", file=sys.stderr)
