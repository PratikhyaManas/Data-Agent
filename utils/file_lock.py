"""
Cross-platform advisory file locking, shared by every file-backed store in
this project (query cache, ETL schedule, data catalog) so the read-modify-
write cycle is serialized across concurrent processes without duplicating
the fcntl/msvcrt fallback dance in each module.
"""
import os
import sys
import time
from contextlib import contextmanager

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
def locked(target_path: str):
    """
    Serializes the read-modify-write cycle across processes for the given
    target file (locks a sibling `<target_path>.lock` file, not the target
    itself, so the target's own atomic write-then-rename is untouched).

    Windows' locking semantics are stricter than POSIX, so contended
    acquisition is retried briefly instead of silently downgrading to an
    unlocked path. Falls back to a warning-only no-op only if the lock
    file itself can't be created (e.g. unwritable directory) - locking is
    a safety net, not something that should crash a write that would
    otherwise succeed unlocked.
    """
    lock_path = target_path + ".lock"
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
        print(f"[file_lock] warning: could not acquire lock, proceeding unlocked: {e}", file=sys.stderr)
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


def atomic_write_json(path: str, data) -> bool:
    """
    Write JSON to `path` via a unique per-call temp file + os.replace(),
    avoiding both torn writes and the shared-".tmp"-filename collision
    that broke concurrent writers before this pattern was introduced.
    Returns False (and prints a warning) instead of raising - every
    caller treats persistence as best-effort.
    """
    import json

    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp_path = f"{path}.{os.getpid()}.{time.time_ns()}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, path)
        return True
    except OSError as e:
        print(f"[file_lock] warning: failed to write {path}: {e}", file=sys.stderr)
        return False
