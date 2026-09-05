"""
File-backed store of column descriptions across tables (data/data_catalog.json),
maintained by agents/data_catalog_agent.py. Mirrors the same atomic-write +
cross-platform-lock pattern used by utils/query_cache.py and utils/scheduler.py.

Deliberately separate from the agent module: this file owns only
deterministic persistence (load/save/merge/diff), so it's fully unit
testable without an LLM, the same way utils/query_cache.py is.
"""
import time
from typing import Any, Dict, List, Tuple

from utils.file_lock import locked, atomic_write_json

CATALOG_PATH = "data/data_catalog.json"


def load_catalog() -> Dict[str, Dict[str, Any]]:
    """{table: {column: {"description": ..., "updated_at": ...}}}"""
    import json
    import os

    if not os.path.exists(CATALOG_PATH):
        return {}
    try:
        with open(CATALOG_PATH) as f:
            return json.load(f)
    except (ValueError, OSError):
        return {}


def _save_catalog(catalog: Dict[str, Dict[str, Any]]) -> bool:
    return atomic_write_json(CATALOG_PATH, catalog)


def columns_needing_descriptions(
    tables_with_columns: Dict[str, List[str]],
    catalog: Dict[str, Dict[str, Any]],
    refresh: bool = False,
) -> List[Tuple[str, str]]:
    """
    Returns [(table, column), ...] that don't yet have a cataloged
    description, or every column if `refresh` is True (used to regenerate
    stale descriptions after a schema change).
    """
    missing = []
    for table, columns in tables_with_columns.items():
        existing = catalog.get(table, {})
        for column in columns:
            if refresh or column not in existing:
                missing.append((table, column))
    return missing


def merge_descriptions(
    catalog: Dict[str, Dict[str, Any]],
    descriptions: List[Dict[str, str]],
) -> Dict[str, Dict[str, Any]]:
    """
    Merges [{"table":..., "column":..., "description":...}, ...] into the
    catalog dict, overwriting only the entries provided (existing
    descriptions for untouched columns are preserved). Returns the merged
    dict; does not persist it - call save_catalog() for that.
    """
    now = time.time()
    merged = {t: dict(cols) for t, cols in catalog.items()}
    for d in descriptions:
        table_entry = merged.setdefault(d["table"], {})
        table_entry[d["column"]] = {"description": d["description"], "updated_at": now}
    return merged


def save_catalog(catalog: Dict[str, Dict[str, Any]]) -> bool:
    with locked(CATALOG_PATH):
        return _save_catalog(catalog)
