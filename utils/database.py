"""
Lightweight DB utility. Uses SQLite by default (zero setup) but the
interface is intentionally small so it's easy to swap in psycopg2 /
PostgreSQL later without touching the agents.
"""
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

FORBIDDEN_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
    "CREATE", "TRUNCATE", "REPLACE", "ATTACH", "PRAGMA",
)

# Schema rarely changes between questions in a single session; re-querying
# sqlite_master + PRAGMA table_info for every table on every question is
# wasted round trips. Cache with a short TTL so a schema change (e.g. a
# fresh feed_db.py run) is picked up within a minute without a restart.
SCHEMA_CACHE_TTL_SECONDS = 60


class DatabaseUtil:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.getenv("DB_PATH", "data/data_agent.db")
        self._schema_cache: Optional[str] = None
        self._schema_cache_at: float = 0.0

    @contextmanager
    def _connection(self):
        """
        Context manager so a connection is always closed, even if the
        query raises partway through (e.g. malformed SQL, sqlite3.Error).
        The previous version opened a connection and closed it only on
        the success path, leaking a handle on every failed query.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def invalidate_schema_cache(self):
        """Call after any operation that changes the schema (e.g. feed_db.py)."""
        self._schema_cache = None

    def schema_details(self, force_refresh: bool = False) -> str:
        """Return a human-readable schema summary the LLM can use as context."""
        now = time.time()
        if (
            not force_refresh
            and self._schema_cache is not None
            and (now - self._schema_cache_at) < SCHEMA_CACHE_TTL_SECONDS
        ):
            return self._schema_cache

        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [r[0] for r in cur.fetchall() if not r[0].startswith("sqlite_")]

            lines = []
            for table in tables:
                # nosec B608 - `table` comes from sqlite_master (our own DB's table
                # list), not from user/LLM input, so there's nothing to inject here.
                cur.execute(f"PRAGMA table_info({table});")
                cols = cur.fetchall()  # (cid, name, type, notnull, dflt, pk)
                col_desc = ", ".join(f"{c[1]} ({c[2]})" for c in cols)
                lines.append(f"Table `{table}`: {col_desc}")

        result = "\n".join(lines) if lines else "No tables found."
        self._schema_cache = result
        self._schema_cache_at = now
        return result

    def list_tables(self) -> List[str]:
        """Table names only, for callers (e.g. the data catalog agent) that
        need to iterate tables individually rather than one combined string."""
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [r[0] for r in cur.fetchall() if not r[0].startswith("sqlite_")]

    def column_info(self, table: str) -> List[Dict[str, Any]]:
        """[{"name": ..., "type": ..., "notnull": bool, "pk": bool}, ...] for one table."""
        if table not in self.list_tables():
            raise ValueError(f"Unknown table: {table}")
        with self._connection() as conn:
            cur = conn.cursor()
            # nosec B608 - `table` is validated against list_tables() (our own
            # sqlite_master) above, not passed through from user/LLM input.
            cur.execute(f"PRAGMA table_info({table});")
            cols = cur.fetchall()  # (cid, name, type, notnull, dflt, pk)
        return [
            {"name": c[1], "type": c[2], "notnull": bool(c[3]), "pk": bool(c[5])}
            for c in cols
        ]

    def sample_rows(self, table: str, n: int = 5) -> List[Dict[str, Any]]:
        """A handful of rows from `table`, for context when generating
        column descriptions. Not user input - `table` is validated the
        same way as column_info() above."""
        if table not in self.list_tables():
            raise ValueError(f"Unknown table: {table}")
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(f"SELECT * FROM {table} LIMIT ?;", (n,))  # nosec B608 - table validated above
            return [dict(r) for r in cur.fetchall()]

    @staticmethod
    def is_query_safe(sql: str) -> bool:
        upper = sql.upper()
        if not upper.strip().startswith("SELECT"):
            return False
        return not any(kw in upper for kw in FORBIDDEN_KEYWORDS)

    def run_query(self, sql: str, row_limit: int = 10) -> List[Dict[str, Any]]:
        # `sql` is LLM-generated, dynamic-by-nature (this is a NL-to-SQL agent -
        # there's no fixed query to parameterize against). SAST will flag the
        # execute() below (bandit B608 / semgrep sqli). The actual mitigation
        # is upstream and layered, not parameterization:
        #   1. is_query_safe() below requires a SELECT and blocks any
        #      destructive/DDL keyword (see FORBIDDEN_KEYWORDS)
        #   2. every query is executed against a connection with no write
        #      permission needed - this class never issues COMMIT-requiring ops
        #   3. a LIMIT is force-appended to cap result size
        #   4. the SQL Analyst agent additionally runs an LLM-as-judge pass
        #      before this ever gets called - see agents/sql_analyst.py
        # Do not remove this check or weaken FORBIDDEN_KEYWORDS without adding
        # an equivalent replacement control.
        if not self.is_query_safe(sql):
            raise ValueError("Refusing to execute unsafe (non-SELECT) query.")

        # Enforce a row cap if the query doesn't already have one
        if "LIMIT" not in sql.upper():
            sql = sql.rstrip().rstrip(";") + f" LIMIT {row_limit};"

        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(sql)  # nosec B608 - see mitigation notes above
            rows = [dict(r) for r in cur.fetchall()]
        return rows
