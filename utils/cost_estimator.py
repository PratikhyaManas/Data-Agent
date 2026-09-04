"""
Deterministic cost estimation for SQL queries. Uses SQLite's own query
planner (EXPLAIN QUERY PLAN) rather than an LLM - this is a mechanical
check, not a judgment call, so it's cheap and fast by design.
"""
import sqlite3
from typing import Any, Dict

# Table sizes above this are expensive to full-scan; below it, a scan is
# cheap enough not to warrant a warning even without an index.
LARGE_TABLE_ROW_THRESHOLD = 5000


def estimate_query_cost(db_path: str, sql: str) -> Dict[str, Any]:
    """
    Returns a dict: {cost_level, plan, notes}
    cost_level is 'low' | 'medium' | 'high'.

    Heuristic:
    - Any 'SCAN' (not 'SEARCH') step in the query plan against a table
      over LARGE_TABLE_ROW_THRESHOLD rows -> 'high' (full scan of a big table)
    - Any 'SCAN' step against a smaller table, or multiple joined scans -> 'medium'
    - Only indexed 'SEARCH' steps -> 'low'
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        try:
            cur.execute(f"EXPLAIN QUERY PLAN {sql}")  # nosec B608 - read-only planner call, same sql already passed is_query_safe()
            plan_rows = cur.fetchall()
        except sqlite3.Error as e:
            return {"cost_level": "low", "plan": [], "notes": f"Could not generate query plan: {e}"}

        plan_lines = [row[-1] for row in plan_rows]  # last column holds the human-readable detail

        scan_tables = []
        for line in plan_lines:
            if line.upper().startswith("SCAN"):
                # SQLite formats vary by version: "SCAN users" (3.x) or
                # "SCAN TABLE users" (older). Handle both.
                parts = line.split()
                if len(parts) >= 2:
                    table = parts[2] if len(parts) > 2 and parts[1].upper() == "TABLE" else parts[1]
                    scan_tables.append(table)

        if not scan_tables:
            return {
                "cost_level": "low",
                "plan": plan_lines,
                "notes": "Query uses indexed lookups only, no full table scans.",
            }

        max_scanned_rows = 0
        scan_details = []
        for table in set(scan_tables):
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")  # nosec B608 - table name enumerated from our own plan output, not external input
                row_count = cur.fetchone()[0]
            except sqlite3.Error:
                row_count = 0
            max_scanned_rows = max(max_scanned_rows, row_count)
            scan_details.append(f"{table} (~{row_count} rows)")

        if max_scanned_rows > LARGE_TABLE_ROW_THRESHOLD:
            cost_level = "high"
            notes = f"Full table scan on large table(s): {', '.join(scan_details)}. Consider adding a WHERE clause or index."
        else:
            cost_level = "medium"
            notes = f"Full table scan on: {', '.join(scan_details)}. Small enough to be inexpensive here, but worth an index if this table grows."

        return {"cost_level": cost_level, "plan": plan_lines, "notes": notes}
    finally:
        conn.close()
