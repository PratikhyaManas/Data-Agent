"""Tests for utils/cost_estimator.py."""
import sqlite3
import pytest

from utils.cost_estimator import estimate_query_cost, LARGE_TABLE_ROW_THRESHOLD


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "cost_test.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE small_table (id INTEGER, name TEXT)")
    conn.executemany("INSERT INTO small_table VALUES (?, ?)", [(i, f"n{i}") for i in range(5)])

    conn.execute("CREATE TABLE big_table (id INTEGER, val TEXT)")
    conn.executemany(
        "INSERT INTO big_table VALUES (?, ?)",
        [(i, f"v{i}") for i in range(LARGE_TABLE_ROW_THRESHOLD + 1000)],
    )
    conn.execute("CREATE INDEX idx_big_id ON big_table(id)")
    conn.commit()
    conn.close()
    return path


def test_indexed_lookup_is_low_cost(db_path):
    result = estimate_query_cost(db_path, "SELECT * FROM big_table WHERE id = 5")
    assert result["cost_level"] == "low"


def test_small_table_full_scan_is_medium_cost(db_path):
    result = estimate_query_cost(db_path, "SELECT * FROM small_table")
    assert result["cost_level"] == "medium"


def test_large_table_full_scan_is_high_cost(db_path):
    result = estimate_query_cost(db_path, "SELECT * FROM big_table")
    assert result["cost_level"] == "high"
    assert "big_table" in result["notes"]


def test_invalid_sql_does_not_crash(db_path):
    result = estimate_query_cost(db_path, "SELECT * FROM nonexistent_table")
    assert result["cost_level"] == "low"
    assert "Could not generate query plan" in result["notes"]
