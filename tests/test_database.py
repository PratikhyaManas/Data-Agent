"""
Tests for utils/database.py. Covers the safety-check logic that gates
every SQL query before execution - this is the most security-critical
piece of the whole system, so it's tested directly rather than only
indirectly through the LLM pipeline.
"""
import os
import sqlite3
import tempfile
import pytest

from utils.database import DatabaseUtil


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT, rating REAL)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice', 4.9)")
    conn.execute("INSERT INTO users VALUES (2, 'Bob', 4.2)")
    conn.commit()
    conn.close()
    return DatabaseUtil(db_path=db_path)


class TestSafetyCheck:
    def test_select_is_safe(self):
        assert DatabaseUtil.is_query_safe("SELECT * FROM users") is True

    def test_select_with_where_is_safe(self):
        assert DatabaseUtil.is_query_safe("SELECT name FROM users WHERE rating > 4.5") is True

    @pytest.mark.parametrize("sql", [
        "DROP TABLE users",
        "DELETE FROM users",
        "UPDATE users SET rating = 0",
        "INSERT INTO users VALUES (3, 'Eve', 1.0)",
        "ALTER TABLE users ADD COLUMN hacked TEXT",
        "TRUNCATE TABLE users",
        "CREATE TABLE evil (id INTEGER)",
        "ATTACH DATABASE '/etc/passwd' AS pwned",
    ])
    def test_destructive_statements_are_unsafe(self, sql):
        assert DatabaseUtil.is_query_safe(sql) is False

    def test_select_disguised_with_destructive_suffix_is_unsafe(self):
        # A naive "starts with SELECT" check would miss this - make sure
        # a stacked/chained statement is still blocked.
        sql = "SELECT * FROM users; DROP TABLE users;"
        assert DatabaseUtil.is_query_safe(sql) is False

    def test_lowercase_select_is_safe(self):
        assert DatabaseUtil.is_query_safe("select * from users") is True

    def test_non_select_statement_is_unsafe(self):
        assert DatabaseUtil.is_query_safe("PRAGMA table_info(users)") is False


class TestQueryExecution:
    def test_run_query_returns_rows(self, db):
        rows = db.run_query("SELECT * FROM users ORDER BY rating DESC")
        assert len(rows) == 2
        assert rows[0]["name"] == "Alice"

    def test_run_query_enforces_row_limit(self, db):
        rows = db.run_query("SELECT * FROM users", row_limit=1)
        assert len(rows) == 1

    def test_run_query_refuses_unsafe_sql(self, db):
        with pytest.raises(ValueError):
            db.run_query("DROP TABLE users")

    def test_schema_details_lists_table_and_columns(self, db):
        details = db.schema_details()
        assert "users" in details
        assert "name" in details
        assert "rating" in details
