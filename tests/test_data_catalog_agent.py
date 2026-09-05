"""
Tests for agents/data_catalog_agent.py's deterministic fast-path: when
every requested column is already cataloged (and refresh=False), no LLM
call should be made at all. This is the only part of the agent testable
without an API key - the actual description generation requires a real
LLM call and is covered by evals/, not tests/.
"""
import sqlite3

import pytest

from agents.data_catalog_agent import generate_descriptions
import agents.data_catalog_agent as catalog_agent_module
from Models.schema import DataCatalogSchema
from utils.database import DatabaseUtil
from utils.data_catalog import save_catalog, merge_descriptions


@pytest.fixture
def seeded_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice')")
    conn.commit()
    conn.close()

    monkeypatch.setattr(catalog_agent_module, "db", DatabaseUtil(db_path=db_path))
    return db_path


def test_fully_cataloged_columns_skip_llm_call(tmp_path, monkeypatch, seeded_db):
    catalog_path = str(tmp_path / "catalog.json")
    monkeypatch.setattr("utils.data_catalog.CATALOG_PATH", catalog_path)

    # Pre-populate the catalog with descriptions for every column in `users`.
    catalog = merge_descriptions(
        catalog={},
        descriptions=[
            {"table": "users", "column": "id", "description": "unique user id"},
            {"table": "users", "column": "name", "description": "display name"},
        ],
    )
    save_catalog(catalog)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("invoke_with_resilience should not be called when nothing is missing")

    monkeypatch.setattr(catalog_agent_module, "invoke_with_resilience", _fail_if_called)

    state = generate_descriptions(DataCatalogSchema())

    assert state.updated_columns == 0
    assert "already cataloged" in state.catalog_report


def test_refresh_forces_llm_call_even_when_cataloged(tmp_path, monkeypatch, seeded_db):
    catalog_path = str(tmp_path / "catalog.json")
    monkeypatch.setattr("utils.data_catalog.CATALOG_PATH", catalog_path)

    catalog = merge_descriptions(
        catalog={},
        descriptions=[
            {"table": "users", "column": "id", "description": "unique user id"},
            {"table": "users", "column": "name", "description": "display name"},
        ],
    )
    save_catalog(catalog)

    calls = []

    class FakeResult:
        class _Desc:
            def model_dump(self_inner):
                return {"table": "users", "column": "id", "description": "refreshed"}
        descriptions = [_Desc()]

    def _fake_invoke(*args, **kwargs):
        calls.append(args)
        return FakeResult()

    monkeypatch.setattr(catalog_agent_module, "invoke_with_resilience", _fake_invoke)

    state = generate_descriptions(DataCatalogSchema(refresh=True))

    assert len(calls) == 1
    assert state.updated_columns == 1
