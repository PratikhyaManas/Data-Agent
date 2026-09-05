"""Tests for utils/data_catalog.py. All deterministic - no LLM involved."""
import json

import utils.data_catalog as catalog_module


def _fresh_catalog(tmp_path, monkeypatch):
    path = str(tmp_path / "catalog.json")
    monkeypatch.setattr(catalog_module, "CATALOG_PATH", path)
    return path


def test_load_catalog_empty_when_missing(tmp_path, monkeypatch):
    _fresh_catalog(tmp_path, monkeypatch)
    assert catalog_module.load_catalog() == {}


def test_columns_needing_descriptions_all_missing_initially():
    tables = {"users": ["id", "name", "rating"]}
    missing = catalog_module.columns_needing_descriptions(tables, catalog={})
    assert set(missing) == {("users", "id"), ("users", "name"), ("users", "rating")}


def test_columns_needing_descriptions_skips_already_cataloged():
    tables = {"users": ["id", "name", "rating"]}
    catalog = {"users": {"id": {"description": "primary key", "updated_at": 0}}}
    missing = catalog_module.columns_needing_descriptions(tables, catalog)
    assert set(missing) == {("users", "name"), ("users", "rating")}


def test_columns_needing_descriptions_refresh_forces_all():
    tables = {"users": ["id", "name"]}
    catalog = {"users": {"id": {"description": "pk", "updated_at": 0}, "name": {"description": "n", "updated_at": 0}}}
    missing = catalog_module.columns_needing_descriptions(tables, catalog, refresh=True)
    assert set(missing) == {("users", "id"), ("users", "name")}


def test_merge_descriptions_adds_new_entries():
    merged = catalog_module.merge_descriptions(
        catalog={},
        descriptions=[{"table": "users", "column": "id", "description": "unique user identifier"}],
    )
    assert merged["users"]["id"]["description"] == "unique user identifier"
    assert "updated_at" in merged["users"]["id"]


def test_merge_descriptions_preserves_untouched_columns():
    existing = {"users": {"id": {"description": "old", "updated_at": 1}}}
    merged = catalog_module.merge_descriptions(
        catalog=existing,
        descriptions=[{"table": "users", "column": "name", "description": "display name"}],
    )
    assert merged["users"]["id"]["description"] == "old"  # untouched
    assert merged["users"]["name"]["description"] == "display name"  # newly added


def test_merge_descriptions_overwrites_refreshed_columns():
    existing = {"users": {"id": {"description": "old", "updated_at": 1}}}
    merged = catalog_module.merge_descriptions(
        catalog=existing,
        descriptions=[{"table": "users", "column": "id", "description": "new"}],
    )
    assert merged["users"]["id"]["description"] == "new"
    assert merged["users"]["id"]["updated_at"] > 1


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    path = _fresh_catalog(tmp_path, monkeypatch)
    catalog = catalog_module.merge_descriptions(
        catalog={}, descriptions=[{"table": "users", "column": "id", "description": "pk"}]
    )
    assert catalog_module.save_catalog(catalog) is True

    with open(path) as f:
        raw = json.load(f)
    assert raw["users"]["id"]["description"] == "pk"

    reloaded = catalog_module.load_catalog()
    assert reloaded["users"]["id"]["description"] == "pk"


def test_corrupt_catalog_file_treated_as_empty(tmp_path, monkeypatch):
    path = _fresh_catalog(tmp_path, monkeypatch)
    with open(path, "w") as f:
        f.write("{not valid json")
    assert catalog_module.load_catalog() == {}
