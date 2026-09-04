"""Tests for utils/query_cache.py."""
import importlib
import os
import time

import utils.query_cache as qc


def _worker_for_concurrent_cache(path, key_id):
    import importlib
    import utils.query_cache as qc_worker

    importlib.reload(qc_worker)
    qc_worker.CACHE_PATH = path
    qc_worker.CACHE_TTL_SECONDS = 86400
    key = qc_worker.build_cache_key(f"question_{key_id}", "schema")
    qc_worker.set_cached_query(key, f"SELECT {key_id}", "")


def _fresh_cache(tmp_path, monkeypatch, ttl=None):
    cache_path = str(tmp_path / "cache.json")
    monkeypatch.setattr(qc, "CACHE_PATH", cache_path)
    if ttl is not None:
        monkeypatch.setattr(qc, "CACHE_TTL_SECONDS", ttl)
    return cache_path


def test_miss_on_empty_cache(tmp_path, monkeypatch):
    _fresh_cache(tmp_path, monkeypatch)
    key = qc.build_cache_key("top 5 users", "schema text")
    assert qc.get_cached_query(key) is None


def test_write_then_hit(tmp_path, monkeypatch):
    _fresh_cache(tmp_path, monkeypatch)
    key = qc.build_cache_key("top 5 users", "schema text")
    qc.set_cached_query(key, "SELECT * FROM users LIMIT 5", "no changes needed")

    entry = qc.get_cached_query(key)
    assert entry is not None
    assert entry["sql"] == "SELECT * FROM users LIMIT 5"
    assert entry["optimizer_notes"] == "no changes needed"


def test_different_schema_produces_different_key(tmp_path, monkeypatch):
    _fresh_cache(tmp_path, monkeypatch)
    key_a = qc.build_cache_key("top 5 users", "schema A")
    key_b = qc.build_cache_key("top 5 users", "schema B")
    assert key_a != key_b


def test_question_normalization_is_case_and_whitespace_insensitive(tmp_path, monkeypatch):
    _fresh_cache(tmp_path, monkeypatch)
    key_a = qc.build_cache_key("Top 5 Users", "schema")
    key_b = qc.build_cache_key("  top 5 users  ", "schema")
    assert key_a == key_b


def test_ttl_expiry(tmp_path, monkeypatch):
    _fresh_cache(tmp_path, monkeypatch, ttl=1)
    key = qc.build_cache_key("q", "schema")
    qc.set_cached_query(key, "SELECT 1", "")
    assert qc.get_cached_query(key) is not None
    time.sleep(1.2)
    assert qc.get_cached_query(key) is None


def test_clear_cache_removes_all_entries(tmp_path, monkeypatch):
    _fresh_cache(tmp_path, monkeypatch)
    key = qc.build_cache_key("q", "schema")
    qc.set_cached_query(key, "SELECT 1", "")
    qc.clear_cache()
    assert qc.get_cached_query(key) is None


def test_corrupt_cache_file_treated_as_empty(tmp_path, monkeypatch):
    cache_path = _fresh_cache(tmp_path, monkeypatch)
    with open(cache_path, "w") as f:
        f.write("{not valid json")
    key = qc.build_cache_key("q", "schema")
    assert qc.get_cached_query(key) is None


def test_expired_entries_purged_from_file_on_next_write(tmp_path, monkeypatch):
    """
    get_cached_query() already treats an expired entry as a miss, but
    without active purging the underlying file would grow forever - every
    distinct question ever asked stays in it permanently. This confirms
    stale entries are actually removed from disk on the next write, not
    just skipped on read.
    """
    import json
    import time

    cache_path = _fresh_cache(tmp_path, monkeypatch, ttl=1)
    key1 = qc.build_cache_key("q1", "s")
    key2 = qc.build_cache_key("q2", "s")
    qc.set_cached_query(key1, "SELECT 1", "")
    qc.set_cached_query(key2, "SELECT 2", "")

    with open(cache_path) as f:
        assert len(json.load(f)) == 2

    time.sleep(1.2)
    key3 = qc.build_cache_key("q3", "s")
    qc.set_cached_query(key3, "SELECT 3", "")  # this write should purge key1, key2

    with open(cache_path) as f:
        raw = json.load(f)
    assert len(raw) == 1
    assert key3 in raw


def test_write_never_raises_on_failure(tmp_path, monkeypatch):
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("x")
    monkeypatch.setattr(qc, "CACHE_PATH", str(blocker / "cache.json"))

    qc.set_cached_query("somekey", "SELECT 1", "")  # must not raise
    qc.clear_cache()  # must not raise either


def test_concurrent_writers_lose_no_updates(tmp_path, monkeypatch):
    """
    Regression test for a real race found by hand: without file locking,
    concurrent writers to the same cache file lost updates (confirmed: 3/30
    lost) and could even crash outright when two processes collided on the
    same shared ".tmp" filename mid-write. Spawns real separate processes
    (not threads - needs true concurrent file access, not just interleaved
    bytecode) each writing a distinct key, and asserts every single one
    survives in the final file.
    """
    import multiprocessing
    import json

    cache_path = str(tmp_path / "concurrent_cache.json")

    n_writers = 25
    procs = [
        multiprocessing.Process(target=_worker_for_concurrent_cache, args=(cache_path, i))
        for i in range(n_writers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    with open(cache_path) as f:
        final = json.load(f)

    assert len(final) == n_writers, (
        f"lost {n_writers - len(final)} update(s) to a concurrent-write race"
    )
