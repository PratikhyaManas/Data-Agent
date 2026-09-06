"""Tests for utils/audit.py."""
import json
import os

import utils.audit as audit_module


def test_log_event_writes_jsonl(tmp_path, monkeypatch):
    log_path = str(tmp_path / "audit.jsonl")
    monkeypatch.setattr(audit_module, "LOG_PATH", log_path)

    audit_module.log_event("test_event", foo="bar", n=1)

    assert os.path.exists(log_path)
    with open(log_path) as f:
        entry = json.loads(f.readline())
    assert entry["event_type"] == "test_event"
    assert entry["foo"] == "bar"
    assert entry["n"] == 1
    assert "timestamp" in entry


def test_read_recent_returns_last_n(tmp_path, monkeypatch):
    log_path = str(tmp_path / "audit.jsonl")
    monkeypatch.setattr(audit_module, "LOG_PATH", log_path)

    for i in range(5):
        audit_module.log_event("test_event", i=i)

    recent = audit_module.read_recent(2)
    assert len(recent) == 2
    assert recent[-1]["i"] == 4


def test_read_recent_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(audit_module, "LOG_PATH", str(tmp_path / "nope.jsonl"))
    assert audit_module.read_recent() == []
