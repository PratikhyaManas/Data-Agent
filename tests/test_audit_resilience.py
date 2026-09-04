"""
Additional tests for utils/audit.py: fail-safe writes and rotation.
(test_audit.py already covers basic write/read; this covers the
resilience improvements added afterward.)
"""
import json
import os

import utils.audit as audit_module


def test_log_event_never_raises_on_write_failure(tmp_path, monkeypatch):
    # parent path component is a file, not a directory - fails for anyone
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("x")
    monkeypatch.setattr(audit_module, "LOG_PATH", str(blocker / "audit.jsonl"))

    audit_module.log_event("test_event", foo="bar")  # must not raise


def test_read_recent_returns_correct_tail_with_many_entries(tmp_path, monkeypatch):
    log_path = str(tmp_path / "audit.jsonl")
    monkeypatch.setattr(audit_module, "LOG_PATH", log_path)

    for i in range(50):
        audit_module.log_event("bulk", i=i)

    recent = audit_module.read_recent(10)
    assert len(recent) == 10
    assert recent[-1]["i"] == 49
    assert recent[0]["i"] == 40


def test_rotation_triggers_at_size_threshold(tmp_path, monkeypatch):
    log_path = str(tmp_path / "audit.jsonl")
    monkeypatch.setattr(audit_module, "LOG_PATH", log_path)
    monkeypatch.setattr(audit_module, "MAX_LOG_SIZE_BYTES", 500)

    for i in range(100):
        audit_module.log_event("rotation_test", i=i, padding="x" * 20)

    assert os.path.exists(log_path + ".1"), "rotation should have created a .1 backup"
    assert os.path.getsize(log_path) < 5000, "current log should be much smaller after rotation"


def test_rotation_keeps_only_one_backup_generation(tmp_path, monkeypatch):
    log_path = str(tmp_path / "audit.jsonl")
    monkeypatch.setattr(audit_module, "LOG_PATH", log_path)
    monkeypatch.setattr(audit_module, "MAX_LOG_SIZE_BYTES", 300)

    for i in range(200):
        audit_module.log_event("rotation_test", i=i, padding="x" * 20)

    # only ONE backup generation should exist, not audit_log.jsonl.1.1 etc
    assert os.path.exists(log_path + ".1")
    assert not os.path.exists(log_path + ".1.1")
