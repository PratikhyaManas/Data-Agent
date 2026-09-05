"""Tests for utils/scheduler.py. All deterministic - the ETL runner is injected."""
import json
import time

import utils.scheduler as sched


def _fresh_schedule(tmp_path, monkeypatch):
    schedule_path = str(tmp_path / "schedule.json")
    monkeypatch.setattr(sched, "SCHEDULE_PATH", schedule_path)
    return schedule_path


def test_resolve_interval_seconds_presets():
    assert sched.resolve_interval_seconds("hourly") == 3600
    assert sched.resolve_interval_seconds("daily") == 86400
    assert sched.resolve_interval_seconds("weekly") == 7 * 86400


def test_resolve_interval_seconds_raw_number():
    assert sched.resolve_interval_seconds(120) == 120
    assert sched.resolve_interval_seconds("300") == 300


def test_resolve_interval_seconds_rejects_unknown_preset():
    import pytest
    with pytest.raises(ValueError):
        sched.resolve_interval_seconds("fortnightly")


def test_resolve_interval_seconds_rejects_non_positive():
    import pytest
    with pytest.raises(ValueError):
        sched.resolve_interval_seconds(0)


def test_add_job_is_due_immediately(tmp_path, monkeypatch):
    _fresh_schedule(tmp_path, monkeypatch)
    job_id = sched.add_job("extract things", interval=60, name="job1")

    jobs = sched.list_jobs()
    assert job_id in jobs
    assert jobs[job_id]["request"] == "extract things"
    assert jobs[job_id]["interval_seconds"] == 60

    due = sched.due_jobs()
    assert len(due) == 1
    assert due[0]["id"] == "job1"


def test_disabled_job_is_never_due(tmp_path, monkeypatch):
    _fresh_schedule(tmp_path, monkeypatch)
    sched.add_job("extract things", interval=60, name="job1", enabled=False)
    assert sched.due_jobs() == []


def test_remove_job(tmp_path, monkeypatch):
    _fresh_schedule(tmp_path, monkeypatch)
    sched.add_job("x", interval=60, name="job1")
    assert sched.remove_job("job1") is True
    assert sched.list_jobs() == {}
    assert sched.remove_job("job1") is False  # already gone


def test_enable_disable_job(tmp_path, monkeypatch):
    _fresh_schedule(tmp_path, monkeypatch)
    sched.add_job("x", interval=60, name="job1")
    assert sched.set_job_enabled("job1", False) is True
    assert sched.list_jobs()["job1"]["enabled"] is False
    assert sched.due_jobs() == []
    sched.set_job_enabled("job1", True)
    assert len(sched.due_jobs()) == 1


def test_run_due_jobs_invokes_runner_and_reschedules(tmp_path, monkeypatch):
    _fresh_schedule(tmp_path, monkeypatch)
    sched.add_job("extract pokemon", interval=100, name="job1")
    now = time.time()  # captured after add_job, so it's >= the job's creation-time next_run_at

    calls = []

    def fake_runner(request):
        calls.append(request)
        return "extracted 20 rows"

    summaries = sched.run_due_jobs(fake_runner, now=now)

    assert calls == ["extract pokemon"]
    assert summaries == [{"id": "job1", "request": "extract pokemon", "result": "extracted 20 rows"}]

    jobs = sched.list_jobs()
    assert jobs["job1"]["last_run_at"] == now
    assert jobs["job1"]["next_run_at"] == now + 100
    assert jobs["job1"]["last_result"]["status"] == "ok"

    # Not due again immediately after rescheduling
    assert sched.due_jobs(now=now + 1) == []
    assert len(sched.due_jobs(now=now + 101)) == 1


def test_run_due_jobs_records_runner_errors_without_raising(tmp_path, monkeypatch):
    _fresh_schedule(tmp_path, monkeypatch)
    sched.add_job("bad request", interval=60, name="job1")
    now = time.time()  # captured after add_job, see comment in the test above

    def failing_runner(request):
        raise RuntimeError("boom")

    summaries = sched.run_due_jobs(failing_runner, now=now)

    assert summaries[0]["error"] == "boom"
    jobs = sched.list_jobs()
    assert jobs["job1"]["last_result"]["status"] == "error"
    assert jobs["job1"]["next_run_at"] == now + 60  # still reschedules despite failure


def test_run_due_jobs_skips_jobs_not_yet_due(tmp_path, monkeypatch):
    _fresh_schedule(tmp_path, monkeypatch)
    sched.add_job("job a", interval=60, name="a")
    sched.add_job("job b", interval=60, name="b")
    now = time.time()  # captured after both add_job calls, see comment above

    # Run once so both jobs are due and get rescheduled ~60s into the future.
    summaries = sched.run_due_jobs(lambda r: "ok", now=now)
    assert {s["id"] for s in summaries} == {"a", "b"}

    # Neither should be due again immediately after rescheduling.
    assert sched.due_jobs(now=now + 1) == []


def test_schedule_persists_to_disk_as_json(tmp_path, monkeypatch):
    schedule_path = _fresh_schedule(tmp_path, monkeypatch)
    sched.add_job("x", interval=60, name="job1")

    with open(schedule_path) as f:
        raw = json.load(f)
    assert "job1" in raw
    assert raw["job1"]["request"] == "x"
