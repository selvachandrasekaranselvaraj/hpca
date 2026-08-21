from pathlib import Path

from hpca.scheduler import FakeScheduler


def test_fake_scheduler_lifecycle_and_dependency(tmp_path: Path):
    scheduler = FakeScheduler(next_job_id=42)
    job = scheduler.submit(tmp_path / "sub.sh", cwd=tmp_path, dependency="afterok:41")
    assert job == "42"
    assert scheduler.alive(job)
    assert scheduler.submissions[0]["dependency"] == "afterok:41"
    scheduler.states[job] = "RUNNING"
    assert scheduler.state(job) == "RUNNING"
    assert scheduler.cancel(job)
    assert scheduler.state(job) == "CANCELLED"
    assert not scheduler.alive(job)


def test_fake_scheduler_unknown_job_is_safe():
    scheduler = FakeScheduler()
    assert scheduler.state("missing") == "UNKNOWN"
    assert not scheduler.cancel("missing")
