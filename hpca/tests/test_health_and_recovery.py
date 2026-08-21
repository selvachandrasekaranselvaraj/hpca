from hpca.monitor.health import project_health
from hpca.orchestrator.state_tracker import ProjectState
from hpca.scheduler.errors import FailureClass, classify_scheduler_failure


def test_scheduler_failure_classification_is_fail_closed():
    assert classify_scheduler_failure("slurmctld is busy") is FailureClass.TRANSIENT
    assert classify_scheduler_failure("invalid account") is FailureClass.PERMANENT
    assert classify_scheduler_failure("") is FailureClass.PERMANENT


def test_health_snapshot_is_read_only_and_reports_failure(tmp_path):
    state = ProjectState(tmp_path)
    state.set_stage("h01_dft", "FAILED", error="bad input")
    before = state.path.read_bytes()
    result = project_health(tmp_path)
    assert not result["healthy"]
    assert result["stage_counts"]["FAILED"] == 1
    assert state.path.read_bytes() == before
