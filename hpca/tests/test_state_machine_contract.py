import json

import pytest

from hpca.orchestrator.state_tracker import ProjectState, StateCorruptionError


def test_state_transitions_are_validated_recorded_and_revisioned(tmp_path):
    state = ProjectState(tmp_path)
    state.set_stage("h01_dft", "RUNNING", job="42")
    state.set_stage("h01_dft", "COMPLETE")
    assert state.get_handler("h01_dft")["history"][-1]["to"] == "COMPLETE"
    persisted = json.loads(state.path.read_text())
    assert persisted["revision"] == 2


def test_terminal_stage_cannot_jump_directly_to_running(tmp_path):
    state = ProjectState(tmp_path)
    state.set_stage("h01_dft", "COMPLETE")
    with pytest.raises(ValueError, match="COMPLETE -> RUNNING"):
        state.set_stage("h01_dft", "RUNNING")


def test_corrupt_schema_is_not_resumed(tmp_path):
    path = tmp_path / "logs" / "orchestrator_state.json"
    path.parent.mkdir()
    path.write_text('{"schema_version": 99, "handlers": {}}')
    original = path.read_text()
    with pytest.raises(StateCorruptionError, match="Refusing to resume"):
        ProjectState(tmp_path)
    assert path.read_text() == original


def test_explicit_restart_clears_failure_metadata_and_attempt_budget(tmp_path):
    state = ProjectState(tmp_path)
    state.state["autonomy"] = {
        "local_attempts": {"h00_design": 5},
        "submissions": {"h01_dft": 2},
        "total_submissions": 2,
        "in_progress": {"h00_design": {"pid": 123}},
    }
    state.set_stage("h00_design", "FAILED", error="budget exhausted", failed_at="now")
    state.set_stage("h01_dft", "FAILED", error="submission failed", job="42")

    assert state.reset_failed_handlers() == ["h00_design", "h01_dft"]
    assert state.get_stage("h00_design") == "PENDING"
    assert "error" not in state.get_handler("h00_design")
    assert "job" not in state.get_handler("h01_dft")
    assert state.state["autonomy"]["local_attempts"] == {}
    assert state.state["autonomy"]["submissions"] == {}
    assert state.state["autonomy"]["total_submissions"] == 0
