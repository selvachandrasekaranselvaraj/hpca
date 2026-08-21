"""Tests for fail-closed unattended execution policy."""
from __future__ import annotations

from pathlib import Path

from hpca.core.autonomy import AutonomyPolicy
from hpca.orchestrator.state_tracker import ProjectState


def test_attended_mode_requires_explicit_design_flag(tmp_path: Path):
    designed = tmp_path / "designed_structures"
    designed.mkdir()
    (designed / "DESIGN_COMPLETE.md").write_text("validated design")
    assert not AutonomyPolicy().design_approved(tmp_path)
    (designed / "simulation_approved.flag").touch()
    assert AutonomyPolicy().design_approved(tmp_path)


def test_unattended_mode_auto_approves_completed_design(tmp_path: Path):
    designed = tmp_path / "designed_structures"
    designed.mkdir()
    (designed / "DESIGN_COMPLETE.md").write_text("validated design")
    policy = AutonomyPolicy.from_project({"autonomy": {"mode": "unattended"}})
    assert policy.design_approved(tmp_path)
    assert (tmp_path / "design" / "simulation_approved.flag").exists()


def test_wizard_v2_autonomous_project_migrates_missing_policy(tmp_path: Path):
    designed = tmp_path / "designed_structures"
    designed.mkdir()
    (designed / "DESIGN_COMPLETE.md").write_text("validated design")
    project = {
        "workflow_version": 2,
        "execution_mode": "slurm",
        "stages": {"design": True, "dft": {"vc_relax": True}, "cmd": True},
    }
    policy = AutonomyPolicy.from_project(project)
    assert policy.unattended
    assert policy.design_approved(tmp_path)
    assert (tmp_path / "design" / "simulation_approved.flag").exists()


def test_explicit_attended_mode_overrides_v2_migration(tmp_path: Path):
    designed = tmp_path / "designed_structures"
    designed.mkdir()
    (designed / "DESIGN_COMPLETE.md").write_text("validated design")
    project = {
        "workflow_version": 2,
        "stages": {"design": True, "cmd": True},
        "autonomy": {"mode": "attended"},
    }
    policy = AutonomyPolicy.from_project(project)
    assert not policy.unattended
    assert not policy.design_approved(tmp_path)
    assert not (tmp_path / "design" / "simulation_approved.flag").exists()


def test_attempt_budget_is_durable_and_fail_closed(tmp_path: Path):
    state = ProjectState(tmp_path)
    policy = AutonomyPolicy(mode="unattended", max_stage_submissions=2,
                            max_total_submissions=2)
    for _ in range(2):
        allowed, reason = policy.may_attempt("h01_dft.opt", state, local=False)
        assert allowed, reason
        policy.record_attempt("h01_dft.opt", state, local=False)
    allowed, reason = policy.may_attempt("h01_dft.opt", state, local=False)
    assert not allowed
    assert "budget exhausted" in reason


def test_stage_allowlist_accepts_parent_for_substage(tmp_path: Path):
    state = ProjectState(tmp_path)
    policy = AutonomyPolicy(mode="unattended", allowed_stages=("h01_dft",))
    assert policy.may_attempt("h01_dft.opt", state, local=False)[0]
    assert not policy.may_attempt("h04_mlip", state, local=False)[0]


def test_graceful_interruption_refunds_local_attempt(tmp_path: Path):
    state = ProjectState(tmp_path)
    policy = AutonomyPolicy(mode="unattended")
    policy.record_attempt("h00_design", state, local=True)
    marker = state.state["autonomy"]["in_progress"]["h00_design"]
    assert state.state["autonomy"]["local_attempts"]["h00_design"] == 1

    assert state.rollback_interrupted_attempts(marker["pid"]) == ["h00_design"]
    assert "h00_design" not in state.state["autonomy"]["local_attempts"]
    assert "h00_design" not in state.state["autonomy"]["in_progress"]


def test_explicit_failure_is_not_refunded_on_shutdown(tmp_path: Path):
    state = ProjectState(tmp_path)
    policy = AutonomyPolicy(mode="unattended")
    policy.record_attempt("h00_design", state, local=True)
    marker = state.state["autonomy"]["in_progress"]["h00_design"]
    state.set_stage("h00_design", "FAILED", error="real failure")

    assert state.rollback_interrupted_attempts(marker["pid"]) == []
    assert state.state["autonomy"]["local_attempts"]["h00_design"] == 1


def test_verified_daemon_success_clears_consecutive_attempt_budget(tmp_path: Path):
    state = ProjectState(tmp_path)
    policy = AutonomyPolicy(mode="unattended")
    policy.record_attempt("h06_analysis", state, local=True)
    policy.finish_attempt("h06_analysis", state, local=True)

    policy.clear_successful_local_attempts("h06_analysis", state)

    assert "h06_analysis" not in state.state["autonomy"]["local_attempts"]
    assert policy.may_attempt("h06_analysis", state, local=True)[0]


def test_migrates_only_synthetic_cumulative_local_budget_failure(tmp_path: Path):
    state = ProjectState(tmp_path)
    state.state["handlers"] = {
        "h06_analysis": {
            "stage": "FAILED",
            "error": "h06_analysis local_attempts budget exhausted (5/5)",
        },
        "h00_design": {"stage": "FAILED", "error": "real packing failure"},
    }
    state.state["autonomy"] = {
        "local_attempts": {"h06_analysis": 5, "h00_design": 5},
    }
    state.save()

    assert state.migrate_exhausted_local_attempts() == ["h06_analysis"]
    assert state.get_stage("h06_analysis") == "PENDING"
    assert state.get_stage("h00_design") == "FAILED"
    assert "h06_analysis" not in state.state["autonomy"]["local_attempts"]
    assert state.migrate_exhausted_local_attempts() == []


def test_migrates_h05_existing_job_wait_without_charging_submission(tmp_path: Path):
    state = ProjectState(tmp_path)
    state.state["handlers"] = {
        "h05_cmd": {
            "stage": "FAILED",
            "error": "sbatch returned None",
            "jobs": {"cmd/nvt/300K": "12345"},
        }
    }
    state.state["autonomy"] = {
        "submissions": {"h05_cmd": 2},
        "total_submissions": 2,
    }
    state.save()

    assert state.migrate_h05_wait_contract()
    assert state.get_stage("h05_cmd") == "PENDING"
    assert state.state["autonomy"]["submissions"]["h05_cmd"] == 1
    assert state.state["autonomy"]["total_submissions"] == 1
    assert not state.migrate_h05_wait_contract()
