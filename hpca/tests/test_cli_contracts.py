from argparse import Namespace

import pytest

from hpca.pipeline import (_reset_failed_project_tree, build_parser, cmd_stop,
                           cmd_validate)
from hpca.orchestrator.state_tracker import ProjectState


def test_parser_exposes_validation_and_health_json_interfaces():
    parser = build_parser()
    assert parser.parse_args(["validate", "/tmp/x", "--json"]).as_json
    assert parser.parse_args(["health", "/tmp/x", "--json"]).as_json
    assert parser.parse_args(["restart", "/tmp/x"]).command == "restart"


def test_validate_uses_exit_2_for_invalid_project(tmp_path, capsys):
    (tmp_path / "project.yaml").write_text("name: incomplete\n")
    with pytest.raises(SystemExit) as exc:
        cmd_validate(Namespace(project_dir=str(tmp_path), as_json=True))
    assert exc.value.code == 2
    assert '"valid": false' in capsys.readouterr().out


def test_stop_writes_only_project_local_control(tmp_path):
    cmd_stop(Namespace(project_dir=str(tmp_path)))
    assert (tmp_path / ".hpca" / "control.json").is_file()


def test_restart_resets_parent_and_direct_child_failures(tmp_path):
    (tmp_path / "project.yaml").write_text("name: parent\n")
    child = tmp_path / "child"
    child.mkdir()
    (child / "project.yaml").write_text("name: child\n")
    for root in (tmp_path, child):
        state = ProjectState(root)
        state.state["autonomy"] = {"local_attempts": {"h00_design": 5}}
        state.set_stage("h00_design", "FAILED", error="exhausted")

    changed = _reset_failed_project_tree(tmp_path)
    assert [path for path, _ in changed] == [tmp_path, child]
    assert ProjectState(tmp_path).get_stage("h00_design") == "PENDING"
    assert ProjectState(child).state["autonomy"]["local_attempts"] == {}
