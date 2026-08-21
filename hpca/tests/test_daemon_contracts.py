import json
from pathlib import Path

import pytest

from hpca.daemon.control import get_desired_state, set_desired_state
from hpca.daemon.schemas import ProjectRequest


def test_project_control_is_local_idempotent_and_versioned(tmp_path: Path):
    path = set_desired_state(tmp_path, "running", "tester")
    first = json.loads(path.read_text())
    path = set_desired_state(tmp_path, "RUNNING", "tester")
    assert path == tmp_path / ".hpca" / "control.json"
    assert first["schema_version"] == 1
    assert get_desired_state(tmp_path) == "RUNNING"


def test_unknown_control_schema_is_rejected(tmp_path: Path):
    path = tmp_path / ".hpca" / "control.json"
    path.parent.mkdir()
    path.write_text('{"schema_version": 99, "desired_state": "RUNNING"}')
    with pytest.raises(ValueError, match="unsupported schema_version"):
        get_desired_state(tmp_path)


def test_request_schema_rejects_extra_fields():
    value = {name: "x" for name in ProjectRequest.__dataclass_fields__}
    value["schema_version"] = 1
    value["unexpected"] = True
    with pytest.raises(ValueError, match="unknown request fields"):
        ProjectRequest.from_mapping(value)
