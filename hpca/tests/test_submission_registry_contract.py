from pathlib import Path

import pytest

from hpca.registry.submission import SUBMISSIONS, SubmissionDefinition, write_submission


def test_registry_metadata_is_complete_and_inspectable():
    assert SUBMISSIONS
    assert all(isinstance(item, SubmissionDefinition) for item in SUBMISSIONS.values())
    assert SUBMISSIONS["submit_fanout"].family == "local"
    assert SUBMISSIONS["vasp_aimd"].required == {"natoms"}


def test_unknown_parameter_is_rejected(tmp_path: Path):
    with pytest.raises(TypeError, match="unknown parameters.*natmoms"):
        write_submission(tmp_path / "sub.sh", "vasp_aimd", "test", natoms=32, natmoms=32)


def test_missing_parameter_is_rejected(tmp_path: Path):
    with pytest.raises(TypeError, match="missing required parameters.*natoms"):
        write_submission(tmp_path / "sub.sh", "vasp_aimd", "test")


def test_writer_creates_only_requested_executable(tmp_path: Path):
    target = tmp_path / "nested" / "sub.sh"
    assert write_submission(target, "vasp", "test") == target
    assert target.stat().st_mode & 0o111
    assert list(tmp_path.rglob("*")) == [target.parent, target]
