import json

import pytest

from hpca.core.artifacts import record_artifact, verify_artifact


def test_artifact_record_is_relative_checksum_bound_and_append_only(tmp_path):
    artifact = tmp_path / "results" / "msd.csv"
    artifact.parent.mkdir()
    artifact.write_text("t,msd\n0,0\n")
    record = record_artifact(tmp_path, artifact, producer="h06_analysis", kind="table")
    assert record.path == "results/msd.csv"
    assert record.format == "csv"
    assert verify_artifact(tmp_path, record)
    record_artifact(tmp_path, artifact, producer="h06_analysis", kind="table")
    lines = (tmp_path / ".hpca" / "artifacts.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["producer"] == "h06_analysis"
    artifact.write_text("changed")
    assert not verify_artifact(tmp_path, record)


def test_artifact_outside_project_is_rejected(tmp_path):
    outside = tmp_path.parent / "outside-artifact.txt"
    outside.write_text("x")
    try:
        with pytest.raises(ValueError, match="inside project root"):
            record_artifact(tmp_path, outside, producer="test", kind="test")
    finally:
        outside.unlink()
