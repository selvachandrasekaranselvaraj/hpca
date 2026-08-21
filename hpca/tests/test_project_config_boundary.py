from pathlib import Path

import pytest

from hpca.core.project_io import migrate_project_yaml, read_project_yaml
from hpca.core.project_schema import migrate, validate


def test_non_mapping_root_has_field_specific_error(tmp_path: Path):
    (tmp_path / "project.yaml").write_text("- not\n- a\n- mapping\n")
    with pytest.raises(ValueError, match="root must be a mapping"):
        read_project_yaml(tmp_path)
    assert validate([]) == ["project root must be a mapping, got list"]


def test_migration_is_non_mutating_and_has_one_canonical_implementation():
    original = {"system": "solid", "aimd_temperatures": [300], "stages": {"classical_md": True}}
    expected = migrate(original)
    assert migrate_project_yaml(original) == expected
    assert original == {"system": "solid", "aimd_temperatures": [300], "stages": {"classical_md": True}}
    assert expected["workflow_version"] == 2


def test_nested_sections_are_type_checked():
    errors = validate({"name": "x", "category": "solid", "T_ref": 300,
                       "simulation": [], "stages": []})
    assert "simulation must be a mapping" in errors
    assert "stages must be a mapping" in errors
