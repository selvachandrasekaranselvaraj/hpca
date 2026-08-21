"""Tests for hpca.core.type_map utilities."""
import pytest
from pathlib import Path
import tempfile

FIXTURES = Path(__file__).parent / "fixtures"


def test_read_type_map_file():
    from hpca.core.type_map import read_type_map
    result = read_type_map(FIXTURES / "type_map.raw")
    assert result == ["Li", "P", "S", "Cl"]


def test_read_type_map_missing():
    from hpca.core.type_map import read_type_map
    assert read_type_map(Path("/nonexistent/type_map.raw")) == []


def test_mobile_type_id_found():
    from hpca.core.type_map import mobile_type_id
    assert mobile_type_id(["Li", "P", "S", "Cl"], "Li") == 1
    assert mobile_type_id(["Li", "P", "S", "Cl"], "S") == 3


def test_mobile_type_id_not_found():
    from hpca.core.type_map import mobile_type_id
    assert mobile_type_id(["Li", "P", "S"], "Na") is None


def test_lammps_data_type_id_missing_file():
    from hpca.core.type_map import lammps_data_type_id
    result = lammps_data_type_id(Path("/nonexistent/data.lammps"), "Li")
    assert result is None


def test_lammps_data_type_id_parse():
    from hpca.core.type_map import lammps_data_type_id
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lammps", delete=False) as f:
        f.write("LAMMPS data file\n\n4 atom types\n\nMasses\n\n")
        f.write("1 6.941 # Li\n2 30.974 # P\n3 32.065 # S\n4 35.453 # Cl\n\n")
        fname = f.name
    from hpca.core.type_map import lammps_data_type_id
    assert lammps_data_type_id(Path(fname), "Li") == 1
    assert lammps_data_type_id(Path(fname), "S") == 3
