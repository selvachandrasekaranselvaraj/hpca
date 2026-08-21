"""Tests for hpca.io.vasp parsing functions."""
import pytest
from pathlib import Path
import tempfile

FIXTURES = Path(__file__).parent / "fixtures"
POSCAR = FIXTURES / "POSCAR_Li6PS5Cl"


def test_natoms_from_poscar():
    from hpca.io.vasp import natoms_from_poscar
    assert natoms_from_poscar(POSCAR) == 52


def test_elements_from_poscar():
    from hpca.io.vasp import elements_from_poscar
    elems = elements_from_poscar(POSCAR)
    assert "Li" in elems
    assert "P" in elems
    assert "S" in elems
    assert "Cl" in elems


def test_poscar_lattice_params():
    from hpca.io.vasp import poscar_lattice_params
    params = poscar_lattice_params(POSCAR)
    assert len(params) == 3
    assert abs(params[0] - 9.86) < 0.1


def test_outcar_converged_false_on_missing():
    from hpca.io.vasp import outcar_converged
    assert outcar_converged(Path("/nonexistent/OUTCAR")) == False


def test_outcar_converged_true():
    from hpca.io.vasp import outcar_converged
    with tempfile.NamedTemporaryFile(mode="w", suffix="OUTCAR", delete=False) as f:
        f.write("some header\n" * 100)
        f.write("reached required accuracy - stopping structural energy minimisation\n")
        fname = f.name
    assert outcar_converged(Path(fname)) == True


def test_outcar_converged_general_timing():
    from hpca.io.vasp import outcar_converged
    with tempfile.NamedTemporaryFile(mode="w", suffix="OUTCAR", delete=False) as f:
        f.write("some header\n" * 100)
        f.write("General timing and accounting informations for this job:\n")
        fname = f.name
    assert outcar_converged(Path(fname)) == True


def test_incar_read_write():
    from hpca.io.vasp import read_incar, write_incar
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "INCAR"
        p.write_text("ENCUT = 520\nNSW = 100\nISMEAR = 0\n")
        d = read_incar(p)
        assert d.get("ENCUT") in ("520", 520, "520.0") or str(d.get("ENCUT")).startswith("520")
