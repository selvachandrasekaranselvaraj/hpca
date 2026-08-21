"""Tests for hpca.io.lammps parsing functions."""
import pytest
from pathlib import Path
import tempfile

FIXTURES = Path(__file__).parent / "fixtures"
LOG_FILE = FIXTURES / "log.lammps"


def test_read_log_thermo_returns_list():
    from hpca.io.lammps import read_log_thermo
    data = read_log_thermo(LOG_FILE)
    assert isinstance(data, list)
    assert len(data) >= 1


def test_read_log_thermo_has_temp():
    from hpca.io.lammps import read_log_thermo
    data = read_log_thermo(LOG_FILE)
    assert all("Temp" in row or "temp" in row or any("T" in k for k in row) for row in data[:1])


def test_count_dump_frames_empty():
    from hpca.io.lammps import count_dump_frames
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lmp", delete=False) as f:
        fname = f.name
    assert count_dump_frames(Path(fname)) == 0


def test_count_dump_frames():
    from hpca.io.lammps import count_dump_frames
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lmp", delete=False) as f:
        for i in range(3):
            f.write(f"ITEM: TIMESTEP\n{i*100}\nITEM: NUMBER OF ATOMS\n4\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n")
            f.write("ITEM: ATOMS id type x y z\n")
            for j in range(4):
                f.write(f"{j+1} 1 {j} {j} {j}\n")
        fname = f.name
    assert count_dump_frames(Path(fname)) == 3
