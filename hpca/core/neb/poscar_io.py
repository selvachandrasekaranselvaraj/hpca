"""hpca/core/neb/poscar_io.py — POSCAR I/O helpers for the NEB subpackage."""
from __future__ import annotations

from pathlib import Path

try:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Poscar
except ImportError:
    import sys
    print("ERROR: pymatgen is required. Install with: pip install pymatgen")
    sys.exit(1)


def read_structure(poscar_path: Path) -> Structure:
    """
    Read a POSCAR/VASP file and return a pymatgen Structure object.

    Args:
        poscar_path (Path): Path to the POSCAR file.

    Returns:
        Structure: Pymatgen Structure object containing cell, sites, and coordinates.
    """
    return Structure.from_file(str(poscar_path))


def write_poscar(structure: Structure, path: Path) -> None:
    """
    Write a pymatgen Structure object to a POSCAR file.

    Args:
        structure (Structure): Pymatgen Structure object.
        path (Path): Output file path.
    """
    poscar = Poscar(structure)
    poscar.write_file(str(path))
