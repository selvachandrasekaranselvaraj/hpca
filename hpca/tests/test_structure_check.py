"""Unit tests for hpca.core.structure_check, including POTCAR-derived (PAW-aware)
per-pair minimum distances."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from hpca.core.structure_check import (
    check_and_fix_poscar,
    check_and_fix_poscar_potcar,
    min_distance_poscar,
    parse_potcar_rcores,
    pairwise_min_distances_from_potcar,
)

_POTCAR = """ PAW_PBE Li 17Jan2003
   parameters from PSCTR are:
   VRHFIN =Li: s1p0
   TITEL  = PAW_PBE Li 17Jan2003
   LEXCH  = PE
   RCORE  =    2.050    outmost cutoff radius
   End of Dataset
 PAW_PBE H 15Jun2001
   parameters from PSCTR are:
   VRHFIN =H
   TITEL  = PAW_PBE H 15Jun2001
   LEXCH  = PE
   RCORE  =    1.100    outmost cutoff radius
   End of Dataset
"""

_TWO_LI_POSCAR = """Two Li, 1.5 A apart
1.0
  10.0  0.0  0.0
  0.0  10.0  0.0
  0.0  0.0  10.0
   Li
   2
Cartesian
  5.00  5.00  5.00
  6.50  5.00  5.00
"""


def test_parse_potcar_rcores(tmp_path: Path):
    potcar = tmp_path / "POTCAR"
    potcar.write_text(_POTCAR)
    assert parse_potcar_rcores(potcar) == {"Li": 2.050, "H": 1.100}


def test_pairwise_min_distances_from_potcar(tmp_path: Path):
    potcar = tmp_path / "POTCAR"
    potcar.write_text(_POTCAR)
    mat = pairwise_min_distances_from_potcar(["Li", "Li", "H"], potcar, factor=0.8)
    assert mat.shape == (3, 3)
    assert np.isclose(mat[0, 1], 0.8 * (2.050 + 2.050))
    assert np.isclose(mat[0, 2], 0.8 * (2.050 + 1.100))


def test_flat_scalar_min_dist_accepts_a_potcar_unsafe_contact(tmp_path: Path):
    """1.5 A clears the flat 1.0 A default even though it violates PAW RCORE overlap."""
    poscar = tmp_path / "POSCAR"
    poscar.write_text(_TWO_LI_POSCAR)
    modified = check_and_fix_poscar(poscar, min_dist=1.0)
    assert not modified
    assert min_distance_poscar(poscar) < 0.8 * (2.050 + 2.050)


def test_potcar_aware_check_pushes_atoms_past_combined_rcore(tmp_path: Path):
    """Regression guard: a flat 1.0 A floor is smaller than combined RCORE for
    elements like Li (2.05 A) — random non-bonded contacts that pass the flat
    check can still overlap PAW augmentation spheres and crash VASP's SCF."""
    poscar = tmp_path / "POSCAR"
    poscar.write_text(_TWO_LI_POSCAR)
    potcar = tmp_path / "POTCAR"
    potcar.write_text(_POTCAR)

    modified = check_and_fix_poscar_potcar(poscar, potcar, factor=0.8)

    assert modified
    assert min_distance_poscar(poscar) >= 0.8 * (2.050 + 2.050)


def test_potcar_aware_check_falls_back_to_scalar_on_unparseable_potcar(tmp_path: Path):
    poscar = tmp_path / "POSCAR"
    poscar.write_text(_TWO_LI_POSCAR)
    potcar = tmp_path / "not_a_potcar.txt"
    potcar.write_text("garbage, no TITEL/RCORE lines here\n")

    # Elements missing from the (unparseable) POTCAR fall back to a distance
    # derived from the flat default, so this must not raise.
    check_and_fix_poscar_potcar(poscar, potcar, factor=0.8)
