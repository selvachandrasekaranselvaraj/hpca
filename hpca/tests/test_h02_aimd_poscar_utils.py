"""Unit tests for h02_aimd's random dataset-box POSCAR generation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from hpca.orchestrator.handlers.h02_aimd._poscar_utils import make_random_poscar

_POSCAR = """Liquid box
1.0
  15.0  0.0  0.0
  0.0  15.0  0.0
  0.0  0.0  15.0
   C   H
  40  80
Direct
""" + "\n".join(f"  {x:.6f}  {x:.6f}  {x:.6f}" for x in np.linspace(0.01, 0.99, 120)) + "\n"


def _min_image_pair_distances(frac: np.ndarray, cell_T: np.ndarray) -> np.ndarray:
    n = frac.shape[0]
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = frac[i] - frac[j]
            diff -= np.round(diff)
            dists.append(np.linalg.norm(cell_T @ diff))
    return np.array(dists)


def test_random_poscar_respects_minimum_atom_distance(tmp_path: Path):
    """No two atoms should land closer than min_dist under periodic boundary conditions.

    Regression guard for the unconstrained-random-placement bug that produced
    overlapping atoms and crashed VASP's SCF on liquid AIMD dataset boxes.
    """
    source = tmp_path / "POSCAR"
    source.write_text(_POSCAR)
    out = tmp_path / "random_poscar" / "POSCAR"

    make_random_poscar(source, out, scale=1.0, rng_seed=7, min_dist=1.2)

    lines = out.read_text().splitlines()
    cell = np.array([[float(v) for v in lines[i].split()[:3]] for i in (2, 3, 4)])
    cell_T = cell.T
    frac = np.array([[float(v) for v in ln.split()[:3]] for ln in lines[8:8 + 120]])

    assert frac.shape == (120, 3)
    dists = _min_image_pair_distances(frac, cell_T)
    assert dists.min() >= 1.2 - 1e-9


def test_random_poscar_is_reproducible_for_a_given_seed(tmp_path: Path):
    source = tmp_path / "POSCAR"
    source.write_text(_POSCAR)
    out_a = tmp_path / "a" / "POSCAR"
    out_b = tmp_path / "b" / "POSCAR"

    make_random_poscar(source, out_a, scale=1.0, rng_seed=3, min_dist=1.2)
    make_random_poscar(source, out_b, scale=1.0, rng_seed=3, min_dist=1.2)

    assert out_a.read_text() == out_b.read_text()
