"""
structure_check.py — Minimum interatomic distance check and repair for POSCAR files.

Used before any VASP/AIMD/LAMMPS submission to prevent crashes caused by
atoms placed too close together (e.g. after random rattling or PACKMOL failures).

Algorithm
---------
All violating pairs are processed simultaneously each iteration: every atom
accumulates repulsion forces from ALL its close neighbours before any atom
moves. This prevents the fix for pair (i,j) from pushing i into atom k.

Public API
----------
check_and_fix_poscar(path, min_dist=1.0, max_iter=200) -> bool
    Read a POSCAR/CONTCAR, fix any atom pairs closer than min_dist Å (a
    scalar, or a per-pair N×N matrix), write the file back in-place.
    Returns True if the file was modified.

check_and_fix_poscar_potcar(poscar, potcar, factor=0.8, max_iter=200) -> bool
    Same repair, but the minimum distance for each atom pair is
    factor * (RCORE_a + RCORE_b) — the PAW augmentation-sphere radii read
    from the POTCAR. A uniform scalar (e.g. 1.0 Å) is too small for pairs
    like Li (RCORE 2.05 Å) or S (1.90 Å): random non-bonded contacts closer
    than the combined RCORE overlap the augmentation spheres and blow up
    VASP's SCF even though the raw distance "looks" fine.

min_distance_poscar(path) -> float
    Return the minimum interatomic distance in the structure (Å).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

log = logging.getLogger("hpca.orch")

def _sc_cfg(key: str, default):
    """Read a tunable from the platform.yaml ``structure_check:`` section."""
    try:
        from hpca.core.config import Config
        return Config.get().section("structure_check").get(key, default)
    except Exception:
        return default


_MIN_DIST_DEFAULT = _sc_cfg("min_dist_A",    1.0)   # Å
_MAX_ITER         = _sc_cfg("max_iter",      200)   # simultaneous-push iterations
_STEP             = _sc_cfg("push_step",     0.4)   # fraction of overlap pushed per iteration
_MARGIN           = _sc_cfg("push_margin_A", 0.05)  # extra Å on top of min_dist
_POTCAR_FACTOR    = _sc_cfg("potcar_min_dist_factor", 0.8)  # × (RCORE_a + RCORE_b)


# ---------------------------------------------------------------------------
# POSCAR I/O
# ---------------------------------------------------------------------------

def _read_poscar(path: Path) -> tuple[list[str], np.ndarray, list[str], np.ndarray, bool]:
    """Return (header_lines, lattice 3×3, element_list, coords N×3, direct)."""
    lines = path.read_text().splitlines()
    scale   = float(lines[1].strip())
    lattice = np.array([l.split() for l in lines[2:5]], dtype=float)
    if scale > 0:
        lattice *= scale
    else:
        vol = abs(np.linalg.det(lattice))
        lattice *= (-scale / vol) ** (1.0 / 3.0)

    elem_line  = lines[5].split()
    count_line = lines[6].split()
    try:
        counts = [int(c) for c in count_line]
        elements = elem_line
        coord_start = 8
    except ValueError:
        counts = [int(c) for c in elem_line]
        elements = [f"X{i}" for i in range(len(counts))]
        coord_start = 7

    coord_type_line = lines[coord_start - 1].strip().lower()
    direct = coord_type_line.startswith("d")

    n_atoms = sum(counts)
    raw = np.array(
        [l.split()[:3] for l in lines[coord_start:coord_start + n_atoms]],
        dtype=float,
    )
    element_labels: list[str] = []
    for el, cnt in zip(elements, counts):
        element_labels.extend([el] * cnt)

    header = lines[:coord_start - 1]
    return header, lattice, element_labels, raw, direct


def _write_poscar(path: Path, header: list[str], lattice: np.ndarray,
                  elements: list[str], coords: np.ndarray, direct: bool) -> None:
    """Write a POSCAR file to *path* using the provided header, lattice, and coords."""
    lines = list(header)
    lines.append("Direct" if direct else "Cartesian")
    for row in coords:
        lines.append(f"  {row[0]: .9f}  {row[1]: .9f}  {row[2]: .9f}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _to_cart(frac: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Convert fractional coordinates to Cartesian using the lattice matrix."""
    return frac @ lattice


def _to_frac(cart: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Convert Cartesian coordinates to fractional using the lattice matrix."""
    return cart @ np.linalg.inv(lattice)


def _wrap_frac(frac: np.ndarray) -> np.ndarray:
    """Wrap fractional coordinates into [0, 1) using periodic boundary conditions."""
    return frac % 1.0


def _all_pairwise(cart: np.ndarray, inv_lat: np.ndarray, lattice: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (distances N×N, displacement_vectors N×N×3) under minimum image convention.
    distances[i,j] = |r_i - r_j| (minimum image).
    vectors[i,j]   = cart[i] - cart[j] (minimum image, pointing i←j).
    """
    n = len(cart)
    diff = cart[:, None, :] - cart[None, :, :]          # N×N×3 Cartesian
    frac_diff = diff @ inv_lat                            # N×N×3 fractional
    frac_diff -= np.round(frac_diff)                      # minimum image
    mic_diff = frac_diff @ lattice                        # back to Cartesian
    dists = np.linalg.norm(mic_diff, axis=-1)             # N×N
    return dists, mic_diff


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def min_distance_poscar(path: Path) -> float:
    """Return the minimum interatomic distance in a POSCAR/CONTCAR (Å)."""
    try:
        header, lattice, elements, raw, direct = _read_poscar(path)
    except Exception as exc:
        log.warning("[structure_check] Cannot read %s: %s", path, exc)
        return float("inf")

    if len(raw) < 2:
        return float("inf")

    cart    = _to_cart(raw, lattice) if direct else raw.copy()
    inv_lat = np.linalg.inv(lattice)
    dists, _ = _all_pairwise(cart, inv_lat, lattice)
    np.fill_diagonal(dists, np.inf)
    return float(dists.min())


def check_and_fix_poscar(path: Path, min_dist: float | np.ndarray = _MIN_DIST_DEFAULT,
                         max_iter: int = _MAX_ITER) -> bool:
    """
    Check and fix close atom contacts in a POSCAR/CONTCAR.

    `min_dist` is either a scalar (Å, applied to every pair) or an N×N matrix
    of per-pair minimum distances — see check_and_fix_poscar_potcar() for the
    latter, built from PAW augmentation-sphere radii.

    Each iteration accumulates repulsive forces from ALL violating pairs
    simultaneously before moving any atom, so fixing pair (i,j) cannot
    push i into atom k. The step size is fractional (< 1) to prevent
    overshoot. Iterates until all pairs satisfy min_dist or max_iter reached.

    The file is written back in-place only when modifications are made.
    Returns True if the file was modified.
    """
    try:
        header, lattice, elements, raw, direct = _read_poscar(path)
    except Exception as exc:
        log.warning("[structure_check] Cannot read %s: %s", path, exc)
        return False

    n = len(raw)
    if n < 2:
        return False

    if isinstance(min_dist, np.ndarray):
        min_dist_mat = min_dist.astype(float)
    else:
        min_dist_mat = np.full((n, n), float(min_dist))
    np.fill_diagonal(min_dist_mat, 0.0)

    cart    = _to_cart(raw, lattice) if direct else raw.copy()
    inv_lat = np.linalg.inv(lattice)
    modified = False
    target  = min_dist_mat + _MARGIN   # push to min_dist + small margin

    for iteration in range(max_iter):
        dists, vecs = _all_pairwise(cart, inv_lat, lattice)
        np.fill_diagonal(dists, np.inf)

        # Find all violating pairs (upper triangle only to avoid double-count)
        mask = (dists < min_dist_mat)
        if not mask.any():
            break  # all clear

        # Accumulate repulsive displacements for every atom from ALL its bad neighbours
        forces = np.zeros_like(cart)
        rows, cols = np.where(np.triu(mask, k=1))
        for i, j in zip(rows, cols):
            d = dists[i, j]
            if d < 1e-10:
                # Perfectly overlapping: push along lattice a-vector direction
                direction = lattice[0] / (np.linalg.norm(lattice[0]) + 1e-30)
            else:
                direction = vecs[i, j] / d   # unit vector pointing i←j

            overlap = target[i, j] - d
            push    = overlap * _STEP        # fractional push — avoids overshoot

            # Equal and opposite: both atoms accumulate their half
            forces[i] += direction * push
            forces[j] -= direction * push

        cart    += forces
        modified = True

        log.debug("[structure_check] iter %d: %d violations, max_push=%.4f Å",
                  iteration + 1, len(rows), float(np.linalg.norm(forces, axis=1).max()))
    else:
        dists2, _ = _all_pairwise(cart, inv_lat, lattice)
        np.fill_diagonal(dists2, np.inf)
        log.warning("[structure_check] %s: close contacts remain after %d iterations "
                    "(min d=%.3f Å)", path.name, max_iter, float(dists2.min()))

    if not modified:
        return False

    # Wrap into cell and convert back to fractional if needed
    frac = _wrap_frac(_to_frac(cart, lattice))
    out_coords = frac if direct else _to_cart(frac, lattice)
    _write_poscar(path, header, lattice, elements, out_coords, direct)

    final_d = min_distance_poscar(path)
    log.info("[structure_check] %s: fixed — min dist (target %.3f–%.3f Å) → %.3f Å",
             path.name, float(min_dist_mat[min_dist_mat > 0].min(initial=0.0)),
             float(min_dist_mat.max()), final_d)
    return True


# ---------------------------------------------------------------------------
# POTCAR-derived (PAW augmentation-sphere) pairwise minimum distances
# ---------------------------------------------------------------------------

def parse_potcar_rcores(potcar: Path) -> dict[str, float]:
    """Return {element: RCORE_Å} parsed from a POTCAR's TITEL/RCORE header blocks.

    Elements and RCORE values appear in matching order, one pair per block.
    """
    text = potcar.read_text(errors="replace")
    titel_elements = re.findall(r"TITEL\s*=\s*\S+\s+(\S+)\s+\S+", text)
    rcores = [float(v) for v in re.findall(r"RCORE\s*=\s*([\d.]+)", text)]
    return dict(zip(titel_elements, rcores))


def pairwise_min_distances_from_potcar(
    elements: list[str], potcar: Path, factor: float = _POTCAR_FACTOR,
) -> np.ndarray:
    """Return an N×N matrix: min_dist[i,j] = factor * (RCORE_i + RCORE_j).

    `elements` is the per-atom element label list (POSCAR atom order).
    Elements missing from the POTCAR fall back to `_MIN_DIST_DEFAULT / (2*factor)`
    so an unparseable POTCAR degrades to roughly the flat scalar default.
    """
    rcores = parse_potcar_rcores(potcar)
    fallback = _MIN_DIST_DEFAULT / (2 * factor)
    r = np.array([rcores.get(el, fallback) for el in elements])
    return factor * (r[:, None] + r[None, :])


def check_and_fix_poscar_potcar(
    poscar: Path, potcar: Path, factor: float = _POTCAR_FACTOR,
    max_iter: int = _MAX_ITER,
) -> bool:
    """Repair close contacts using POTCAR-derived (PAW-aware) per-pair minimum distances.

    A flat scalar min_dist (e.g. 1.0 Å) is smaller than the combined PAW
    augmentation-sphere radius for most element pairs (Li RCORE=2.05 Å,
    S=1.90 Å, C/N/O/F≈1.5 Å) — a random, non-bonded contact that "looks" far
    enough apart under a flat threshold can still overlap augmentation
    spheres and blow up VASP's SCF. This builds the per-pair threshold from
    the actual POTCAR instead of guessing one global number.
    """
    try:
        _, _, elements, raw, _ = _read_poscar(poscar)
    except Exception as exc:
        log.warning("[structure_check] Cannot read %s: %s", poscar, exc)
        return False
    if len(raw) < 2:
        return False
    try:
        min_dist_mat = pairwise_min_distances_from_potcar(elements, potcar, factor)
    except Exception as exc:
        log.warning("[structure_check] Cannot parse RCORE from %s (%s) — "
                    "falling back to scalar min_dist=%.2f Å", potcar, exc, _MIN_DIST_DEFAULT)
        return check_and_fix_poscar(poscar, min_dist=_MIN_DIST_DEFAULT, max_iter=max_iter)
    return check_and_fix_poscar(poscar, min_dist=min_dist_mat, max_iter=max_iter)


def check_poscar_log(path: Path, min_dist: float = _MIN_DIST_DEFAULT) -> None:
    """Log a warning if any atom pair is closer than min_dist Å (read-only)."""
    d = min_distance_poscar(path)
    if d < min_dist:
        log.warning("[structure_check] %s: min distance %.3f Å < %.1f Å threshold",
                    path.name, d, min_dist)
    else:
        log.debug("[structure_check] %s: min distance %.3f Å OK", path.name, d)
