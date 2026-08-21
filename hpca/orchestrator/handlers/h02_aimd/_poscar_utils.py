"""
_poscar_utils.py — Standalone POSCAR manipulation functions for h02_aimd.
"""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

import numpy as np

log = logging.getLogger("hpca.orch")

# Rattle sigma / min-distance pulled from h02_aimd_constants via late import to avoid circular deps
try:
    from ..h02_aimd_constants import _RATTLE_SIGMA, _MIN_ATOM_DISTANCE
except ImportError:
    _RATTLE_SIGMA = 0.08
    _MIN_ATOM_DISTANCE = 1.2


def read_poscar_lines(poscar: Path) -> list[str]:
    """Read POSCAR, strip blank lines, return non-empty stripped lines."""
    return [
        line.strip()
        for line in poscar.read_text(errors="replace").splitlines()
        if line.strip()
    ]


def get_poscar_elements(poscar: Path) -> list[str]:
    """Return element list from POSCAR line 6 (empty list if not found)."""
    try:
        lines = read_poscar_lines(poscar)
        if len(lines) < 7:
            return []
        tokens = lines[5].split()
        if all(re.fullmatch(r"[A-Z][a-z]?", t) for t in tokens):
            return tokens
        return []
    except OSError:
        return []


def count_atoms_poscar(poscar: Path, default: int = 200) -> int:
    """Return total atom count from POSCAR."""
    try:
        lines = read_poscar_lines(poscar)
        if len(lines) < 7:
            return default
        line6 = lines[5].split()
        line7 = lines[6].split()
        if all(re.fullmatch(r"[A-Z][a-z]?", t) for t in line6):
            return sum(int(x) for x in line7)
        return sum(int(x) for x in line6)
    except (OSError, ValueError, IndexError):
        log.warning("[h02_aimd] Cannot count atoms in %s", poscar)
        return default


def parse_temperature(aimd_dir: Path, default: int = 300) -> int:
    """Infer temperature (K) from directory name."""
    match = re.search(r"(\d+)", aimd_dir.name)
    if match:
        return int(match.group(1))
    log.warning("[h02_aimd] Cannot infer T from %s; using %d K", aimd_dir, default)
    return default


def make_deformed_poscar(source: Path, out: Path, scale: float) -> None:
    """Write POSCAR with cell scaled uniformly by `scale`; fractional coords preserved."""
    lines = source.read_text(errors="replace").splitlines()
    if len(lines) < 5:
        shutil.copy2(source, out)
        return
    result = list(lines)
    try:
        existing = float(result[1].strip())
    except (ValueError, IndexError):
        existing = 1.0
    result[1] = f"  {existing * scale:.8f}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(result) + "\n")


def _min_image_distances(candidate: np.ndarray, placed: np.ndarray, cell_T: np.ndarray) -> np.ndarray:
    """Return the periodic (minimum-image) Cartesian distance from `candidate` to each row of `placed`."""
    diff_frac = candidate - placed
    diff_frac -= np.round(diff_frac)
    diff_cart = diff_frac @ cell_T.T
    return np.linalg.norm(diff_cart, axis=1)


def _random_fractional_positions(
    cell_T: np.ndarray, n_atoms: int, rng: np.random.Generator,
    min_dist: float, max_attempts: int = 300,
) -> np.ndarray:
    """Return (n_atoms, 3) random fractional coords with no pair closer than `min_dist` Å.

    Uses rejection sampling under the minimum-image convention so liquid dataset
    boxes don't land atoms on top of each other (which blows up VASP's SCF within
    the first few electronic steps). Falls back to the least-bad candidate after
    max_attempts rather than looping forever on a cell too small for min_dist.
    """
    placed = np.empty((0, 3))
    for _ in range(n_atoms):
        best_frac, best_margin = rng.random(3), -np.inf
        for _ in range(max_attempts):
            candidate = rng.random(3)
            if placed.shape[0] == 0:
                best_frac = candidate
                break
            margin = float(_min_image_distances(candidate, placed, cell_T).min())
            if margin >= min_dist:
                best_frac = candidate
                break
            if margin > best_margin:
                best_margin, best_frac = margin, candidate
        placed = np.vstack([placed, best_frac])
    return placed


def make_random_poscar(
    source: Path, out: Path, scale: float, rng_seed: int = 42,
    min_dist: float = _MIN_ATOM_DISTANCE,
) -> None:
    """Scale cell by `scale` and randomize all atom fractional coordinates.

    Rejection-samples positions under the minimum-image convention so no two
    atoms land closer than `min_dist` Å — fully unconstrained placement produced
    overlapping atoms often enough to crash VASP's SCF on liquid boxes.
    """
    rng   = np.random.default_rng(rng_seed)
    lines = source.read_text(errors="replace").splitlines()
    if len(lines) < 9:
        shutil.copy2(source, out)
        return
    result = list(lines)
    try:
        existing = float(result[1].strip())
    except (ValueError, IndexError):
        existing = 1.0
    new_scale = existing * scale
    result[1] = f"  {new_scale:.8f}"
    try:
        mat = np.array([[float(v) for v in result[i].split()[:3]] for i in (2, 3, 4)])
        cell_T = (new_scale * mat).T  # columns = a, b, c → cart = cell_T @ frac
    except (ValueError, IndexError):
        shutil.copy2(source, out)
        return
    # Count atoms from line 6 (or 5 if no element names)
    try:
        n_atoms = sum(int(x) for x in result[6].split())
    except (ValueError, IndexError):
        shutil.copy2(source, out)
        return
    # Find coordinate block: look for "Direct"/"Cartesian"/"Selective dynamics"
    coord_header = 8  # fallback
    for i in range(7, min(10, len(result))):
        first = result[i].strip().lower()
        if first.startswith(("d", "c", "k")):
            coord_header = i + 1
            break
        if first.startswith("s"):  # Selective dynamics
            coord_header = i + 2
            break
    n_atoms = min(n_atoms, len(result) - coord_header)
    positions = _random_fractional_positions(cell_T, n_atoms, rng, min_dist)
    for offset, (x, y, z) in enumerate(positions):
        result[coord_header + offset] = f"  {x:.8f}  {y:.8f}  {z:.8f}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(result) + "\n")


def make_rattled_poscar(
    source: Path, out: Path, scale: float,
    sigma: float = _RATTLE_SIGMA, rng_seed: int = 0
) -> None:
    """Scale cell by `scale` and displace each atom by Gaussian noise (sigma Å, Cartesian).

    Unlike make_random_poscar, lattice topology is preserved — atoms stay near
    equilibrium. Used for crystal/SSE dataset generation where long-range order matters.
    """
    rng = np.random.default_rng(rng_seed)
    lines = source.read_text(errors="replace").splitlines()
    if len(lines) < 9:
        shutil.copy2(source, out)
        return
    result = list(lines)
    try:
        existing = float(result[1].strip())
    except (ValueError, IndexError):
        existing = 1.0
    new_scale = existing * scale
    result[1] = f"  {new_scale:.8f}"
    try:
        mat = np.array([[float(v) for v in result[i].split()[:3]] for i in (2, 3, 4)])
        cell = new_scale * mat  # rows = a, b, c lattice vectors (Å)
    except (ValueError, IndexError):
        shutil.copy2(source, out)
        return
    try:
        n_atoms = sum(int(x) for x in result[6].split())
    except (ValueError, IndexError):
        shutil.copy2(source, out)
        return
    coord_header = 8
    for i in range(7, min(11, len(result))):
        tok = result[i].strip().lower()
        if tok.startswith("s"):        # Selective dynamics
            coord_header = i + 2
            break
        if tok.startswith(("d", "c", "k")):
            coord_header = i + 1
            break
    cell_T     = cell.T                  # columns = a, b, c → cart = cell_T @ frac
    cell_T_inv = np.linalg.inv(cell_T)
    for idx in range(coord_header, min(coord_header + n_atoms, len(result))):
        parts = result[idx].split()
        if len(parts) < 3:
            continue
        frac  = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
        cart  = cell_T @ frac
        cart += rng.normal(0.0, sigma, 3)
        frac_new = (cell_T_inv @ cart) % 1.0
        result[idx] = f"  {frac_new[0]:.8f}  {frac_new[1]:.8f}  {frac_new[2]:.8f}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(result) + "\n")
