"""Pure-NumPy linear NEB fallback and selective-dynamics algorithms."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger("hpca.neb")


# ---------------------------------------------------------------------------
# Internal data structure
# ---------------------------------------------------------------------------

@dataclass
class _Poscar:
    """Lightweight dataclass holding all parsed fields from a VASP POSCAR file."""

    comment: str
    scale: float
    lattice: np.ndarray          # shape (3, 3), rows = a, b, c vectors
    elements: list[str]
    counts: list[int]
    n_atoms: int
    coords_direct: np.ndarray   # shape (n_atoms, 3), always fractional
    selective: bool
    flags: list[tuple[str, str, str]]   # length n_atoms; ("T","T","T") or ("F","F","F")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_poscar(path: Path) -> _Poscar:
    """Parse a VASP5 (or VASP4) POSCAR and return a _Poscar with fractional coords."""
    path = Path(path)
    lines = path.read_text().splitlines()

    comment = lines[0]
    scale = float(lines[1].split()[0])

    lattice = np.array(
        [list(map(float, lines[i].split()[:3])) for i in range(2, 5)],
        dtype=float,
    )

    # Detect VASP4 vs VASP5: if line 5 starts with a digit it's counts (VASP4)
    vasp4 = lines[5].split()[0][0].isdigit()
    if vasp4:
        elements = []          # unknown in VASP4
        counts = list(map(int, lines[5].split()))
        body_start = 6
    else:
        elements = lines[5].split()
        counts = list(map(int, lines[6].split()))
        body_start = 7

    n_atoms = sum(counts)

    # Optional "Selective dynamics" line
    selective = False
    if lines[body_start].strip().lower().startswith("s"):
        selective = True
        body_start += 1

    # Coordinate mode
    coord_mode = lines[body_start].strip().lower()
    is_direct = coord_mode.startswith("d")
    body_start += 1

    # Parse coordinates (and optional SD flags)
    coords_raw = np.zeros((n_atoms, 3), dtype=float)
    flags: list[tuple[str, str, str]] = []
    for i in range(n_atoms):
        parts = lines[body_start + i].split()
        coords_raw[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
        if selective and len(parts) >= 6:
            flags.append((parts[3], parts[4], parts[5]))
        else:
            flags.append(("T", "T", "T"))

    # Convert to fractional if needed
    if is_direct:
        coords_direct = coords_raw
    else:
        # Cartesian coords (Angstroms if scale>0).
        # frac = cart @ inv(lattice).  Scale cancels: cart = scale * raw, lattice_phys = scale * lattice
        # => frac = (scale * raw) @ inv(scale * lattice) = raw @ inv(lattice)
        inv_lat = np.linalg.inv(lattice)
        coords_direct = coords_raw @ inv_lat

    return _Poscar(
        comment=comment,
        scale=scale,
        lattice=lattice,
        elements=elements,
        counts=counts,
        n_atoms=n_atoms,
        coords_direct=coords_direct,
        selective=selective,
        flags=flags,
    )


def _write_poscar(path: Path, p: _Poscar, coords_direct: np.ndarray) -> None:
    """Write a POSCAR in Direct (fractional) format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(p.comment)
    lines.append(f"  {p.scale:.10f}")

    for row in p.lattice:
        lines.append("  " + "  ".join(f"{v:20.16f}" for v in row))

    if p.elements:
        lines.append("  " + "  ".join(p.elements))

    lines.append("  " + "  ".join(str(c) for c in p.counts))

    if p.selective:
        lines.append("Selective dynamics")

    lines.append("Direct")

    for i in range(p.n_atoms):
        coord_str = "  ".join(f"{v:20.16f}" for v in coords_direct[i])
        if p.selective:
            fx, fy, fz = p.flags[i]
            lines.append(f"  {coord_str}  {fx}  {fy}  {fz}")
        else:
            lines.append(f"  {coord_str}")

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_neb_images(
    i_poscar: Path,
    f_poscar: Path,
    n_images: int,
    output_dir: Path,
) -> list[Path]:
    """
    Generate n_images intermediate NEB image POSCARs by linear interpolation.

    Image directories are created as output_dir/{k:02d}/POSCAR for k in 1..n_images.
    Minimum-image convention is applied so interpolation takes the shortest path
    across periodic boundaries.

    Returns list of written POSCAR paths.
    """
    i_poscar = Path(i_poscar)
    f_poscar = Path(f_poscar)
    output_dir = Path(output_dir)

    pi = _read_poscar(i_poscar)
    pf = _read_poscar(f_poscar)

    if pi.n_atoms != pf.n_atoms:
        raise ValueError(
            f"Endpoint POSCARs have different atom counts: "
            f"{pi.n_atoms} vs {pf.n_atoms}"
        )

    fi = pi.coords_direct
    ff = pf.coords_direct

    diff = ff - fi
    # Minimum image convention: wrap displacements to (-0.5, 0.5]
    diff -= np.round(diff)

    written: list[Path] = []
    for k in range(1, n_images + 1):
        t = k / (n_images + 1)
        coords = (fi + t * diff) % 1.0

        img_dir = output_dir / f"{k:02d}"
        img_dir.mkdir(parents=True, exist_ok=True)
        poscar_path = img_dir / "POSCAR"

        _write_poscar(poscar_path, pi, coords)
        written.append(poscar_path)

    log.info("[neb_tools] Created %d NEB images in %s", n_images, output_dir)
    return written


def apply_selective_dynamics(
    poscar_path: Path,
    mobile_indices: Sequence[int],
    *,
    inplace: bool = True,
) -> Path:
    """
    Add selective dynamics flags to a POSCAR.

    Atoms in mobile_indices (0-based) get "T T T"; all others get "F F F".
    If inplace=True the file is overwritten; otherwise written to poscar_path + "_sd".

    Returns the path of the written file.
    """
    poscar_path = Path(poscar_path)
    p = _read_poscar(poscar_path)

    mobile_set = set(mobile_indices)
    p.selective = True
    p.flags = [
        ("T", "T", "T") if i in mobile_set else ("F", "F", "F")
        for i in range(p.n_atoms)
    ]

    out_path = poscar_path if inplace else Path(str(poscar_path) + "_sd")
    _write_poscar(out_path, p, p.coords_direct)
    return out_path


def find_migrating_atom(
    i_poscar: Path,
    f_poscar: Path,
    mobile_element: str = "Li",
) -> int:
    """
    Return the 0-based index of the atom with the largest Cartesian displacement
    between the initial and final POSCARs.

    If mobile_element atoms exist, only those are candidates (non-mobile atoms
    are masked to displacement -1).  Minimum image convention is applied.
    """
    i_poscar = Path(i_poscar)
    f_poscar = Path(f_poscar)

    pi = _read_poscar(i_poscar)
    pf = _read_poscar(f_poscar)

    if pi.n_atoms != pf.n_atoms:
        raise ValueError(
            f"Endpoint POSCARs have different atom counts: "
            f"{pi.n_atoms} vs {pf.n_atoms}"
        )

    diff = pf.coords_direct - pi.coords_direct
    diff -= np.round(diff)   # minimum image in fractional space

    # Convert fractional displacement to Cartesian
    # cart_diff[i] = diff[i] @ (scale * lattice)  — but scale only for physical units
    phys_lattice = pi.scale * pi.lattice
    cart_diff = diff @ phys_lattice          # shape (n_atoms, 3)
    displacements = np.linalg.norm(cart_diff, axis=1)

    # Build element list per atom
    atom_elements: list[str] = []
    for el, cnt in zip(pi.elements, pi.counts):
        atom_elements.extend([el] * cnt)

    if mobile_element and mobile_element in atom_elements:
        # Mask non-mobile atoms by setting their displacement to -1
        for i, el in enumerate(atom_elements):
            if el != mobile_element:
                displacements[i] = -1.0

    return int(np.argmax(displacements))


def freeze_sphere(
    poscar_path: Path,
    center_idx: int,
    radius_A: float,
    *,
    inplace: bool = True,
) -> Path:
    """
    Apply selective dynamics: atoms within radius_A of center_idx are free (T T T);
    all atoms beyond that radius are frozen (F F F).

    Distances are computed in Cartesian space with minimum image convention.

    Returns the path of the written file.
    """
    poscar_path = Path(poscar_path)
    p = _read_poscar(poscar_path)

    phys_lattice = p.scale * p.lattice
    inv_lat = np.linalg.inv(p.lattice)   # fractional-space inverse

    center_frac = p.coords_direct[center_idx]

    mobile_indices: list[int] = []
    for i in range(p.n_atoms):
        diff_frac = p.coords_direct[i] - center_frac
        diff_frac -= np.round(diff_frac)          # minimum image in fractional space
        diff_cart = diff_frac @ phys_lattice
        dist = float(np.linalg.norm(diff_cart))
        if dist <= radius_A:
            mobile_indices.append(i)

    return apply_selective_dynamics(poscar_path, mobile_indices, inplace=inplace)
