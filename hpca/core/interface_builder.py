"""
interface_builder.py — Solid electrode slab + liquid electrolyte sandwich builder.

Builds a fully periodic, double-sided electrode|electrolyte|electrode interface
cell from an existing equilibrated bulk electrode structure (e.g. an NPT/NVT
amorphous hard-carbon box) by:

  1. Slicing a slab-shaped void out of the periodic electrode box (removing one
     contiguous z-window of atoms). Because the box is periodic, the remaining
     electrode wraps continuously through the periodic boundary — this yields
     a genuine double-sided sandwich (two exposed electrode faces, one gap)
     without any vacuum or explicit mirroring.
  2. Sizing an electrolyte fill for that gap from a target salt molarity and an
     approximate starting mass density (refined later by NPT equilibration,
     same as HPCA's existing liquid-electrolyte pipeline).
  3. Packing the electrolyte into the gap with PACKMOL, treating every kept
     electrode atom as a fixed obstacle other molecules must avoid.

This is new capability — HPCA's existing `slab_builder.build_interface` only
joins two crystalline POSCARs (film + substrate); nothing in the platform
previously combined a solid slab with a PACKMOL liquid region.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

log = logging.getLogger("hpca.interface_builder")

_NA = 6.02214076e23  # Avogadro's number

# Molecular weights (g/mol) — mirrors hpca.orchestrator.handlers.h00_design._MW
_MW = {"NaPF6": 167.95, "EC": 88.06, "DEC": 118.13}

# Atoms per molecule — mirrors hpca.orchestrator.handlers.h00_design._NATOMS_PER_MOL
_NATOMS = {"NaPF6": 8, "EC": 10, "DEC": 18}


# ---------------------------------------------------------------------------
# 1. Read the source electrode structure (LAMMPS dump format)
# ---------------------------------------------------------------------------

class ElectrodeBox:
    """Parsed LAMMPS dump: orthogonal box lengths + per-atom (element, x, y, z)."""

    def __init__(self, box_lengths: np.ndarray, elements: list[str], xyz: np.ndarray):
        self.box_lengths = box_lengths   # (3,) Å, orthogonal
        self.elements = elements         # len N
        self.xyz = xyz                   # (N, 3) Å, wrapped into [0, box)


def read_lammps_dump(path: Path) -> ElectrodeBox:
    """Parse a LAMMPS dump snapshot with an 'element' column (orthogonal box only)."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[3])
    bounds = np.array([[float(v) for v in lines[i].split()[:2]] for i in (5, 6, 7)])
    box_lengths = bounds[:, 1] - bounds[:, 0]

    header = lines[8].split()[2:]  # tokens after "ITEM: ATOMS"
    idx_x, idx_y, idx_z = header.index("x"), header.index("y"), header.index("z")
    idx_el = header.index("element")

    elements: list[str] = []
    xyz = np.empty((n, 3))
    for i, raw in enumerate(lines[9:9 + n]):
        parts = raw.split()
        elements.append(parts[idx_el])
        xyz[i] = (float(parts[idx_x]) - bounds[0, 0],
                   float(parts[idx_y]) - bounds[1, 0],
                   float(parts[idx_z]) - bounds[2, 0])
    return ElectrodeBox(box_lengths, elements, xyz)


def read_poscar_electrode(path: Path) -> ElectrodeBox:
    """Parse a VASP POSCAR/CONTCAR (Direct or Cartesian) into an ElectrodeBox.

    Box lengths are taken as the lattice-vector norms — a good approximation
    for the near-orthogonal, lightly relaxed cells this is used for (small
    DFT host structures), not intended for strongly triclinic cells.
    """
    lines = Path(path).read_text().splitlines()
    scale = float(lines[1])
    lat = np.array([[float(v) for v in lines[i].split()] for i in (2, 3, 4)]) * scale
    species = lines[5].split()
    counts = [int(c) for c in lines[6].split()]
    direct = lines[7].strip().lower().startswith("d")
    n = sum(counts)
    raw = np.array([[float(v) for v in lines[8 + i].split()[:3]] for i in range(n)])
    cart = raw @ lat if direct else raw

    elements: list[str] = []
    for sp, c in zip(species, counts):
        elements.extend([sp] * c)

    box_lengths = np.linalg.norm(lat, axis=1)
    return ElectrodeBox(box_lengths, elements, cart)


def open_gap_by_extending_box(box: ElectrodeBox, gap_thickness: float, axis: int = 2):
    """Open a slab-shaped void by EXTENDING the box along `axis` rather than
    removing atoms — appropriate for a small, already-complete electrode cell
    (e.g. a DFT-relaxed host) where every atom should be kept, unlike
    `slice_slab_gap`'s use on a large bulk box with atoms to spare.

    Existing atoms are left at their original Cartesian coordinates, occupying
    a contiguous [0, L_orig] window; the new gap is appended at the end,
    [L_orig, L_orig + gap_thickness). Because the box is periodic, that gap
    sits between the top face (z = L_orig) and — via periodic wraparound —
    the bottom face (z = 0) as well, giving the same double-sided sandwich
    topology as `slice_slab_gap` without altering a single existing bond.

    Returns (elements, xyz, new_box_lengths, z_lo, z_hi).
    """
    l_orig = box.box_lengths[axis]
    new_lengths = box.box_lengths.copy()
    new_lengths[axis] = l_orig + gap_thickness

    # Wrap into [0, original_length) per axis first — a lattice built from a
    # (near-)triclinic CONTCAR via vector norms can leave a few Cartesian
    # coordinates just outside [0, L) even though they're valid under the
    # true (tilted) lattice. PACKMOL's PBC mode requires fixed atoms strictly
    # inside the box it's given, so they must be wrapped into the ORIGINAL
    # per-axis length (not the extended one) — atoms belong to the kept
    # electrode block, not the newly appended gap.
    xyz = box.xyz.copy()
    xyz = xyz - box.box_lengths * np.floor(xyz / box.box_lengths)

    return list(box.elements), xyz, new_lengths, l_orig, l_orig + gap_thickness


# ---------------------------------------------------------------------------
# 2. Slice a slab-shaped void out of the periodic box
# ---------------------------------------------------------------------------

def slice_slab_gap(box: ElectrodeBox, gap_thickness: float, axis: int = 2):
    """Remove a contiguous `gap_thickness`-wide window along `axis`, centered
    in the box. Returns (kept_elements, kept_xyz, z_lo, z_hi).

    The kept atoms wrap through the periodic boundary opposite the gap, so the
    result is a single continuous electrode slab with two exposed faces at
    z_lo and z_hi — a double-sided sandwich once the gap is filled.
    """
    length = box.box_lengths[axis]
    if gap_thickness >= length:
        raise ValueError(f"gap_thickness {gap_thickness} >= box length {length}")
    center = length / 2.0
    lo, hi = center - gap_thickness / 2.0, center + gap_thickness / 2.0

    coord = box.xyz[:, axis]
    in_gap = (coord >= lo) & (coord < hi)
    kept_idx = np.where(~in_gap)[0]

    kept_elements = [box.elements[i] for i in kept_idx]
    kept_xyz = box.xyz[kept_idx]
    return kept_elements, kept_xyz, lo, hi


# ---------------------------------------------------------------------------
# 3. Electrolyte sizing for the gap volume
# ---------------------------------------------------------------------------

def electrolyte_counts_for_gap(
    box_x: float, box_y: float, gap_thickness: float,
    molarity: float, target_density_gcm3: float = 1.10,
    solvent_ratio: tuple[float, float] = (1.0, 1.0),  # EC : DEC
) -> dict[str, int]:
    """Molecule counts (NaPF6, EC, DEC) to fill the gap volume at `molarity`
    NaPF6 and an approximate starting mass density (NPT-refined later).
    """
    vol_A3 = box_x * box_y * gap_thickness
    vol_cm3 = vol_A3 * 1e-24
    vol_L = vol_A3 * 1e-27

    n_salt = max(1, round(molarity * vol_L * _NA))
    mass_total_g = target_density_gcm3 * vol_cm3
    mass_salt_g = n_salt * _MW["NaPF6"] / _NA
    mass_solvent_g = max(0.0, mass_total_g - mass_salt_g)

    r_ec, r_dec = solvent_ratio
    r_tot = r_ec + r_dec
    # mass_solvent = n_unit * (r_ec*MW_EC + r_dec*MW_DEC) / r_tot   [n_unit = mole-count unit]
    denom = (r_ec * _MW["EC"] + r_dec * _MW["DEC"]) / r_tot
    n_unit = mass_solvent_g * _NA / denom if denom > 0 else 0.0
    n_ec = max(1, round(n_unit * r_ec / r_tot))
    n_dec = max(1, round(n_unit * r_dec / r_tot))

    return {"NaPF6": n_salt, "EC": n_ec, "DEC": n_dec}


# ---------------------------------------------------------------------------
# 4. PACKMOL packing (electrode fixed, electrolyte confined to the gap)
# ---------------------------------------------------------------------------

def _write_xyz(elements: list[str], xyz: np.ndarray, path: Path, comment: str = "") -> None:
    lines = [str(len(elements)), comment]
    for el, (x, y, z) in zip(elements, xyz):
        lines.append(f"{el}  {x:.6f}  {y:.6f}  {z:.6f}")
    path.write_text("\n".join(lines) + "\n")


def _vasp_to_xyz(vasp_path: Path, xyz_path: Path) -> None:
    from pymatgen.core import Structure
    struct = Structure.from_file(str(vasp_path))
    lines = [str(len(struct)), vasp_path.stem]
    for site in struct:
        x, y, z = site.coords
        lines.append(f"{site.species_string}  {x:.6f}  {y:.6f}  {z:.6f}")
    xyz_path.write_text("\n".join(lines) + "\n")


def pack_electrolyte_gap(
    kept_elements: list[str], kept_xyz: np.ndarray,
    box_lengths: np.ndarray, z_lo: float, z_hi: float,
    mol_counts: dict[str, int], mol_vasp_paths: dict[str, Path],
    packmol_bin: str, tolerance: float = 2.0, timeout: int = 3600,
) -> tuple[list[str], np.ndarray] | None:
    """Pack `mol_counts` molecules into the [z_lo, z_hi) slab of the box,
    avoiding the fixed electrode atoms. Returns (elements, xyz) for the full
    merged cell, or None on failure.
    """
    bx, by, bz = box_lengths
    has_electrode = len(kept_elements) > 0
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        if has_electrode:
            electrode_xyz = tmp / "electrode.xyz"
            _write_xyz(kept_elements, kept_xyz, electrode_xyz, "electrode")

        mol_xyz_paths: dict[str, Path] = {}
        for name in mol_counts:
            xyz_p = tmp / f"{name}.xyz"
            _vasp_to_xyz(mol_vasp_paths[name], xyz_p)
            mol_xyz_paths[name] = xyz_p

        out_xyz = tmp / "packed.xyz"
        inp_lines = [
            f"tolerance {tolerance}",
            "seed -1",
            "maxit 500",
            "nloop 1000",
            "filetype xyz",
            f"output {out_xyz}",
            # Whole cell is periodic (electrode wraps through the z boundary) —
            # without this, PACKMOL treats box edges as free surfaces and will
            # happily place two molecules on opposite faces of the SAME
            # periodic boundary, overlapping once minimum-image wrapping is
            # applied downstream.
            f"pbc 0.0 0.0 0.0 {bx:.4f} {by:.4f} {bz:.4f}",
            "",
        ]
        if has_electrode:
            inp_lines += [
                f"structure {electrode_xyz}",
                "  number 1",
                "  fixed 0. 0. 0. 0. 0. 0.",
                "end structure",
                "",
            ]
        for name, count in mol_counts.items():
            inp_lines += [
                f"structure {mol_xyz_paths[name]}",
                f"  number {count}",
                f"  inside box 0.0 0.0 {z_lo:.4f} {bx:.4f} {by:.4f} {z_hi:.4f}",
                "end structure",
                "",
            ]
        inp_file = tmp / "packmol.inp"
        inp_file.write_text("\n".join(inp_lines) + "\n")

        with open(inp_file) as fin:
            result = subprocess.run(
                [packmol_bin], stdin=fin,
                capture_output=True, text=True, timeout=timeout,
            )
        if not out_xyz.exists():
            log.error("[interface_builder] PACKMOL failed: %s", result.stdout[-2000:])
            return None

        lines = out_xyz.read_text().splitlines()
        n = int(lines[0])
        elements: list[str] = []
        xyz = np.empty((n, 3))
        for i, raw in enumerate(lines[2:2 + n]):
            parts = raw.split()
            elements.append(parts[0])
            xyz[i] = [float(v) for v in parts[1:4]]
        return elements, xyz


# ---------------------------------------------------------------------------
# 5. Write final merged POSCAR
# ---------------------------------------------------------------------------

def write_sandwich_poscar(
    elements: list[str], xyz: np.ndarray, box_lengths: np.ndarray,
    out_path: Path, comment: str = "electrode|electrolyte sandwich",
) -> None:
    """Write an orthogonal-cell POSCAR, species grouped and counted."""
    order: list[str] = []
    for e in elements:
        if e not in order:
            order.append(e)
    idx_by_el = {e: [i for i, el in enumerate(elements) if el == e] for e in order}
    counts = [len(idx_by_el[e]) for e in order]

    lines = [
        comment, "1.0",
        f"  {box_lengths[0]:.6f}  0.000000  0.000000",
        f"  0.000000  {box_lengths[1]:.6f}  0.000000",
        f"  0.000000  0.000000  {box_lengths[2]:.6f}",
        "  " + "  ".join(order),
        "  " + "  ".join(str(c) for c in counts),
        "Cartesian",
    ]
    for e in order:
        for i in idx_by_el[e]:
            x, y, z = xyz[i]
            lines.append(f"  {x:14.8f}  {y:14.8f}  {z:14.8f}")
    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 6. Overlap sanity check — INTERmolecular only
# ---------------------------------------------------------------------------
#
# A naive nearest-neighbor distance check over ALL atoms is useless here: every
# molecule's own covalent bonds (C-H ~1.09 Å, O-H ~0.96 Å) and the electrode's
# own C-C network (~1.2-1.5 Å) are always the closest contacts and are not
# overlaps. Only cross-molecule (electrode-vs-electrolyte, or between two
# different electrolyte molecules) distances indicate a packing defect.

def build_molecule_ids(n_electrode: int, mol_counts: dict[str, int],
                        mol_natoms: dict[str, int]) -> np.ndarray:
    """Return a per-atom molecule id: 0 for every electrode atom (one frozen
    rigid body), then a distinct id per electrolyte molecule instance, in the
    same (name, instance) order used to build the PACKMOL input / merged array.
    """
    mol_id = np.zeros(n_electrode, dtype=int)
    next_id = 1
    for name, count in mol_counts.items():
        natoms = mol_natoms[name]
        for _ in range(count):
            mol_id = np.concatenate([mol_id, np.full(natoms, next_id)])
            next_id += 1
    return mol_id


def min_intermolecular_distance(
    elements: list[str], xyz: np.ndarray, box_lengths: np.ndarray,
    mol_id: np.ndarray, k_neighbors: int = 8,
) -> tuple[float, int, int]:
    """Return (min_dist, atom_i, atom_j) for the closest pair of atoms that
    belong to DIFFERENT molecules (minimum-image, periodic). Atoms within the
    same molecule/electrode are excluded — those distances are bond lengths,
    not overlaps.
    """
    from scipy.spatial import cKDTree

    n = len(elements)
    k_neighbors = min(k_neighbors, n)  # cKDTree can't return more neighbors than exist

    xyz_w = xyz - box_lengths * np.floor(xyz / box_lengths)
    tree = cKDTree(xyz_w, boxsize=box_lengths)
    dists, idx = tree.query(xyz_w, k=k_neighbors)
    min_inter = np.full(n, np.inf)
    partner = np.full(n, -1)
    for k in range(1, k_neighbors):
        diff_mol = mol_id[idx[:, k]] != mol_id
        better = diff_mol & (dists[:, k] < min_inter)
        min_inter = np.where(better, dists[:, k], min_inter)
        partner = np.where(better, idx[:, k], partner)

    i = int(np.argmin(min_inter))
    j = int(partner[i])
    return float(min_inter[i]), i, j
