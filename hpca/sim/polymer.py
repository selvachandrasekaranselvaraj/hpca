"""
polymer.py — PVDF-HFP copolymer chain builder for LAMMPS OPLS-AA simulations.

Reference: Zhang et al., Nature Energy 9, 386-400 (2024)
  VDF:HFP = 4:1 (hfp_fraction=0.20), 50 monomers/chain, OPLS-AA force field.

Public API
----------
build_sequence(n_units, hfp_fraction=0.20, seed=42) -> list[str]
    Returns list of "VDF" / "HFP" strings of length n_units.

build_pvdf_hfp_chain(sequence) -> tuple[list[dict], list[tuple[int,int]]]
    Returns (atoms, bonds) where each atom is
      {"element": str, "opls_type": str, "x": float, "y": float, "z": float, "charge": float}
    and bonds are (i, j) 0-indexed atom pairs.

class PolymerMolData:
    @classmethod def from_sequence(cls, sequence, name="PVDF_HFP", count=1) -> "PolymerMolData"
    @classmethod def pvdf_hfp(cls, n_units=50, hfp_fraction=0.20, seed=42, count=1) -> "PolymerMolData"

    Attributes: name (str), count (int), atoms (list[dict]), bonds (list[tuple]),
                mw (float), formula (str)

    Method: to_mol_data() -> "MolData" — converts to forcefield.MolData for build_mixed_system
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import List, Tuple, Dict

# ---------------------------------------------------------------------------
# Monomer molecular weights
# ---------------------------------------------------------------------------
_MW_VDF = 2 * 12.011 + 2 * 1.008 + 2 * 19.000   # 64.034 g/mol
_MW_HFP = 2 * 12.011 + 2 * 19.000 + 12.011 + 3 * 19.000  # 150.034 g/mol (CF2-CF-CF3 total)
_MW_H   = 1.008  # end-cap H

# ---------------------------------------------------------------------------
# Bond lengths and geometry constants
# ---------------------------------------------------------------------------
_BL_CC   = 1.54   # C-C backbone (Å)
_BL_CH   = 1.09   # C-H (Å)
_BL_CF   = 1.34   # C-F (Å)
_BL_CCF3 = 1.52   # C-C(CF₃) branch (Å)

# Zigzag backbone geometry
# projected bond length along x: BL_CC * sin(35.25°)
# projected bond length along z: BL_CC * cos(35.25°)
_SIN35 = math.sin(math.radians(35.25))   # 0.5774
_COS35 = math.cos(math.radians(35.25))   # 0.8165

_BACKBONE_DX = _BL_CC * _SIN35   # 0.889 Å
_BACKBONE_DZ = _BL_CC * _COS35   # 1.257 Å (alternates 0 / +1.257)

# Tetrahedral substituent offsets
# sub_y = ±bond_len * sin(54.75°), sub_z = ±bond_len * cos(54.75°)
_SIN5475 = math.sin(math.radians(54.75))   # 0.8165
_COS5475 = math.cos(math.radians(54.75))   # 0.5774


# ---------------------------------------------------------------------------
# Public function 1: build_sequence
# ---------------------------------------------------------------------------

def build_sequence(
    n_units: int,
    hfp_fraction: float = 0.20,
    seed: int = 42,
) -> List[str]:
    """
    Return a list of "VDF" / "HFP" strings of length *n_units*.

    Parameters
    ----------
    n_units : int
        Total number of repeat units (monomers) in the chain.
    hfp_fraction : float
        Mole fraction of HFP (default 0.20 → VDF:HFP = 4:1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[str]
        e.g. ["VDF", "VDF", "HFP", "VDF", ...]
    """
    if not 0.0 <= hfp_fraction <= 1.0:
        raise ValueError(f"hfp_fraction must be in [0, 1], got {hfp_fraction}")
    rng = random.Random(seed)
    return rng.choices(
        ["VDF", "HFP"],
        weights=[1.0 - hfp_fraction, hfp_fraction],
        k=n_units,
    )


# ---------------------------------------------------------------------------
# Internal geometry helper
# ---------------------------------------------------------------------------

def _backbone_position(backbone_idx: int) -> Tuple[float, float, float]:
    """Return (x, y, z) for backbone atom *backbone_idx* in the zigzag."""
    x = backbone_idx * _BACKBONE_DX
    y = 0.0
    z = (backbone_idx % 2) * _BACKBONE_DZ
    return x, y, z


def _substituent_positions(
    backbone_idx: int,
    x_b: float,
    y_b: float,
    z_b: float,
    bond_len: float,
    n_sub: int,
) -> List[Tuple[float, float, float]]:
    """
    Return positions for 1 or 2 substituents on backbone atom *backbone_idx*.

    The local "up" direction is +y for even backbone index, −y for odd.
    Offsets maintain approximate tetrahedral geometry away from backbone.

    Parameters
    ----------
    n_sub : int
        Number of substituents (1 or 2).
    """
    # sign of z-offset alternates with backbone position
    direction = 1 if backbone_idx % 2 == 0 else -1

    sub_y = bond_len * _SIN5475   # ~0.8165 * bond_len
    sub_z = bond_len * _COS5475   # ~0.5774 * bond_len

    if n_sub == 1:
        return [(x_b, y_b + sub_y * direction, z_b + sub_z * direction)]
    else:
        # two substituents: one at +y side, one at −y side
        pos1 = (x_b, y_b + sub_y,  z_b + sub_z * direction)
        pos2 = (x_b, y_b - sub_y,  z_b - sub_z * direction)
        return [pos1, pos2]


# ---------------------------------------------------------------------------
# Public function 2: build_pvdf_hfp_chain
# ---------------------------------------------------------------------------

def build_pvdf_hfp_chain(
    sequence: List[str],
) -> Tuple[List[Dict], List[Tuple[int, int]]]:
    """
    Build 3-D coordinates and OPLS-AA atom types for a PVDF-HFP chain.

    Parameters
    ----------
    sequence : list[str]
        Output of :func:`build_sequence`; each entry is "VDF" or "HFP".

    Returns
    -------
    atoms : list[dict]
        Each entry: {"element": str, "opls_type": str,
                     "x": float, "y": float, "z": float, "charge": float}
    bonds : list[tuple[int, int]]
        0-indexed atom-pair bonds.
    """
    atoms: List[Dict] = []
    bonds: List[Tuple[int, int]] = []

    # Each monomer contributes 2 backbone carbons.
    n_backbone = 2 * len(sequence)
    backbone_indices: List[int] = []  # atom indices of backbone C atoms

    # ── Pass 1: build backbone atoms ─────────────────────────────────────────
    for mono_idx, mono in enumerate(sequence):
        for local in range(2):                  # 0 = even C, 1 = odd C
            bi = 2 * mono_idx + local           # backbone position index
            x, y, z = _backbone_position(bi)

            if mono == "VDF":
                if local == 0:
                    # CH₂
                    atom = {"element": "C", "opls_type": "CT_H2",
                            "x": x, "y": y, "z": z, "charge": +0.176}
                else:
                    # CF₂
                    atom = {"element": "C", "opls_type": "CT_F2",
                            "x": x, "y": y, "z": z, "charge": -0.212}
            else:  # HFP
                if local == 0:
                    # CF₂
                    atom = {"element": "C", "opls_type": "CT_F2",
                            "x": x, "y": y, "z": z, "charge": -0.212}
                else:
                    # CF bearing CF₃
                    atom = {"element": "C", "opls_type": "CT_F1",
                            "x": x, "y": y, "z": z, "charge": -0.107}

            backbone_indices.append(len(atoms))
            atoms.append(atom)

    # ── Pass 2: backbone C–C bonds ────────────────────────────────────────────
    for i in range(n_backbone - 1):
        bonds.append((backbone_indices[i], backbone_indices[i + 1]))

    # ── Pass 3: substituents on each backbone carbon ──────────────────────────
    for mono_idx, mono in enumerate(sequence):
        for local in range(2):
            bi = 2 * mono_idx + local
            c_idx = backbone_indices[bi]
            xb, yb, zb = _backbone_position(bi)

            if mono == "VDF":
                if local == 0:
                    # CH₂ — two H substituents
                    sub_positions = _substituent_positions(bi, xb, yb, zb, _BL_CH, 2)
                    for sx, sy, sz in sub_positions:
                        h_idx = len(atoms)
                        atoms.append({"element": "H", "opls_type": "HP",
                                      "x": sx, "y": sy, "z": sz, "charge": -0.088})
                        bonds.append((c_idx, h_idx))
                else:
                    # CF₂ — two F substituents
                    sub_positions = _substituent_positions(bi, xb, yb, zb, _BL_CF, 2)
                    for sx, sy, sz in sub_positions:
                        f_idx = len(atoms)
                        atoms.append({"element": "F", "opls_type": "FP",
                                      "x": sx, "y": sy, "z": sz, "charge": +0.106})
                        bonds.append((c_idx, f_idx))

            else:  # HFP
                if local == 0:
                    # CF₂ — two F substituents
                    sub_positions = _substituent_positions(bi, xb, yb, zb, _BL_CF, 2)
                    for sx, sy, sz in sub_positions:
                        f_idx = len(atoms)
                        atoms.append({"element": "F", "opls_type": "FP",
                                      "x": sx, "y": sy, "z": sz, "charge": +0.106})
                        bonds.append((c_idx, f_idx))
                else:
                    # CT_F1 — one backbone F + CF₃ branch
                    # --- backbone F ---
                    direction = 1 if bi % 2 == 0 else -1
                    sub_y = _BL_CF * _SIN5475
                    sub_z = _BL_CF * _COS5475
                    fx = xb
                    fy = yb + sub_y * direction
                    fz = zb + sub_z * direction
                    bf_idx = len(atoms)
                    atoms.append({"element": "F", "opls_type": "FP",
                                  "x": fx, "y": fy, "z": fz, "charge": +0.106})
                    bonds.append((c_idx, bf_idx))

                    # --- CF₃ branch carbon ---
                    # Place branch C on opposite side from backbone F
                    branch_x = xb
                    branch_y = yb - sub_y * direction
                    branch_z = zb - _BL_CCF3 * direction
                    cf3_c_idx = len(atoms)
                    atoms.append({"element": "C", "opls_type": "CT_F3",
                                  "x": branch_x, "y": branch_y, "z": branch_z,
                                  "charge": +0.580})
                    bonds.append((c_idx, cf3_c_idx))

                    # --- 3 F atoms on CF₃ ---
                    # Distribute around branch C with tetrahedral geometry
                    # F1: along -y from branch C
                    # F2 and F3: symmetric about the C-C(branch) axis
                    for k in range(3):
                        angle = k * (2.0 * math.pi / 3.0)
                        # local frame: branch axis is roughly along z from branch_C
                        # F atoms radiate outward in a cone
                        cone_half = math.radians(109.5 / 2.0)
                        tf_x = branch_x + _BL_CF * math.sin(cone_half) * math.cos(angle)
                        tf_y = branch_y + _BL_CF * math.sin(cone_half) * math.sin(angle)
                        tf_z = branch_z - _BL_CF * math.cos(cone_half) * direction
                        tf_idx = len(atoms)
                        atoms.append({"element": "F", "opls_type": "FP3",
                                      "x": tf_x, "y": tf_y, "z": tf_z,
                                      "charge": -0.193})
                        bonds.append((cf3_c_idx, tf_idx))

    return atoms, bonds


# ---------------------------------------------------------------------------
# Helper: molecular formula string
# ---------------------------------------------------------------------------

def _formula_from_atoms(atoms: List[Dict]) -> str:
    """Return Hill-notation molecular formula string for the given atom list."""
    counts = Counter(a["element"] for a in atoms)
    order = ["C", "H", "F", "O", "N", "S", "P", "Li"]
    parts = []
    for el in order:
        if el in counts:
            parts.append(f"{el}{counts[el]}" if counts[el] > 1 else el)
    for el in sorted(counts):
        if el not in order:
            parts.append(f"{el}{counts[el]}" if counts[el] > 1 else el)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Helper: molecular weight
# ---------------------------------------------------------------------------

_ATOMIC_MASS: Dict[str, float] = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 19.000,
    "P": 30.974,
    "S": 32.060,
    "Li": 6.941,
}


def _mw_from_atoms(atoms: List[Dict]) -> float:
    """Compute chain MW by summing atomic masses of all atoms in the chain."""
    return sum(_ATOMIC_MASS.get(a["element"], 12.0) for a in atoms)


def _mw_from_sequence(sequence: List[str]) -> float:
    """Compute chain MW from sequence (sums per-monomer atomic masses)."""
    atoms, _ = build_pvdf_hfp_chain(sequence)
    return _mw_from_atoms(atoms)


# ---------------------------------------------------------------------------
# Public class: PolymerMolData
# ---------------------------------------------------------------------------

class PolymerMolData:
    """
    Container for a PVDF-HFP polymer chain's topology and force-field data.

    Attributes
    ----------
    name : str
    count : int
        Number of identical chains to place in the simulation box.
    atoms : list[dict]
        {"element", "opls_type", "x", "y", "z", "charge"}
    bonds : list[tuple[int,int]]
        0-indexed atom pairs.
    mw : float
        Molecular weight in g/mol.
    formula : str
        Hill-notation molecular formula.
    """

    def __init__(
        self,
        name: str,
        atoms: List[Dict],
        bonds: List[Tuple[int, int]],
        mw: float,
        formula: str,
        count: int = 1,
    ) -> None:
        """Initialise a PolymerMolData record with topology and metadata."""
        self.name    = name
        self.count   = count
        self.atoms   = atoms
        self.bonds   = bonds
        self.mw      = mw
        self.formula = formula

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_sequence(
        cls,
        sequence: List[str],
        name: str = "PVDF_HFP",
        count: int = 1,
    ) -> "PolymerMolData":
        """
        Build a PolymerMolData from an explicit monomer sequence.

        Parameters
        ----------
        sequence : list[str]
            Each element is "VDF" or "HFP" (output of :func:`build_sequence`).
        name : str
            Molecule name used in LAMMPS output.
        count : int
            Number of identical chains to place in the simulation box.
        """
        atoms, bonds = build_pvdf_hfp_chain(sequence)
        mw      = _mw_from_sequence(sequence)
        formula = _formula_from_atoms(atoms)
        return cls(name=name, atoms=atoms, bonds=bonds,
                   mw=mw, formula=formula, count=count)

    @classmethod
    def pvdf_hfp(
        cls,
        n_units: int = 50,
        hfp_fraction: float = 0.20,
        seed: int = 42,
        count: int = 1,
    ) -> "PolymerMolData":
        """
        Convenience constructor matching the Zhang et al. (2024) parameters.

        Parameters
        ----------
        n_units : int
            Number of repeat units (default 50).
        hfp_fraction : float
            Mole fraction of HFP (default 0.20 → VDF:HFP = 4:1).
        seed : int
            Random seed for monomer sequence generation.
        count : int
            Number of identical chains in the simulation box.
        """
        sequence = build_sequence(n_units, hfp_fraction=hfp_fraction, seed=seed)
        n_vdf = sequence.count("VDF")
        n_hfp = sequence.count("HFP")
        name = f"PVDF_HFP_{n_vdf}VDF_{n_hfp}HFP"
        return cls.from_sequence(sequence, name=name, count=count)

    # ------------------------------------------------------------------
    # Conversion to forcefield.MolData
    # ------------------------------------------------------------------

    def to_mol_data(self) -> "MolData":  # type: ignore[name-defined]
        """Convert to forcefield.MolData for use with build_mixed_system."""
        from hpca.sim.forcefield import (
            MolData, build_angles, build_dihedrals, build_impropers,
        )
        # bonds in polymer.py are 0-indexed tuples; MolData expects 1-indexed
        bonds_1idx = [(a + 1, b + 1) for a, b in self.bonds]
        # atoms in polymer.py are dicts; MolData expects (element, x, y, z) tuples
        atoms_tuple = [(a["element"], a["x"], a["y"], a["z"]) for a in self.atoms]
        types = [a["opls_type"] for a in self.atoms]

        md = MolData.__new__(MolData)
        md.name      = self.name
        md.count     = self.count
        md.atoms     = atoms_tuple
        md.bonds     = bonds_1idx
        md.types     = types
        md.mw        = self.mw
        md.angles    = build_angles(bonds_1idx)
        md.dihedrals = build_dihedrals(bonds_1idx)
        md.impropers = build_impropers(atoms_tuple, bonds_1idx, types)
        return md

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise string representation for debugging."""
        return (
            f"PolymerMolData(name={self.name!r}, "
            f"n_atoms={len(self.atoms)}, "
            f"n_bonds={len(self.bonds)}, "
            f"mw={self.mw:.1f} g/mol, "
            f"formula={self.formula}, "
            f"count={self.count})"
        )
