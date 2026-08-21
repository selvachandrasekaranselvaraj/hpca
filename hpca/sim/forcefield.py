"""
forcefield.py — OPLS-AA force field assignment for liquid electrolytes.

Covers ether solvents (DME, DMB, DOL), carbonates (EC, DMC, EMC),
and ionic species (Li+, FSI-/LiFSI, TFSI-, PF6-).

Workflow
--------
1. parse_molecule(path)              → (atoms, bonds)   # VASP / PDB / XYZ
2. assign_atom_types(atoms, bonds)   → list[str]
3. write_lmp(atoms, bonds, types, path)
4. build_mixed_system(components, out_path)   # multi-molecule LAMMPS data
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
# OPLS-AA parameter tables  (kcal/mol, Å)
# ══════════════════════════════════════════════════════════════════════════════

# (OPLS_label, mass, epsilon kcal/mol, sigma Å, description)
OPLS_ATOM_TYPES: dict[str, tuple] = {
    # ── ether / solvent ────────────────────────────────────────────────────
    "OS":   ("OS",   15.999, 0.140, 2.900, "ether oxygen"),
    "CT_O": ("CT_O", 12.011, 0.066, 3.500, "CH2 bonded to ether O"),
    "CT_M": ("CT_M", 12.011, 0.066, 3.500, "CH3 methyl ether"),
    "CT_C": ("CT_C", 12.011, 0.066, 3.500, "CH2 in ester/carbonate ring"),
    # ── carbonate ─────────────────────────────────────────────────────────
    "C_CO": ("C_CO", 12.011, 0.105, 3.750, "carbonyl C"),
    "O_CO": ("O_CO", 15.999, 0.210, 2.960, "carbonyl O"),
    "OS_E": ("OS_E", 15.999, 0.140, 2.900, "ester/carbonate bridging O"),
    # ── FSI- / LiFSI ──────────────────────────────────────────────────────
    "NI":   ("NI",   14.007, 0.170, 3.250, "imide N in FSI-/TFSI-"),
    "SF":   ("SF",   32.060, 0.250, 3.550, "sulfonyl S bonded to F"),
    "OY":   ("OY",   15.999, 0.210, 2.960, "sulfonyl O in FSI-"),
    "FS":   ("FS",   19.000, 0.061, 2.940, "F on S in FSI-"),
    # ── TFSI- extra ───────────────────────────────────────────────────────
    "CF":   ("CF",   12.011, 0.066, 3.500, "CF3 carbon in TFSI- (bonded to S)"),
    "FT":   ("FT",   19.000, 0.061, 2.940, "F on CF3 in TFSI-"),
    "SY":   ("SY",   32.060, 0.250, 3.550, "sulfonyl S in TFSI- (not bonded to F)"),
    "NT":   ("NT",   14.007, 0.170, 3.250, "N fallback (chain-end or unknown env)"),
    "P_F6": ("P_F6", 30.974, 0.200, 3.740, "P in PF6- (Lopes 2004)"),
    "F_F6": ("F_F6", 19.000, 0.061, 2.940, "F in PF6- (Lopes 2004)"),
    # ── Li ion ────────────────────────────────────────────────────────────
    "Li":   ("Li",   6.941,  0.018, 1.582, "Li+ ion"),
    # ── PVDF-HFP fluoropolymer ─────────────────────────────────────
    "CT_H2": ("CT_H2", 12.011, 0.066, 3.500, "CH2 in PVDF backbone"),
    "CT_F2": ("CT_F2", 12.011, 0.066, 3.500, "CF2 in PVDF/HFP backbone"),
    "CT_F1": ("CT_F1", 12.011, 0.066, 3.500, "CF in HFP bearing CF3 branch"),
    "CT_F3": ("CT_F3", 12.011, 0.066, 3.500, "CF3 carbon in HFP branch"),
    "FP":    ("FP",    19.000, 0.053, 2.950, "F on CF2/CHF polymer backbone (PVDF/HFP)"),
    "FP3":   ("FP3",   19.000, 0.053, 2.950, "F on CF3 branch (HFP/PTFEP)"),
    "HP":    ("HP",    1.008,  0.030, 2.500, "H on CH2 in PVDF backbone (CT_H2)"),
    # ── PTFEP (poly(bis(trifluoroethoxy)phosphazene)) ──────────────
    "P_N":   ("P_N",   30.974, 0.200, 3.740, "phosphazene P in PTFEP"),
    "N_P":   ("N_P",   14.007, 0.170, 3.250, "phosphazene N (P=N) in PTFEP"),
    "CT_B":  ("CT_B",  12.011, 0.066, 3.500, "CH2 backbone in PMMA/PTFEP"),
    "CQ":    ("CQ",    12.011, 0.105, 3.750, "quaternary C in PMMA (no H)"),
    # ── generic ───────────────────────────────────────────────────────────
    "HC":    ("HC",    1.008,  0.030, 2.500, "H on C (alkyl)"),
    "HC_O":  ("HC_O",  1.008,  0.030, 2.500, "H on ether alpha C (CT_O/CT_M), q=+0.030"),
    "HOS":   ("HOS",   1.008,  0.030, 2.500, "H on terminal ether-O (OH end-group), q=+0.200"),
}

OPLS_CHARGES: dict[str, float] = {
    # Ether O and C: OPLS-AA with HC_O=+0.030 for H on ether CH2 (CT_O).
    # CT_M adjusted to +0.020 so DME is neutral:
    #   DME: 2×OS(-0.400)+2×CT_O(+0.140)+2×CT_M(+0.020)+4×HC_O(+0.030)+6×HC(+0.060)=0 ✓
    # PEO: OS(-0.400)+2×CT_O(+0.140)+4×HC_O(+0.030)=0 ✓
    "OS":    -0.4000,   # ether oxygen
    "CT_O":  +0.1400,   # CH2 bonded to ether O
    "CT_M":  +0.0200,   # CH3 terminal methyl ether (was -0.040; updated for DME neutrality with HC_O)
    "HC_O":  +0.0300,   # H on ether CH2 (CT_O); PEO neutral: 2×CT_O+4×HC_O+OS=0
    "HOS":   +0.2000,   # H on terminal ether-O (OH); terminal group neutral: OS+CT_O+2HC_O+HOS=0
    "CT_C":  -0.0908,
    "C_CO":   0.5100,
    "O_CO":  -0.4300,
    "OS_E":  -0.3300,
    "NI":    -0.6600,
    "SF":     1.0200,
    "OY":    -0.5300,
    "FS":    -0.1300,
    "CF":     0.3500,
    "FT":    -0.1600,
    "SY":     1.0200,
    "NT":    -0.6600,
    "P_F6":   1.1980,
    "F_F6":  -0.3663,
    "Li":     1.0000,
    "HC":     0.0600,
    # PVDF-HFP (Byutner & Smith 2000 + Ameduri group)
    # Signs: F is electronegative → FP/FP3 negative; CF2-C is electron-poor → CT_F2 positive
    "CT_H2":  +0.176,   # CH₂ backbone carbon in PVDF
    "CT_F2":  +0.212,   # CF₂ backbone carbon (was -0.212, sign was inverted)
    "CT_F1":  +0.107,   # CHF carbon in HFP (was -0.107)
    "CT_F3":  +0.580,   # CF₃ branch carbon (quaternary, electron-poor)
    "FP":     -0.106,   # F on CF₂/CHF backbone (was +0.106, sign inverted)
    "FP3":    -0.193,   # F on CF₃ branch (already correct)
    "HP":     -0.088,   # H on CH₂ backbone
    # PTFEP
    "P_N":    1.100,
    "N_P":   -0.680,
    "CT_B":  -0.084,
    "CQ":     0.240,
}

OPLS_BONDS: dict[tuple, tuple] = {
    ("OS",   "CT_O"):  (320.0, 1.410),
    ("OS",   "CT_M"):  (320.0, 1.410),
    ("CT_O", "CT_O"):  (268.0, 1.529),
    ("CT_O", "CT_M"):  (268.0, 1.529),
    ("CT_O", "HC"):    (340.0, 1.090),
    ("CT_M", "HC"):    (340.0, 1.090),
    ("CT_O", "HC_O"):  (340.0, 1.090),   # ether alpha C–H (same LJ, q=+0.030)
    ("CT_M", "HC_O"):  (340.0, 1.090),
    ("OS",   "HOS"):   (553.0, 0.945),   # terminal ether-O to OH hydrogen (OPLS-AA O-H)
    ("C_CO", "O_CO"):  (799.0, 1.200),
    ("C_CO", "OS_E"):  (340.0, 1.344),
    ("OS_E", "CT_O"):  (320.0, 1.410),
    ("OS_E", "CT_M"):  (320.0, 1.410),
    ("CT_C", "HC"):    (340.0, 1.090),
    ("CT_C", "OS_E"):  (320.0, 1.410),
    ("CT_C", "CT_C"):  (268.0, 1.529),
    ("CT_C", "CT_O"):  (268.0, 1.529),
    # FSI-
    ("NI",   "SF"):    (400.0, 1.570),
    ("SF",   "OY"):    (637.0, 1.440),
    ("SF",   "FS"):    (520.0, 1.575),
    # TFSI-
    ("SF",   "CF"):    (300.0, 1.818),
    ("CF",   "FT"):    (367.0, 1.323),
    ("NI",   "SY"):    (400.0, 1.570),
    ("SY",   "OY"):    (637.0, 1.440),
    ("SY",   "CF"):    (300.0, 1.818),
    # LiPF6
    ("P_F6", "F_F6"):  (287.0, 1.606),
    # PVDF-HFP bonds
    ("CT_H2", "CT_F2"):  (268.0, 1.529),
    ("CT_F2", "CT_F1"):  (268.0, 1.529),
    ("CT_F1", "CT_F3"):  (268.0, 1.520),
    ("CT_H2", "HP"):     (340.0, 1.090),
    ("CT_F2", "FP"):     (367.0, 1.340),
    ("CT_F1", "FP"):     (367.0, 1.340),
    ("CT_F3", "FP3"):    (367.0, 1.332),
    ("CT_B",  "FP3"):    (367.0, 1.332),
    # PTFEP bonds
    ("P_N",  "N_P"):     (400.0, 1.580),
    ("P_N",  "OS"):      (320.0, 1.620),
    ("CT_B", "CT_F3"):   (268.0, 1.520),
    # PMMA bonds
    ("CQ",   "CT_B"):    (268.0, 1.529),
    ("CQ",   "CT_M"):    (268.0, 1.529),
    ("CQ",   "C_CO"):    (317.0, 1.522),
    ("CT_B", "HC"):      (340.0, 1.090),
}

OPLS_ANGLES: dict[tuple, tuple] = {
    ("CT_M", "OS",  "CT_O"):  (60.0, 109.5),
    ("CT_M", "OS",  "CT_M"):  (60.0, 109.5),
    ("CT_O", "OS",  "CT_O"):  (60.0, 109.5),
    ("OS",   "CT_O","CT_O"):  (50.0, 109.5),
    ("OS",   "CT_O","HC"):    (35.0, 109.5),
    ("CT_O", "CT_O","HC"):    (37.5, 110.7),
    ("HC",   "CT_O","HC"):    (33.0, 107.8),
    ("OS",   "CT_M","HC"):    (35.0, 109.5),
    ("HC",   "CT_M","HC"):    (33.0, 107.8),
    # HC_O (ether alpha H, q=+0.030) — same force constants as HC
    ("OS",   "CT_O","HC_O"):  (35.0, 109.5),
    ("CT_O", "CT_O","HC_O"):  (37.5, 110.7),
    ("HC_O", "CT_O","HC_O"):  (33.0, 107.8),
    ("OS",   "CT_M","HC_O"):  (35.0, 109.5),
    ("HC_O", "CT_M","HC_O"):  (33.0, 107.8),
    ("OS_E", "CT_O","HC_O"):  (35.0, 109.5),
    ("OS_E", "CT_M","HC_O"):  (35.0, 109.5),
    ("O_CO", "C_CO","OS_E"):  (80.0, 123.4),
    ("OS_E", "C_CO","OS_E"):  (74.0, 113.1),
    ("C_CO", "OS_E","CT_O"):  (60.0, 116.9),
    ("C_CO", "OS_E","CT_M"):  (60.0, 116.9),
    ("OS_E", "CT_O","HC"):    (35.0, 109.5),
    ("OS_E", "CT_C","HC"):    (35.0, 109.5),
    ("OS_E", "CT_C","CT_C"):  (50.0, 109.5),
    ("CT_C", "OS_E","C_CO"):  (60.0, 116.9),
    ("HC",   "CT_C","HC"):    (33.0, 107.8),
    ("CT_C", "CT_C","OS_E"):  (50.0, 109.5),
    ("CT_C", "CT_C","HC"):    (37.5, 110.7),
    # FSI-
    ("SF",   "NI",  "SF"):    (83.0, 125.6),
    ("NI",   "SF",  "OY"):    (100.0,113.6),
    ("NI",   "SF",  "FS"):    (93.3, 100.2),
    ("OY",   "SF",  "OY"):    (116.0,118.5),
    ("OY",   "SF",  "FS"):    (115.0,107.5),
    # TFSI- extra
    ("NI",   "SF",  "CF"):    (85.7, 103.5),
    ("SF",   "CF",  "FT"):    (75.0, 112.0),
    ("FT",   "CF",  "FT"):    (77.0, 107.0),
    ("SY",   "NI",  "SY"):    (83.0, 125.6),
    ("SF",   "NI",  "SY"):    (83.0, 125.6),
    ("NI",   "SY",  "OY"):    (100.0,113.6),
    ("OY",   "SY",  "OY"):    (116.0,118.5),
    ("NI",   "SY",  "CF"):    (85.7, 103.5),
    ("SY",   "CF",  "FT"):    (75.0, 112.0),
    # LiPF6
    ("F_F6", "P_F6","F_F6"):  (100.0, 90.0),
    # PVDF-HFP angles
    ("HP",   "CT_H2", "HP"):    (33.0,  107.8),
    ("HP",   "CT_H2", "CT_F2"): (37.5,  110.7),
    ("CT_H2","CT_F2", "FP"):    (40.0,  108.5),
    ("FP",   "CT_F2", "FP"):    (77.0,  107.1),
    ("CT_F2","CT_F1", "FP"):    (40.0,  108.5),
    ("CT_F1","CT_F3", "FP3"):   (75.0,  111.7),
    ("FP3",  "CT_F3", "FP3"):   (77.0,  107.0),
    ("CT_F2","CT_H2", "HP"):    (37.5,  110.7),
    ("CT_B", "CT_F3", "FP3"):   (75.0,  111.7),
    # PTFEP angles
    ("N_P",  "P_N",  "OS"):     (100.0, 101.5),
    ("P_N",  "N_P",  "P_N"):    (130.0, 133.5),
    ("P_N",  "OS",  "CT_B"):    (60.0,  120.0),
    # PMMA angles
    ("CT_B", "CQ",   "CT_M"):   (58.0,  109.5),
    ("CT_B", "CQ",   "C_CO"):   (63.0,  109.5),
    ("CQ",   "C_CO", "OS_E"):   (60.0,  116.9),
    ("HC",   "CT_B", "CQ"):     (37.5,  110.7),
    ("HC",   "CT_B", "HC"):     (33.0,  107.8),
}

OPLS_DIHEDRALS: dict[tuple, tuple] = {
    ("X", "OS",   "CT_O", "X"): (0.000,  0.000,  0.760, 0.000),
    ("X", "OS",   "CT_M", "X"): (0.000,  0.000,  0.760, 0.000),
    ("X", "CT_O", "CT_O", "X"): (0.650, -0.250,  0.670, 0.000),
    ("OS","CT_O", "CT_O", "OS"): (-0.550, 0.000,  0.000, 0.000),
    ("X", "CT_O", "OS",   "X"): (0.000,  0.000,  0.468, 0.000),
    ("X", "C_CO", "OS_E", "X"): (3.700,  8.600,  0.000, 0.000),
    ("X", "OS_E", "CT_O", "X"): (0.000,  0.000,  0.468, 0.000),
    ("X", "OS_E", "CT_M", "X"): (0.000,  0.000,  0.468, 0.000),
    ("X", "OS_E", "CT_C", "X"): (0.000,  0.000,  0.468, 0.000),
    ("X", "CT_C", "CT_C", "X"): (0.650, -0.250,  0.670, 0.000),
    ("X", "HC",   "CT_O", "X"): (0.000,  0.000,  0.300, 0.000),
    ("X", "HC",   "CT_M", "X"): (0.000,  0.000,  0.300, 0.000),
    ("X", "HC",   "CT_C", "X"): (0.000,  0.000,  0.300, 0.000),
    ("X", "HC_O", "CT_O", "X"): (0.000,  0.000,  0.300, 0.000),
    ("X", "HC_O", "CT_M", "X"): (0.000,  0.000,  0.300, 0.000),
    # FSI-
    ("X", "NI",   "SF",   "X"): (0.000,  0.000,  0.000, 0.000),
    ("X", "SF",   "NI",   "X"): (0.000,  0.000,  0.000, 0.000),
    # TFSI- (SY = sulfonyl S not bonded to F)
    ("X", "NI",   "SY",   "X"): (0.000,  0.000,  0.000, 0.000),
    ("X", "SY",   "NI",   "X"): (0.000,  0.000,  0.000, 0.000),
}

OPLS_IMPROPERS: dict[tuple, tuple] = {
    ("C_CO",): (10.5, -1, 2),
}

# Built-in molecule geometries (atoms, bonds, resname)
# Used as fallback when no VASP file is found.
MOLECULES: dict[str, dict] = {
    "DME": {
        "resname": "DME",
        "atoms": [
            ("O",  1.748,  0.495, -0.011), ("O", -1.748, -0.495, -0.011),
            ("C",  0.644, -0.401,  0.000), ("C", -0.644,  0.401,  0.000),
            ("C",  2.985, -0.204,  0.011), ("C", -2.985,  0.203,  0.011),
            ("H",  0.679, -1.053, -0.882), ("H",  0.684, -1.042,  0.889),
            ("H", -0.679,  1.052, -0.882), ("H", -0.684,  1.042,  0.889),
            ("H",  3.796,  0.530, -0.001), ("H",  3.080, -0.843, -0.872),
            ("H",  3.071, -0.803,  0.922), ("H", -3.071,  0.803,  0.922),
            ("H", -3.796, -0.530, -0.001), ("H", -3.080,  0.843, -0.872),
        ],
        "bonds": [(1,3),(1,5),(2,4),(2,6),(3,4),(3,7),(3,8),
                  (4,9),(4,10),(5,11),(5,12),(5,13),(6,14),(6,15),(6,16)],
    },
    "DOL": {
        "resname": "DOL",
        "atoms": [
            ("O",  1.200,  0.693,  0.000), ("O", -1.200,  0.693,  0.000),
            ("C",  0.000,  1.386,  0.000), ("C",  1.336, -0.700,  0.000),
            ("C", -1.336, -0.700,  0.000), ("H",  0.000,  2.015,  0.890),
            ("H",  0.000,  2.015, -0.890), ("H",  1.923, -0.950,  0.893),
            ("H",  1.923, -0.950, -0.893), ("H", -1.923, -0.950,  0.893),
            ("H", -1.923, -0.950, -0.893),
        ],
        "bonds": [(1,3),(1,4),(2,3),(2,5),(4,5),
                  (3,6),(3,7),(4,8),(4,9),(5,10),(5,11)],
    },
    "EC": {
        "resname": "EC",
        "atoms": [
            ("C",  0.000,  0.000,  0.000), ("O",  1.220,  0.000,  0.000),
            ("O", -0.672,  1.177,  0.000), ("O", -0.672, -1.177,  0.000),
            ("C", -2.055,  1.080,  0.000), ("C", -2.055, -1.080,  0.000),
            ("H", -2.449,  1.578,  0.890), ("H", -2.449,  1.578, -0.890),
            ("H", -2.449, -1.578,  0.890), ("H", -2.449, -1.578, -0.890),
        ],
        "bonds": [(1,2),(1,3),(1,4),(3,5),(4,6),(5,6),
                  (5,7),(5,8),(6,9),(6,10)],
    },
    "DMC": {
        "resname": "DMC",
        "atoms": [
            ("C",  0.000,  0.000,  0.000), ("O",  0.000,  1.220,  0.000),
            ("O",  1.355, -0.550,  0.000), ("O", -1.355, -0.550,  0.000),
            ("C",  2.550,  0.278,  0.000), ("C", -2.550,  0.278,  0.000),
            ("H",  2.541,  0.913,  0.890), ("H",  2.541,  0.913, -0.890),
            ("H",  3.442, -0.350,  0.000), ("H", -2.541,  0.913,  0.890),
            ("H", -2.541,  0.913, -0.890), ("H", -3.442, -0.350,  0.000),
        ],
        "bonds": [(1,2),(1,3),(1,4),(3,5),(4,6),
                  (5,7),(5,8),(5,9),(6,10),(6,11),(6,12)],
    },
    "EMC": {
        "resname": "EMC",
        "atoms": [
            ("C",  0.000,  0.000,  0.000), ("O",  0.000,  1.220,  0.000),
            ("O",  1.355, -0.550,  0.000), ("O", -1.355, -0.550,  0.000),
            ("C",  2.550,  0.278,  0.000), ("C", -2.550,  0.278,  0.000),
            ("C", -3.970, -0.272,  0.000),
            ("H",  2.541,  0.913,  0.890), ("H",  2.541,  0.913, -0.890),
            ("H",  3.442, -0.350,  0.000), ("H", -2.521,  0.923,  0.890),
            ("H", -2.521,  0.923, -0.890), ("H", -3.990, -0.900,  0.890),
            ("H", -3.990, -0.900, -0.890), ("H", -4.763,  0.470,  0.000),
        ],
        "bonds": [(1,2),(1,3),(1,4),(3,5),(4,6),(6,7),
                  (5,8),(5,9),(5,10),(6,11),(6,12),(7,13),(7,14),(7,15)],
    },
    "DEC": {
        # Diethyl carbonate — symmetric linear carbonate (both arms ethyl).
        # Carbonate core positioned identically to DMC/EMC; each arm mirrors
        # EMC's ethyl arm (C6/C7) across x=0.
        "resname": "DEC",
        "atoms": [
            ("C",  0.000,  0.000,  0.000), ("O",  0.000,  1.220,  0.000),
            ("O",  1.355, -0.550,  0.000), ("O", -1.355, -0.550,  0.000),
            ("C",  2.550,  0.278,  0.000), ("C", -2.550,  0.278,  0.000),
            ("C",  3.970, -0.272,  0.000), ("C", -3.970, -0.272,  0.000),
            ("H",  2.521,  0.923,  0.890), ("H",  2.521,  0.923, -0.890),
            ("H", -2.521,  0.923,  0.890), ("H", -2.521,  0.923, -0.890),
            ("H",  3.990, -0.900,  0.890), ("H",  3.990, -0.900, -0.890),
            ("H",  4.763,  0.470,  0.000),
            ("H", -3.990, -0.900,  0.890), ("H", -3.990, -0.900, -0.890),
            ("H", -4.763,  0.470,  0.000),
        ],
        "bonds": [(1,2),(1,3),(1,4),(3,5),(4,6),(5,7),(6,8),
                  (5,9),(5,10),(6,11),(6,12),
                  (7,13),(7,14),(7,15),(8,16),(8,17),(8,18)],
    },
    "LIFSI": {
        "resname": "LiFSI",
        "atoms": [
            # LiFSI: Li+ + [N(SO2F)2]- (9 atoms total)
            ("Li",  0.000,  0.000,  0.000),
            ("N",   3.200,  0.000,  0.000),
            ("S",   4.770,  0.900,  0.000), ("S",  1.630,  0.900,  0.000),
            ("O",   5.900,  0.200,  0.600), ("O",  4.800,  2.200,  0.000),
            ("O",   0.500,  0.200,  0.600), ("O",  1.600,  2.200,  0.000),
            ("F",   5.000, -0.400, -1.200), ("F",  1.400, -0.400, -1.200),
        ],
        "bonds": [(2,3),(2,4),(3,5),(3,6),(3,9),(4,7),(4,8),(4,10)],
    },
    "LITFSI": {
        "resname": "LiTFSI",
        "atoms": [
            ("Li",  0.000,  0.000,  0.000),
            ("N",   3.200,  0.000,  0.000),
            ("S",   4.770,  0.900,  0.000), ("S",  1.630,  0.900,  0.000),
            ("O",   5.900,  0.200,  0.600), ("O",  4.800,  2.200,  0.000),
            ("O",   0.500,  0.200,  0.600), ("O",  1.600,  2.200,  0.000),
            ("C",   5.200, -0.500, -1.300), ("C",  1.200, -0.500, -1.300),
            ("F",   5.200, -1.800, -1.000), ("F",  6.400,  0.000, -1.400),
            ("F",   4.200, -0.100, -2.300), ("F",  1.200, -1.800, -1.000),
            ("F",   2.400,  0.000, -1.400), ("F",  0.200, -0.100, -2.300),
        ],
        "bonds": [(2,3),(2,4),(3,5),(3,6),(3,9),(4,7),(4,8),(4,10),
                  (9,11),(9,12),(9,13),(10,14),(10,15),(10,16)],
    },
    "PEO": {
        # Repeat unit: -CH2-CH2-O- (trimeric fragment, 9 atoms)
        "resname": "PEO",
        "atoms": [
            ("O",  0.000,  0.000,  0.000),
            ("C",  1.410,  0.000,  0.000), ("C",  2.050,  1.300,  0.000),
            ("O",  3.460,  1.300,  0.000),
            ("C",  4.100,  2.600,  0.000), ("C",  5.510,  2.600,  0.000),
            ("O",  6.150,  3.900,  0.000),
            ("H",  1.780, -0.540,  0.890), ("H",  1.780, -0.540, -0.890),
            ("H",  1.680,  1.840,  0.890), ("H",  1.680,  1.840, -0.890),
            ("H",  3.730,  3.140,  0.890), ("H",  3.730,  3.140, -0.890),
            ("H",  5.880,  2.060,  0.890), ("H",  5.880,  2.060, -0.890),
        ],
        "bonds": [(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),
                  (2,8),(2,9),(3,10),(3,11),(5,12),(5,13),(6,14),(6,15)],
    },
    "PMMA": {
        # Repeat unit: -CH2-C(CH3)(COOCH3)- (10 heavy atoms + H)
        "resname": "PMMA",
        "atoms": [
            ("C",  0.000,  0.000,  0.000),                      # CT_B
            ("C",  1.529,  0.000,  0.000),                      # CQ
            ("C",  2.180,  1.430,  0.000),                      # CT_M (CH3 on CQ)
            ("C",  2.180, -0.700, -1.200),                      # C_CO (ester)
            ("O",  3.380, -0.600, -1.200),                      # O_CO
            ("O",  1.520, -1.360, -1.400),                      # OS_E
            ("C",  1.900, -2.600, -0.800),                      # CT_M (OCH3)
            ("H",  0.000,  0.000,  0.000),                      # placeholder H on CT_B
            ("H", -0.370,  1.020,  0.000), ("H", -0.370, -0.510,  0.890),
            ("H", -0.370, -0.510, -0.890),
            ("H",  1.800,  2.020,  0.890), ("H",  1.800,  2.020, -0.890),
            ("H",  3.260,  1.380,  0.000),
            ("H",  1.500, -3.400, -1.400), ("H",  3.000, -2.700, -0.900),
            ("H",  1.700, -2.700,  0.260),
        ],
        "bonds": [(1,2),(2,3),(2,4),(4,5),(4,6),(6,7),
                  (1,9),(1,10),(1,11),(3,12),(3,13),(3,14),(7,15),(7,16),(7,17)],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Molecular file readers
# ══════════════════════════════════════════════════════════════════════════════

def read_pdb(path: Path) -> tuple[list, list]:
    """Parse PDB ATOM/HETATM records → (atoms, bonds)."""
    atoms: list[tuple] = []
    conect: dict[int, set] = defaultdict(set)
    with open(path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec in ("ATOM", "HETATM"):
                el = line[76:78].strip() if len(line) > 76 else ""
                if not el:
                    name = line[12:16].strip().lstrip("0123456789")
                    el = name[0] if name else "X"
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                atoms.append((el.capitalize(), x, y, z))
            elif rec == "CONECT":
                parts = line.split()
                src = int(parts[1])
                for dst in parts[2:]:
                    conect[src].add(int(dst))
    bonds = [(a, b) for a, ns in conect.items() for b in sorted(ns) if b > a]
    if not bonds:
        bonds = _guess_bonds(atoms)
    return atoms, bonds


def read_xyz(path: Path) -> tuple[list, list]:
    """Parse XYZ file → (atoms, bonds)."""
    atoms: list[tuple] = []
    with open(path) as fh:
        n = int(fh.readline())
        fh.readline()
        for _ in range(n):
            parts = fh.readline().split()
            atoms.append((parts[0].capitalize(),
                          float(parts[1]), float(parts[2]), float(parts[3])))
    return atoms, _guess_bonds(atoms)


_VASP_INLINE_EL = frozenset(
    ["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S",
     "Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
     "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru",
     "Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce",
     "Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf",
     "Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn"]
)


def _extract_inline_element(raw_line: str) -> str | None:
    """Return element symbol from `! Elem` or `# Elem` inline comment, or None."""
    for sep in ("!", "#"):
        if sep in raw_line:
            after = raw_line.split(sep, 1)[1].strip().split()
            if after:
                candidate = after[0].capitalize()
                if candidate in _VASP_INLINE_EL:
                    return candidate
    return None


def read_vasp(path: Path) -> tuple[list, list]:
    """Parse VASP POSCAR/CONTCAR/.vasp → (atoms, bonds).

    Returns fractional coordinates converted to Cartesian Å.
    Honors inline element comments (`! C`, `# F`, …) that some VASP writers
    embed in coordinate lines when atoms are stored in non-standard (interleaved)
    order — overriding the species-count header for those atoms.
    """
    # Manual POSCAR parser — handles both standard and interleaved element order
    lines = Path(path).read_text().splitlines()
    scale = float(lines[1].strip())
    lat = [[float(v) * scale for v in lines[i].split()] for i in range(2, 5)]
    species_line = lines[5].split()
    counts_line  = lines[6].split()
    try:
        counts = [int(c) for c in counts_line]
        species = species_line
        coord_start = 8
        direct = lines[7].strip().lower().startswith("d")
    except ValueError:
        counts = [int(c) for c in lines[7].split()]
        species = counts_line
        coord_start = 9
        direct = lines[8].strip().lower().startswith("d")

    element_list: list[str] = []
    for el, cnt in zip(species, counts):
        element_list.extend([el] * cnt)

    coord_lines = lines[coord_start: coord_start + len(element_list)]

    # Check whether inline element comments are present and differ from header.
    # If > 30% of lines carry an inline element token, use inline over header.
    inline_els: list[str | None] = [_extract_inline_element(l) for l in coord_lines]
    n_inline = sum(1 for e in inline_els if e is not None)
    use_inline = n_inline > 0.3 * len(element_list)

    atoms: list[tuple] = []
    for i, raw in enumerate(coord_lines):
        vals = [float(v) for v in raw.split()[:3]]
        if direct:
            x = sum(vals[j] * lat[j][0] for j in range(3))
            y = sum(vals[j] * lat[j][1] for j in range(3))
            z = sum(vals[j] * lat[j][2] for j in range(3))
        else:
            x, y, z = vals
        el = (inline_els[i] if use_inline and inline_els[i] else element_list[i])
        atoms.append((el, x, y, z))
    return atoms, _guess_bonds(atoms)


def parse_molecule(path: Path) -> tuple[list, list]:
    """Auto-detect file format and parse → (atoms, bonds)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        return read_pdb(path)
    if suffix == ".xyz":
        return read_xyz(path)
    if suffix in (".vasp", ".poscar", ""):
        return read_vasp(path)
    name = path.name.upper()
    if name in ("POSCAR", "CONTCAR"):
        return read_vasp(path)
    return read_vasp(path)


def write_pdb(atoms: list, bonds: list, path: Path, resname: str = "MOL") -> None:
    """Write a minimal PDB file from (atoms, bonds)."""
    lines = [f"COMPND    {resname}",
             "AUTHOR    GENERATED BY HPCA FORCEFIELD MODULE"]
    for i, (el, x, y, z) in enumerate(atoms, 1):
        lines.append(
            f"HETATM{i:5d}  {el:<3s} {resname:3s}     1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {el:>2s}"
        )
    for a1, a2 in bonds:
        lines.append(f"CONECT{a1:5d}{a2:5d}")
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Bond guesser
# ══════════════════════════════════════════════════════════════════════════════

_COV_RAD: dict[str, float] = {
    "H": 0.31, "C": 0.76, "O": 0.66, "N": 0.71,
    "S": 1.05, "F": 0.57, "P": 1.07, "Li": 1.28,
}
_MAX_VAL: dict[str, int] = {
    "H": 1, "C": 4, "O": 2, "N": 3, "S": 6, "F": 1, "P": 6, "Li": 0,
}


def _dist(a: tuple, b: tuple) -> float:
    """Return Euclidean distance between two (el, x, y, z) atom tuples."""
    return math.sqrt((a[1]-b[1])**2 + (a[2]-b[2])**2 + (a[3]-b[3])**2)


def _guess_bonds(atoms: list, tol: float = 0.15) -> list:
    """Covalent-radius bond guesser; respects max valence."""
    candidates: list[tuple] = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            ri = _COV_RAD.get(atoms[i][0], 0.70)
            rj = _COV_RAD.get(atoms[j][0], 0.70)
            d = _dist(atoms[i], atoms[j])
            if d < ri + rj + tol:
                candidates.append((d, i + 1, j + 1))

    candidates.sort()
    bonds: list[tuple] = []
    valence: dict[int, int] = defaultdict(int)
    for _d, i, j in candidates:
        ei, ej = atoms[i-1][0], atoms[j-1][0]
        if (valence[i] < _MAX_VAL.get(ei, 4) and
                valence[j] < _MAX_VAL.get(ej, 4)):
            bonds.append((i, j))
            valence[i] += 1
            valence[j] += 1
    return bonds


# ══════════════════════════════════════════════════════════════════════════════
# Atom-type assignment
# ══════════════════════════════════════════════════════════════════════════════

def assign_atom_types(atoms: list, bonds: list) -> list[str]:
    """Assign OPLS-AA types from element + bonding environment."""
    neighbors: dict[int, list] = defaultdict(list)
    for a, b in bonds:
        neighbors[a].append(b)
        neighbors[b].append(a)

    types: list[str] = []
    for idx in range(1, len(atoms) + 1):
        el   = atoms[idx-1][0]
        nbrs = neighbors[idx]

        def n_el(e: str) -> int:
            """Count neighbors of element e around the current atom."""
            return sum(1 for nb in nbrs if atoms[nb-1][0] == e)

        if el == "Li":
            atype = "Li"

        elif el == "F":
            s_nbrs = [nb for nb in nbrs if atoms[nb-1][0] == "S"]
            p_nbrs = [nb for nb in nbrs if atoms[nb-1][0] == "P"]
            if s_nbrs:
                atype = "FS"
            elif p_nbrs:
                atype = "F_F6"
            else:
                c_nbrs = [nb for nb in nbrs if atoms[nb-1][0] == "C"]
                if c_nbrs and any(atoms[x-1][0] == "S" for x in neighbors[c_nbrs[0]]):
                    atype = "FT"   # F on CF3 bonded to S (TFSI-)
                else:
                    atype = "FP"   # F on polymer C (PVDF/HFP/PTFEP)

        elif el == "N":
            if n_el("S") >= 2:
                atype = "NI"   # imide N in FSI-/TFSI-
            elif n_el("P") >= 1:
                atype = "N_P"  # phosphazene N in PTFEP
            else:
                atype = "NT"   # N fallback

        elif el == "P":
            atype = "P_F6" if n_el("F") >= 4 else "P_N"

        elif el == "S":
            # S bonded to F → SF (FSI-); generic otherwise
            atype = "SF" if n_el("F") >= 1 else "SY"

        elif el == "H":
            atype = "HC"
            # will be corrected to HP if bonded to CT_H2 in post-pass

        elif el == "O":
            c_nbrs = [nb for nb in nbrs if atoms[nb-1][0] == "C"]
            s_nbrs = [nb for nb in nbrs if atoms[nb-1][0] == "S"]
            if s_nbrs:
                atype = "OY"    # sulfonyl O (bonded to S)
            elif len(c_nbrs) == 2:
                is_ester = any(
                    sum(1 for nb2 in neighbors[cn] if atoms[nb2-1][0] == "O") >= 2
                    for cn in c_nbrs
                )
                atype = "OS_E" if is_ester else "OS"
            else:
                # 1 C neighbor: carbonyl O only if that C has >=2 O and 0 H
                # (e.g. ester/ketone C=O). Terminal ether O (e.g. PEO end) has
                # 1 C neighbor but that C is saturated → assign OS instead.
                cn = c_nbrs[0] if c_nbrs else None
                if cn is not None:
                    c_has_2o = sum(1 for nb2 in neighbors[cn] if atoms[nb2-1][0] == "O") >= 2
                    c_has_no_h = sum(1 for nb2 in neighbors[cn] if atoms[nb2-1][0] == "H") == 0
                    atype = "O_CO" if (c_has_2o and c_has_no_h) else "OS"
                else:
                    atype = "O_CO"  # lone O, treat as carbonyl

        elif el == "C":
            o_nbrs = [nb for nb in nbrs if atoms[nb-1][0] == "O"]
            h_nbrs = [nb for nb in nbrs if atoms[nb-1][0] == "H"]
            f_nbrs = [nb for nb in nbrs if atoms[nb-1][0] == "F"]
            s_nbrs_c = [nb for nb in nbrs if atoms[nb-1][0] == "S"]

            if f_nbrs:
                n_f = len(f_nbrs)
                if s_nbrs_c:
                    atype = "CF"      # TFSI- CF3 bonded to S
                elif n_f >= 3:
                    atype = "CT_F3"   # CF3 branch (HFP/PTFEP)
                elif n_f == 2:
                    atype = "CT_F2"   # CF2 backbone (PVDF)
                else:
                    atype = "CT_F1"   # CHF backbone
            elif len(o_nbrs) >= 2 and len(h_nbrs) == 0:
                atype = "C_CO"  # carbonyl C
            elif len(o_nbrs) == 1:
                if len(h_nbrs) >= 3:
                    atype = "CT_M"   # terminal methyl ether CH3-O (e.g. DME end)
                else:
                    on = o_nbrs[0]
                    on_c_nbrs = [x for x in neighbors[on] if atoms[x-1][0] == "C"]
                    if len(on_c_nbrs) == 2:
                        is_e = any(
                            sum(1 for nb3 in neighbors[cn2] if atoms[nb3-1][0] == "O") >= 2
                            for cn2 in on_c_nbrs
                        )
                        atype = "CT_C" if is_e else "CT_O"
                    else:
                        atype = "CT_O"
            elif len(o_nbrs) == 0:
                if len(h_nbrs) >= 3:
                    # PVDF-type terminal CH₃ (adjacent C has F) → CT_H2 so HP is assigned
                    c_nbrs_c = [nb for nb in nbrs if atoms[nb-1][0] == "C"]
                    pvdf_ch3 = any(
                        any(atoms[nb2-1][0] == "F" for nb2 in neighbors[cn])
                        for cn in c_nbrs_c
                    )
                    atype = "CT_H2" if pvdf_ch3 else "CT_M"
                elif len(h_nbrs) == 2:
                    # CH₂ with F on an adjacent C → PVDF backbone CH₂
                    c_nbrs_c = [nb for nb in nbrs if atoms[nb-1][0] == "C"]
                    pvdf_ch2 = any(
                        any(atoms[nb2-1][0] == "F" for nb2 in neighbors[cn])
                        for cn in c_nbrs_c
                    )
                    atype = "CT_H2" if pvdf_ch2 else "CT_O"
                else:
                    atype = "CT_O"
            else:
                atype = "CT_O"
        else:
            atype = el  # unknown fallback

        types.append(atype)

    # Post-pass: H bonded to CT_H2 → HP (PVDF backbone H, q=-0.088)
    for idx in range(1, len(atoms) + 1):
        if types[idx-1] == "HC":
            for nb in neighbors[idx]:
                if types[nb-1] == "CT_H2":
                    types[idx-1] = "HP"
                    break

    # Post-pass: F bonded to CT_F3 → FP3 (CF3 branch, q=-0.193)
    for idx in range(1, len(atoms) + 1):
        if types[idx-1] == "FP":
            for nb in neighbors[idx]:
                if types[nb-1] == "CT_F3":
                    types[idx-1] = "FP3"
                    break

    # Post-pass: H bonded to CT_O → HC_O (q=+0.030, makes PEO repeat unit neutral)
    # HC on CT_M stays HC (+0.060); CT_M is +0.020 so each ether group sums to 0.
    for idx in range(1, len(atoms) + 1):
        if types[idx-1] == "HC":
            for nb in neighbors[idx]:
                if types[nb-1] == "CT_O":
                    types[idx-1] = "HC_O"
                    break

    # Post-pass: H bonded only to O (terminal OH proton, e.g. PEO chain ends) → HOS
    # q=+0.200 makes terminal group [OS+CT_O+2HC_O+HOS] neutral.
    for idx in range(1, len(atoms) + 1):
        if types[idx-1] == "HC":
            nbrs_idx = neighbors[idx]
            if len(nbrs_idx) == 1 and atoms[nbrs_idx[0]-1][0] == "O":
                types[idx-1] = "HOS"

    return types


# ══════════════════════════════════════════════════════════════════════════════
# Topology builders
# ══════════════════════════════════════════════════════════════════════════════

def build_angles(bonds: list) -> list:
    """Enumerate all 1-3 angle triplets (i, center, k) from a bond list."""
    neighbors: dict[int, list] = defaultdict(list)
    for a, b in bonds:
        neighbors[a].append(b)
        neighbors[b].append(a)
    angles: set = set()
    for center, nbrs in neighbors.items():
        snbrs = sorted(nbrs)
        for i in range(len(snbrs)):
            for j in range(i + 1, len(snbrs)):
                a1, a3 = snbrs[i], snbrs[j]
                angles.add((min(a1, a3), center, max(a1, a3)))
    return sorted(angles)


def build_dihedrals(bonds: list) -> list:
    """Enumerate all 1-4 proper dihedral quadruplets (i, j, k, l) from a bond list."""
    neighbors: dict[int, list] = defaultdict(list)
    for a, b in bonds:
        neighbors[a].append(b)
        neighbors[b].append(a)
    dihedrals: set = set()
    for b1, b2 in bonds:
        for a1 in neighbors[b1]:
            if a1 == b2:
                continue
            for a4 in neighbors[b2]:
                if a4 == b1 or a4 == a1:
                    continue
                d = tuple(sorted([(a1, b1, b2, a4), (a4, b2, b1, a1)])[0])
                dihedrals.add(d)
    return sorted(dihedrals)


def build_impropers(atoms: list, bonds: list, types: list[str]) -> list:
    """Return improper dihedral quadruplets for all carbonyl-C (C_CO) atoms."""
    neighbors: dict[int, list] = defaultdict(list)
    for a, b in bonds:
        neighbors[a].append(b)
        neighbors[b].append(a)
    impropers: list = []
    for idx in range(1, len(atoms) + 1):
        if types[idx-1] == "C_CO":
            nbrs = sorted(neighbors[idx])
            if len(nbrs) >= 3:
                impropers.append((idx, nbrs[0], nbrs[1], nbrs[2]))
    return impropers


# ══════════════════════════════════════════════════════════════════════════════
# Parameter lookups
# ══════════════════════════════════════════════════════════════════════════════

def lookup_bond(t1: str, t2: str) -> tuple:
    """Return (k, r0) OPLS-AA bond parameters for type pair (t1, t2); falls back to a generic C–C value."""
    for key in [(t1, t2), (t2, t1)]:
        if key in OPLS_BONDS:
            return OPLS_BONDS[key]
    return (268.0, 1.529)


def lookup_angle(t1: str, t2: str, t3: str) -> tuple:
    """Return (k, theta0) OPLS-AA angle parameters for triplet (t1, t2, t3); falls back to a generic sp3 angle."""
    for key in [(t1, t2, t3), (t3, t2, t1)]:
        if key in OPLS_ANGLES:
            return OPLS_ANGLES[key]
    return (35.0, 109.5)


def lookup_dihedral(t1: str, t2: str, t3: str, t4: str) -> tuple:
    """Return (V1, V2, V3, V4) Fourier dihedral coefficients; wildcard "X" lookup used as fallback."""
    for key in [(t1, t2, t3, t4), (t4, t3, t2, t1)]:
        if key in OPLS_DIHEDRALS:
            return OPLS_DIHEDRALS[key]
    for key in [("X", t2, t3, "X"), ("X", t3, t2, "X")]:
        if key in OPLS_DIHEDRALS:
            return OPLS_DIHEDRALS[key]
    return (0.0, 0.0, 0.300, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Single-molecule LAMMPS .lmp writer
# ══════════════════════════════════════════════════════════════════════════════

def write_lmp(
    atoms: list,
    bonds: list,
    types: list[str],
    out_path: Path,
    mol_name: str = "MOL",
) -> None:
    """Write a complete single-molecule LAMMPS data file."""
    angles    = build_angles(bonds)
    dihedrals = build_dihedrals(bonds)
    impropers = build_impropers(atoms, bonds, types)

    # Unique parameter sets
    ubond: dict  = {}
    for a, b in bonds:
        p = lookup_bond(types[a-1], types[b-1])
        ubond.setdefault(p, len(ubond) + 1)

    uangle: dict = {}
    for a1, c, a3 in angles:
        p = lookup_angle(types[a1-1], types[c-1], types[a3-1])
        uangle.setdefault(p, len(uangle) + 1)

    udih: dict = {}
    for a1, a2, a3, a4 in dihedrals:
        p = lookup_dihedral(types[a1-1], types[a2-1], types[a3-1], types[a4-1])
        udih.setdefault(p, len(udih) + 1)

    # Unique atom types
    uniq_at: list[str] = []
    atid: dict[str, int] = {}
    for t in types:
        if t not in atid:
            atid[t] = len(uniq_at) + 1
            uniq_at.append(t)

    # Box (single molecule + 25 Å padding)
    xs = [a[1] for a in atoms]
    ys = [a[2] for a in atoms]
    zs = [a[3] for a in atoms]
    pad = 25.0

    imp_k = 10.5
    n_uimp = 1 if impropers else 0

    L: list[str] = []
    L.append(f"LAMMPS data file for {mol_name}\n")
    L.append(f"  {len(atoms)} atoms")
    L.append(f"  {len(bonds)} bonds")
    L.append(f"  {len(angles)} angles")
    L.append(f"  {len(dihedrals)} dihedrals")
    L.append(f"  {len(impropers)} impropers\n")
    L.append(f"  {len(uniq_at)} atom types")
    L.append(f"  {len(ubond)} bond types")
    L.append(f"  {len(uangle)} angle types")
    L.append(f"  {len(udih)} dihedral types")
    L.append(f"  {n_uimp} improper types\n")
    L.append(f"  {min(xs)-pad:.4f}  {max(xs)+pad:.4f} xlo xhi")
    L.append(f"  {min(ys)-pad:.4f}  {max(ys)+pad:.4f} ylo yhi")
    L.append(f"  {min(zs)-pad:.4f}  {max(zs)+pad:.4f} zlo zhi\n")

    L.append("Masses\n")
    for t in uniq_at:
        info = OPLS_ATOM_TYPES.get(t, (t, 12.011, 0.066, 3.5, ""))
        L.append(f"  {atid[t]:4d}  {info[1]:.3f}  # {t}")

    L.append("\nPair Coeffs\n")
    for t in uniq_at:
        info = OPLS_ATOM_TYPES.get(t, (t, 12.011, 0.066, 3.5, ""))
        L.append(f"  {atid[t]:4d}  {info[2]:.3f}  {info[3]:.7f}  # {t}")

    L.append("\nBond Coeffs\n")
    for p, pid in sorted(ubond.items(), key=lambda x: x[1]):
        L.append(f"  {pid:4d}  {p[0]:.4f}  {p[1]:.4f}")

    L.append("\nAngle Coeffs\n")
    for p, pid in sorted(uangle.items(), key=lambda x: x[1]):
        L.append(f"  {pid:4d}  {p[0]:.3f}  {p[1]:.3f}")

    L.append("\nDihedral Coeffs\n")
    for p, pid in sorted(udih.items(), key=lambda x: x[1]):
        L.append(f"  {pid:4d}  {p[0]:.3f}  {p[1]:.3f}  {p[2]:.3f}  {p[3]:.3f}")

    if n_uimp:
        L.append("\nImproper Coeffs\n")
        L.append(f"     1  {imp_k:.3f}  -1  2")

    L.append("\nAtoms\n")
    for i, ((el, x, y, z), t) in enumerate(zip(atoms, types), 1):
        q = OPLS_CHARGES.get(t, 0.0)
        L.append(f"  {i:5d}  1  {atid[t]:3d}  {q:12.8f}  {x:10.6f}  {y:10.6f}  {z:10.6f}  # {t}")

    L.append("\nBonds\n")
    for i, (a, b) in enumerate(bonds, 1):
        p = lookup_bond(types[a-1], types[b-1])
        L.append(f"  {i:5d}  {ubond[p]:3d}  {a:5d}  {b:5d}")

    L.append("\nAngles\n")
    for i, (a1, c, a3) in enumerate(angles, 1):
        p = lookup_angle(types[a1-1], types[c-1], types[a3-1])
        L.append(f"  {i:5d}  {uangle[p]:3d}  {a1:5d}  {c:5d}  {a3:5d}")

    L.append("\nDihedrals\n")
    for i, (a1, a2, a3, a4) in enumerate(dihedrals, 1):
        p = lookup_dihedral(types[a1-1], types[a2-1], types[a3-1], types[a4-1])
        L.append(f"  {i:5d}  {udih[p]:3d}  {a1:5d}  {a2:5d}  {a3:5d}  {a4:5d}")

    if impropers:
        L.append("\nImpropers\n")
        for i, (c, n1, n2, n3) in enumerate(impropers, 1):
            L.append(f"  {i:5d}  1  {c:5d}  {n1:5d}  {n2:5d}  {n3:5d}")

    Path(out_path).write_text("\n".join(L) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Multi-molecule system builder
# ══════════════════════════════════════════════════════════════════════════════

class MolData:
    """Per-molecule-type topology + FF data."""

    def __init__(
        self,
        name:  str,
        atoms: list,
        bonds: list,
        types: list[str],
        count: int = 1,
    ):
        """Store per-molecule topology and derive angles, dihedrals, and impropers."""
        self.name   = name
        self.atoms  = atoms
        self.bonds  = bonds
        self.types  = types
        self.count  = count
        self.angles    = build_angles(bonds)
        self.dihedrals = build_dihedrals(bonds)
        self.impropers = build_impropers(atoms, bonds, types)

    @classmethod
    def from_file(cls, path: Path, name: Optional[str] = None,
                  count: int = 1) -> "MolData":
        """Parse a molecule file (VASP/PDB/XYZ) and return a MolData instance."""
        atoms, bonds = parse_molecule(path)
        types = assign_atom_types(atoms, bonds)
        return cls(name or Path(path).stem, atoms, bonds, types, count)

    @classmethod
    def from_builtin(cls, mol_name: str, count: int = 1) -> "MolData":
        """Build a MolData instance from the built-in MOLECULES library by name."""
        mol = MOLECULES[mol_name]
        atoms = list(mol["atoms"])
        bonds = list(mol["bonds"])
        types = assign_atom_types(atoms, bonds)
        return cls(mol_name, atoms, bonds, types, count)


def _estimate_box(mols: list[MolData], target_density: float = 0.9) -> float:
    """Estimate cubic box side (Å) from molecular masses and target density g/cm³."""
    _MASSES = {"H": 1.008, "C": 12.011, "O": 15.999, "N": 14.007,
               "S": 32.060, "F": 19.000, "P": 30.974, "Li": 6.941}
    total_mass = 0.0
    for m in mols:
        mol_mass = sum(_MASSES.get(el, 12.0) for el, *_ in m.atoms)
        total_mass += mol_mass * m.count
    # V = mass[amu] * 1.66054e-24 g / density [g/cm³] * 1e24 Å³/cm³
    vol_A3 = (total_mass * 1.66054) / target_density
    return vol_A3 ** (1.0 / 3.0)


def build_mixed_system(
    mols: list[MolData],
    out_path: Path,
    box_size: Optional[float] = None,
    target_density: float = 0.9,
) -> float:
    """Write a multi-component LAMMPS data file (full atom style).

    Molecules are placed on a 3D grid; LAMMPS NPT will relax the density.
    Returns the box side length used (Å).
    """
    explicit_box = box_size is not None
    if box_size is None:
        box_size = _estimate_box(mols, target_density)
    L_box = float(box_size)

    # ── Collect global atom/bond/angle/dihedral/improper types ──────────────
    global_atypes: list[str] = []
    gat_id: dict[str, int] = {}

    def _gat(t: str) -> int:
        """Register atom type t in the global type list and return its 1-based index."""
        if t not in gat_id:
            gat_id[t] = len(global_atypes) + 1
            global_atypes.append(t)
        return gat_id[t]

    global_bparams: dict[tuple, int] = {}
    global_aparams: dict[tuple, int] = {}
    global_dparams: dict[tuple, int] = {}

    def _gbp(p: tuple) -> int:
        """Register bond parameter tuple p and return its global 1-based index."""
        global_bparams.setdefault(p, len(global_bparams) + 1)
        return global_bparams[p]

    def _gap(p: tuple) -> int:
        """Register angle parameter tuple p and return its global 1-based index."""
        global_aparams.setdefault(p, len(global_aparams) + 1)
        return global_aparams[p]

    def _gdp(p: tuple) -> int:
        """Register dihedral parameter tuple p and return its global 1-based index."""
        global_dparams.setdefault(p, len(global_dparams) + 1)
        return global_dparams[p]

    # Pre-scan all molecule types to populate global type tables
    for mol in mols:
        for t in mol.types:
            _gat(t)
        for a, b in mol.bonds:
            _gbp(lookup_bond(mol.types[a-1], mol.types[b-1]))
        for a1, c, a3 in mol.angles:
            _gap(lookup_angle(mol.types[a1-1], mol.types[c-1], mol.types[a3-1]))
        for a1, a2, a3, a4 in mol.dihedrals:
            _gdp(lookup_dihedral(mol.types[a1-1], mol.types[a2-1],
                                 mol.types[a3-1], mol.types[a4-1]))

    # Count totals
    total_mols  = sum(m.count for m in mols)
    total_atoms = sum(len(m.atoms) * m.count for m in mols)
    total_bonds = sum(len(m.bonds) * m.count for m in mols)
    total_ang   = sum(len(m.angles) * m.count for m in mols)
    total_dih   = sum(len(m.dihedrals) * m.count for m in mols)
    total_imp   = sum(len(m.impropers) * m.count for m in mols)

    # Grid for molecule placement.  When the caller specifies an explicit box
    # (PACKMOL-derived density or NPT target), use it directly — LAMMPS CG
    # minimize resolves any initial overlaps.  When no box is given, expand to
    # ensure the grid is non-overlapping so the minimizer starts from a
    # reasonable configuration.
    cbrt = max(1, math.ceil(total_mols ** (1.0 / 3.0)))

    if not explicit_box:
        def _mol_extent(mol: MolData) -> float:
            """Return the maximum bounding-box dimension (Å) of a molecule."""
            if not mol.atoms:
                return 0.0
            xs = [a[1] for a in mol.atoms]
            ys = [a[2] for a in mol.atoms]
            zs = [a[3] for a in mol.atoms]
            return max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))

        max_extent = max((_mol_extent(m) for m in mols), default=5.0)
        min_spacing = max_extent + 3.0
        min_box_no_overlap = cbrt * min_spacing
        L_box = max(L_box, min_box_no_overlap)

    spacing = L_box / cbrt

    def _grid_pos(mol_idx: int) -> tuple:
        """Return the (x, y, z) grid origin for the mol_idx-th molecule in the cubic lattice."""
        iz = mol_idx // (cbrt * cbrt)
        iy = (mol_idx % (cbrt * cbrt)) // cbrt
        ix = mol_idx % cbrt
        return ix * spacing, iy * spacing, iz * spacing

    # ── Build data sections ─────────────────────────────────────────────────
    atom_lines: list[str] = []
    bond_lines: list[str] = []
    ang_lines:  list[str] = []
    dih_lines:  list[str] = []
    imp_lines:  list[str] = []

    atom_idx = 0
    bond_idx = 0
    ang_idx  = 0
    dih_idx  = 0
    imp_idx  = 0
    mol_count = 0

    # Interleave molecule types so the grid is mixed (like PACKMOL random packing),
    # not segregated (all DMB first, then all LiFSI).  Use a fixed seed for
    # reproducibility.
    import random as _random
    _rng = _random.Random(42)
    mol_sequence: list[MolData] = []
    for mol in mols:
        mol_sequence.extend([mol] * mol.count)
    _rng.shuffle(mol_sequence)

    # Pre-compute each molecule's centroid once
    _centroids: dict[int, tuple] = {}
    for mol in mols:
        if id(mol) not in _centroids:
            if mol.atoms:
                cx = sum(a[1] for a in mol.atoms) / len(mol.atoms)
                cy = sum(a[2] for a in mol.atoms) / len(mol.atoms)
                cz = sum(a[3] for a in mol.atoms) / len(mol.atoms)
            else:
                cx = cy = cz = 0.0
            _centroids[id(mol)] = (cx, cy, cz)

    for mol in mol_sequence:
        cx, cy, cz = _centroids[id(mol)]
        ox, oy, oz = _grid_pos(mol_count)
        offset = atom_idx
        mol_id = mol_count + 1

        for i, ((el, x, y, z), t) in enumerate(zip(mol.atoms, mol.types)):
            atom_idx += 1
            q = OPLS_CHARGES.get(t, 0.0)
            gid = gat_id[t]
            # Write true (unwrapped) coordinates so LAMMPS remaps atoms
            # into the box on read_data and assigns correct image flags.
            # This avoids "Inconsistent image flags" from pre-wrapped
            # coords with forced-zero image flags when molecules span PBC.
            tx = ox + (x - cx)
            ty = oy + (y - cy)
            tz = oz + (z - cz)
            atom_lines.append(
                f"  {atom_idx:7d}  {mol_id:6d}  {gid:3d}  {q:10.6f}"
                f"  {tx:12.6f}  {ty:12.6f}  {tz:12.6f}"
                f"     0     0     0  # {t}"
            )

        # Bonds
        for a, b in mol.bonds:
            bond_idx += 1
            p = lookup_bond(mol.types[a-1], mol.types[b-1])
            bond_lines.append(
                f"  {bond_idx:7d}  {global_bparams[p]:4d}"
                f"  {a+offset:7d}  {b+offset:7d}"
            )

        # Angles
        for a1, c, a3 in mol.angles:
            ang_idx += 1
            p = lookup_angle(mol.types[a1-1], mol.types[c-1], mol.types[a3-1])
            ang_lines.append(
                f"  {ang_idx:7d}  {global_aparams[p]:4d}"
                f"  {a1+offset:7d}  {c+offset:7d}  {a3+offset:7d}"
            )

        # Dihedrals
        for a1, a2, a3, a4 in mol.dihedrals:
            dih_idx += 1
            p = lookup_dihedral(mol.types[a1-1], mol.types[a2-1],
                                mol.types[a3-1], mol.types[a4-1])
            dih_lines.append(
                f"  {dih_idx:7d}  {global_dparams[p]:4d}"
                f"  {a1+offset:7d}  {a2+offset:7d}"
                f"  {a3+offset:7d}  {a4+offset:7d}"
            )

        # Impropers
        for c, n1, n2, n3 in mol.impropers:
            imp_idx += 1
            imp_lines.append(
                f"  {imp_idx:7d}  1"
                f"  {c+offset:7d}  {n1+offset:7d}"
                f"  {n2+offset:7d}  {n3+offset:7d}"
            )

        mol_count += 1

    # ── Charge neutralization (uniform correction for any residual imbalance) ──
    _charges = [float(line.split()[3]) for line in atom_lines]
    _net_q = sum(_charges)
    if abs(_net_q) > 0.01:
        _corr = -_net_q / len(_charges)
        import logging as _log
        _log.getLogger("hpca.orch").warning(
            "[forcefield] Net charge %.3f e; applying uniform correction %.6f e/atom",
            _net_q, _corr)
        _new_lines = []
        for _line in atom_lines:
            _parts = _line.split()
            _q_idx = 3
            _q_new = float(_parts[_q_idx]) + _corr
            _parts[_q_idx] = f"{_q_new:10.6f}"
            _new_lines.append("  " + "  ".join(_parts[:3]) + "  " + _parts[3] + "  " +
                              "  ".join(_parts[4:]))
        atom_lines = _new_lines

    # ── Write file ──────────────────────────────────────────────────────────
    H: list[str] = [
        f"LAMMPS data file — mixed system ({total_mols} molecules)\n",
        f"  {total_atoms} atoms",
        f"  {total_bonds} bonds",
        f"  {total_ang} angles",
        f"  {total_dih} dihedrals",
        f"  {total_imp} impropers\n",
        f"  {len(global_atypes)} atom types",
        f"  {len(global_bparams)} bond types",
        f"  {len(global_aparams)} angle types",
        f"  {len(global_dparams)} dihedral types",
        f"  {1 if total_imp else 0} improper types\n",
        f"  0.0  {L_box:.4f} xlo xhi",
        f"  0.0  {L_box:.4f} ylo yhi",
        f"  0.0  {L_box:.4f} zlo zhi\n",
        "Masses\n",
    ]
    for t in global_atypes:
        info = OPLS_ATOM_TYPES.get(t, (t, 12.011, 0.066, 3.5, ""))
        H.append(f"  {gat_id[t]:4d}  {info[1]:.3f}  # {t}")

    H.append("\nPair Coeffs\n")
    for t in global_atypes:
        info = OPLS_ATOM_TYPES.get(t, (t, 12.011, 0.066, 3.5, ""))
        H.append(f"  {gat_id[t]:4d}  {info[2]:.3f}  {info[3]:.7f}  # {t}")

    H.append("\nBond Coeffs\n")
    for p, pid in sorted(global_bparams.items(), key=lambda x: x[1]):
        H.append(f"  {pid:4d}  {p[0]:.4f}  {p[1]:.4f}")

    H.append("\nAngle Coeffs\n")
    for p, pid in sorted(global_aparams.items(), key=lambda x: x[1]):
        H.append(f"  {pid:4d}  {p[0]:.3f}  {p[1]:.3f}")

    H.append("\nDihedral Coeffs\n")
    for p, pid in sorted(global_dparams.items(), key=lambda x: x[1]):
        H.append(f"  {pid:4d}  {p[0]:.3f}  {p[1]:.3f}  {p[2]:.3f}  {p[3]:.3f}")

    if total_imp:
        H.append("\nImproper Coeffs\n")
        H.append("     1  10.500  -1  2")

    H.append("\nAtoms  # full\n")
    H.extend(atom_lines)

    H.append("\nBonds\n")
    H.extend(bond_lines)

    H.append("\nAngles\n")
    H.extend(ang_lines)

    H.append("\nDihedrals\n")
    H.extend(dih_lines)

    if total_imp:
        H.append("\nImpropers\n")
        H.extend(imp_lines)

    Path(out_path).write_text("\n".join(H) + "\n")
    return L_box


def build_system_data_from_poscar(
    poscar_path: Path,
    mol_data: list["MolData"],
    out_path: Path,
) -> float:
    """Write LAMMPS data file from a MACE-preopt POSCAR + OPLS-AA topology.

    The POSCAR must be element-sorted (produced by ASE sort=True or pymatgen
    get_sorted_structure). Within each element block, atoms appear in PACKMOL
    molecule-copy order because pymatgen/ASE use a stable sort.

    mol_data must be in the same order as the PACKMOL structure blocks
    (solvents first in combo order, then salt/extras).  The function pulls
    coordinates from each element's positional block in lock-step with the
    molecule-template atom list, so the two orderings must match exactly.

    Returns the box side length (Å) read from the POSCAR cell.

    Raises ValueError if element counts in POSCAR do not match mol_data totals
    (indicates a PACKMOL-order mismatch between mol_data and the POSCAR).
    """
    from ase.io import read as _ase_read

    atoms_ase = _ase_read(str(poscar_path), format="vasp")
    cell  = atoms_ase.get_cell()
    L_box = float(cell[0, 0])

    # Group POSCAR positions by element, preserving file (= PACKMOL molecule) order
    elem_coords: dict[str, list] = {}
    for atom in atoms_ase:
        elem_coords.setdefault(atom.symbol, [])
        elem_coords[atom.symbol].append(atom.position.tolist())
    elem_cursor: dict[str, int] = {e: 0 for e in elem_coords}

    # ── Global type tables (same as build_mixed_system) ───────────────────────
    global_atypes: list[str] = []
    gat_id: dict[str, int] = {}
    global_bparams: dict[tuple, int] = {}
    global_aparams: dict[tuple, int] = {}
    global_dparams: dict[tuple, int] = {}

    def _gat(t: str) -> int:
        """Register atom type t in the global type list and return its 1-based index."""
        if t not in gat_id:
            gat_id[t] = len(global_atypes) + 1
            global_atypes.append(t)
        return gat_id[t]

    def _gbp(p: tuple) -> int:
        """Register bond parameter tuple p and return its global 1-based index."""
        global_bparams.setdefault(p, len(global_bparams) + 1)
        return global_bparams[p]

    def _gap(p: tuple) -> int:
        """Register angle parameter tuple p and return its global 1-based index."""
        global_aparams.setdefault(p, len(global_aparams) + 1)
        return global_aparams[p]

    def _gdp(p: tuple) -> int:
        """Register dihedral parameter tuple p and return its global 1-based index."""
        global_dparams.setdefault(p, len(global_dparams) + 1)
        return global_dparams[p]

    for mol in mol_data:
        for t in mol.types:
            _gat(t)
        for a, b in mol.bonds:
            _gbp(lookup_bond(mol.types[a - 1], mol.types[b - 1]))
        for a1, c, a3 in mol.angles:
            _gap(lookup_angle(mol.types[a1 - 1], mol.types[c - 1], mol.types[a3 - 1]))
        for a1, a2, a3, a4 in mol.dihedrals:
            _gdp(lookup_dihedral(mol.types[a1 - 1], mol.types[a2 - 1],
                                  mol.types[a3 - 1], mol.types[a4 - 1]))

    total_mols  = sum(m.count for m in mol_data)
    total_atoms = sum(len(m.atoms) * m.count for m in mol_data)
    total_bonds = sum(len(m.bonds) * m.count for m in mol_data)
    total_ang   = sum(len(m.angles) * m.count for m in mol_data)
    total_dih   = sum(len(m.dihedrals) * m.count for m in mol_data)
    total_imp   = sum(len(m.impropers) * m.count for m in mol_data)

    # ── Assign MACE coordinates to molecules in PACKMOL order ─────────────────
    atom_lines: list[str] = []
    bond_lines: list[str] = []
    ang_lines:  list[str] = []
    dih_lines:  list[str] = []
    imp_lines:  list[str] = []

    atom_idx  = 0
    bond_idx  = 0
    ang_idx   = 0
    dih_idx   = 0
    imp_idx   = 0
    mol_count = 0

    for mol in mol_data:
        for _ in range(mol.count):
            offset = atom_idx
            mol_id = mol_count + 1

            for (el, _tx, _ty, _tz), t in zip(mol.atoms, mol.types):
                atom_idx += 1
                cur = elem_cursor.get(el, 0)
                pool = elem_coords.get(el, [])
                if cur >= len(pool):
                    raise ValueError(
                        f"[build_system_data_from_poscar] Element '{el}' exhausted "
                        f"at position {cur} of {len(pool)} — "
                        f"mol_data order does not match PACKMOL block order in POSCAR. "
                        f"Current mol: {mol.name}"
                    )
                x, y, z = pool[cur]
                elem_cursor[el] = cur + 1
                q   = OPLS_CHARGES.get(t, 0.0)
                gid = gat_id[t]
                atom_lines.append(
                    f"  {atom_idx:7d}  {mol_id:6d}  {gid:3d}  {q:10.6f}"
                    f"  {x:12.6f}  {y:12.6f}  {z:12.6f}"
                    f"     0     0     0  # {t}"
                )

            for a, b in mol.bonds:
                bond_idx += 1
                p = lookup_bond(mol.types[a - 1], mol.types[b - 1])
                bond_lines.append(
                    f"  {bond_idx:7d}  {global_bparams[p]:4d}"
                    f"  {a + offset:7d}  {b + offset:7d}"
                )

            for a1, c, a3 in mol.angles:
                ang_idx += 1
                p = lookup_angle(mol.types[a1 - 1], mol.types[c - 1], mol.types[a3 - 1])
                ang_lines.append(
                    f"  {ang_idx:7d}  {global_aparams[p]:4d}"
                    f"  {a1 + offset:7d}  {c + offset:7d}  {a3 + offset:7d}"
                )

            for a1, a2, a3, a4 in mol.dihedrals:
                dih_idx += 1
                p = lookup_dihedral(mol.types[a1 - 1], mol.types[a2 - 1],
                                    mol.types[a3 - 1], mol.types[a4 - 1])
                dih_lines.append(
                    f"  {dih_idx:7d}  {global_dparams[p]:4d}"
                    f"  {a1 + offset:7d}  {a2 + offset:7d}"
                    f"  {a3 + offset:7d}  {a4 + offset:7d}"
                )

            for c, n1, n2, n3 in mol.impropers:
                imp_idx += 1
                imp_lines.append(
                    f"  {imp_idx:7d}  1"
                    f"  {c + offset:7d}  {n1 + offset:7d}"
                    f"  {n2 + offset:7d}  {n3 + offset:7d}"
                )

            mol_count += 1

    # Charge neutralization (same as build_mixed_system)
    _charges = [float(line.split()[3]) for line in atom_lines]
    _net_q = sum(_charges)
    if abs(_net_q) > 0.01:
        _corr = -_net_q / len(_charges)
        import logging as _log
        _log.getLogger("hpca.orch").warning(
            "[forcefield] Net charge %.3f e; applying uniform correction %.6f e/atom",
            _net_q, _corr)
        _new_lines = []
        for _line in atom_lines:
            _parts = _line.split()
            _q_new = float(_parts[3]) + _corr
            _parts[3] = f"{_q_new:10.6f}"
            _new_lines.append("  " + "  ".join(_parts[:3]) + "  " + _parts[3] + "  " +
                              "  ".join(_parts[4:]))
        atom_lines = _new_lines

    # Write LAMMPS data file (identical format to build_mixed_system)
    H: list[str] = [
        f"LAMMPS data file — preopt system ({total_mols} molecules, MACE geometry)\n",
        f"  {total_atoms} atoms",
        f"  {total_bonds} bonds",
        f"  {total_ang} angles",
        f"  {total_dih} dihedrals",
        f"  {total_imp} impropers\n",
        f"  {len(global_atypes)} atom types",
        f"  {len(global_bparams)} bond types",
        f"  {len(global_aparams)} angle types",
        f"  {len(global_dparams)} dihedral types",
        f"  {1 if total_imp else 0} improper types\n",
        f"  0.0  {L_box:.4f} xlo xhi",
        f"  0.0  {L_box:.4f} ylo yhi",
        f"  0.0  {L_box:.4f} zlo zhi\n",
        "Masses\n",
    ]
    for t in global_atypes:
        info = OPLS_ATOM_TYPES.get(t, (t, 12.011, 0.066, 3.5, ""))
        H.append(f"  {gat_id[t]:4d}  {info[1]:.3f}  # {t}")

    H.append("\nPair Coeffs\n")
    for t in global_atypes:
        info = OPLS_ATOM_TYPES.get(t, (t, 12.011, 0.066, 3.5, ""))
        H.append(f"  {gat_id[t]:4d}  {info[2]:.3f}  {info[3]:.7f}  # {t}")

    H.append("\nBond Coeffs\n")
    for p, pid in sorted(global_bparams.items(), key=lambda x: x[1]):
        H.append(f"  {pid:4d}  {p[0]:.4f}  {p[1]:.4f}")

    H.append("\nAngle Coeffs\n")
    for p, pid in sorted(global_aparams.items(), key=lambda x: x[1]):
        H.append(f"  {pid:4d}  {p[0]:.3f}  {p[1]:.3f}")

    H.append("\nDihedral Coeffs\n")
    for p, pid in sorted(global_dparams.items(), key=lambda x: x[1]):
        H.append(f"  {pid:4d}  {p[0]:.3f}  {p[1]:.3f}  {p[2]:.3f}  {p[3]:.3f}")

    if total_imp:
        H.append("\nImproper Coeffs\n")
        H.append("     1  10.500  -1  2")

    H.append("\nAtoms  # full\n")
    H.extend(atom_lines)
    H.append("\nBonds\n")
    H.extend(bond_lines)
    H.append("\nAngles\n")
    H.extend(ang_lines)
    H.append("\nDihedrals\n")
    H.extend(dih_lines)
    if total_imp:
        H.append("\nImpropers\n")
        H.extend(imp_lines)

    Path(out_path).write_text("\n".join(H) + "\n")
    return L_box
