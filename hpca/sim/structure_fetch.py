"""
structure_fetch.py — Auto-download and convert structure files for project components.

Priority for each component:
  Organic solvents  → PubChem 3D SDF (urllib, no extra packages needed)
  Inorganic (salts, SSEs, electrodes) → Materials Project API (mp_api + pymatgen)

Both sources are reachable from Kestrel login nodes.  Compute nodes may not
have outbound internet — run the wizard on the login node (default behaviour).
"""
from __future__ import annotations

import os
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional


# ── Common name → PubChem CID lookup (fast-path, avoids name-search latency) ─

_PUBCHEM_CIDS: dict[str, int] = {
    # 2026-08-21: audited every entry in this table against a live PubChem
    # name search after discovering "DMB" (74117) actually resolves to
    # N-(cyclohexylideneamino)-2,4-dinitroaniline, not 1,2-dimethoxybutane --
    # every Fluorine_free_solvent/DMB_LiFSI_* system (CMD/AIMD/MLMD) had been
    # simulating that wrong molecule. 9 of the 17 original entries were wrong
    # (DMB, DOL, EMC, FEC, TEGDME, TMS, SN, GBL, DEE); corrected below,
    # verified by CID -> IUPACName lookup. TMS's original value carried an
    # "(example)" comment suggesting it was never a real lookup -- kept the
    # trimethylsilane CID but flagged, since "TMS" more plausibly means
    # sulfolane (tetramethylene sulfone, CID 31347) in an electrolyte-solvent
    # context; needs a domain-expert call before trusting either.
    "DMB":    14849271, # 1,2-Dimethoxybutane (was 74117 -- wrong compound, see above)
    "DME":    8071,     # 1,2-Dimethoxyethane -- verified correct
    "DOL":    12586,    # 1,3-Dioxolane (was 12239 -> dibutyl sulfate)
    "EC":     7303,     # Ethylene carbonate -- verified correct
    "DMC":    12021,    # Dimethyl carbonate -- verified correct
    "EMC":    522046,   # Ethyl methyl carbonate (was 522272 -> an azabicyclooctane salt)
    "DEC":    7766,     # Diethyl carbonate -- verified correct
    "PC":     7924,     # Propylene carbonate -- verified correct
    "FEC":    2769656,  # Fluoroethylene carbonate (was 5284534 -> an unrelated macrolide)
    "TEGDME": 8925,     # Tetraglyme (was 2723775 -> an unrelated piperazine)
    "ACN":    6342,     # Acetonitrile -- verified correct
    "TMS":    10026,    # UNVERIFIED -- literally "trimethylsilane" (a gas; a poor
                         # electrolyte solvent) but the intended meaning in battery
                         # literature is very often sulfolane instead (CID 31347).
                         # Confirm which was meant before trusting any "TMS" system.
    "SN":     8062,     # Succinonitrile (was 5667 -> an unrelated alkaloid macrocycle)
    "GBL":    7302,     # gamma-Butyrolactone (was 7326 -> an unrelated terpenoid ester)
    "DMSO":   679,      # Dimethyl sulfoxide -- verified correct
    "THF":    8028,     # Tetrahydrofuran -- verified correct
    "DEE":    3283,     # Diethyl ether (was 3286 -> an unrelated phosphorothioate)
}

# Common name aliases → canonical name used in _PUBCHEM_CIDS
_ALIASES: dict[str, str] = {
    "dme":    "DME",  "dol": "DOL",  "ec": "EC",   "dmc": "DMC",
    "emc":    "EMC",  "pc":  "PC",   "fec":"FEC",  "dmb": "DMB",
    "dec":    "DEC",
    "tegdme": "TEGDME", "acn": "ACN",
}

# Inorganic / crystalline: formula → preferred Materials Project query
# Only list species that are true crystalline solids findable in MP.
# Organic salts (LiFSI, LiTFSI, ...) are handled via SMILES below.
_INORGANIC_FORMULAS: dict[str, str] = {
    "LiPF6":    "LiPF6",
    "LiClO4":   "LiClO4",
    "LiCl":     "LiCl",
    "NaCl":     "NaCl",
    "NaPF6":    "NaPF6",
    "Li2ZrCl6": "Li2ZrCl6",
    "Li3YCl6":  "Li3YCl6",
    "Li6PS5Cl": "Li6PS5Cl",
    "LGPS":     "Li10GeP2S12",
    "LLZO":     "Li7La3Zr2O12",
}

# SMILES for organic salts and polymers (used when PubChem/MP both fail or aren't suitable)
_SMILES: dict[str, str] = {
    # Organic lithium salts
    "LiFSI":  "[Li+].[N-](S(=O)(=O)F)S(=O)(=O)F",
    "LiTFSI": "[Li+].[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    "LiBF4":  "[Li+].[B-](F)(F)(F)F",
    "LiPF6":  "[Li+].[P-](F)(F)(F)(F)(F)F",
    "LiClO4": "[Li+].[Cl-](=O)(=O)(=O)=O",
    # Organic sodium salts
    "NaFSI":  "[Na+].[N-](S(=O)(=O)F)S(=O)(=O)F",
    "NaTFSI": "[Na+].[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    "NaPF6":  "[Na+].[P-](F)(F)(F)(F)(F)F",
    # Additional solvents whose PubChem 3D conformer is unavailable
    "EMC":    "CCOC(=O)OC",                         # ethyl methyl carbonate
    "DEC":    "CCOC(=O)OCC",                        # diethyl carbonate
    "EC":     "O=C1OCCO1",                          # ethylene carbonate
    "FEC":    "O=C1OC[C@@H](F)O1",                  # fluoroethylene carbonate
    "SN":     "N#CCCC#N",                           # succinonitrile
    # Salts without PubChem 3D
    "LiDFOB": "[Li+].[B-]1(OC(=O)C(F)(F)O1)(F)F", # lithium difluoro(oxalato)borate
    # Polymer repeat units
    "PEO":    "COCCO",          # single EO repeat unit
    "PVDF":   "FC(F)CC(F)F",   # single VDF repeat unit
}


# ── SDF → POSCAR converter ────────────────────────────────────────────────────

def _parse_sdf(sdf_text: str) -> tuple[list[str], list[tuple[float,float,float]]]:
    """
    Parse a V2000 SDF file.
    Returns (elements, coords_angstrom).
    Raises ValueError if the SDF cannot be parsed.
    """
    lines = sdf_text.splitlines()

    # Find the counts line (line index 3)
    try:
        counts_line = lines[3]
        natoms = int(counts_line[0:3])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot parse SDF counts line: {exc}") from exc

    elements: list[str] = []
    coords:   list[tuple[float,float,float]] = []

    for i in range(4, 4 + natoms):
        if i >= len(lines):
            raise ValueError("SDF truncated before end of atom block")
        line = lines[i]
        try:
            x    = float(line[0:10])
            y    = float(line[10:20])
            z    = float(line[20:30])
            elem = line[31:34].strip()
            elements.append(elem)
            coords.append((x, y, z))
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Bad atom line {i}: {line!r}") from exc

    return elements, coords


def _fix_cation_geometry(elements: list[str], xyz) -> None:
    """Move cations that landed too close to anion atoms (< 1.5 Å) to a valid position.

    Disconnected SMILES ion pairs (e.g. [Li+].[N-]...) can result in the cation
    being placed at the anion centroid by MMFF / centering, causing < 1 Å contacts.
    This places each cation 2.0 Å from the nearest O, along the direction from the
    anion centroid outward — a valid coordination geometry for DFT/MD starting points.
    """
    import numpy as np

    CATIONS = {"Li", "Na", "K", "Mg", "Ca", "Al", "Zn"}
    MIN_OK   = 1.5   # distances below this trigger a fix
    COORD_R  = 2.0   # target cation-O coordination distance

    n = len(elements)
    for i, elem in enumerate(elements):
        if elem not in CATIONS:
            continue
        anion_idx = [j for j in range(n) if elements[j] not in CATIONS and j != i]
        if not anion_idx:
            continue
        dists = np.linalg.norm(xyz[anion_idx] - xyz[i], axis=1)
        if dists.min() >= MIN_OK:
            continue  # already valid

        O_idx = [j for j in anion_idx if elements[j] == "O"] or anion_idx
        anion_cent = xyz[anion_idx].mean(axis=0)
        O_coords   = xyz[O_idx]
        # Pick the O that the cation is closest to (best coordination site)
        best_O = O_idx[np.linalg.norm(O_coords - xyz[i], axis=1).argmin()]
        direction  = xyz[best_O] - anion_cent
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction /= norm
        xyz[i] = xyz[best_O] + COORD_R * direction


def _write_poscar(elements: list[str], coords: list[tuple[float,float,float]],
                   out_path: Path, name: str = "molecule") -> None:
    """
    Write a VASP POSCAR with the molecule centred in a 20×20×20 Å box.
    """
    import numpy as np

    xyz = np.array(coords, dtype=float)

    # Fix cation positions BEFORE centering (handles disconnected SMILES ion pairs)
    _fix_cation_geometry(elements, xyz)

    # Centre molecule in a large box
    box  = 20.0
    cent = xyz.mean(axis=0)
    xyz  = xyz - cent + box / 2

    # Count elements
    from collections import Counter
    order = []
    for e in elements:
        if e not in order:
            order.append(e)
    counts = [elements.count(e) for e in order]

    # Sort: heavy atoms first (by atomic number), then H
    from pymatgen.core.periodic_table import Element as _Elem
    order_sorted = sorted(order, key=lambda e: (1 if e == "H" else 0, -_Elem(e).Z))
    idx_map = {e: [i for i, el in enumerate(elements) if el == e]
               for e in order_sorted}

    lines = [
        f"{name}\n",
        "1.0\n",
        f"  {box:.4f}  0.0000  0.0000\n",
        f"  0.0000  {box:.4f}  0.0000\n",
        f"  0.0000  0.0000  {box:.4f}\n",
        "  " + "  ".join(order_sorted) + "\n",
        "  " + "  ".join(str(len(idx_map[e])) for e in order_sorted) + "\n",
        "Cartesian\n",
    ]
    for e in order_sorted:
        for i in idx_map[e]:
            x, y, z = xyz[i]
            lines.append(f"  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")

    out_path.write_text("".join(lines))


# ── PubChem fetcher ───────────────────────────────────────────────────────────

def _pubchem_cid(name: str) -> Optional[int]:
    """Return PubChem CID for a molecule name, first via cache then via REST."""
    canon = _ALIASES.get(name.lower(), name.upper())
    if canon in _PUBCHEM_CIDS:
        return _PUBCHEM_CIDS[canon]
    # Fall back to PubChem name search
    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           f"{urllib.parse.quote(name)}/cids/JSON")
    try:
        data = json.loads(urllib.request.urlopen(url, timeout=10).read())
        return data["IdentifierList"]["CID"][0]
    except Exception:
        return None


import urllib.parse

def fetch_from_pubchem(name: str, out_path: Path) -> bool:
    """
    Download 3D structure from PubChem and write as VASP POSCAR.
    Returns True on success.
    """
    print(f"  [fetch] {name} → PubChem 3D ...", end="", flush=True)

    cid = _pubchem_cid(name)
    if cid is None:
        print(f" FAIL (CID not found for '{name}')")
        return False

    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
           f"{cid}/record/SDF?record_type=3d")
    try:
        sdf = urllib.request.urlopen(url, timeout=20).read().decode()
    except Exception as exc:
        print(f" FAIL ({exc})")
        return False

    try:
        elements, coords = _parse_sdf(sdf)
    except ValueError as exc:
        print(f" FAIL (SDF parse: {exc})")
        return False

    try:
        _write_poscar(elements, coords, out_path, name=name)
        print(f" → {out_path.name}  ({len(elements)} atoms)")
        return True
    except Exception as exc:
        print(f" FAIL (POSCAR write: {exc})")
        return False


# ── Materials Project fetcher ─────────────────────────────────────────────────

def fetch_from_mp(name: str, out_path: Path,
                   api_key: str = None) -> bool:
    """
    Download lowest-energy structure from Materials Project and write as POSCAR.
    Returns True on success.
    """
    print(f"  [fetch] {name} → Materials Project ...", end="", flush=True)

    key = api_key or os.environ.get("MP_API_KEY", "")
    if not key:
        print(" FAIL (MP_API_KEY not set)")
        return False

    formula = _INORGANIC_FORMULAS.get(name, name)

    try:
        from mp_api.client import MPRester
        with MPRester(key) as mpr:
            docs = mpr.summary.search(
                formula=formula,
                fields=["structure", "material_id", "energy_above_hull"],
            )
        if not docs:
            print(f" FAIL (no MP entry for '{formula}')")
            return False
        docs.sort(key=lambda d: d.energy_above_hull)
        docs[0].structure.to(fmt="poscar", filename=str(out_path))
        mid = docs[0].material_id
        print(f" → {out_path.name}  ({mid})")
        return True
    except Exception as exc:
        print(f" FAIL ({exc})")
        return False


# ── SMILES-based 3D generator ─────────────────────────────────────────────────

def fetch_from_smiles(name: str, smiles: str, out_path: Path) -> bool:
    """Generate 3D structure from SMILES using RDKit and write as VASP POSCAR."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  [fetch] {name} SMILES invalid")
            return False
        mol = Chem.AddHs(mol)
        ps  = AllChem.ETKDGv3()
        ps.randomSeed = 42
        result = AllChem.EmbedMolecule(mol, ps)
        if result != 0:
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result != 0:
            print(f"  [fetch] {name} SMILES embed failed")
            return False
        AllChem.MMFFOptimizeMolecule(mol)

        conf     = mol.GetConformer()
        elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coords   = [(conf.GetAtomPosition(i).x,
                     conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z)
                    for i in range(mol.GetNumAtoms())]
        _write_poscar(elements, coords, out_path, name=name)
        print(f"  [fetch] {name} → SMILES-generated  ({len(elements)} atoms)")
        return True
    except ImportError:
        print(f"  [fetch] {name}: RDKit not available — cannot generate from SMILES")
        return False
    except Exception as exc:
        print(f"  [fetch] {name} SMILES generation failed: {exc}")
        return False


# ── Role-based dispatcher ─────────────────────────────────────────────────────

def fetch_structure(name: str, role: str, project_dir: Path,
                     api_key: str = None) -> bool:
    """
    Fetch a structure file for a component.

    role: 'solvent'        → PubChem 3D, then SMILES
          'salt'           → PubChem, then MP (for true crystals), then SMILES
          'halide_sse' etc → MP first, then SMILES
    """
    out = project_dir / f"{name}.vasp"
    if out.exists():
        print(f"  [fetch] {name}.vasp already exists — skipping")
        return True

    if role == "solvent":
        ok = fetch_from_pubchem(name, out)
        if not ok and name in _SMILES:
            ok = fetch_from_smiles(name, _SMILES[name], out)
        if not ok:
            print(f"  [fetch] Tip: place {name}.vasp manually in {project_dir}")
        return ok

    elif role == "salt":
        # Organic salts (LiFSI, LiTFSI, ...): try PubChem → SMILES → MP
        ok = fetch_from_pubchem(name, out)
        if not ok and name in _SMILES:
            ok = fetch_from_smiles(name, _SMILES[name], out)
        if not ok and name in _INORGANIC_FORMULAS:
            ok = fetch_from_mp(name, out, api_key=api_key)
        if not ok:
            print(f"  [fetch] Tip: place {name}.vasp manually in {project_dir}")
        return ok

    else:
        # True inorganic (SSE, electrode, coating): MP first, then SMILES fallback
        ok = fetch_from_mp(name, out, api_key=api_key)
        if not ok and name in _SMILES:
            ok = fetch_from_smiles(name, _SMILES[name], out)
        if not ok:
            print(f"  [fetch] Tip: download from Materials Project and save as {name}.vasp")
        return ok


def check_missing(project_dir: Path, structure_files: list[dict]) -> list[dict]:
    """
    Return subset of structure_files whose .vasp files are missing.
    """
    missing = []
    for sf in structure_files:
        p = project_dir / sf["file"]
        if not p.exists():
            missing.append(sf)
    return missing
