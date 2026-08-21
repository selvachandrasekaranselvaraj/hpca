"""
hpca/io/lammps.py — Consolidated LAMMPS file I/O.

Replaces scattered parsing in:
  tools/lammps.py, h05_cmd.py (_parse_npt_thermo, _parse_elements_from_data,
  _count_atom_types), h05_lammps.py, h06_analysis.py (_parse_dump_lammps)

No non-stdlib dependencies.

Usage:
    from hpca.io.lammps import (
        read_log_thermo,
        validate_npt,
        elements_from_data,
        natoms_from_data,
        count_dump_frames,
        read_dump_positions,
        write_nvt_start_alias,
    )
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


# ── LAMMPS data file ──────────────────────────────────────────────────────────

def natoms_from_data(data_path: Path | str) -> int | None:
    """Return total atom count from LAMMPS data file header."""
    try:
        for line in Path(data_path).read_text(encoding="utf-8",
                                               errors="ignore").splitlines()[:30]:
            if "atoms" in line and "atom types" not in line:
                parts = line.split()
                if parts and parts[0].isdigit():
                    return int(parts[0])
    except Exception:
        pass
    return None


def atom_types_from_data(data_path: Path | str) -> int | None:
    """Return number of atom types from LAMMPS data file header."""
    try:
        for line in Path(data_path).read_text(encoding="utf-8",
                                               errors="ignore").splitlines()[:30]:
            if "atom types" in line:
                return int(line.split()[0])
    except Exception:
        pass
    return None


# OPLS atom-type label → element symbol (for Masses section comment parsing)
_OPLS_ELEMENT: dict[str, str] = {}


def _load_opls_map() -> dict[str, str]:
    """Load the OPLS atom-type-label → element-symbol mapping, caching the result."""
    global _OPLS_ELEMENT
    if not _OPLS_ELEMENT:
        try:
            from hpca.data import load
            _OPLS_ELEMENT = load("opls_elements")
        except Exception:
            # Inline fallback for the most common types
            _OPLS_ELEMENT = {
                "OS": "O", "CT_O": "C", "CT_M": "C", "CT_C": "C",
                "C_CO": "C", "O_CO": "O", "NI": "N", "SF": "S",
                "OY": "O", "FS": "F", "CF": "C", "FT": "F",
                "SY": "S", "NT": "N", "P_F6": "P", "F_F6": "F",
                "Li": "Li", "Na": "Na",
                "CT_H2": "C", "CT_F2": "C", "CT_F1": "C", "CT_F3": "C",
                "FP": "F", "FP3": "F", "HP": "H",
                "P_N": "P", "N_P": "N", "CT_B": "C", "CQ": "C",
                "HC": "H", "HC_O": "H", "HOS": "H",
            }
    return _OPLS_ELEMENT


def elements_from_data(data_path: Path | str,
                       n_types: int | None = None) -> list[str]:
    """
    Return element symbols in atom-type-ID order from a LAMMPS data file.
    Reads the Masses section comments for OPLS-AA type labels.
    n_types: if given, truncate to this many types.
    """
    opls = _load_opls_map()
    elements: dict[int, str] = {}
    try:
        in_masses = False
        for line in Path(data_path).read_text(encoding="utf-8",
                                               errors="ignore").splitlines():
            stripped = line.strip()
            if stripped == "Masses":
                in_masses = True
                continue
            if in_masses:
                if not stripped:
                    continue
                if not stripped[0].isdigit():
                    break
                parts = stripped.split()
                if len(parts) >= 2:
                    tid = int(parts[0])
                    opls_type = stripped.split("#", 1)[1].strip() if "#" in stripped else ""
                    elements[tid] = opls.get(opls_type, _mass_to_element(float(parts[1])))
    except Exception:
        pass
    result = [elements[k] for k in sorted(elements.keys())]
    return result[:n_types] if n_types else result


def _mass_to_element(mass: float) -> str:
    """Infer element symbol from atomic mass (fallback when no OPLS comment)."""
    table = {
        1.008: "H", 6.941: "Li", 12.011: "C", 14.007: "N",
        15.999: "O", 18.998: "F", 22.990: "Na", 30.974: "P",
        32.06: "S", 35.453: "Cl",
    }
    for m, el in table.items():
        if abs(mass - m) < 0.05:
            return el
    return "X"


# ── LAMMPS log (thermo) ────────────────────────────────────────────────────────

def read_log_thermo(log_path: Path | str,
                    tail_bytes: int = 16384) -> list[dict[str, float]]:
    """
    Parse thermo output from a LAMMPS log file.

    Returns list of dicts keyed by the thermo column headers
    (step, temp, press, density, pe, ke, etotal, vol, …).
    Reads the last *tail_bytes* of the file for speed.
    """
    path = Path(log_path)
    if not path.exists():
        return []

    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - tail_bytes))
            text = fh.read().decode("utf-8", errors="replace")
    except Exception:
        return []

    rows: list[dict[str, float]] = []
    headers: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("step"):
            headers = stripped.lower().split()
            continue
        if headers and stripped and stripped[0].isdigit():
            parts = stripped.split()
            if len(parts) == len(headers):
                try:
                    rows.append({h: float(v) for h, v in zip(headers, parts)})
                except ValueError:
                    pass
    return rows


def last_thermo_row(log_path: Path | str) -> dict[str, float] | None:
    """Return the last thermo row from a LAMMPS log (or None on failure)."""
    rows = read_log_thermo(log_path)
    return rows[-1] if rows else None


# ── NPT validation gate ───────────────────────────────────────────────────────

def validate_npt(log_path: Path | str,
                 T_target: float = 300.0,
                 cfg=None) -> tuple[bool, str]:
    """
    Check NPT equilibration result from LAMMPS log.
    Returns (ok: bool, message: str).

    Checks:
      - density within [density_min, density_max] g/cm³
      - temperature within T_target ± tolerance_K
    """
    if cfg is None:
        from hpca.core.config import Config
        cfg = Config.get()

    val   = cfg.npt_validation()
    d_min = val.get("density_min_g_cm3", 0.5)
    d_max = val.get("density_max_g_cm3", 3.0)
    t_tol = val.get("temperature_tolerance_K", 100)

    row = last_thermo_row(log_path)
    if row is None:
        return False, "Could not parse LAMMPS log thermo"

    density = row.get("density") or row.get("dens")
    temp    = row.get("temp")

    if density is None:
        return False, "No density column in LAMMPS thermo"
    if temp is None:
        return False, "No temp column in LAMMPS thermo"

    if not (d_min <= density <= d_max):
        return False, (f"Density {density:.3f} g/cm³ out of range "
                       f"[{d_min},{d_max}]")
    if abs(temp - T_target) > t_tol:
        return False, (f"Temperature {temp:.1f} K deviates > {t_tol} K "
                       f"from target {T_target} K")

    return True, f"NPT OK: ρ={density:.3f} g/cm³, T={temp:.1f} K"


# ── LAMMPS dump file ───────────────────────────────────────────────────────────

def count_dump_frames(dump_path: Path | str) -> int:
    """Count ITEM: TIMESTEP blocks in a LAMMPS dump file (fast grep)."""
    path = Path(dump_path)
    if not path.exists():
        return 0
    count = 0
    try:
        with open(path, "rb") as fh:
            for line in fh:
                if line.startswith(b"ITEM: TIMESTEP"):
                    count += 1
    except Exception:
        pass
    return count


def read_dump_positions(dump_path: Path | str,
                        element_col: str = "element",
                        frame_indices: list[int] | None = None
                        ) -> list[dict[str, Any]]:
    """
    Parse a LAMMPS dump file.

    Returns list of frames; each frame is:
        {'timestep': int, 'box': [[xlo,xhi],[ylo,yhi],[zlo,zhi]],
         'atoms': [{'id':int,'type':int,'x':float,'y':float,'z':float,
                    'element':str, ...}]}

    frame_indices: if given, only return those frame indices (0-based).
    """
    path = Path(dump_path)
    if not path.exists():
        return []

    frames: list[dict] = []
    frame_set = set(frame_indices) if frame_indices is not None else None
    frame_idx = -1

    current: dict | None = None
    col_map: dict[str, int] = {}
    in_atoms = False

    try:
        with open(path, "rb") as fh:
            for raw_line in fh:
                line = raw_line.decode("utf-8", errors="ignore").rstrip()
                if line.startswith("ITEM: TIMESTEP"):
                    frame_idx += 1
                    if current is not None:
                        if frame_set is None or (frame_idx - 1) in frame_set:
                            frames.append(current)
                    if frame_set is not None and frame_idx > max(frame_set, default=0) + 1:
                        break
                    current = {"timestep": 0, "box": [], "atoms": []}
                    in_atoms = False
                    col_map  = {}
                    continue

                if current is None:
                    continue

                if line.startswith("ITEM: NUMBER"):
                    continue
                if line.startswith("ITEM: BOX"):
                    current["box"] = []
                    continue
                if line.startswith("ITEM: ATOMS"):
                    headers = line.split()[2:]
                    col_map = {h: i for i, h in enumerate(headers)}
                    in_atoms = True
                    continue

                if not in_atoms:
                    if "TIMESTEP" in line or "NUMBER" in line:
                        pass
                    elif current["box"] is not None and len(current["box"]) < 3:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                current["box"].append(
                                    [float(parts[0]), float(parts[1])])
                            except ValueError:
                                pass
                    else:
                        try:
                            current["timestep"] = int(line.strip())
                        except ValueError:
                            pass
                    continue

                parts = line.split()
                if not parts:
                    continue
                try:
                    atom: dict[str, Any] = {}
                    for col, idx in col_map.items():
                        if idx < len(parts):
                            try:
                                atom[col] = float(parts[idx])
                            except ValueError:
                                atom[col] = parts[idx]
                    if "id" in atom:
                        atom["id"] = int(atom["id"])
                    if "type" in atom:
                        atom["type"] = int(atom["type"])
                    current["atoms"].append(atom)
                except Exception:
                    pass

        if current is not None:
            if frame_set is None or frame_idx in frame_set:
                frames.append(current)
    except Exception:
        pass

    return frames


# ── nvt_start.dat canonical naming ────────────────────────────────────────────

NVT_START_NAME = "nvt_start.dat"


def write_nvt_start_alias(npt_dir: Path | str,
                          source_name: str = "dump_npt.lmp") -> Path | None:
    """
    After NPT completes, ensure nvt_start.dat exists in npt_dir.

    If the LAMMPS script wrote 'write_data nvt_start.dat' → already correct.
    If the script wrote a different name, copy/symlink to nvt_start.dat.
    Returns the path to nvt_start.dat, or None if no source found.
    """
    npt = Path(npt_dir)
    dest = npt / NVT_START_NAME
    if dest.exists():
        return dest

    candidates = [source_name, "npt_final.data", "npt_final.lmp",
                  "equilibrated.data", "final.lmp"]
    for name in candidates:
        src = npt / name
        if src.exists():
            import shutil
            shutil.copy2(src, dest)
            return dest
    return None
