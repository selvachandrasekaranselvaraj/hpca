"""
type_map.py — Unified type_map.raw reading utilities for DeepMD and LAMMPS.

Replaces three separate implementations in h04_mlip, h05_lammps, h06_analysis.
Cross-ref: hpca/core/paths.py, hpca/orchestrator/handlers/h04_mlip.py
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("hpca.core")


def read_type_map(path: Path) -> list[str]:
    """Read element list from type_map.raw. Returns [] if file missing or empty."""
    try:
        if not path.exists():
            return []
        lines = path.read_text().splitlines()
        return [l.strip() for l in lines if l.strip()]
    except Exception:
        return []


def find_type_map(project_dir: Path) -> list[str]:
    """Search standard locations for type_map.raw and return element list.

    Search order:
      1. mlmd/mlff/00.data/type_map.raw
      2. mlmd/mlff/dataset_data/type_map.raw
      3. dft/opt/type_map.raw
    Falls back to ["Li"] if none found.
    """
    from hpca.core.paths import mlmd_mlff, dft_opt
    candidates = [
        mlmd_mlff(project_dir) / "00.data" / "type_map.raw",
        mlmd_mlff(project_dir) / "dataset_data" / "type_map.raw",
        dft_opt(project_dir) / "type_map.raw",
    ]
    for c in candidates:
        tm = read_type_map(c)
        if tm:
            return tm
    return ["Li"]


def mobile_type_id(elements: list[str], mobile_element: str) -> int | None:
    """Return 1-based LAMMPS type ID for mobile_element in elements list.

    Returns None if mobile_element is not in the list.
    """
    try:
        return elements.index(mobile_element) + 1
    except ValueError:
        return None


def lammps_data_type_id(system_data: Path, mobile_ion: str) -> int | None:
    """Read LAMMPS data file Masses section and return type ID for mobile_ion.

    Handles common OPLS aliases (Na+ → Na, Li+ → Li, etc.).
    Returns None if not found or file missing.
    """
    _OPLS_ALIASES: dict[str, str] = {
        "Li+": "Li", "Na+": "Na", "K+": "K", "Mg2+": "Mg",
        "Ca2+": "Ca", "Zn2+": "Zn", "Al3+": "Al",
        "F-": "F", "Cl-": "Cl", "Br-": "Br", "I-": "I",
        "O2-": "O", "S2-": "S",
    }
    if not system_data.exists():
        return None
    try:
        in_masses = False
        for line in system_data.read_text().splitlines():
            stripped = line.strip()
            if stripped == "Masses":
                in_masses = True
                continue
            if in_masses:
                if not stripped:
                    continue
                if stripped and not stripped[0].isdigit():
                    break
                parts = stripped.split()
                if len(parts) < 2:
                    continue
                type_id = int(parts[0])
                label = parts[-1].lstrip("#").strip() if len(parts) > 2 else ""
                resolved = _OPLS_ALIASES.get(label, label)
                if resolved == mobile_ion:
                    return type_id
    except Exception as exc:
        log.debug("[type_map] lammps_data_type_id failed: %s", exc)
    return None
