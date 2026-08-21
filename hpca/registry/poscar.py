"""Canonical POSCAR source registry.

The canonical implementation lives in hpca.core.poscar_source (find_poscar /
require_poscar).  This module exposes a get_poscar_source() wrapper that maps
the stage name used in older code ("dft_followup" → "aimd", etc.) to the
canonical stage names understood by find_poscar().
"""
from __future__ import annotations

from pathlib import Path

from hpca.core.poscar_source import find_poscar


# Mapping from legacy/convenience stage names to canonical poscar_source stages.
_STAGE_ALIASES: dict[str, str] = {
    "dft_followup":  "aimd",
    "dft_opt":       "dft",
    "vc_relax":      "dft",
    "neb_host":      "neb",
    "classical_md":  "cmd",
    "cmd_followup":  "cmd",
    "mlip_train":    "mlip",
}


def get_poscar_source(stage: str, project_dir: Path | str) -> Path | None:
    """Return the best available structure file for *stage*, or None.

    Accepts both canonical stage names understood by find_poscar()
    ("dft", "aimd", "neb", "cmd", "mlip", "mace_preopt") and the
    legacy aliases defined in _STAGE_ALIASES.
    """
    project_dir = Path(project_dir)
    canonical = _STAGE_ALIASES.get(stage, stage)
    try:
        return find_poscar(project_dir, canonical)
    except ValueError:
        return None
