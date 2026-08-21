"""Canonical VASP INCAR template registry.

Templates are defined in platform.yaml under incar_templates:.
All numbers live in platform.yaml — this file contains no hardcoded values.

Usage:
    from hpca.registry.incar import get_incar, build_incar, write_incar

    # Raw template dict:
    d = get_incar("opt")

    # Fully patched for a specific project:
    d = build_incar("opt", poscar_path=p / "dft/opt/POSCAR",
                    natoms=120, project_yaml=yaml_dict)

    # Write to disk:
    write_incar(p / "dft/opt/INCAR", d)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from hpca.core.paths import load_platform_config as _lpc

# ── Load templates from platform.yaml ────────────────────────────────────────
# All INCAR parameter values live in platform.yaml incar_templates:.
# ENCUT and NCORE are placeholders overridden by build_incar() at call time.
# NSW is overridden by the caller (from limits.slurm.* or project.yaml).
# TEBEG/TEEND are overridden per temperature.

def _load_registry() -> dict[str, dict[str, Any]]:
    """Load INCAR templates from platform.yaml and return them as a dict of dicts."""
    templates = _lpc().get("incar_templates", {})
    return {k: dict(v) for k, v in templates.items()}

_REGISTRY: dict[str, dict[str, Any]] = _load_registry()


def get_incar(key: str, overrides: dict | None = None) -> dict:
    """Return a copy of the INCAR template dict, optionally with overrides applied."""
    if key not in _REGISTRY:
        raise KeyError(f"incar_registry: unknown key '{key}'. "
                       f"Valid: {sorted(_REGISTRY)}")
    d = dict(_REGISTRY[key])
    if overrides:
        d.update(overrides)
    return d


def build_incar(
    key: str,
    *,
    natoms: int = 0,
    nsw: int | None = None,
    poscar_path: Path | None = None,
    encut: float | None = None,
    project_yaml: dict | None = None,
    tebeg: int | None = None,
    teend: int | None = None,
    extra: dict | None = None,
) -> dict:
    """
    Return fully-patched INCAR dict ready to write.

    Patches applied (in order):
      1. Base template from registry
      2. NCORE from Config.ncore(natoms) (skip for neb_images which uses NPAR)
      3. ENCUT from encut arg, or poscar_path POTCAR scan, or keep template value
      4. NSW from nsw arg or project_yaml simulation override
      5. TEBEG/TEEND from tebeg/teend args
      6. LREAL="F" for natoms < 50 (avoids ghost forces on small cells)
      7. extra overrides dict
    """
    from hpca.core.config import Config
    cfg = Config.get()

    d = get_incar(key)

    # NCORE from atom-count rule.
    # Skip if the template or caller extra uses NPAR (NEB images, active-learning).
    uses_npar = "NPAR" in d or "NPAR" in (extra or {})
    if uses_npar:
        d.pop("NCORE", None)
    elif natoms > 0:
        d["NCORE"] = cfg.ncore(natoms)["ncore"]

    # ENCUT
    if encut is not None:
        d["ENCUT"] = encut
    elif poscar_path is not None:
        ec = _encut_from_potcar(Path(poscar_path).parent, cfg)
        if ec:
            d["ENCUT"] = ec

    # NSW override
    if nsw is not None:
        d["NSW"] = nsw
    elif project_yaml:
        sim = project_yaml.get("simulation", {})
        nsw_key = f"{key}_nsw"
        if nsw_key in sim:
            d["NSW"] = int(sim[nsw_key])

    # TEBEG / TEEND for AIMD/NPT
    if tebeg is not None:
        d["TEBEG"] = tebeg
    if teend is not None:
        d["TEEND"] = teend if teend is not None else tebeg

    # Small cell: force LREAL=F
    if natoms > 0 and natoms < 50 and "LREAL" in d:
        d["LREAL"] = "F"

    # Liquids, polymers, and molecular mixtures have no meaningful crystal
    # symmetry.  ISYM=0 still makes VASP classify the direct and reciprocal
    # Bravais lattices and can abort distorted boxes as a SICK_JOB; ISYM=-1
    # disables symmetry analysis completely.
    if project_yaml:
        from hpca.core.categories import is_molecular
        if is_molecular(project_yaml.get("category", "")):
            d["ISYM"] = -1
            d.pop("SYMPREC", None)

    if extra:
        d.update(extra)

    return d


def write_incar(path: Path | str, d: dict) -> None:
    """Write INCAR dict to file (key = value, one per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f" {k} = {v}" for k, v in d.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _encut_from_potcar(calc_dir: Path, cfg) -> float | None:
    """Return ENCUT = factor × max(ENMAX) from POTCAR in calc_dir."""
    potcar = calc_dir / "POTCAR"
    if not potcar.exists():
        return None
    factor = cfg.perf("encut_factor", 1.3)
    try:
        enmax_values: list[float] = []
        for line in potcar.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "ENMAX" in line:
                parts = line.split("=")
                if len(parts) >= 2:
                    val_str = parts[1].split(";")[0].split()[0]
                    enmax_values.append(float(val_str))
        if enmax_values:
            return round(max(enmax_values) * factor, 2)
    except Exception:
        pass
    return None
