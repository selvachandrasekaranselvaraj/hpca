"""
poscar_source.py — Canonical structure-file lookup chain for all simulation stages.

Use find_poscar(project_dir, stage) to get the best available starting structure
rather than each handler implementing its own ad-hoc search.
"""
from __future__ import annotations

from pathlib import Path

from hpca.core.paths import (
    contcar_preopt, designed_structures, dft_opt, dft_vc, cmd_npt,
)


def _first_existing(*paths: Path) -> Path | None:
    """Return the first path in *paths that exists on disk, or None."""
    for p in paths:
        if p.exists():
            return p
    return None


def _any_aimd_contcar(project_dir: Path) -> Path | None:
    """Scan aimd/<T>/ subdirectories and return the first CONTCAR found."""
    aimd_base = project_dir / "aimd"
    if not aimd_base.is_dir():
        return None
    for subdir in sorted(aimd_base.iterdir()):
        contcar = subdir / "CONTCAR"
        if contcar.exists():
            return contcar
    return None


def find_poscar(project_dir: Path, stage: str) -> Path | None:
    """Return the best available structure file for *stage*, or None.

    Priority chains (highest priority first):

    stage="dft"
        1. dft/preopt/CONTCAR                      (preoptimization output)
        2. designed_structures/poscar_dft.vasp
        3. dft/opt/POSCAR
        4. dft/vc/POSCAR

    stage="aimd"
        1. dft/opt/CONTCAR
        2. dft/vc/CONTCAR
        3. designed_structures/poscar_aimd.vasp

    stage="neb"
        1. neb/preopt/atomopt/CONTCAR              (doped preopt chain)
        2. dft/opt/CONTCAR
        3. dft/vc/CONTCAR

    stage="cmd"
        1. preopt/contcar_cmd_preopt.vasp
        2. designed_structures/system_cmd.data     (LAMMPS data format)
        3. designed_structures/poscar_cmd.vasp

    stage="mlip"
        1. dft/opt/CONTCAR
        2. aimd/<T>/CONTCAR                         (first available temperature)

    stage="mace_preopt"
        1. designed_structures/poscar_dft.vasp
        2. designed_structures/poscar_cmd.vasp
    """
    if stage == "dft":
        return _first_existing(
            contcar_preopt(project_dir, "dft"),
            designed_structures(project_dir) / "poscar_dft.vasp",
            dft_opt(project_dir) / "POSCAR",
            dft_vc(project_dir) / "POSCAR",
        )

    if stage == "aimd":
        return _first_existing(
            dft_opt(project_dir) / "CONTCAR",
            dft_vc(project_dir) / "CONTCAR",
            designed_structures(project_dir) / "poscar_aimd.vasp",
        )

    if stage == "neb":
        return _first_existing(
            project_dir / "neb" / "preopt" / "atomopt" / "CONTCAR",
            dft_opt(project_dir) / "CONTCAR",
            dft_vc(project_dir) / "CONTCAR",
        )

    if stage == "cmd":
        return _first_existing(
            contcar_preopt(project_dir, "cmd"),
            designed_structures(project_dir) / "system_cmd.data",
            designed_structures(project_dir) / "poscar_cmd.vasp",
        )

    if stage == "mlip":
        # dft_aimd requires a temperature; scan for any available CONTCAR
        p = _first_existing(
            dft_opt(project_dir) / "CONTCAR",
        )
        if p is not None:
            return p
        return _any_aimd_contcar(project_dir)

    if stage == "mace_preopt":
        return _first_existing(
            designed_structures(project_dir) / "poscar_dft.vasp",
            designed_structures(project_dir) / "poscar_cmd.vasp",
        )

    raise ValueError(f"Unknown stage {stage!r}. "
                     "Valid stages: 'dft', 'aimd', 'neb', 'cmd', 'mlip', 'mace_preopt'.")


def require_poscar(project_dir: Path, stage: str) -> Path:
    """Like find_poscar but raises FileNotFoundError if no structure is found."""
    result = find_poscar(project_dir, stage)
    if result is None:
        raise FileNotFoundError(
            f"No POSCAR source for stage={stage!r} in {project_dir}"
        )
    return result


def find_lammps_data(project_dir: Path) -> Path | None:
    """Return the best available LAMMPS data file, or None.

    Priority:
        1. designed_structures/system_cmd.data
        2. cmd/npt/system_npt_final.data
    """
    return _first_existing(
        designed_structures(project_dir) / "system_cmd.data",
        cmd_npt(project_dir) / "system_npt_final.data",
    )
