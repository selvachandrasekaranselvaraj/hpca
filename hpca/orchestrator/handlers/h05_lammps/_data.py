"""
_data.py — LAMMPS data file preparation and type-map reading utilities.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

log = logging.getLogger("hpca.orch")


def prepare_lammps_data_from(
    src_vasp: Path,
    work_dir: Path,
    yaml_data: dict,
    hpc_site_packages: str,
) -> None:
    """Convert a VASP POSCAR/CONTCAR to data.lammps in work_dir."""
    from hpca.core.paths import dft_opt
    if not src_vasp.exists():
        # Fallback to dft/opt/CONTCAR for backwards compatibility
        src_vasp = dft_opt(work_dir.parent.parent) / "CONTCAR"
    if not src_vasp.exists():
        log.warning("[h05_lammps] No source structure for data.lammps")
        return

    try:
        sys.path.insert(0, hpc_site_packages)
        from pymatgen.core import Structure
        from pymatgen.io.lammps.data import LammpsData
        struct = Structure.from_file(str(src_vasp))
        lammps_data = LammpsData.from_structure(struct, atom_style="atomic")
        lammps_data.write_file(str(work_dir / "data.lammps"))
        log.info("[h05_lammps] Converted %s → data.lammps", src_vasp.name)
    except Exception as exc:
        log.warning("[h05_lammps] pymatgen conversion failed (%s) — ASE fallback", exc)
        try:
            from ase.io import read, write
            atoms = read(str(src_vasp))
            write(str(work_dir / "data.lammps"), atoms, format="lammps-data")
        except Exception as exc2:
            log.error("[h05_lammps] Both conversion methods failed: %s", exc2)


def read_type_map_for_project(project_dir: Path) -> list[str]:
    """Return element list for LAMMPS type map, from type_map.raw or POSCAR."""
    from hpca.core.paths import mlmd_mlff, dft_opt
    type_map_file = mlmd_mlff(project_dir) / "00.data" / "type_map.raw"
    if type_map_file.exists():
        return [l.strip() for l in type_map_file.read_text().splitlines() if l.strip()]
    poscar = dft_opt(project_dir) / "CONTCAR"
    if not poscar.exists():
        poscar = dft_opt(project_dir) / "POSCAR"
    if poscar.exists():
        lines = poscar.read_text().splitlines()
        if len(lines) >= 6:
            return lines[5].split()
    return ["Li"]
