"""
h05_lammps — MLMD handler: NPT equilibration → NVT production (SLURM gpu-h100).

Workflow:
  1. mlmd/npt/     — NPT at 300 K, up to 100 ps (SLURM)
                     Source: preopt/contcar_mlmd_preopt.vasp
  2. mlmd/nvt/{T}/ — NVT at each temperature for up to 1 ns (SLURM)
                     Source: mlmd/npt/ final frame

Temperatures (per-category from platform.yaml limits):
  SSE:       300, 320, 340, 360, 380, 400, 500, 600, 650, 700 K
  Molecular: 300, 320, 340, 360, 380, 400, 450, 500 K

Atom limits (from platform.yaml limits):
  slurm: mlmd_atoms ≤ 6000

Cross-ref:
  hpca/core/paths.py              — mlmd_npt(), mlmd_nvt(), contcar_preopt(), mlmd_mlff()
  hpca/config/platform.yaml       — limits and hpc paths
  hpca/orchestrator/handlers/h00_design.py — writes preopt/contcar_mlmd_preopt.vasp
  hpca/orchestrator/handlers/h04_mlip.py  — writes mlmd/mlff/pot_com.pb or MACE_model.pt
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from ..base import SimulationHandler
from hpca.core.categories import is_sse as _cat_is_sse, is_crystalline as _cat_is_crystalline
from hpca.core.lammps_job import (
    mass_block as _lj_mass_block,
    pair_style_block as _lj_pair_style,
    write_nvt_input as _lj_write_nvt,
    write_npt_input as _lj_write_npt,
    dump_valid as _lj_dump_valid,
)
from hpca.registry.submission import write_submission as _write_sub

from ._benchmark import benchmark_lammps_ntasks
from ._data import prepare_lammps_data_from, read_type_map_for_project

if TYPE_CHECKING:
    from hpca.orchestrator.state_tracker import ProjectState

log = logging.getLogger("hpca.orch")

# Layout: see hpca/core/paths.py
# Cross-ref: hpca/config/platform.yaml — HPC paths and simulation limits
from hpca.core.paths import dft_opt, mlmd_mlff, mlmd_nvt, mlmd_npt, contcar_preopt

# GPU/CPU partition and deepmd venv read from platform.yaml at call time via cls.platform_config()
# Cross-ref: hpca/config/platform.yaml hpc.deepmd_cpu_venv, hpc.partitions.gpu, hpc.accounts.*
DUMP_MIN_SIZE_PLACEHOLDER = None  # placeholder so old references resolve; remove when refactoring
DUMP_MIN_SIZE = 1_000_000  # 1 MB minimum for a valid dump file
NPT_DONE_FLAG = "NPT_COMPLETE"  # written to mlmd/npt/ when equilibration finishes

# _ATOMIC_MASS and _mass_block moved to hpca.core.lammps_job / hpca.data.atomic_masses



class LAMMPSHandler(SimulationHandler):
    """SLURM handler: MLMD NPT equilibration → NVT production at canonical temperatures.

    Prerequisite: mlmd/mlff/pot_com.pb or MACE_model.pt (from h04_mlip)
                  preopt/contcar_mlmd_preopt.vasp (from h00_design)
    Produces:     mlmd/npt/  — NPT equilibrated structure + dump
                  mlmd/nvt/{T}/  — NVT dump at each temperature
    """

    name = "h05_lammps"
    is_daemon = False

    def _nvt_temps(self, yaml: dict) -> list[int]:
        """Return NVT temperature list appropriate for the project category."""
        from hpca.core.categories import is_molecular as _is_mol
        cat = yaml.get("category", "")
        if _is_mol(cat):
            return self.platform_config().get("limits", {}).get(
                "nvt_temperatures_mol", [300, 320, 340, 360, 380, 400, 450, 500])
        return self.platform_config().get("limits", {}).get(
            "nvt_temperatures_sse", [300, 320, 340, 360, 380, 400, 500, 600, 650, 700])

    @staticmethod
    def _npt_density_ok(npt_dir: Path, yaml: dict,
                        lo: float = 0.5, hi: float = 3.0) -> bool:
        """Return True if NPT final density is within [lo, hi] g/cm³ (or log file absent)."""
        log_file = npt_dir / "log.lammps"
        if not log_file.exists():
            return True
        try:
            from hpca.io.lammps import read_log_thermo as _rlt
            thermo = _rlt(log_file)
            if not thermo:
                return True
            last = thermo[-1]
            density = last.get("density", last.get("c_rho", None))
            if density is None:
                return True
            ok = lo <= density <= hi
            log.info("[h05_lammps] NPT density = %.3f g/cm³ (%s)", density, "OK" if ok else "FAIL")
            return ok
        except Exception as exc:
            log.debug("[h05_lammps] density check failed: %s", exc)
            return True

    def _active_backends(self, project_dir: Path) -> list[str]:
        """Return list of backends for which a trained model exists."""
        mlff_dir = mlmd_mlff(project_dir)
        backends = []
        if (mlff_dir / "pot_com.pb").exists():
            backends.append("deepmd")
        if (mlff_dir / "MACE_model.pt").exists():
            backends.append("mace")
        return backends

    def _use_backend_prefix(self, project_dir: Path) -> bool:
        """True for dual-backend projects (SSE or mlip_backend=both) — use mlmd/{backend}/ dirs."""
        yaml_data = self.read_project_yaml(project_dir)
        be  = yaml_data.get("mlip_backend", "")
        cat = yaml_data.get("category", "")
        return be == "both" or _cat_is_sse(cat)

    def can_run(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True when a trained potential and a valid MLMD preopt CONTCAR are both present."""
        deepmd_ready = (mlmd_mlff(project_dir) / "pot_com.pb").exists()
        mace_ready   = (mlmd_mlff(project_dir) / "MACE_model.pt").exists()
        preopt_vasp  = contcar_preopt(project_dir, "mlmd")
        if not self._poscar_is_valid(preopt_vasp):
            log.warning("[h05_lammps] contcar_mlmd_preopt.vasp missing — waiting for h00_design")
            return False
        return deepmd_ready or mace_ready

    def is_complete(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True when NPT_COMPLETE flag and all NVT dump files exist for every active backend."""
        backends = self._active_backends(project_dir)
        if not backends:
            return False
        yaml = self.read_project_yaml(project_dir)
        use_prefix = self._use_backend_prefix(project_dir)
        temps = self._nvt_temps(yaml)
        for be in backends:
            be_pfx = be if use_prefix else None
            if not (mlmd_npt(project_dir, be_pfx) / NPT_DONE_FLAG).exists():
                return False
            for T in temps:
                dump = mlmd_nvt(project_dir, T, be_pfx) / "dump_unwrapped.lmp"
                try:
                    if not dump.exists() or dump.stat().st_size < DUMP_MIN_SIZE:
                        return False
                except OSError:
                    return False
        return True

    def submit(self, project_dir: Path, state: "ProjectState") -> str | None:
        """Submit NPT equilibration then NVT production jobs for each active backend and temperature."""
        yaml = self.read_project_yaml(project_dir)

        # LAMMPS force-field parameters from platform.yaml
        timestep_ps   = self.plat("lammps_md", "timestep_fs_mlmd",    0.5) * 0.001
        nvt_damp      = self.plat("lammps_md", "mlmd_nvt_temp_damp",  0.05)
        npt_temp_damp = self.plat("lammps_md", "mlmd_npt_temp_damp",  0.1)
        npt_pres_damp = self.plat("lammps_md", "mlmd_npt_press_damp", 1.0)

        # Simulation parameters — prefer project.yaml, fall back to platform limits
        # Cross-ref: hpca/config/platform.yaml limits section
        nvt_ns    = yaml.get("mlmd_nvt_ns",    self.sim_limit("slurm", "mlmd_nvt_ns", 1))
        npt_ps    = yaml.get("mlmd_npt_ps",    self.sim_limit("slurm", "mlmd_npt_ps", 100))
        n_steps   = int(nvt_ns * 1e6)
        npt_steps = int(npt_ps * 1000)

        npt_dump_every = self.plat("lammps_md", "mlmd_npt_dump_freq", 500)
        nvt_dump_every = self.plat("lammps_md", "mlmd_nvt_dump_freq", 500)
        min_dump_bytes = DUMP_MIN_SIZE

        active_backends = self._active_backends(project_dir)
        if not active_backends:
            log.error("[h05_lammps] No trained potential found")
            return None

        use_prefix = self._use_backend_prefix(project_dir)
        mlff_dir   = mlmd_mlff(project_dir)
        handler_state = state.get_handler("h05_lammps")
        submitted_jobs: dict = handler_state.get("jobs", {})
        first_job_id: str | None = None
        project_name = yaml.get("name", project_dir.name)
        type_map = self._read_type_map(project_dir)
        natoms = int(yaml.get("tiers", {}).get("mlmd", {}).get("natoms", 1000))
        temps = self._nvt_temps(yaml)

        from hpca.core.categories import is_molecular as _is_mol

        for be in active_backends:
            be_pfx    = be if use_prefix else None
            pot_path  = mlff_dir / ("pot_com.pb" if be == "deepmd" else "MACE_model.pt")
            pot_type  = be
            npt_key   = f"{be}:npt" if use_prefix else "npt"

            # ── NPT equilibration ──────────────────────────────────────────────
            npt_dir  = mlmd_npt(project_dir, be_pfx)
            npt_done = (npt_dir / NPT_DONE_FLAG).exists()
            npt_job  = submitted_jobs.get(npt_key)

            if not npt_done:
                npt_dump = npt_dir / "dump_npt.lmp"
                if npt_dump.exists() and npt_dump.stat().st_size > min_dump_bytes:
                    (npt_dir / NPT_DONE_FLAG).touch()
                    log.info("[h05_lammps] [%s] NPT complete", be)
                    npt_done = True
                elif not (npt_job and self.job_alive(npt_job)):
                    npt_dir.mkdir(parents=True, exist_ok=True)
                    npt_preopt = contcar_preopt(project_dir, "mlmd")
                    if not (npt_dir / "data.lammps").exists():
                        self._prepare_lammps_data_from(npt_preopt, npt_dir, yaml)
                    npt_temp = self.platform_config().get("limits", {}).get("npt_temperature", 300)
                    self._write_npt_input(
                        npt_dir / "in.lammps",
                        T=npt_temp, pot_path=str(pot_path), pot_type=pot_type,
                        n_steps=npt_steps, dump_every=npt_dump_every, elements=type_map,
                        timestep_ps=timestep_ps, npt_temp_damp=npt_temp_damp,
                        npt_pres_damp=npt_pres_damp,
                    )
                    self._write_sub_sh(npt_dir / "sub.sh",
                                       job_name=f"{project_name}_{be}_npt", pot_path=pot_path)
                    job_id = self.sbatch(npt_dir / "sub.sh", cwd=npt_dir)
                    if job_id:
                        submitted_jobs[npt_key] = job_id
                        first_job_id = first_job_id or job_id
                        log.info("[h05_lammps] [%s] Submitted NPT job=%s", be, job_id)
                    continue  # wait for this backend's NPT before its NVT
                else:
                    log.debug("[h05_lammps] [%s] NPT job %s still running", be, npt_job)

            # ── NPT density gate for molecular projects ────────────────────────
            if _is_mol(yaml.get("category", "")) and npt_done:
                if not self._npt_density_ok(npt_dir, yaml):
                    log.warning(
                        "[h05_lammps] [%s] NPT density outside 0.5–3 g/cm³ — resubmitting NPT", be
                    )
                    (npt_dir / NPT_DONE_FLAG).unlink(missing_ok=True)
                    npt_done = False
                    npt_dump = npt_dir / "dump_npt.lmp"
                    npt_dump.unlink(missing_ok=True)

            if not npt_done:
                continue

            # ── NVT: benchmark core count for deepmd ───────────────────────────
            ncores = 104
            if pot_type == "deepmd":
                bench_data = npt_dir / "data.lammps"
                if not bench_data.exists():
                    bench_data = contcar_preopt(project_dir, "mlmd")
                if bench_data.exists():
                    ncores = self._benchmark_lammps_ntasks(pot_path, bench_data)

            # ── NVT production at each temperature ────────────────────────────
            for T in temps:
                t_key = f"{be}:{T}" if use_prefix else str(T)
                existing_job = submitted_jobs.get(t_key)
                if existing_job and self.job_alive(existing_job):
                    log.debug("[h05_lammps] [%s] T=%s running job=%s", be, T, existing_job)
                    continue

                nvt_dir = mlmd_nvt(project_dir, T, be_pfx)
                dump = nvt_dir / "dump_unwrapped.lmp"
                if dump.exists() and dump.stat().st_size >= DUMP_MIN_SIZE:
                    log.info("[h05_lammps] [%s] T=%s already done", be, T)
                    continue

                nvt_dir.mkdir(parents=True, exist_ok=True)
                npt_final = npt_dir / "nvt_start.dat"
                if not (nvt_dir / "data.lammps").exists():
                    if npt_final.exists():
                        shutil.copy(npt_final, nvt_dir / "data.lammps")
                    else:
                        self._prepare_lammps_data_from(
                            contcar_preopt(project_dir, "mlmd"), nvt_dir, yaml)

                nvt_seed = hash(project_name + str(T)) % 900000 + 100000
                self._write_lammps_input(
                    nvt_dir / "in.lammps", T=T, pot_path=str(pot_path),
                    pot_type=pot_type, n_steps=n_steps, dump_every=nvt_dump_every,
                    elements=type_map, timestep_ps=timestep_ps, nvt_damp=nvt_damp,
                    seed=nvt_seed,
                )

                sub_sh = nvt_dir / "sub.sh"
                if pot_type == "deepmd":
                    self._write_cpu_nvt_sub_sh(sub_sh,
                                               job_name=f"{project_name}_{be}_{T}K",
                                               pot_path=pot_path, ncores=ncores)
                else:
                    self._write_sub_sh(sub_sh, job_name=f"{project_name}_{be}_{T}K",
                                       time=self.slurm_time("mlmd_nvt"), pot_path=pot_path)

                job_id = self.sbatch(sub_sh, cwd=nvt_dir)
                if job_id:
                    submitted_jobs[t_key] = job_id
                    first_job_id = first_job_id or job_id
                    log.info("[h05_lammps] [%s] Submitted NVT T=%s K, job=%s", be, T, job_id)

        state.set_stage("h05_lammps", "RUNNING", jobs=submitted_jobs)
        return first_job_id

    def check_progress(self, project_dir: Path, state: "ProjectState") -> None:
        """Log dump file sizes and clear dead NVT jobs so submit() can requeue them."""
        yaml = self.read_project_yaml(project_dir)
        temps = self._nvt_temps(yaml)
        handler_state = state.get_handler("h05_lammps")
        submitted_jobs = handler_state.get("jobs", {})
        dump_sizes: dict = {}

        npt_dump = mlmd_npt(project_dir) / "dump_npt.lmp"
        dump_sizes["npt"] = npt_dump.stat().st_size if npt_dump.exists() else 0

        for T in temps:
            dump = mlmd_nvt(project_dir, T) / "dump_unwrapped.lmp"
            size = dump.stat().st_size if dump.exists() else 0
            dump_sizes[str(T)] = size

            job_id = submitted_jobs.get(str(T))
            if job_id and not self.job_alive(job_id) and size < DUMP_MIN_SIZE:
                log.warning("[h05_lammps] T=%s job %s dead, dump=%d bytes — cleared for resubmit",
                            T, job_id, size)
                # Remove from submitted_jobs so submit() resubmits on next poll
                del submitted_jobs[str(T)]

        log.info("[h05_lammps] Sizes: %s", {k: f"{v/1e6:.1f}MB" for k, v in dump_sizes.items()})
        state.set_stage("h05_lammps", "RUNNING", jobs=submitted_jobs,
                        dump_sizes=dump_sizes)

    # ── File writers ────────────────────────────────────────────────────────────

    @classmethod
    def _write_lammps_input(
        cls, path: Path,
        T: int,
        pot_path: str,
        pot_type: str,
        n_steps: int = 1_000_000,
        dump_every: int = 1000,
        elements: list | None = None,
        timestep_ps: float = 0.001,
        nvt_damp: float = 0.05,
        seed: int = 12345,
    ) -> None:
        """Write NVT LAMMPS input script via lammps_job module."""
        _lj_write_nvt(path, T=T, pot_path=pot_path, pot_type=pot_type,
                      n_steps=n_steps, dump_every=dump_every,
                      elements=elements or ["Li"],
                      timestep_ps=timestep_ps, nvt_damp=nvt_damp, seed=seed)

    @classmethod
    def _write_npt_input(
        cls, path: Path, T: int, pot_path: str, pot_type: str,
        n_steps: int, dump_every: int, elements: list | None = None,
        timestep_ps: float = 0.001,
        npt_temp_damp: float = 0.1,
        npt_pres_damp: float = 1.0,
    ) -> None:
        """Write NPT equilibration LAMMPS input. Saves restart for NVT seeding."""
        _lj_write_npt(path, T=T, pot_path=pot_path, pot_type=pot_type,
                      n_steps=n_steps, dump_every=dump_every,
                      elements=elements or ["Li"],
                      timestep_ps=timestep_ps,
                      npt_temp_damp=npt_temp_damp, npt_pres_damp=npt_pres_damp)

    @classmethod
    def _benchmark_lammps_ntasks(cls, pot_path: Path, data_path: Path) -> int:
        """Run 10-step DeepMD LAMMPS benchmarks on daemon (104→32 cores, step -8).

        Result cached in pot_path.parent/_ncores_cached.
        Falls back to 104 if all benchmarks fail.
        """
        return benchmark_lammps_ntasks(
            pot_path, data_path, cls.platform_config().get("hpc", {})
        )

    @classmethod
    def _write_sub_sh(cls, path: Path, job_name: str, time: str = "",
                      pot_path: "Path | None" = None) -> None:
        """Write GPU LAMMPS SLURM submission script for NPT or MACE NVT jobs."""
        wall = time or cls.slurm_time("mlmd_npt")
        _write_sub(path, "lammps_gpu", job_name, time=wall, pot_path=pot_path)

    @staticmethod
    def _pair_style_block(pot_path: str, pot_type: str) -> str:
        """Return the LAMMPS pair_style + pair_coeff block string for the given potential."""
        return _lj_pair_style(pot_path, pot_type)

    def _prepare_lammps_data_from(
        self, src_vasp: Path, work_dir: Path, yaml_data: dict
    ) -> None:
        """Convert a VASP POSCAR/CONTCAR to data.lammps in work_dir."""
        prepare_lammps_data_from(
            src_vasp, work_dir, yaml_data,
            self.hpc_path("cladue_site_packages", ""),
        )

    def _prepare_lammps_data(
        self, project_dir: Path, dlmd_dir: Path, yaml_data: dict
    ) -> None:
        """Legacy: convert dft/opt/CONTCAR to LAMMPS data format."""
        contcar = dft_opt(project_dir) / "CONTCAR"
        poscar = dft_opt(project_dir) / "POSCAR"
        src = contcar if contcar.exists() else poscar

        if not src.exists():
            log.warning("[h05_lammps] No CONTCAR/POSCAR found for LAMMPS data conversion")
            return

        try:
            _cladue_site = self.hpc_path("cladue_site_packages", "")
            sys.path.insert(0, _cladue_site)
            from pymatgen.core import Structure
            from pymatgen.io.lammps.data import LammpsData

            struct = Structure.from_file(str(src))
            lammps_data = LammpsData.from_structure(struct, atom_style="atomic")
            lammps_data.write_file(str(dlmd_dir / "data.lammps"))
            log.info("[h05_lammps] Converted %s → data.lammps", src.name)
        except Exception as exc:
            log.warning("[h05_lammps] pymatgen LAMMPS conversion failed (%s) — using ASE fallback", exc)
            try:
                from ase.io import read, write
                atoms = read(str(src))
                write(str(dlmd_dir / "data.lammps"), atoms, format="lammps-data")
                log.info("[h05_lammps] ASE wrote data.lammps")
            except Exception as exc2:
                log.error("[h05_lammps] Both conversion methods failed: %s", exc2)

    @staticmethod
    def _read_type_map(project_dir: Path) -> list[str]:
        """Return the element type-map list for this project (e.g. ['Li', 'C', 'O'])."""
        return read_type_map_for_project(project_dir)

    @staticmethod
    def _read_type_map_z(project_dir: Path) -> dict[int, int]:
        """Return {lammps_type: atomic_number} and {atomic_number: lammps_type} merged."""
        from ase.data import atomic_numbers
        elements = LAMMPSHandler._read_type_map(project_dir)
        # Z_of_type for read_lammps_data: {lammps_type(1-based): atomic_number}
        return {(i + 1): atomic_numbers[el] for i, el in enumerate(elements)}

    @classmethod
    def _write_ase_md_script(
        cls,
        path: Path,
        T: int,
        pot_path: str,
        n_steps: int,
        dump_every: int,
        type_map_z: dict,
    ) -> None:
        """Write a standalone Python ASE+DeepMD NVT MD script that outputs a LAMMPS dump."""
        hpc            = cls.platform_config().get("hpc", {})
        dp_venv        = hpc.get("deepmd_cpu_venv", "")
        cladue_site    = hpc.get("cladue_site_packages", "")
        # Derive site-packages path from the dp venv python
        import glob as _glob
        dp_site_dirs   = _glob.glob(f"{dp_venv}/lib/python3.*/site-packages")
        dp_site        = dp_site_dirs[0] if dp_site_dirs else f"{dp_venv}/lib/python3.11/site-packages"
        z_of_type_str = repr(type_map_z)  # {lammps_type: Z}
        z_to_type_str = repr({v: k for k, v in type_map_z.items()})  # {Z: lammps_type}
        script = f"""\"\"\"NVT MD via ASE + DeepMD calculator; writes LAMMPS dump.\"\"\"
import sys
sys.path.insert(0, "{dp_site}")
sys.path.insert(0, "{cladue_site}")
from pathlib import Path
import numpy as np
from hpca.core.config import account_fallback as _account_fallback
from ase.io.lammpsdata import read_lammps_data
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from deepmd.calculator import DP

work_dir = Path(__file__).parent
atoms = read_lammps_data(str(work_dir / "data.lammps"), Z_of_type={z_of_type_str}, style="atomic")
print(f"Read {{len(atoms)}} atoms: {{atoms.get_chemical_formula()}}")
atoms.calc = DP(model="{pot_path}")
MaxwellBoltzmannDistribution(atoms, temperature_K={T})
friction = 0.05 / (1000 * units.fs)
dyn = Langevin(atoms, 1.0 * units.fs, temperature_K={T}, friction=friction)

z_to_type = {z_to_type_str}  # {{atomic_number: lammps_type}}
dump_file = work_dir / "dump_unwrapped.lmp"
def write_frame(atoms, step, fh):
    cell = atoms.cell.array
    pos = atoms.get_positions()
    fh.write(f"ITEM: TIMESTEP\\n{{step}}\\n")
    fh.write(f"ITEM: NUMBER OF ATOMS\\n{{len(atoms)}}\\n")
    fh.write(f"ITEM: BOX BOUNDS pp pp pp\\n")
    fh.write(f"0.0 {{cell[0,0]:.6f}}\\n0.0 {{cell[1,1]:.6f}}\\n0.0 {{cell[2,2]:.6f}}\\n")
    fh.write("ITEM: ATOMS id type xu yu zu\\n")
    for i, (p, z) in enumerate(zip(pos, atoms.get_atomic_numbers())):
        fh.write(f"{{i+1}} {{z_to_type.get(z,1)}} {{p[0]:.6f}} {{p[1]:.6f}} {{p[2]:.6f}}\\n")

with open(dump_file, "w") as fh:
    for s in range(0, {n_steps} + 1, {dump_every}):
        if s > 0:
            dyn.run({dump_every})
        write_frame(atoms, s, fh)
        e = atoms.get_potential_energy()
        print(f"Step {{s:7d}}: T={{atoms.get_temperature():.1f}}K PE={{e:.4f}} eV")
print(f"Done. Dump: {{dump_file}}")
"""
        path.write_text(script)

    @classmethod
    def _write_cpu_nvt_sub_sh(cls, path: Path, job_name: str, pot_path: Path,
                              ncores: int = 104) -> None:
        """Write SLURM sub.sh for CPU LAMMPS NVT using deepmd-lammps-cpu_2023.
        ncores is determined by _benchmark_lammps_ntasks() (cached in pot_path.parent)."""
        _write_sub(path, "lammps_cpu", job_name,
                   ncores=ncores, time=cls.slurm_time("mlmd_nvt"))

    @classmethod
    def _write_ase_sub_sh(cls, path: Path, job_name: str) -> None:
        """Write a single-CPU SLURM script that runs the ASE DeepMD NVT Python script."""
        hpc          = cls.platform_config().get("hpc", {})
        cpu_dp_venv  = hpc.get("deepmd_cpu_venv", "")
        account      = hpc.get("accounts", {}).get("standard") or _account_fallback()
        work_dir     = str(path.parent)
        script = (
            "#!/bin/bash\n"
            f"#SBATCH --account={account}\n"
            "#SBATCH --nodes=1\n"
            "#SBATCH --ntasks-per-node=1\n"
            "#SBATCH --cpus-per-task=16\n"
            "#SBATCH --time=12:00:00\n"
            f"#SBATCH --job-name={job_name}\n"
            "#SBATCH --mem=64G\n"
            f"#SBATCH --error={work_dir}/%J.stderr\n"
            f"#SBATCH --output={work_dir}/%J.stdout\n"
            "module purge\n"
            f"source {cpu_dp_venv}/bin/activate\n"
            "export DP_DISABLE_CUDA=1\n"
            "export TF_ENABLE_ONEDNN_OPTS=0\n"
            "export OMP_NUM_THREADS=16\n"
            "export DP_INTRA_OP_PARALLELISM_THREADS=8\n"
            "export DP_INTER_OP_PARALLELISM_THREADS=4\n"
            f"cd {work_dir}\n"
            f"{cpu_dp_venv}/bin/python3 run_ase_md.py\n"
        )
        path.write_text(script)
        path.chmod(0o755)
