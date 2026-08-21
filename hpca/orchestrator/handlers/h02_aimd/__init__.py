"""
h02_aimd — AIMD handler: multi-temperature NVT/NPT VASP AIMD (SLURM).

Handles three material geometries after the canonical DFT relaxation sequence:
  crystal   — one supercell → aimd/{T}/ per temperature
  liquid    — dft/opt/CONTCAR → short dataset trajectories
  polymer   — crystal-like cell, optional NPT equilibration

Canonical relaxation is owned exclusively by h01_dft for every material type:
  [doped solid only: dft/aimd_relax] → dft/vc (ISIF=3)
  → dft/opt (ISIF=2) → h02 AIMD dataset generation
"""
from __future__ import annotations

import csv
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..base import SimulationHandler
if TYPE_CHECKING:
    from hpca.orchestrator.state_tracker import ProjectState

log = logging.getLogger("hpca.orch")

# Layout: see hpca/core/paths.py
from hpca.core.paths import dft_opt, dft_aimd, mlmd_mlff
from hpca.core.structure_check import check_and_fix_poscar, check_and_fix_poscar_potcar
from hpca.registry.incar import build_incar as _build_incar
from hpca.core.categories import is_molecular as _cat_is_molecular, is_sse as _cat_is_sse

from ._submit_writers import incar_key_for_npt as _incar_key_for_npt
from ._poscar_utils import (
    read_poscar_lines as _read_poscar_lines,
    get_poscar_elements as _get_poscar_elements,
    count_atoms_poscar as _count_atoms_poscar,
    parse_temperature as _parse_temperature,
    make_deformed_poscar as _make_deformed_poscar,
    make_random_poscar as _make_random_poscar,
    make_rattled_poscar as _make_rattled_poscar,
)

from hpca.core.vasp_job import write_incar as _write_incar, generate_potcar as _gen_potcar
from hpca.registry.submission import write_submission as _write_sub
from hpca.core.kpoints import write_kpoints as _write_kp, kpoints_from_poscar as _kp_from_poscar

try:
    from ..h02_aimd_constants import (
        MAX_LIQUID_SUBMIT,
        SMALL_CELL_THRESHOLD,
        _PARTIAL_THRESHOLD_STEPS,
        _VASP_NODES_SMALL,
        _VASP_NODES_MEDIUM,
        _VASP_NODES_LARGE,
        _DEFORM_SCALES,
        _RANDOM_SCALES,
        _DATASET_TEMPS,
        _RATTLE_SIGMA,
    )
except ImportError:
    SMALL_CELL_THRESHOLD = 100
    _PARTIAL_THRESHOLD_STEPS = 3500
    MAX_LIQUID_SUBMIT = 2000
    _VASP_NODES_SMALL  = (1, 52)
    _VASP_NODES_MEDIUM = (1, 104)
    _VASP_NODES_LARGE  = (1, 104)
    _DEFORM_SCALES     = [0.90, 0.95, 1.00, 1.05, 1.10]
    _RANDOM_SCALES     = [0.95, 1.00, 1.05]
    _DATASET_TEMPS     = [300, 500]
    _RATTLE_SIGMA      = 0.08


class AIMDHandler(SimulationHandler):
    """SLURM handler: submits per-temperature NVT VASP AIMD jobs."""

    name = "h02_aimd"
    is_daemon = False

    # =========================================================================
    # ABC interface
    # =========================================================================

    def can_run(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True when the h01_dft.opt CONTCAR (or a prior AIMD POSCAR) is available to start AIMD."""
        yaml = self.read_project_yaml(project_dir)
        category = yaml.get("category", "inorganic_sse")
        # All systems: AIMD uses h01_dft.opt CONTCAR as starting structure.
        opt_contcar = dft_opt(project_dir) / "CONTCAR"
        if opt_contcar.exists() and opt_contcar.stat().st_size > 50:
            return True
        if _cat_is_molecular(category):
            # Restart: per-temperature POSCARs already set up by a prior submit()
            aimd_dirs = yaml.get("aimd_dirs", [])
            if any((project_dir / d / "POSCAR").exists() for d in aimd_dirs):
                return True
            log.warning("[h02_aimd] waiting for h01_dft.opt CONTCAR")
            return False
        # Crystal fallback: opt disabled in project.yaml
        stages = yaml.get("stages", {})
        dft_cfg = stages.get("dft", True)
        opt_enabled = (dft_cfg is True) or (
            isinstance(dft_cfg, dict) and dft_cfg.get("opt", True)
        )
        if not opt_enabled:
            return (dft_opt(project_dir) / "POSCAR").exists()
        log.warning("[h02_aimd] waiting for h01_dft.opt CONTCAR")
        return False

    def is_complete(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True when all per-temperature XDATCAR files reach the target step count and the DeepMD dataset exists."""
        yaml = self.read_project_yaml(project_dir)
        category = yaml.get("category", "inorganic_sse")
        if _cat_is_molecular(category):
            return self._is_complete_liquid(project_dir, yaml)
        sim = yaml.get("simulation", {})
        temps  = sim.get("aimd_temps") or yaml.get("aimd_temps", [300, 600])
        target = self.resolve(project_dir, "aimd_steps", 3000)
        for T in temps:
            xdatcar = dft_aimd(project_dir, T) / "XDATCAR"
            if not xdatcar.exists():
                return False
            if self.grep_count(xdatcar, "Direct configuration=") < target:
                return False
        # dataset_data/ matches the path h04_mlip.can_run() checks (same as liquid)
        dataset_ready = (
            mlmd_mlff(project_dir) / "dataset_data" / "set.000" / "energy.npy"
        ).exists()
        return bool(temps) and dataset_ready

    def submit(self, project_dir: Path, state: "ProjectState") -> str | None:
        """Dispatch to the liquid or crystal AIMD submission pipeline based on project category."""
        yaml = self.read_project_yaml(project_dir)
        category = yaml.get("category", "inorganic_sse")
        if _cat_is_molecular(category):
            return self._submit_liquid(project_dir, yaml, state)
        return self._submit_crystal(project_dir, yaml, state)

    def check_progress(self, project_dir: Path, state: "ProjectState") -> None:
        """Update handler state with per-temperature step counts and restart dead trajectories."""
        yaml = self.read_project_yaml(project_dir)
        category = yaml.get("category", "inorganic_sse")
        if _cat_is_molecular(category):
            self._check_progress_liquid(project_dir, yaml, state)
        else:
            self._check_progress_crystal(project_dir, yaml, state)

    def on_complete(self, project_dir: Path, state: "ProjectState") -> None:
        """Write the DeepMD training dataset from finished AIMD trajectories."""
        yaml = self.read_project_yaml(project_dir)
        category = yaml.get("category", "inorganic_sse")
        if _cat_is_molecular(category):
            self._prepare_dataset_liquid(project_dir, yaml)
        else:
            self._prepare_dataset_crystal(project_dir, yaml)

    def auto_fix(self, project_dir: Path, state: "ProjectState") -> bool:
        """Resubmit dead AIMD jobs; returns True if any corrective action was taken."""
        yaml = self.read_project_yaml(project_dir)
        category = yaml.get("category", "inorganic_sse")
        if _cat_is_molecular(category):
            return self._auto_fix_liquid(project_dir, yaml, state)
        handler_state = state.get_handler("h02_aimd")
        submitted_jobs: dict = handler_state.get("jobs", {})
        dead_any = any(
            jid and not self.job_alive(jid)
            for jid in submitted_jobs.values()
        )
        if dead_any:
            self._submit_crystal(project_dir, yaml, state)
            return True
        return False

    # =========================================================================
    # POSCAR utilities (thin wrappers over _poscar_utils standalone functions)
    # =========================================================================

    @staticmethod
    def _read_poscar_lines(poscar: Path) -> list[str]:
        """Read and return all lines of a POSCAR file."""
        return _read_poscar_lines(poscar)

    def _get_poscar_elements(self, poscar: Path) -> list[str]:
        """Return the list of element symbols from a POSCAR header."""
        return _get_poscar_elements(poscar)

    def _count_atoms_poscar(self, poscar: Path, default: int = 200) -> int:
        """Return the total atom count from a POSCAR, falling back to default on parse failure."""
        return _count_atoms_poscar(poscar, default)

    @staticmethod
    def _parse_temperature(aimd_dir: Path, default: int = 300) -> int:
        """Extract the temperature (K) from an AIMD directory name, returning default if not found."""
        return _parse_temperature(aimd_dir, default)

    # =========================================================================
    # Dataset-box POSCAR generation
    # =========================================================================

    @staticmethod
    def _make_deformed_poscar(source: Path, out: Path, scale: float) -> None:
        """Write POSCAR with cell scaled uniformly by `scale`; fractional coords preserved."""
        _make_deformed_poscar(source, out, scale)

    @staticmethod
    def _make_random_poscar(source: Path, out: Path, scale: float, rng_seed: int = 42) -> None:
        """Scale cell by `scale` and randomize all atom fractional coordinates."""
        _make_random_poscar(source, out, scale, rng_seed=rng_seed)

    @staticmethod
    def _make_rattled_poscar(
        source: Path, out: Path, scale: float, sigma: float = _RATTLE_SIGMA, rng_seed: int = 0
    ) -> None:
        """Scale cell by `scale` and displace each atom by Gaussian noise (sigma Å, Cartesian).

        Unlike _make_random_poscar, lattice topology is preserved — atoms stay near
        equilibrium. Used for crystal/SSE dataset generation where long-range order matters.
        """
        _make_rattled_poscar(source, out, scale, sigma=sigma, rng_seed=rng_seed)

    @staticmethod
    def _dataset_box_specs(
        composition_dir: Path, rand_kind: str = "random"
    ) -> list[tuple[str, Path, float, int, str]]:
        """Return (name, box_dir, scale, T, kind) for all 16 dataset boxes. No side effects.

        rand_kind: "random" for liquids (fully randomised coords),
                   "rattle" for crystals (Gaussian noise around equilibrium).
        """
        specs: list[tuple[str, Path, float, int, str]] = []
        for scale in _DEFORM_SCALES:
            si = int(round(scale * 100))
            for T in _DATASET_TEMPS:
                name = f"deform_{si:03d}_{T}K"
                specs.append((name, composition_dir / "dataset" / name, scale, T, "deform"))
        for scale in _RANDOM_SCALES:
            si = int(round(scale * 100))
            for T in _DATASET_TEMPS:
                name = f"{rand_kind}_{si:03d}_{T}K"
                specs.append((name, composition_dir / "dataset" / name, scale, T, rand_kind))
        return specs

    def _dataset_box_done(self, box_dir: Path, n_steps: int) -> bool:
        """Return True if XDATCAR in box_dir contains at least n_steps ionic frames."""
        xdatcar = box_dir / "XDATCAR"
        return (
            xdatcar.exists()
            and self.grep_count(xdatcar, "Direct configuration=") >= n_steps
        )

    def _dataset_box_attempted(self, box_dir: Path) -> bool:
        """Return True if VASP was already run in box_dir (success or failure)."""
        return (box_dir / "OUTCAR").exists() or (box_dir / "out").exists()

    @staticmethod
    def _count_xdatcar_frames(box_dir: Path) -> int:
        """Count ionic frames written to XDATCAR in box_dir."""
        xdatcar = box_dir / "XDATCAR"
        if not xdatcar.exists():
            return 0
        try:
            return sum(1 for ln in xdatcar.read_text(errors="ignore").splitlines()
                       if ln.startswith("Direct configuration="))
        except Exception:
            return 0

    def _ensure_box_poscar(
        self,
        box_dir: Path,
        optimized_poscar: Path,
        scale: float,
        kind: str,
        rng_seed: int = 0,
    ) -> Path:
        """Generate deformed/random POSCAR directly inside box_dir.

        Writes to box_dir/POSCAR so no intermediate files appear at the
        composition level. Returns box_dir/POSCAR.
        """
        si   = int(round(scale * 100))
        dest = box_dir / "POSCAR"
        box_dir.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.stat().st_size < 50:
            if kind == "deform":
                self._make_deformed_poscar(optimized_poscar, dest, scale)
            elif kind == "rattle":
                self._make_rattled_poscar(optimized_poscar, dest, scale, rng_seed=si + rng_seed)
            else:
                self._make_random_poscar(optimized_poscar, dest, scale, rng_seed=si + rng_seed)
        return dest

    def _setup_dataset_box(
        self,
        box_dir: Path,
        poscar: Path,
        shared_potcar: Path,
        T: int,
        n_steps: int,
        encut: float,
        n_atoms: int,
    ) -> None:
        """Write VASP inputs for one dataset AIMD box (ISIF=2 NVT)."""
        box_dir.mkdir(parents=True, exist_ok=True)
        if not (box_dir / "POSCAR").exists() or (box_dir / "POSCAR").stat().st_size < 50:
            shutil.copy2(poscar, box_dir / "POSCAR")
        check_and_fix_poscar_potcar(box_dir / "POSCAR", shared_potcar)
        shutil.copy2(shared_potcar, box_dir / "POTCAR")
        incar = _build_incar(
            "nvt_dataset",
            natoms=n_atoms,
            encut=encut,
            nsw=n_steps,
            tebeg=T,
            teend=T,
        )
        _write_incar(box_dir / "INCAR", incar, system_name=f"ds_{box_dir.name}"[:48])
        self._write_kpoints(box_dir / "KPOINTS")

    # =========================================================================
    # POTCAR / INCAR / SLURM utilities
    # =========================================================================

    @staticmethod
    def _generate_potcar(poscar: Path, potcar: Path) -> None:
        """Generate POTCAR from POSCAR element list; logs an error on failure without raising."""
        try:
            _gen_potcar(poscar, potcar)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            log.error("[h02_aimd] POTCAR generation failed: %s", exc)


    @staticmethod
    def _write_kpoints(path: Path, gamma_only: bool = True, poscar: Path | None = None) -> None:
        """Write a KPOINTS file; uses Gamma-only (1×1×1) by default, or auto-mesh from POSCAR."""
        if gamma_only:
            _write_kp(path, (1, 1, 1), gamma=True)
        else:
            mesh = _kp_from_poscar(poscar) if poscar is not None and poscar.exists() else (1, 1, 1)
            _write_kp(path, mesh, gamma=True)

    def _write_sub_sh(
        self,
        path: Path,
        job_name: str,
        n_atoms: int,
        account: str = "",
        time: str = "48:00:00",
    ) -> None:
        """Write VASP SLURM submission script scaled to atom count."""
        _write_sub(path, "vasp_aimd", job_name, natoms=n_atoms, time=time)

    def _write_batch_sub_sh(
        self,
        path: Path,
        box_dirs: "list[Path]",
        n_tasks_per_job: int,
        job_name: str,
        time: str = "48:00:00",
    ) -> None:
        """Write a SLURM script that runs multiple VASP AIMD boxes in parallel on one node.

        Each box gets n_tasks_per_job MPI ranks via a background srun; all run simultaneously.
        Total node allocation = len(box_dirs) * n_tasks_per_job.
        """
        _write_sub(path, "vasp_batch", job_name, box_dirs=box_dirs,
                   ntasks_per_job=n_tasks_per_job, time=time)

    # =========================================================================
    # Dataset-box execution (16 short AIMD runs for MLIP training diversity)
    # =========================================================================

    def _submit_dataset_boxes_slurm(
        self,
        composition_dir: Path,
        optimized_poscar: Path,
        shared_potcar: Path,
        yaml: dict,
        submitted_jobs: dict,
        rand_kind: str = "random",
        project_dir: Path = None,
    ) -> list[str]:
        """Submit all dataset boxes as SLURM jobs; return list of new job IDs.

        rand_kind: "random" for liquid/polymer, "rattle" for crystal/SSE.
        If ncore_opt chose the parallel 6×16 config, boxes are batched into groups
        of n_parallel and submitted as a single node-filling SLURM job each.
        Otherwise each box gets its own job (original behaviour).
        """
        from hpca.orchestrator.handlers.h01_dft import DFTHandler as _DFT
        if project_dir is None:
            project_dir = composition_dir.parent.parent
        n_steps  = int(self.resolve(project_dir, "aimd_dataset_steps", 3000))
        encut    = self.project_encut(project_dir, yaml)
        n_atoms  = self._count_atoms_poscar(optimized_poscar)
        new_jids: list[str] = []
        comp_tag = composition_dir.name

        n_parallel       = _DFT._get_optimal_nparallel(project_dir)
        n_tasks_per_job  = _DFT._get_optimal_ntasks(project_dir, default=96)

        if n_parallel <= 1:
            # Original: one SLURM job per dataset box
            for name, box_dir, scale, T, kind in self._dataset_box_specs(composition_dir, rand_kind):
                if self._dataset_box_done(box_dir, n_steps):
                    continue
                job_key = f"ds_{comp_tag}_{name}"
                if self.job_alive(submitted_jobs.get(job_key)):
                    continue
                poscar = self._ensure_box_poscar(box_dir, optimized_poscar, scale, kind)
                self._setup_dataset_box(box_dir, poscar, shared_potcar, T, n_steps, encut, n_atoms)
                self._write_sub_sh(box_dir / "sub.sh", job_name=f"ds_{name}"[:48], n_atoms=n_atoms)
                jid = self.sbatch(box_dir / "sub.sh", cwd=box_dir)
                if jid:
                    submitted_jobs[job_key] = jid
                    new_jids.append(jid)
                    log.info("[h02_aimd] dataset box %s → job %s", name, jid)
            return new_jids

        # Parallel mode: pack n_parallel boxes onto one node (e.g. 6 × 16-core)
        pending: list[tuple[str, "Path", str]] = []  # (name, box_dir, job_key)
        for name, box_dir, scale, T, kind in self._dataset_box_specs(composition_dir, rand_kind):
            if self._dataset_box_done(box_dir, n_steps):
                continue
            job_key = f"ds_{comp_tag}_{name}"
            if self.job_alive(submitted_jobs.get(job_key)):
                continue
            poscar = self._ensure_box_poscar(box_dir, optimized_poscar, scale, kind)
            self._setup_dataset_box(box_dir, poscar, shared_potcar, T, n_steps, encut, n_atoms)
            pending.append((name, box_dir, job_key))

        for batch_i, start in enumerate(range(0, len(pending), n_parallel)):
            batch = pending[start : start + n_parallel]
            batch_script = composition_dir / f"batch_{batch_i}_sub.sh"
            batch_name   = f"ds_{comp_tag}_b{batch_i}"
            self._write_batch_sub_sh(
                batch_script,
                box_dirs=[bd for _, bd, _ in batch],
                n_tasks_per_job=n_tasks_per_job,
                job_name=batch_name[:48],
                time=self.slurm_time("aimd_dataset"),
            )
            jid = self.sbatch(batch_script, cwd=composition_dir)
            if jid:
                for name, _, job_key in batch:
                    submitted_jobs[job_key] = jid
                new_jids.append(jid)
                log.info("[h02_aimd] batch %s (%d boxes, %dx%d cores) → job %s",
                         batch_name, len(batch), len(batch), n_tasks_per_job, jid)
        return new_jids

    # =========================================================================
    # Crystal workflow
    # =========================================================================

    def _submit_crystal(
        self,
        project_dir: Path,
        yaml: dict,
        state: "ProjectState",
    ) -> str | None:
        """Submit per-temperature NVT AIMD for crystal/SSE materials."""
        sim    = yaml.get("simulation", {})
        temps  = sim.get("aimd_temps") or yaml.get("aimd_temps", [300, 400, 500, 600, 700, 800])
        target = self.resolve(project_dir, "aimd_steps", 3000)

        contcar  = dft_opt(project_dir) / "CONTCAR"
        poscar_src = contcar if contcar.exists() else dft_opt(project_dir) / "POSCAR"
        if not poscar_src.exists():
            log.error("[h02_aimd] No source POSCAR for crystal AIMD in %s", project_dir)
            return None

        potcar_src = dft_opt(project_dir) / "POTCAR"
        if not potcar_src.exists():
            potcar_src = project_dir / "POTCAR"
        if not potcar_src.exists():
            self._generate_potcar(poscar_src, potcar_src)

        handler_state = state.get_handler("h02_aimd")
        submitted_jobs: dict = handler_state.get("jobs", {})
        first_job_id: str | None = None

        n_atoms = self._count_atoms_poscar(poscar_src)
        encut   = self.project_encut(project_dir, yaml)
        category = yaml.get("category", "inorganic_sse")

        # ── NPT Step 0: cell equilibration before per-T NVT ──────────────────
        npt0_dir    = dft_aimd(project_dir, "NPT")
        npt0_key    = "npt_step0"
        npt0_outcar = npt0_dir / "OUTCAR"
        npt0_done   = (
            npt0_outcar.exists()
            and "Total CPU time used" in npt0_outcar.read_text(errors="ignore")
            and (npt0_dir / "CONTCAR").exists()
        )
        if npt0_done:
            poscar_src = npt0_dir / "CONTCAR"
        elif not self.job_alive(submitted_jobs.get(npt0_key)):
            npt0_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(poscar_src, npt0_dir / "POSCAR")
            shutil.copy2(potcar_src, npt0_dir / "POTCAR")
            check_and_fix_poscar_potcar(npt0_dir / "POSCAR", npt0_dir / "POTCAR")

            T0 = temps[0] if temps else 300
            incar_key = _incar_key_for_npt(category)
            npt_nsw   = self.resolve(project_dir, "npt_steps_aimd", 3000)
            incar_npt0 = _build_incar(
                incar_key,
                natoms=n_atoms,
                encut=encut,
                nsw=npt_nsw,
                tebeg=T0,
                teend=T0,
            )
            job_name_npt0 = f"npt0_{project_dir.name}"[:48]
            _write_incar(npt0_dir / "INCAR", incar_npt0, system_name=job_name_npt0)
            self._write_kpoints(npt0_dir / "KPOINTS", gamma_only=False, poscar=npt0_dir / "POSCAR")
            self._write_sub_sh(npt0_dir / "sub.sh", job_name=job_name_npt0,
                               n_atoms=n_atoms, time=self.slurm_time("aimd_npt"))

            jid = self.sbatch(npt0_dir / "sub.sh", cwd=npt0_dir)
            if jid:
                submitted_jobs[npt0_key] = jid
                first_job_id = jid
                log.info("[h02_aimd] Crystal NPT Step 0 submitted → %s", jid)
            state.set_handler("h02_aimd", {"jobs": submitted_jobs, "target": target})
            # Pass job= so the orchestrator tracks the NPT job (not the old first submission)
            state.set_stage("h02_aimd", "RUNNING", jobs=submitted_jobs, target=target,
                            job=first_job_id)
            return first_job_id

        for T in temps:
            aimd_dir = dft_aimd(project_dir, T)
            xdatcar  = aimd_dir / "XDATCAR"
            steps_done = (
                self.grep_count(xdatcar, "Direct configuration=")
                if xdatcar.exists() else 0
            )
            if steps_done >= target:
                continue

            job_key = f"aimd_{T}"
            if self.job_alive(submitted_jobs.get(job_key)):
                continue

            aimd_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(poscar_src, aimd_dir / "POSCAR")
            potcar_dst = aimd_dir / "POTCAR"
            shutil.copy2(potcar_src, potcar_dst)
            check_and_fix_poscar_potcar(aimd_dir / "POSCAR", potcar_dst)

            incar = _build_incar(
                "nvt_production",
                natoms=n_atoms,
                encut=encut,
                nsw=target,
                tebeg=T,
                teend=T,
            )
            job_name = f"aimd_{project_dir.name}_{T}K"[:48]
            _write_incar(aimd_dir / "INCAR", incar, system_name=job_name)
            self._write_kpoints(aimd_dir / "KPOINTS", gamma_only=False, poscar=aimd_dir / "POSCAR")
            self._write_sub_sh(aimd_dir / "sub.sh", job_name=job_name, n_atoms=n_atoms,
                               time=self.slurm_time("aimd_nvt"))

            jid = self.sbatch(aimd_dir / "sub.sh", cwd=aimd_dir)
            if jid:
                submitted_jobs[job_key] = jid
                first_job_id = first_job_id or jid
                log.info("[h02_aimd] Crystal AIMD T=%dK submitted → %s", T, jid)

        # Submit 16-box dataset (deform + rattle) in parallel with per-T NVT.
        aimd_base = project_dir / "aimd"
        aimd_base.mkdir(parents=True, exist_ok=True)
        shared_potcar = potcar_src
        if not shared_potcar.exists():
            self._generate_potcar(poscar_src, aimd_base / "POTCAR")
            shared_potcar = aimd_base / "POTCAR"
        ds_jids = self._submit_dataset_boxes_slurm(
            aimd_base, poscar_src, shared_potcar, yaml, submitted_jobs,
            rand_kind="rattle", project_dir=project_dir,
        )
        for jid in ds_jids:
            first_job_id = first_job_id or jid

        state.set_handler("h02_aimd", {"jobs": submitted_jobs, "target": target})
        state.set_stage("h02_aimd", "RUNNING", jobs=submitted_jobs, target=target)
        return first_job_id

    def _check_progress_crystal(
        self,
        project_dir: Path,
        yaml: dict,
        state: "ProjectState",
    ) -> None:
        """Update handler state with per-temperature NVT and dataset-box step counts for crystal systems."""
        sim    = yaml.get("simulation", {})
        temps  = sim.get("aimd_temps") or yaml.get("aimd_temps", [300])
        target = self.resolve(project_dir, "aimd_steps", 3000)
        steps: dict = {}
        for T in temps:
            xdatcar = dft_aimd(project_dir, T) / "XDATCAR"
            steps[T] = (
                self.grep_count(xdatcar, "Direct configuration=")
                if xdatcar.exists() else 0
            )
        handler_state = state.get_handler("h02_aimd")
        handler_state["steps"] = steps

        # Track dataset box progress
        aimd_base = project_dir / "aimd"
        ds_target = int(self.resolve(project_dir, "aimd_dataset_steps", 3000))
        ds_specs  = self._dataset_box_specs(aimd_base, rand_kind="rattle")
        ds_done   = sum(1 for _, bd, *_ in ds_specs if self._dataset_box_done(bd, ds_target))
        handler_state["dataset_boxes"] = f"{ds_done}/{len(ds_specs)}"

        state.set_handler("h02_aimd", handler_state)
        nvt_done = sum(1 for v in steps.values() if v >= target)
        log.info("[h02_aimd] Crystal: %d/%d temps at target, %d/%d dataset boxes done",
                 nvt_done, len(temps), ds_done, len(ds_specs))

        # Prepare dataset only after per-T NVT AND all 16 dataset boxes complete.
        all_done = (nvt_done == len(temps) and ds_done == len(ds_specs) and temps)
        if all_done:
            set_npy = mlmd_mlff(project_dir) / "dataset_data" / "set.000" / "energy.npy"
            if not set_npy.exists():
                log.info("[h02_aimd] Crystal AIMD + dataset boxes done — preparing DeepMD dataset")
                try:
                    self._prepare_dataset_crystal(project_dir, yaml)
                except Exception as exc:
                    log.error("[h02_aimd] _prepare_dataset_crystal failed: %s", exc)

    def _prepare_dataset_crystal(self, project_dir: Path, yaml: dict) -> None:
        """Write DeepMD training set from crystal AIMD OUTCARs: per-T NVT + 16 dataset boxes.

        Writes to mlmd_mlff/dataset_data/ — same path h04_mlip.can_run() checks,
        matching the liquid workflow so h04_mlip code is universal.
        """
        sim   = yaml.get("simulation", {})
        temps = sim.get("aimd_temps") or yaml.get("aimd_temps", [300])

        dataset_root = mlmd_mlff(project_dir) / "dataset_data"
        set_dir = dataset_root / "set.000"
        set_dir.mkdir(parents=True, exist_ok=True)

        all_coords: list[np.ndarray] = []
        all_boxes:  list[np.ndarray] = []
        all_energies: list[float]    = []
        all_forces: list[np.ndarray] = []
        type_map: list[str] = []
        types_ref: list[int] | None = None

        def _collect_outcar(outcar: Path, skip_frac: float = 0.2, stride: int = 10) -> None:
            """Append frames from outcar into the accumulated dataset lists, skipping equilibration."""
            nonlocal types_ref
            if not outcar.exists():
                return
            frames = self._parse_vasp_trajectory(outcar)
            skip = max(0, int(len(frames) * skip_frac))
            for atoms in frames[skip::stride]:
                syms = atoms.get_chemical_symbols()
                if types_ref is None:
                    for s in syms:
                        if s not in type_map:
                            type_map.append(s)
                    types_ref = [type_map.index(s) for s in syms]
                try:
                    all_coords.append(atoms.get_positions().reshape(-1))
                    all_boxes.append(atoms.get_cell().array.reshape(-1))
                    all_energies.append(float(atoms.get_potential_energy()))
                    all_forces.append(atoms.get_forces().reshape(-1))
                except Exception:
                    pass

        # Per-temperature NVT trajectories (skip first 20%, every 10th frame)
        for T in temps:
            _collect_outcar(dft_aimd(project_dir, T) / "OUTCAR", skip_frac=0.2)

        # Dataset boxes: deformed + rattled cells (no equilibration skip, every 10th frame)
        aimd_base = project_dir / "aimd"
        for _, box_dir, *_ in self._dataset_box_specs(aimd_base, rand_kind="rattle"):
            _collect_outcar(box_dir / "OUTCAR", skip_frac=0.0)

        if not all_coords or types_ref is None:
            log.warning("[h02_aimd] No crystal frames for DeepMD dataset")
            return

        np.save(set_dir / "coord.npy",  np.array(all_coords,   dtype=np.float64))
        np.save(set_dir / "box.npy",    np.array(all_boxes,    dtype=np.float64))
        np.save(set_dir / "energy.npy", np.array(all_energies, dtype=np.float64))
        np.save(set_dir / "force.npy",  np.array(all_forces,   dtype=np.float64))
        (dataset_root / "type.raw").write_text(
            "\n".join(str(t) for t in types_ref) + "\n"
        )
        (dataset_root / "type_map.raw").write_text("\n".join(type_map) + "\n")
        log.info("[h02_aimd] Crystal dataset_data: %d frames (%d temps + %d dataset boxes) → %s",
                 len(all_coords), len(temps), len(self._dataset_box_specs(aimd_base)), dataset_root)

    # =========================================================================
    # Liquid workflow — dataset generation from canonical dft/opt output
    # =========================================================================

    def _group_liquid_dirs(
        self,
        project_dir: Path,
        aimd_dirs: list[str],
    ) -> dict[Path, list[tuple[str, Path]]]:
        """Group per-temperature AIMD directories by their shared composition parent."""
        groups: dict[Path, list[tuple[str, Path]]] = defaultdict(list)
        for rel_dir in aimd_dirs:
            aimd_dir = project_dir / rel_dir
            groups[aimd_dir.parent].append((rel_dir, aimd_dir))
        return dict(groups)

    def _ensure_composition_inputs(
        self,
        project_dir: Path,
        composition_dir: Path,
        temp_entries: list[tuple[str, Path]],
    ) -> tuple[Path, Path] | None:
        """Create shared POSCAR_initial + POTCAR for a composition (once only)."""
        source_poscar = composition_dir / "POSCAR_initial"
        shared_potcar = composition_dir / "POTCAR"

        if not source_poscar.exists():
            composition_dir.mkdir(parents=True, exist_ok=True)
            # Prefer h01_dft.opt CONTCAR: DFT-relaxed cell at target density.
            opt_contcar = dft_opt(project_dir) / "CONTCAR"
            if opt_contcar.exists() and opt_contcar.stat().st_size > 50:
                shutil.copy2(opt_contcar, source_poscar)
                log.info("[h02_aimd] POSCAR_initial from h01_dft.opt CONTCAR: %s", source_poscar)
            else:
                # Restart: copy from an existing per-temperature POSCAR
                for _, temp_dir in temp_entries:
                    candidate = temp_dir / "POSCAR"
                    if candidate.exists() and candidate.stat().st_size > 50:
                        shutil.copy2(candidate, source_poscar)
                        log.info("[h02_aimd] POSCAR_initial from prior AIMD dir: %s", source_poscar)
                        break

        if not source_poscar.exists():
            log.error("[h02_aimd] No usable POSCAR for %s", composition_dir)
            return None

        if not shared_potcar.exists():
            self._generate_potcar(source_poscar, shared_potcar)

        if not shared_potcar.exists() or shared_potcar.stat().st_size == 0:
            log.error("[h02_aimd] Shared POTCAR missing for %s", composition_dir)
            return None

        return source_poscar, shared_potcar

    def _trajectory_steps(self, aimd_dir: Path) -> int:
        """Count XDATCAR configurations including any restart sibling."""
        xdatcar = aimd_dir / "XDATCAR"
        base = (
            self.grep_count(xdatcar, "Direct configuration=")
            if xdatcar.exists() else 0
        )
        rst_xdatcar = Path(str(aimd_dir) + "_restart") / "XDATCAR"
        rst = (
            self.grep_count(rst_xdatcar, "Direct configuration=")
            if rst_xdatcar.exists() else 0
        )
        return base + max(0, rst - 1)

    def _submit_liquid(
        self,
        project_dir: Path,
        yaml: dict,
        state: "ProjectState",
    ) -> str | None:
        """Submit molecular dataset AIMD from the canonical dft/opt CONTCAR."""
        sim          = yaml.get("simulation", {})
        target       = self.resolve(project_dir, "aimd_steps", 3000)
        run_npt      = bool(sim.get("run_npt", False))
        npt_steps    = self.resolve(project_dir, "npt_steps_aimd", 3000)
        liquid_potim = str(sim.get("potim_fs", "1.0"))
        aimd_dirs    = yaml.get("aimd_dirs", [])

        discard = max(0, int(sim.get("discard_initial_steps", 500)))
        stride  = max(1, int(sim.get("frame_stride", 10)))

        handler_state  = state.get_handler("h02_aimd")
        submitted_jobs: dict = handler_state.get("jobs", {})
        first_job_id: str | None = None
        n_submitted = 0

        for composition_dir, temp_entries in self._group_liquid_dirs(
            project_dir, aimd_dirs
        ).items():
            if n_submitted >= MAX_LIQUID_SUBMIT:
                break
            inputs = self._ensure_composition_inputs(
                project_dir, composition_dir, temp_entries
            )
            if inputs is None:
                continue
            optimized_poscar, shared_potcar = inputs

            n_atoms = self._count_atoms_poscar(optimized_poscar)
            encut   = self.project_encut(project_dir, yaml)

            # Submit dataset boxes (16 short AIMD runs for MLIP training set)
            ds_jids = self._submit_dataset_boxes_slurm(
                composition_dir, optimized_poscar, shared_potcar, yaml, submitted_jobs
            )
            for jid in ds_jids:
                n_submitted += 1
                first_job_id = first_job_id or jid

            for rel_dir, aimd_dir in temp_entries:
                if n_submitted >= MAX_LIQUID_SUBMIT:
                    break
                T       = self._parse_temperature(aimd_dir)
                npt_dir = aimd_dir / "NPT"

                if not run_npt or n_submitted >= MAX_LIQUID_SUBMIT:
                    continue

                # ── NPT ──────────────────────────────────────────────────────
                npt_key     = rel_dir + "_npt"
                npt_xdatcar = npt_dir / "XDATCAR"
                npt_done    = (
                    self.grep_count(npt_xdatcar, "Direct configuration=")
                    if npt_xdatcar.exists() else 0
                )
                if npt_done < npt_steps and not self.job_alive(submitted_jobs.get(npt_key)):
                    npt_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(optimized_poscar, npt_dir / "POSCAR")
                    check_and_fix_poscar_potcar(npt_dir / "POSCAR", shared_potcar)
                    shutil.copy2(shared_potcar,    npt_dir / "POTCAR")

                    incar_npt = _build_incar(
                        "npt_step0_mol",
                        natoms=n_atoms,
                        encut=encut,
                        nsw=npt_steps,
                        tebeg=T,
                        teend=T,
                        extra={"POTIM": liquid_potim, "LORBIT": 0},
                    )

                    job_name = f"npt_{rel_dir.replace('/', '_')}"[:48]
                    _write_incar(npt_dir / "INCAR", incar_npt, system_name=job_name)
                    self._write_kpoints(npt_dir / "KPOINTS")
                    self._write_sub_sh(npt_dir / "sub.sh", job_name, n_atoms)

                    jid = self.sbatch(npt_dir / "sub.sh", cwd=npt_dir)
                    if jid:
                        submitted_jobs[npt_key] = jid
                        n_submitted += 1
                        first_job_id = first_job_id or jid
                        log.info("[h02_aimd] NPT submitted %s T=%dK → %s", rel_dir, T, jid)

        state.set_handler(
            "h02_aimd",
            {"jobs": submitted_jobs, "target": target,
             "discard_initial_steps": discard, "frame_stride": stride},
        )
        state.set_stage("h02_aimd", "RUNNING", jobs=submitted_jobs, target=target)
        log.info("[h02_aimd] Submitted %d new liquid jobs", n_submitted)
        return first_job_id

    def _restart_partial_liquid(
        self,
        nvt_dir: Path,
        target: int,
        cur: int,
        yaml: dict,
        submitted_jobs: dict,
        base_key: str,
    ) -> str | None:
        """Restart a partially completed NVT AIMD from CONTCAR."""
        contcar = nvt_dir / "CONTCAR"
        if not contcar.exists() or contcar.stat().st_size < 50:
            log.warning("[h02_aimd] No CONTCAR for restart in %s", nvt_dir)
            return None

        restart_dir = Path(str(nvt_dir) + "_restart")
        restart_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(contcar, restart_dir / "POSCAR")

        potcar = nvt_dir / "POTCAR"
        if potcar.exists():
            shutil.copy2(potcar, restart_dir / "POTCAR")
            check_and_fix_poscar_potcar(restart_dir / "POSCAR", restart_dir / "POTCAR")
        else:
            check_and_fix_poscar(restart_dir / "POSCAR")

        # Inherit INCAR, update NSW for remaining steps
        incar: dict = {}
        incar_src = nvt_dir / "INCAR"
        if incar_src.exists():
            for line in incar_src.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, _, v = line.partition("=")
                    incar[k.strip()] = v.strip()
        incar["NSW"]    = str(max(1000, target - cur))
        incar["ISTART"] = "0"

        job_name = f"rst_{base_key.replace('/', '_')}"[:48]
        _write_incar(restart_dir / "INCAR", incar, system_name=job_name)
        self._write_kpoints(restart_dir / "KPOINTS")

        n_atoms = self._count_atoms_poscar(restart_dir / "POSCAR")
        self._write_sub_sh(restart_dir / "sub.sh", job_name, n_atoms)

        jid = self.sbatch(restart_dir / "sub.sh", cwd=restart_dir)
        if jid:
            log.info("[h02_aimd] Restart submitted %s → %s", restart_dir, jid)
        return jid

    def _check_progress_liquid(
        self,
        project_dir: Path,
        yaml: dict,
        state: "ProjectState",
    ) -> None:
        """Track liquid NVT jobs; restart dead trajectories."""
        sim       = yaml.get("simulation", {})
        aimd_dirs = yaml.get("aimd_dirs", [])
        target    = self.resolve(project_dir, "aimd_steps", 3000)

        handler_state  = state.get_handler("h02_aimd")
        submitted_jobs: dict = handler_state.get("jobs", {})
        steps: dict = {}
        n_done = 0

        for rel_dir in aimd_dirs:
            nvt_dir = project_dir / rel_dir / "NVT"
            cur = self._trajectory_steps(nvt_dir)
            steps[rel_dir] = cur

            if cur >= _PARTIAL_THRESHOLD_STEPS:
                n_done += 1
                continue

            nvt_key = rel_dir + "_nvt"
            rst_key = rel_dir + "_nvt_restart"
            nvt_alive = self.job_alive(submitted_jobs.get(nvt_key))
            rst_alive = self.job_alive(submitted_jobs.get(rst_key))

            if submitted_jobs.get(nvt_key) and not nvt_alive and not rst_alive:
                log.warning(
                    "[h02_aimd] NVT dead for %s at %d/%d; restarting", rel_dir, cur, target
                )
                rid = self._restart_partial_liquid(
                    nvt_dir, target, cur, yaml, submitted_jobs, nvt_key
                )
                if rid:
                    submitted_jobs[rst_key] = rid

        state.set_handler("h02_aimd", {"steps": steps, "jobs": submitted_jobs})
        log.info(
            "[h02_aimd] Liquid NVT: %d/%d usable (>= %d frames)",
            n_done, len(aimd_dirs), _PARTIAL_THRESHOLD_STEPS,
        )
        self._submit_liquid(project_dir, yaml, state)

    def _is_complete_liquid(self, project_dir: Path, yaml: dict) -> bool:
        """Complete when all dataset boxes are done. NVT runs are not required —
        LAMMPS (h05_lammps) provides production trajectories after MLIP training."""
        aimd_dirs = yaml.get("aimd_dirs", [])
        if not aimd_dirs:
            return False
        ds_steps = self.resolve(project_dir, "aimd_dataset_steps", 3000)
        dft_contcar = dft_opt(project_dir) / "CONTCAR"
        if not dft_contcar.exists() or dft_contcar.stat().st_size < 50:
            return False
        for composition_dir in self._group_liquid_dirs(project_dir, aimd_dirs):
            for _name, box_dir, _scale, _T, _kind in self._dataset_box_specs(composition_dir):
                if not self._dataset_box_done(box_dir, ds_steps):
                    return False
        return True

    def _auto_fix_liquid(
        self,
        project_dir: Path,
        yaml: dict,
        state: "ProjectState",
    ) -> bool:
        """Re-submit stalled liquid jobs and check trajectory progress; always returns True."""
        handler_state  = state.get_handler("h02_aimd")
        submitted_jobs: dict = handler_state.get("jobs", {})
        aimd_dirs: list[str] = yaml.get("aimd_dirs", [])

        active_keys: set[str] = set()
        for composition_dir, temp_entries in self._group_liquid_dirs(
            project_dir, aimd_dirs
        ).items():
            for rel_dir, _ in temp_entries:
                active_keys.update({
                    rel_dir + "_nvt",
                    rel_dir + "_npt",
                    rel_dir + "_nvt_restart",
                })

        alive = any(
            self.job_alive(submitted_jobs.get(k)) for k in active_keys
        )
        if alive:
            self._submit_liquid(project_dir, yaml, state)
            return True
        self._check_progress_liquid(project_dir, yaml, state)
        return True

    # =========================================================================
    # Dataset generation (shared NVT/NPT liquid path)
    # =========================================================================

    def _parse_vasp_trajectory(self, outcar: Path) -> list:
        """Read VASP OUTCAR frames via ASE."""
        try:
            from ase.io import read
            if not outcar.exists():
                return []
            frames = read(str(outcar), index=":")
            if not isinstance(frames, list):
                frames = [frames]
            valid = []
            for atoms in frames:
                try:
                    atoms.get_potential_energy()
                    atoms.get_forces()
                    valid.append(atoms)
                except Exception:
                    continue
            return valid
        except Exception as exc:
            log.warning("[h02_aimd] ASE parse failed for %s: %s", outcar, exc)
            return []

    def _write_deepmd_dataset(
        self,
        project_dir: Path,
        yaml: dict,
        ensemble: str,
    ) -> None:
        """Write DeepMD dataset for NVT or NPT frames."""
        sim       = yaml.get("simulation", {})
        aimd_dirs = yaml.get("aimd_dirs", [])
        discard   = int(sim.get("discard_initial_steps",
                                self.plat("aimd_dataset", "discard_initial_steps", 0)))
        stride    = max(1, int(sim.get("frame_stride",
                                       self.plat("aimd_dataset", "frame_stride", 10))))
        target    = self.resolve(project_dir, "aimd_steps", 3000)
        if target < discard * 2:
            discard = max(0, target // 4)  # keep 75% of frames for short runs

        dataset_root = mlmd_mlff(project_dir) / f"{ensemble.lower()}_data"
        set_dir = dataset_root / "set.000"
        set_dir.mkdir(parents=True, exist_ok=True)

        all_coords:   list[np.ndarray] = []
        all_boxes:    list[np.ndarray] = []
        all_energies: list[float]      = []
        all_forces:   list[np.ndarray] = []
        metadata_rows: list[dict]      = []
        symbols_ref: list[str] | None  = None
        type_map:    list[str]         = []
        types_ref:   list[int] | None  = None
        n_atoms: int | None            = None

        for rel_dir in aimd_dirs:
            run_dir = project_dir / rel_dir / ensemble
            outcar  = run_dir / "OUTCAR"
            frames  = self._parse_vasp_trajectory(outcar)
            if not frames:
                continue
            T = self._parse_temperature(project_dir / rel_dir)
            for ionic_step, atoms in enumerate(frames[discard::stride], start=discard):
                syms = atoms.get_chemical_symbols()
                if symbols_ref is None:
                    symbols_ref = syms
                    for s in syms:
                        if s not in type_map:
                            type_map.append(s)
                    types_ref = [type_map.index(s) for s in syms]
                    n_atoms   = len(syms)
                elif syms != symbols_ref:
                    log.error(
                        "[h02_aimd] Incompatible atom ordering in %s — write separate dataset",
                        run_dir,
                    )
                    continue
                try:
                    coords  = atoms.get_positions().reshape(-1)
                    box     = atoms.get_cell().array.reshape(-1)
                    energy  = float(atoms.get_potential_energy())
                    forces  = atoms.get_forces().reshape(-1)
                except Exception:
                    continue
                if n_atoms is None:
                    continue
                if len(coords) != 3 * n_atoms or len(forces) != 3 * n_atoms:
                    continue
                all_coords.append(coords)
                all_boxes.append(box)
                all_energies.append(energy)
                all_forces.append(forces)
                metadata_rows.append({
                    "frame_id":     len(all_coords) - 1,
                    "composition":  str((project_dir / rel_dir).parent.relative_to(project_dir)),
                    "temperature_K": T,
                    "ensemble":     ensemble,
                    "source_dir":   str(run_dir.relative_to(project_dir)),
                    "ionic_step":   ionic_step,
                    "volume_A3":    float(atoms.get_volume()),
                })

        if not all_coords or types_ref is None:
            log.warning("[h02_aimd] No valid %s frames found", ensemble)
            return

        np.save(set_dir / "coord.npy",  np.array(all_coords,   dtype=np.float64))
        np.save(set_dir / "box.npy",    np.array(all_boxes,    dtype=np.float64))
        np.save(set_dir / "energy.npy", np.array(all_energies, dtype=np.float64))
        np.save(set_dir / "force.npy",  np.array(all_forces,   dtype=np.float64))
        (dataset_root / "type.raw").write_text(
            "\n".join(str(t) for t in types_ref) + "\n"
        )
        (dataset_root / "type_map.raw").write_text("\n".join(type_map) + "\n")

        if metadata_rows:
            meta_path = dataset_root / "frame_metadata.csv"
            with meta_path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=metadata_rows[0].keys())
                writer.writeheader()
                writer.writerows(metadata_rows)

        log.info(
            "[h02_aimd] Wrote %s DeepMD dataset: %d frames, %d atoms",
            ensemble, len(all_coords), n_atoms,
        )

    def _write_deepmd_dataset_boxes(self, project_dir: Path, yaml: dict) -> None:
        """Collect frames from the 16 dataset boxes → mlmd_mlff/dataset_data/."""
        aimd_dirs = yaml.get("aimd_dirs", [])
        stride = max(1, int(self.plat("aimd_dataset", "frame_stride", 10)))
        discard = 0

        dataset_root = mlmd_mlff(project_dir) / "dataset_data"
        set_dir = dataset_root / "set.000"
        set_dir.mkdir(parents=True, exist_ok=True)

        all_coords:   list[np.ndarray] = []
        all_boxes:    list[np.ndarray] = []
        all_energies: list[float]      = []
        all_forces:   list[np.ndarray] = []
        symbols_ref: list[str] | None  = None
        type_map:    list[str]         = []
        types_ref:   list[int] | None  = None
        n_atoms: int | None            = None

        for composition_dir in self._group_liquid_dirs(project_dir, aimd_dirs):
            for _name, box_dir, _scale, _T, _kind in self._dataset_box_specs(composition_dir):
                frames = self._parse_vasp_trajectory(box_dir / "OUTCAR")
                for atoms in frames[discard::stride]:
                    syms = atoms.get_chemical_symbols()
                    if symbols_ref is None:
                        symbols_ref = syms
                        for s in syms:
                            if s not in type_map:
                                type_map.append(s)
                        types_ref = [type_map.index(s) for s in syms]
                        n_atoms   = len(syms)
                    elif syms != symbols_ref:
                        continue
                    try:
                        all_coords.append(atoms.get_positions().reshape(-1))
                        all_boxes.append(atoms.get_cell().array.reshape(-1))
                        all_energies.append(float(atoms.get_potential_energy()))
                        all_forces.append(atoms.get_forces().reshape(-1))
                    except Exception:
                        continue

        if not all_coords or types_ref is None:
            log.warning("[h02_aimd] No dataset box frames collected")
            return

        np.save(set_dir / "coord.npy",  np.array(all_coords,   dtype=np.float64))
        np.save(set_dir / "box.npy",    np.array(all_boxes,    dtype=np.float64))
        np.save(set_dir / "energy.npy", np.array(all_energies, dtype=np.float64))
        np.save(set_dir / "force.npy",  np.array(all_forces,   dtype=np.float64))
        (dataset_root / "type.raw").write_text(
            "\n".join(str(t) for t in types_ref) + "\n"
        )
        (dataset_root / "type_map.raw").write_text("\n".join(type_map) + "\n")
        log.info("[h02_aimd] Dataset boxes: %d frames → %s", len(all_coords), dataset_root)

    def _prepare_dataset_liquid(self, project_dir: Path, yaml: dict) -> None:
        """Collect NVT (and optionally NPT) frames plus dataset boxes into the DeepMD dataset."""
        self._write_deepmd_dataset(project_dir, yaml, ensemble="NVT")
        if bool(yaml.get("simulation", {}).get("run_npt", False)):
            self._write_deepmd_dataset(project_dir, yaml, ensemble="NPT")
        self._write_deepmd_dataset_boxes(project_dir, yaml)
        # h04_mlip now reads directly from dataset_data/ and builds 00.data/training_data
        # + 00.data/validation_data itself. No copy needed here.
