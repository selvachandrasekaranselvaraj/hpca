"""
h01_dft.py — DFT handler: vc-relax, opt, Bader, DOS, static (SLURM).

Subtasks (each tracked separately in state as h01_dft.{subtask}):
  vc_relax     → h01_dft.vc_relax
  opt          → h01_dft.opt
  bader        → h01_dft.bader
  dos_scf      → h01_dft.dos_scf
  dos_nonscf   → h01_dft.dos_nonscf
  static       → h01_dft.static
  echem_static → h01_dft.echem_static
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .base import SimulationHandler
from hpca.core.categories import (
    is_sse as _cat_is_sse,
    is_crystalline as _cat_is_crystalline,
    is_molecular as _cat_is_molecular,
)
from hpca.core.paths import dft_opt, dft_vc, dft_base, contcar_preopt
from hpca.core.structure_check import check_and_fix_poscar
from hpca.core.config import Config
from hpca.core.potcar import build_potcar as _build_potcar
from hpca.core.kpoints import kpoints_from_poscar as _kp_from_poscar, write_kpoints as _write_kpoints
from hpca.core.vasp_job import write_incar as _write_incar, write_kpoints_from_poscar as _write_kpoints_from_poscar
from hpca.registry.incar import build_incar as _build_incar
from hpca.registry.submission import write_submission as _write_sub

if TYPE_CHECKING:
    from hpca.orchestrator.state_tracker import ProjectState

log = logging.getLogger("hpca.orch")
# Layout: see hpca/core/paths.py

# MPI process counts to benchmark (sequential phase)
_NTASKS_CANDIDATES = [96, 80, 72, 64, 48, 32, 16]

def _ncore_for_ntasks(ntasks: int) -> int:
    """NCORE rule: ≥40 MPI processes → NCORE=8; <40 → NCORE=4."""
    return 8 if ntasks >= 40 else 4

# Dependency chain: each subtask depends on the previous ones
SUBTASK_DEPS: dict[str, list[str]] = {
    "aimd_relax":  [],
    "vc_relax":    ["aimd_relax"],
    "opt":         ["vc_relax"],
    "bader":       ["opt"],
    "dos_scf":     ["opt"],
    "dos_nonscf":  ["dos_scf"],
    "static":      ["opt"],
    "echem_static": ["static"],
}
ALL_SUBTASKS = ["aimd_relax", "vc_relax", "opt", "bader", "dos_scf", "dos_nonscf", "static", "echem_static"]
DEFAULT_SUBTASKS = ["vc_relax", "opt"]
SSE_SUBTASKS = ["vc_relax", "opt", "bader", "dos_scf", "dos_nonscf", "static", "echem_static"]


class DFTHandler(SimulationHandler):
    """SLURM handler: manages sequential/parallel DFT subtasks."""

    name = "h01_dft"
    is_daemon = False

    def migrate_molecular_sick_job(self, project_dir: Path, state: "ProjectState",
                                   project_yaml: dict) -> list[str]:
        """Recover molecular DFT retries exhausted by the old SYMPREC policy."""
        from hpca.core.categories import is_molecular
        if not is_molecular(project_yaml.get("category", "")):
            return []
        autonomy = state.state.setdefault("autonomy", {})
        migrations = autonomy.setdefault("migrations", {})
        migration = "molecular_isym_minus_one_v2"
        if migrations.get(migration):
            return []
        recovered: list[str] = []
        for sub in self._enabled_subtasks(project_dir):
            key = f"h01_dft.{sub}"
            value = state.get_handler(key)
            if not (
                value.get("stage") == "FAILED"
                and value.get("error") == "FIX_BUDGET_EXHAUSTED"
                and value.get("fixed") in ("SICK_JOB", "SICK_JOB_SYMPREC")
            ):
                continue
            incar = self._workdir(project_dir, sub) / "INCAR"
            if not incar.exists():
                continue
            from hpca.orchestrator.auto_fix import incar_set, incar_remove
            incar_set(incar, "ISYM", "-1")
            incar_remove(incar, "SYMPREC")
            history = list(value.get("history", []))
            history.append({"from": "FAILED", "to": "PENDING",
                            "at": datetime.now().isoformat(),
                            "reason": migration})
            value["stage"] = "PENDING"
            value["history"] = history[-100:]
            value["fix_count"] = 0
            value.pop("error", None)
            value.pop("failed_at", None)
            value.pop("job", None)
            recovered.append(key)
        migrations[migration] = True
        state.save()
        return recovered

    def can_run(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True when a valid POSCAR or MACE-preopt CONTCAR is available for the DFT work dir."""
        # Preopt contcar takes priority: polymer gel DFT POSCARs have intentional
        # Phase-2 grid-placement overlaps (<1 Å) that _poscar_is_valid flags as
        # invalid, but the MACE-preopt contcar resolves them.
        if contcar_preopt(project_dir, "dft").exists():
            return True
        poscar_dft = project_dir / "designed_structures" / "poscar_dft.vasp"
        if poscar_dft.exists():
            if not self._poscar_is_valid(poscar_dft):
                log.warning("[h01_dft] poscar_dft.vasp has overlapping atoms and no "
                            "preopt contcar — waiting for h00_design")
                return False
            return True
        return (
            (dft_opt(project_dir) / "POSCAR").exists()
            or (dft_vc(project_dir) / "POSCAR").exists()
        )

    def is_complete(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True when every enabled DFT subtask is in the COMPLETE state."""
        enabled = self._enabled_subtasks(project_dir)
        for sub in enabled:
            key = f"h01_dft.{sub}"
            if state.get_stage(key) != "COMPLETE":
                return False
        return True

    def submit(self, project_dir: Path, state: "ProjectState") -> str | None:
        """Submit the next ready DFT subtask; auto-complete skipped ones and return the SLURM job ID."""
        enabled = self._enabled_subtasks(project_dir)
        yaml_data = self.read_project_yaml(project_dir)
        encut_override = yaml_data.get("encut")   # explicit user value; None → compute from POTCAR

        # Auto-complete any subtask that is PENDING but not needed for this sub-project.
        # This handles host/pure SSE systems where aimd_relax is in the category-level
        # enabled list but not in this project's inner enabled list (not doped).
        # Without this, the outer dep-graph would see aimd_relax=PENDING forever and
        # block other handlers that check it as a dependency.
        enabled_set = set(enabled)
        for sub_all in ALL_SUBTASKS:
            key_all = f"h01_dft.{sub_all}"
            if sub_all not in enabled_set and state.get_stage(key_all) == "PENDING":
                state.set_stage(key_all, "COMPLETE")
                log.debug("[h01_dft] Auto-completing skipped subtask %s (not needed)", key_all)

        # Find the next subtask to submit
        for sub in enabled:
            key = f"h01_dft.{sub}"
            stage = state.get_stage(key)
            if stage in ("RUNNING",):
                # Already running — check progress
                self.check_progress(project_dir, state)
                continue
            if stage == "COMPLETE":
                continue
            if stage == "FAILED":
                log.warning("[h01_dft] subtask %s FAILED; attempting auto_fix", sub)
                fixed = self.auto_fix(project_dir, state)
                if not fixed:
                    continue

            # Check if dependencies are met; skip deps not in enabled list
            deps_ok = all(
                state.get_stage(f"h01_dft.{d}") == "COMPLETE"
                for d in SUBTASK_DEPS[sub]
                if d in enabled
            )
            if not deps_ok:
                log.debug("[h01_dft] Waiting for deps of %s", sub)
                continue

            # Submit this subtask
            job_id = self._submit_subtask(project_dir, sub, yaml_data, encut_override)
            if job_id:
                state.set_stage(key, "RUNNING", job=job_id)
                log.info("[h01_dft] Submitted %s job=%s", sub, job_id)
                # For parallel-eligible subtasks (bader + dos_scf), submit both
                if sub == "dos_scf" and "bader" in enabled:
                    bader_key = "h01_dft.bader"
                    if state.get_stage(bader_key) == "PENDING":
                        j2 = self._submit_subtask(project_dir, "bader", yaml_data, encut_override)
                        if j2:
                            state.set_stage(bader_key, "RUNNING", job=j2)
                            log.info("[h01_dft] Also submitted bader job=%s", j2)
                return job_id

        return None

    def _submit_subtask(
        self, project_dir: Path, sub: str,
        yaml_data: dict, encut_override: float | None = None
    ) -> str | None:
        """Prepare VASP input files for one subtask and submit via sbatch; return job ID or None."""
        work_dir = self._workdir(project_dir, sub)
        work_dir.mkdir(parents=True, exist_ok=True)
        system_name = yaml_data.get("name", project_dir.name)
        # Use benchmarked ntasks for DFT jobs; ncore_opt always requests 96 internally
        tasks = self._get_optimal_ntasks(project_dir,
                    default=yaml_data.get("tasks_dft", self.sim_limit("slurm", "vasp_ntasks", 96)))

        # ── Standard DFT subtasks ───────────────────────────────────────────────
        # echem_static: build delithiated POSCAR from opt/ CONTCAR
        if sub == "echem_static":
            opt_contcar = dft_opt(project_dir) / "CONTCAR"
            opt_poscar  = dft_opt(project_dir) / "POSCAR"
            src = opt_contcar if opt_contcar.exists() else opt_poscar
            if not src.exists():
                log.warning("[h01_dft] echem_static: no opt CONTCAR/POSCAR found")
                return None
            mobile_ion = yaml_data.get("mobile_ion", "Li")
            n_removed = self._make_delithiated_poscar(src, work_dir / "POSCAR", mobile_ion, fraction=0.5)
            log.info("[h01_dft] echem_static: removed %d %s atoms (50%% delithiation)", n_removed, mobile_ion)
        else:
            # Source POSCAR
            poscar_src = self._poscar_source(project_dir, sub)
            if poscar_src and poscar_src.exists():
                if poscar_src.resolve() != (work_dir / "POSCAR").resolve():
                    shutil.copy(poscar_src, work_dir / "POSCAR")
            else:
                log.warning("[h01_dft] No POSCAR source for %s", sub)
                return None
        # Fix close atomic contacts before SLURM submission
        check_and_fix_poscar(work_dir / "POSCAR")
        n_atoms = self._count_atoms_poscar(work_dir / "POSCAR")

        # Copy POTCAR from first available source; generate from POTPAW if none found.
        # vc_relax runs before opt/, so dft/opt/POTCAR doesn't exist on the first pass —
        # must also check dft/vc/ and designed_structures/, then fall back to generation.
        potcar_dst = work_dir / "POTCAR"
        if not potcar_dst.exists():
            for potcar_src in [
                dft_opt(project_dir) / "POTCAR",
                dft_vc(project_dir) / "POTCAR",
                project_dir / "designed_structures" / "POTCAR",
            ]:
                if potcar_src.exists() and potcar_src.resolve() != potcar_dst.resolve():
                    shutil.copy(potcar_src, potcar_dst)
                    break
            if not potcar_dst.exists():
                self._generate_potcar_dft(work_dir / "POSCAR", potcar_dst)
        # Also copy WAVECAR/CHGCAR for restarts
        if sub == "dos_nonscf":
            for f in ("WAVECAR", "CHGCAR", "OUTCAR"):
                src = project_dir / "dos" / "scf" / f
                if src.exists():
                    shutil.copy(src, work_dir / f)

        # ENCUT: project-wide value (same for DFT/AIMD/NEB); local POTCAR is fallback
        encut = self.project_encut(project_dir, yaml_data) if encut_override is None \
                else float(encut_override)
        log.info("[h01_dft] %s ENCUT=%.1f eV", sub, encut)

        kpoints_mesh = _kp_from_poscar(work_dir / "POSCAR") if (work_dir / "POSCAR").exists() else (1, 1, 1)
        nsw: int | None = None
        tebeg: int | None = None
        teend: int | None = None
        if sub == "aimd_relax":
            nsw   = self.resolve(project_dir, "aimd_relax_steps")
            tebeg = self.resolve(project_dir, "aimd_relax_temp")
            teend = tebeg
        elif sub == "vc_relax":
            nsw = self.resolve(project_dir, "dft_nsw_vcrelax")
        elif sub == "opt":
            nsw = self.resolve(project_dir, "dft_nsw_opt")
        incar = _build_incar("static" if sub == "echem_static" else sub,
                             natoms=n_atoms, encut=encut, nsw=nsw,
                             tebeg=tebeg, teend=teend,
                             project_yaml=yaml_data)
        _write_incar(work_dir / "INCAR", incar, system_name=f"{system_name}_{sub}")
        _write_kpoints(work_dir / "KPOINTS", kpoints_mesh, gamma=True)

        nodes = yaml_data.get("nodes_dft", 1)
        time_map = {
            "aimd_relax":   self.slurm_time("dft_aimd_relax", default="48:00:00"),
            "vc_relax":     self.slurm_time("dft_vc"),
            "opt":          self.slurm_time("dft_opt"),
            "bader":        self.slurm_time("dft_bader"),
            "dos_scf":      self.slurm_time("dft_dos"),
            "dos_nonscf":   self.slurm_time("dft_dos"),
            "static":       self.slurm_time("dft_dos"),
            "echem_static": self.slurm_time("dft_echem_static", default=self.slurm_time("dft_dos")),
        }
        time = time_map.get(sub, self.slurm_time("dft_dos"))
        sub_sh = work_dir / "sub.sh"
        _write_sub(sub_sh, "vasp", f"{system_name}_{sub}", nodes=nodes, ntasks=tasks, time=time)

        return self.sbatch(sub_sh, cwd=work_dir)

    def check_progress(self, project_dir: Path, state: "ProjectState") -> None:
        """Poll OUTCAR convergence and SLURM job status for all running DFT subtasks."""
        enabled = self._enabled_subtasks(project_dir)
        for sub in enabled:
            key = f"h01_dft.{sub}"
            if state.get_stage(key) != "RUNNING":
                continue
            job_id = state.get_handler(key).get("job")
            work_dir = self._workdir(project_dir, sub)

            outcar = work_dir / "OUTCAR"

            # MD and static subtasks never write "reached required accuracy";
            # detect completion by "User time" in OUTCAR tail instead.
            _static_subs = ("aimd_relax", "bader", "dos_scf", "dos_nonscf", "static", "echem_static")
            if sub in _static_subs:
                if outcar.exists() and "User time" in outcar.read_text(errors="replace")[-3000:]:
                    if sub == "bader":
                        self._run_bader_program(work_dir)
                    state.set_stage(key, "COMPLETE")
                    log.info("[h01_dft] %s COMPLETE (OUTCAR finished)", sub)
                    continue
            elif self._outcar_converged(outcar):
                state.set_stage(key, "COMPLETE")
                log.info("[h01_dft] %s COMPLETE (OUTCAR converged)", sub)
                continue
            elif sub in ("vc_relax", "opt"):
                # LORBIT=11 writes projected DOS tables into OUTCAR after geometry
                # converges, pushing "reached required accuracy" beyond the 4000-byte
                # tail window. Accept CONTCAR + "User time" as equivalent evidence.
                contcar = work_dir / "CONTCAR"
                if (contcar.exists() and contcar.stat().st_size > 50
                        and outcar.exists()
                        and "User time" in outcar.read_text(errors="replace")[-3000:]):
                    state.set_stage(key, "COMPLETE")
                    log.info("[h01_dft] %s COMPLETE (CONTCAR + OUTCAR finished, LORBIT tail)", sub)
                    continue

            if not self.job_alive(job_id):
                if not outcar.exists():
                    # No OUTCAR in work_dir: orchestrator stamped the wrong subtask RUNNING
                    # (submit() submitted a prerequisite subtask whose output is elsewhere).
                    # Reset to PENDING so the next poll can submit the real job.
                    log.warning("[h01_dft] %s: job dead but no OUTCAR in %s — resetting to PENDING",
                                sub, work_dir)
                    state.set_stage(key, "PENDING")
                    continue
                err = None
                try:
                    from hpca.orchestrator.auto_fix import detect_vasp_error
                    err = detect_vasp_error(work_dir)
                except ImportError:
                    pass
                if err:
                    log.warning("[h01_dft] %s dead with error %s — attempting fix", sub, err)
                    state.set_stage(key, "FAILED", error=err)
                else:
                    log.warning("[h01_dft] %s job dead, no error detected — FAILED", sub)
                    state.set_stage(key, "FAILED")

    def auto_fix(self, project_dir: Path, state: "ProjectState") -> bool:
        """Detect VASP error, apply fix, and resubmit if successful."""
        enabled = self._enabled_subtasks(project_dir)
        for sub in enabled:
            key = f"h01_dft.{sub}"
            # Allow both FAILED and RUNNING (called while job just died, stage may still be RUNNING)
            if state.get_stage(key) not in ("FAILED", "RUNNING"):
                continue
            work_dir = self._workdir(project_dir, sub)
            incar_path = work_dir / "INCAR"

            try:
                from hpca.orchestrator.auto_fix import (
                    detect_vasp_error, fix_incar, detect_walltime,
                    incar_set, incar_remove, fix_potcar_species,
                    fix_wavecar_zero_byte, within_fix_budget,
                    increment_fix_count,
                )
            except ImportError:
                log.error("[h01_dft] auto_fix: cannot import auto_fix module")
                return False

            if not within_fix_budget(state, key):
                log.warning("[h01_dft] %s: auto-fix budget exhausted — marking FAILED", sub)
                state.set_stage(key, "FAILED", error="FIX_BUDGET_EXHAUSTED")
                continue

            err = state.get_handler(key).get("error") or detect_vasp_error(work_dir)
            if err is None and detect_walltime(work_dir):
                err = "WALLTIME"

            # Transient SLURM failure (job limit, scheduler busy): reset to PENDING for retry.
            # No VASP ran so there's nothing to fix — just let the next poll try again.
            if err == "sbatch returned None":
                log.info("[h01_dft] %s failed due to sbatch transient error — resetting to PENDING", sub)
                state.set_stage(key, "PENDING")
                return True

            # Handle POTCAR mismatch: try to find matching POTCAR, then resubmit
            if err == "POTCAR_MISMATCH":
                poscar = work_dir / "POSCAR"
                potcar = work_dir / "POTCAR"
                if fix_potcar_species(poscar, potcar):
                    log.info("[h01_dft] Fixed POTCAR species mismatch for %s", sub)
                    yaml_data = self.read_project_yaml(project_dir)
                    job_id = self._submit_subtask(
                        project_dir, sub, yaml_data, yaml_data.get("encut")   # None → _submit_subtask computes from POTCAR
                    )
                    if job_id:
                        increment_fix_count(state, key)
                        state.set_stage(key, "RUNNING", job=job_id, fixed=err)
                        return True
                continue

            # Handle negative volume: swap b↔c vectors in POSCAR
            if err == "NEGATIVE_VOLUME":
                from hpca.orchestrator.auto_fix import fix_negative_volume
                poscar = work_dir / "POSCAR"
                if fix_negative_volume(poscar):
                    log.info("[h01_dft] Fixed negative volume POSCAR for %s", sub)
                    yaml_data = self.read_project_yaml(project_dir)
                    job_id = self._submit_subtask(
                        project_dir, sub, yaml_data, yaml_data.get("encut")   # None → _submit_subtask computes from POTCAR
                    )
                    if job_id:
                        increment_fix_count(state, key)
                        state.set_stage(key, "RUNNING", job=job_id, fixed=err)
                        return True
                continue

            # ZPOTRF: delete 0-byte WAVECAR first, then patch INCAR (LREAL=F)
            if err == "ZPOTRF":
                fix_wavecar_zero_byte(work_dir)

            molecular_sick_job = (
                err in ("SICK_JOB", "SICK_JOB_SYMPREC")
                and _cat_is_molecular(
                    self.read_project_yaml(project_dir).get("category", ""))
            )
            fixed_incar = False
            if molecular_sick_job:
                incar_set(incar_path, "ISYM", "-1")
                incar_remove(incar_path, "SYMPREC")
                fixed_incar = True
                log.warning("[h01_dft] %s: disabled symmetry for molecular SICK_JOB", sub)
            elif err:
                fixed_incar = fix_incar(incar_path, err)

            if err and fixed_incar:
                # SICK_JOB/SICK_JOB_SYMPREC: restart from scratch (ISTART=0) with new
                # SYMPREC — reusing the WAVECAR from the failing run is unsafe.
                if err in ("SICK_JOB", "SICK_JOB_SYMPREC"):
                    incar_remove(incar_path, "ISTART")
                    incar_remove(incar_path, "ICHARG")
                else:
                    # For other errors: warm-restart from WAVECAR if it exists and is valid
                    wavecar = work_dir / "WAVECAR"
                    if wavecar.exists() and wavecar.stat().st_size > 0:
                        incar_set(incar_path, "ISTART", "1")
                        incar_set(incar_path, "ICHARG", "1")
                # Remove NELMDL if it was set (causes NaN with ISTART=1)
                incar_remove(incar_path, "NELMDL")
                yaml_data = self.read_project_yaml(project_dir)
                job_id = self._submit_subtask(
                    project_dir, sub, yaml_data, yaml_data.get("encut")   # None → _submit_subtask computes from POTCAR
                )
                if job_id:
                    increment_fix_count(state, key)
                    state.set_stage(key, "RUNNING", job=job_id, fixed=err)
                    log.info("[h01_dft] Resubmitted %s after fixing %s, new job=%s", sub, err, job_id)
                    return True

        return False

    # ── Internal helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _get_optimal_ntasks(project_dir: Path, default: int = 96) -> int:
        """Read optimal ntasks from ncore_opt result, or return default."""
        result = project_dir / "dft" / "ncore_opt" / "ncore_best.txt"
        try:
            return int(result.read_text().strip())
        except Exception:
            return default

    @staticmethod
    def _get_optimal_ncore(project_dir: Path, default: int = 8) -> int:
        """Derive NCORE from optimal ntasks: ≥40 → 8, <40 → 4."""
        ntasks = DFTHandler._get_optimal_ntasks(project_dir, default=96)
        return _ncore_for_ntasks(ntasks)

    @staticmethod
    def _get_optimal_nparallel(project_dir: Path) -> int:
        """Number of parallel VASP jobs per node from ncore_opt (1 = sequential, 6 = parallel batch)."""
        p = project_dir / "dft" / "ncore_opt" / "ncore_parallel.txt"
        try:
            return int(p.read_text().strip())
        except Exception:
            return 1

    def _write_ncore_phase1_sub_sh(self, path: Path, poscar: Path, potcar: Path,
                                   kpoints_mesh: tuple, job_name: str) -> None:
        """Phase 1: sequential ntasks benchmark on one node → writes best_ntasks.txt + best_time.txt."""
        _write_sub(path, "vasp_ncore_phase1", job_name,
                   work_dir=str(path.parent), poscar=str(poscar), potcar=str(potcar),
                   kpoints_mesh=kpoints_mesh, candidates=_NTASKS_CANDIDATES)

    def _write_ncore_phase2_sub_sh(self, path: Path, poscar: Path, potcar: Path,
                                   kpoints_mesh: tuple, job_name: str) -> None:
        """Phase 2: parallel 6×16-core test on a SEPARATE node → writes par_time.txt + par_ok.txt."""
        _write_sub(path, "vasp_ncore_phase2", job_name,
                   work_dir=str(path.parent), poscar=str(poscar), potcar=str(potcar),
                   kpoints_mesh=kpoints_mesh)

    def _write_ncore_finalize_sub_sh(self, path: Path, job_name: str) -> None:
        """Finalize: compare Phase 1 + Phase 2 results → write ncore_best.txt + ncore_parallel.txt."""
        _write_sub(path, "vasp_ncore_finalize", job_name, work_dir=str(path.parent))

    def _enabled_subtasks(self, project_dir: Path) -> list[str]:
        """Return the ordered list of DFT subtasks to run, derived from project category and project.yaml."""
        yaml_data = self.read_project_yaml(project_dir)
        category = yaml_data.get("category", "")
        is_sse = _cat_is_sse(category)

        # aimd_relax only for doped SSE structures: substitution creates local
        # distortions that need thermal equilibration before vc_relax.
        # Host/pure systems (doping_n=0, no doping_elements) go straight to vc_relax.
        is_doped = (
            bool(yaml_data.get("doping_n", 0))        # single-dopant (cation or anion)
            or bool(yaml_data.get("doping_elements"))  # di/multi-dopant
            or bool(yaml_data.get("doping_element"))   # fallback: explicit element field
        )
        needs_aimd_relax = is_sse and is_doped

        sse_defaults = (["aimd_relax"] + SSE_SUBTASKS) if needs_aimd_relax else (
            list(SSE_SUBTASKS) if is_sse else list(DEFAULT_SUBTASKS)
        )
        stages_cfg = yaml_data.get("stages", {}).get("dft", {})
        if not stages_cfg:
            return sse_defaults
        if isinstance(stages_cfg, bool):
            return sse_defaults
        if isinstance(stages_cfg, list):
            return [s for s in ALL_SUBTASKS if s in stages_cfg]
        return [s for s in ALL_SUBTASKS if stages_cfg.get(s, s in sse_defaults)]

    def _workdir(self, project_dir: Path, sub: str) -> Path:
        """Return the filesystem path for a given DFT subtask working directory."""
        mapping = {
            "ncore_opt":    project_dir / "dft" / "ncore_opt",
            "aimd_relax":   project_dir / "dft" / "aimd_relax",
            "vc_relax":     dft_vc(project_dir),
            "opt":          dft_opt(project_dir),
            "bader":        project_dir / "bader",
            "dos_scf":      project_dir / "dos" / "scf",
            "dos_nonscf":   project_dir / "dos" / "nonscf",
            "static":       project_dir / "static",
            "echem_static": project_dir / "echem_static",
        }
        return mapping[sub]

    def _poscar_source(self, project_dir: Path, sub: str) -> Path | None:
        """Return the POSCAR source path for a given subtask.

        Workflow: dft/preopt/CONTCAR → [dft/aimd_relax/] → dft/vc/ → dft/opt/
        Cross-ref: hpca/core/paths.py contcar_preopt(), dft_vc(), dft_opt()
        """
        if sub == "aimd_relax":
            # AIMD pre-relax starts from MACE-preopt structure
            preopt_src = contcar_preopt(project_dir, "dft")
            if preopt_src.exists():
                return preopt_src
            designed = project_dir / "designed_structures" / "poscar_dft.vasp"
            if designed.exists():
                return designed
            return dft_vc(project_dir) / "POSCAR"
        if sub == "vc_relax":
            # Prefer thermally-equilibrated structure from AIMD pre-relax
            aimd_relax_contcar = project_dir / "dft" / "aimd_relax" / "CONTCAR"
            if aimd_relax_contcar.exists() and aimd_relax_contcar.stat().st_size > 50:
                return aimd_relax_contcar
            # Fall back to MACE-preopt structure (from interactive preopt or h00_design)
            preopt_src = contcar_preopt(project_dir, "dft")
            if preopt_src.exists():
                return preopt_src
            # Fresh project: designed_structures/poscar_dft.vasp
            designed = project_dir / "designed_structures" / "poscar_dft.vasp"
            if designed.exists():
                return designed
            # Fallback: existing POSCAR in vc/ (e.g. user-placed)
            p = dft_vc(project_dir) / "POSCAR"
            if p.exists():
                return p
            return dft_opt(project_dir) / "POSCAR"
        if sub == "opt":
            # Prefer CONTCAR from vc_relax (ISIF=3 → ISIF=2)
            contcar = dft_vc(project_dir) / "CONTCAR"
            if contcar.exists():
                return contcar
            # vc_relax disabled or not yet run: fall back to MACE preopt then designed
            preopt_src = contcar_preopt(project_dir, "dft")
            if preopt_src.exists():
                return preopt_src
            designed = project_dir / "designed_structures" / "poscar_dft.vasp"
            if designed.exists():
                return designed
            return dft_opt(project_dir) / "POSCAR"
        # bader, dos_scf, static: use relaxed geometry from opt/
        contcar = dft_opt(project_dir) / "CONTCAR"
        if contcar.exists():
            return contcar
        return dft_opt(project_dir) / "POSCAR"

    def _run_bader_program(self, work_dir: Path) -> None:
        """Run Henkelman bader program on VASP AECCAR/CHGCAR outputs to produce ACF.dat."""
        import subprocess as _sp
        bader_bin = self.hpc_path("bader_bin") or "bader"
        chgcar = work_dir / "CHGCAR"
        aeccar2 = work_dir / "AECCAR2"
        if not chgcar.exists():
            log.warning("[h01_dft] bader: CHGCAR not found in %s", work_dir)
            return
        ref_arg = ["-ref", str(aeccar2)] if aeccar2.exists() else []
        cmd = [bader_bin, str(chgcar)] + ref_arg
        log.info("[h01_dft] Running bader: %s", " ".join(cmd))
        try:
            r = _sp.run(cmd, cwd=str(work_dir), capture_output=True, text=True, timeout=300)
            if (work_dir / "ACF.dat").exists():
                log.info("[h01_dft] bader ACF.dat written (%s)", work_dir)
            else:
                log.warning("[h01_dft] bader ran (rc=%d) but ACF.dat missing; stderr: %s",
                            r.returncode, r.stderr[:200])
        except Exception as exc:
            log.warning("[h01_dft] bader program failed: %s", exc)

    @staticmethod
    def _make_delithiated_poscar(source: Path, dest: Path, mobile_ion: str,
                                  fraction: float = 0.5) -> int:
        """Remove fraction of mobile_ion atoms from POSCAR. Returns count removed."""
        lines = source.read_text().splitlines()
        # Handle optional Selective Dynamics line
        header_end = 8
        if lines[7].strip().lower().startswith("s"):
            header_end = 9
        elem_line  = lines[5].split()
        count_line = lines[6].split()
        new_counts = list(count_line)
        n_removed  = 0
        for i, elem in enumerate(elem_line):
            if elem == mobile_ion:
                n_total  = int(count_line[i])
                n_keep   = max(1, int(round(n_total * (1.0 - fraction))))
                n_removed = n_total - n_keep
                new_counts[i] = str(n_keep)
        lines[6] = "  ".join(new_counts)
        # Rebuild atom coordinate section
        atom_lines = lines[header_end:]
        result_atoms: list[str] = []
        idx = 0
        for elem, cnt_str in zip(elem_line, count_line):
            n = int(cnt_str)
            block = atom_lines[idx: idx + n]
            idx += n
            if elem == mobile_ion:
                n_keep = max(1, int(round(n * (1.0 - fraction))))
                block = block[:n_keep]
            result_atoms.extend(block)
        dest.write_text("\n".join(lines[:header_end] + result_atoms) + "\n")
        return n_removed

    @staticmethod
    def _generate_potcar_dft(poscar: Path, potcar: Path) -> None:
        """Generate POTCAR from POTPAW library for the elements in poscar, writing to potcar."""
        from hpca.core.vasp_job import generate_potcar
        try:
            generate_potcar(poscar, potcar)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            log.error("[h01_dft] POTCAR generation failed: %s", exc)

    @staticmethod
    def _outcar_converged(outcar_path: Path) -> bool:
        """Return True if OUTCAR contains 'reached required accuracy' in the last 200 lines."""
        if not outcar_path.exists():
            return False
        try:
            # Read last 200 lines — convergence message can appear 50+ lines before EOF
            # due to memory stats and timing info written after the SCF loop
            tail = outcar_path.read_text(errors="replace").splitlines()[-200:]
            return any("reached required accuracy" in l for l in tail)
        except Exception:
            return False
