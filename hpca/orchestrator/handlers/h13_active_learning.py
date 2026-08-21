"""
h13_active_learning.py — MLIP active learning loop handler.

Runs after h04_mlip. Checks RMSE thresholds; if above threshold, identifies
gap temperatures, submits new AIMD jobs, merges datasets, and retrains.
Up to MAX_CYCLES iterations before giving up and continuing to h05_lammps.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from .base import SimulationHandler
from hpca.core.paths import dft_opt, mlmd_mlff, dft_aimd
from hpca.registry.incar import build_incar as _build_incar, write_incar as _write_incar
from hpca.registry.submission import write_submission as _write_sub
from hpca.io.rmse import rmse_summary as _rmse_summary

if TYPE_CHECKING:
    from hpca.orchestrator.state_tracker import ProjectState

log = logging.getLogger("hpca.orch")
# Layout: see hpca/core/paths.py

E_RMSE_THRESHOLD = 5.0    # meV/atom  (5e-3 eV/atom)
F_RMSE_THRESHOLD = 100.0  # meV/Å     (0.10 eV/Å)
MAX_CYCLES = 3

# CPU DeepMD env read from platform.yaml at call time via cls.platform_config()
# Cross-ref: hpca/config/platform.yaml hpc.deepmd_cpu_venv


class ActiveLearningHandler(SimulationHandler):
    """
    Daemon handler: MLIP active learning loop.

    Lifecycle:
    1. can_run()   — h04_mlip COMPLETE + test results exist + not yet converged + cycle < MAX_CYCLES
    2. submit()    — parse RMSE; if OK → mark converged; else submit new AIMD jobs
    3. check_progress() — when new AIMD jobs done: merge datasets, retrain
    4. is_complete() — converged flag OR max cycles reached
    """

    name = "h13_active_learning"
    is_daemon = True  # runs in orchestrator process, not via sbatch

    # ── Lifecycle methods ──────────────────────────────────────────────────────

    def can_run(self, project_dir: Path, state: "ProjectState") -> bool:
        """h04_mlip COMPLETE + test results present + not converged + cycle < MAX_CYCLES."""
        if state.get_stage("h11_manuscript") != "COMPLETE":
            log.debug("[h13_active_learning] Waiting for h11_manuscript to complete")
            return False
        if state.get_stage("h04_mlip") != "COMPLETE":
            return False

        al_state = state.get_handler(self.name)
        if al_state.get("converged"):
            return False
        if al_state.get("cycle", 0) >= MAX_CYCLES:
            return False

        # Need either lcurve.out or test_results.txt to exist
        mlff_dir = mlmd_mlff(project_dir)
        has_lcurve = (mlff_dir / "lcurve.out").exists()
        has_test   = (mlff_dir / "test_results.txt").exists()
        return has_lcurve or has_test

    def is_complete(self, project_dir: Path, state: "ProjectState") -> bool:
        """Converged flag OR max cycles reached."""
        al_state = state.get_handler(self.name)
        if al_state.get("converged"):
            return True
        if al_state.get("cycle", 0) >= MAX_CYCLES:
            log.info("[h13_al] MAX_CYCLES=%d reached — marking complete", MAX_CYCLES)
            return True
        return False

    def submit(self, project_dir: Path, state: "ProjectState") -> str | None:
        """
        Main entry point after h04_mlip completes.
        1. Parse RMSE. If within threshold → mark converged immediately.
        2. Otherwise: ask AI for gap temperatures, submit AIMD jobs.
        """
        al_state = state.get_handler(self.name)
        cycle = al_state.get("cycle", 0) + 1
        log.info("[h13_al] Active learning cycle %d/%d for %s", cycle, MAX_CYCLES, project_dir.name)

        # Parse RMSE
        test_results = self._parse_test_results(project_dir)
        E_rmse = test_results.get("e_rmse", float("inf"))
        F_rmse = test_results.get("f_rmse", float("inf"))
        n_frames = test_results.get("n_frames", 0)

        log.info("[h13_al] RMSE: E=%.4f meV/atom (threshold=%.1f), F=%.4f meV/Å (threshold=%.1f)",
                 E_rmse, E_RMSE_THRESHOLD, F_rmse, F_RMSE_THRESHOLD)

        # Check convergence
        if E_rmse <= E_RMSE_THRESHOLD and F_rmse <= F_RMSE_THRESHOLD:
            log.info("[h13_al] MLIP quality ACCEPTED (E=%.3f meV/atom, F=%.1f meV/Å) — done",
                     E_rmse, F_rmse)
            state.set_handler(self.name, {
                "converged": True,
                "cycle": cycle,
                "e_rmse": E_rmse,
                "f_rmse": F_rmse,
                "n_frames": n_frames,
            })
            state.set_stage(self.name, "COMPLETE",
                            converged=True, cycle=cycle, e_rmse=E_rmse, f_rmse=F_rmse)
            return None

        # Not converged — query AI for gap temperatures
        try:
            from ai_advisor import AIAdvisor
            advisor = AIAdvisor()
            al_plan = advisor.plan_active_learning(project_dir, E_rmse / 1000.0, F_rmse / 1000.0)
        except Exception as exc:
            log.warning("[h13_al] AI advisor unavailable: %s — using fallback gap fill", exc)
            al_plan = {"add_temperatures": [], "add_configurations": 200, "reasoning": "fallback"}

        add_temps = al_plan.get("add_temperatures", [])
        add_configs = al_plan.get("add_configurations", 200)
        log.info("[h13_al] AI plan: add_temps=%s, add_configs=%d, reason=%s",
                 add_temps, add_configs, al_plan.get("reasoning", ""))

        # If AI gave no temperatures, use gap-filling heuristic
        if not add_temps:
            yaml_data = self.read_project_yaml(project_dir)
            existing_temps = yaml_data.get("aimd_temps", [300, 600])
            add_temps = self._identify_gap_temperatures(existing_temps)
            log.info("[h13_al] Fallback gap temps: %s", add_temps)

        if not add_temps:
            log.warning("[h13_al] No gap temperatures identified — marking COMPLETE")
            state.set_stage(self.name, "COMPLETE",
                            converged=False, cycle=cycle,
                            note="No gap temperatures to add")
            return None

        # Submit AIMD jobs for new temperatures
        new_jobs: dict[str, str] = {}
        for T in add_temps[:2]:  # cap at 2
            job_id = self._submit_aimd_for_temperature(project_dir, T, add_configs)
            if job_id:
                new_jobs[str(T)] = job_id
                log.info("[h13_al] Submitted AIMD T=%s K, job=%s", T, job_id)

        if not new_jobs:
            log.warning("[h13_al] No new AIMD jobs submitted — marking COMPLETE (no further action)")
            state.set_stage(self.name, "COMPLETE",
                            converged=False, cycle=cycle,
                            note="AIMD submission failed")
            return None

        state.set_stage(self.name, "RUNNING",
                        cycle=cycle,
                        e_rmse=E_rmse,
                        f_rmse=F_rmse,
                        n_frames=n_frames,
                        new_aimd_jobs=new_jobs,
                        add_configs=add_configs)
        return next(iter(new_jobs.values()))

    def check_progress(self, project_dir: Path, state: "ProjectState") -> None:
        """Check if new AIMD jobs from active learning cycle are done; if so, retrain."""
        al_state = state.get_handler(self.name)
        new_jobs: dict = al_state.get("new_aimd_jobs", {})
        if not new_jobs:
            return

        add_configs = al_state.get("add_configs", 200)

        # Check each new AIMD job
        all_done = True
        still_running = []
        for T_str, job_id in new_jobs.items():
            T = int(T_str)
            xdatcar = dft_aimd(project_dir, f"al_{T}") / "XDATCAR"
            steps = self.grep_count(xdatcar, "Direct configuration=") if xdatcar.exists() else 0
            if steps >= add_configs:
                log.info("[h13_al] New AIMD T=%s done: %d steps", T, steps)
            elif self.job_alive(job_id):
                all_done = False
                still_running.append(T_str)
                log.debug("[h13_al] New AIMD T=%s still running (job=%s, steps=%d/%d)",
                          T, job_id, steps, add_configs)
            else:
                # Job dead but not enough steps — consider it done anyway (will have partial data)
                log.warning("[h13_al] New AIMD T=%s job %s dead with %d/%d steps",
                            T, job_id, steps, add_configs)

        if still_running:
            log.info("[h13_al] Waiting for new AIMD jobs: %s", still_running)
            return

        # All new AIMD jobs done — merge datasets and retrain
        log.info("[h13_al] All new AIMD jobs complete — merging datasets and retraining")
        self._merge_and_retrain(project_dir, state)

    # ── Internal Helpers ───────────────────────────────────────────────────────

    def _parse_test_results(self, project_dir: Path) -> dict:
        """Parse RMSE from mlff/. Returns {e_rmse: meV/atom, f_rmse: meV/Å, n_frames: int}."""
        result = _rmse_summary(mlmd_mlff(project_dir))
        if not result:
            log.warning("[h13_al] Could not parse RMSE — assuming not converged")
            return {"e_rmse": float("inf"), "f_rmse": float("inf"), "n_frames": 0}
        return {
            "e_rmse":   result.get("e_rmse_eV",   float("inf")) * 1000,
            "f_rmse":   result.get("f_rmse_eV_A", float("inf")) * 1000,
            "n_frames": result.get("n_frames", 0),
        }

    def _identify_gap_temperatures(self, existing_temps: list) -> list:
        """Return up to 2 new temperatures filling gaps or extending range."""
        if not existing_temps:
            return [500, 700]

        sorted_T = sorted(set(int(t) for t in existing_temps))
        candidates = []

        # Find the largest gap between consecutive points
        gaps = []
        for i in range(len(sorted_T) - 1):
            gap_size = sorted_T[i + 1] - sorted_T[i]
            mid = (sorted_T[i] + sorted_T[i + 1]) // 2
            if gap_size > 100 and mid not in sorted_T:
                gaps.append((gap_size, mid))
        gaps.sort(reverse=True)
        for _, mid_T in gaps[:2]:
            candidates.append(mid_T)

        # Extend range if few candidates found
        if len(candidates) < 2 and sorted_T[0] > 250:
            candidates.insert(0, max(200, sorted_T[0] - 100))
        if len(candidates) < 2 and sorted_T[-1] < 900:
            candidates.append(min(1000, sorted_T[-1] + 100))

        # Remove duplicates vs existing
        new_temps = [T for T in candidates[:2] if T not in sorted_T]
        return new_temps

    def _submit_aimd_for_temperature(self, project_dir: Path, T: int,
                                     nsw: int = 200) -> str | None:
        """
        Write VASP AIMD inputs for temperature T and submit via sbatch.
        Uses al_{T} subdirectory to avoid polluting original aimd/ dirs.
        Returns job_id string or None.
        """
        yaml_data = self.read_project_yaml(project_dir)
        project_name = yaml_data.get("name", project_dir.name)
        encut = yaml_data.get("encut_aimd", 400.8232)

        aimd_dir = dft_aimd(project_dir, f"al_{T}")
        aimd_dir.mkdir(parents=True, exist_ok=True)

        # Copy POSCAR from dft/opt/
        poscar_src = (dft_opt(project_dir) / "CONTCAR"
                      if (dft_opt(project_dir) / "CONTCAR").exists()
                      else dft_opt(project_dir) / "POSCAR")
        if poscar_src.exists():
            shutil.copy(poscar_src, aimd_dir / "POSCAR")
        else:
            log.error("[h13_al] No POSCAR source found at %s", dft_opt(project_dir))
            return None

        # Copy POTCAR
        potcar_src = dft_opt(project_dir) / "POTCAR"
        if potcar_src.exists():
            shutil.copy(potcar_src, aimd_dir / "POTCAR")

        # Count atoms (build_incar uses this to set LREAL=F for small cells)
        try:
            poscar_lines = (aimd_dir / "POSCAR").read_text().splitlines()
            n_atoms = sum(int(x) for x in poscar_lines[6].split())
        except Exception:
            n_atoms = 0

        _write_incar(aimd_dir / "INCAR", _build_incar(
            "nvt_production",
            natoms=n_atoms,
            encut=encut,
            nsw=nsw,
            tebeg=T,
            teend=T,
            extra={"NPAR": 8, "LORBIT": 11, "WEIMIN": 0,
                   "SYSTEM": f"{project_name}_AL_AIMD_{T}K"},
        ))

        # Write KPOINTS
        (aimd_dir / "KPOINTS").write_text(
            "Automatic mesh\n0\nGamma\n  1  1  1\n  0   0   0\n"
        )

        # Write submission script
        sub_sh = aimd_dir / "sub.sh"
        _write_sub(sub_sh, "vasp", f"{project_name}_al_{T}K",
                   nodes=2, ntasks=104, time="24:00:00")

        return self.sbatch(sub_sh, cwd=aimd_dir)

    def _merge_and_retrain(self, project_dir: Path, state: "ProjectState") -> None:
        """
        Merge new AIMD frames into the existing DeepMD dataset and submit a retraining job.

        Steps: parse al_{T}/ XDATCAR/OUTCAR → concatenate coord/box/energy/force npy files
        → write updated deepmd_input.json → sbatch sub_retrain.sh → reset h04_mlip to RUNNING.
        """
        try:
            import numpy as np
        except ImportError:
            log.error("[h13_al] numpy unavailable — cannot merge dataset")
            return

        mlff_dir = mlmd_mlff(project_dir)
        set_dir = mlff_dir / "00.data" / "set.000"

        # Backup original dataset
        backup_dir = mlff_dir / "00.data" / "set.000_backup"
        if not backup_dir.exists() and set_dir.exists():
            shutil.copytree(str(set_dir), str(backup_dir))
            log.info("[h13_al] Backed up original dataset to %s", backup_dir)

        # Load existing data
        def _load_npy(path: Path):
            """Load a .npy array from path, returning None if the file does not exist."""
            return np.load(str(path)) if path.exists() else None

        existing_coord  = _load_npy(set_dir / "coord.npy")
        existing_box    = _load_npy(set_dir / "box.npy")
        existing_energy = _load_npy(set_dir / "energy.npy")
        existing_force  = _load_npy(set_dir / "force.npy")

        from hpca.tools.deepmd import _parse_xdatcar, _parse_outcar_ef

        al_state = state.get_handler(self.name)
        new_jobs: dict = al_state.get("new_aimd_jobs", {})

        new_coords: list = []
        new_boxes: list = []
        new_energies: list = []
        new_forces: list = []
        n_atoms = existing_coord.shape[1] // 3 if existing_coord is not None else 0

        for T_str in new_jobs:
            T = int(T_str)
            aimd_dir = dft_aimd(project_dir, f"al_{T}")
            xdatcar = aimd_dir / "XDATCAR"
            outcar  = aimd_dir / "OUTCAR"

            if not xdatcar.exists():
                log.warning("[h13_al] XDATCAR missing for al_%d", T)
                continue

            frames, box_vecs = _parse_xdatcar(xdatcar)
            if not frames:
                continue

            if n_atoms == 0:
                n_atoms = frames[0].shape[0] // 3  # frames are flat (3*n_atoms,)

            energies, forces = _parse_outcar_ef(outcar, len(frames))
            n_use = min(len(frames), len(energies)) if energies else len(frames)

            for i in range(n_use):
                new_coords.append(frames[i].reshape(-1))
                new_boxes.append(box_vecs[i].reshape(-1))
                if energies:
                    new_energies.append(energies[i])
                if forces and i < len(forces):
                    new_forces.append(forces[i].reshape(-1))

            log.info("[h13_al] al_%d: %d frames added", T, n_use)

        if not new_coords:
            log.warning("[h13_al] No new frames parsed — retraining skipped")
            return

        # Merge with existing
        def _merge(existing, new_list):
            """Concatenate new_list (as float64 array) onto existing array, or return new array alone."""
            new_arr = np.array(new_list, dtype=np.float64)
            if existing is not None:
                return np.concatenate([existing, new_arr], axis=0)
            return new_arr

        merged_coord  = _merge(existing_coord,  new_coords)
        merged_box    = _merge(existing_box,    new_boxes)
        merged_energy = _merge(existing_energy, new_energies) if new_energies else existing_energy
        merged_force  = _merge(existing_force,  new_forces)  if new_forces  else existing_force

        # Write merged dataset back
        set_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(set_dir / "coord.npy"),  merged_coord)
        np.save(str(set_dir / "box.npy"),    merged_box)
        if merged_energy is not None:
            np.save(str(set_dir / "energy.npy"), merged_energy)
        if merged_force is not None:
            np.save(str(set_dir / "force.npy"), merged_force)

        n_new = len(new_coords)
        n_total = merged_coord.shape[0]
        log.info("[h13_al] Merged dataset: +%d frames → %d total frames", n_new, n_total)

        # Submit retraining job
        yaml_data = self.read_project_yaml(project_dir)
        project_name = yaml_data.get("name", project_dir.name)

        # Load existing deepmd_input.json or build minimal one
        input_json_path = mlff_dir / "deepmd_input.json"
        if input_json_path.exists():
            try:
                inp = json.loads(input_json_path.read_text())
                # Update numb_steps for additional training
                inp["training"]["numb_steps"] = inp["training"].get("numb_steps", 500000)
            except Exception:
                inp = None
        else:
            inp = None

        if inp is None:
            log.warning("[h13_al] No deepmd_input.json found — cannot retrain")
            return

        # Save updated input
        input_json_path.write_text(json.dumps(inp, indent=2))

        # Write retrain submission script
        sub_sh = mlff_dir / "sub_retrain.sh"
        _write_sub(sub_sh, "deepmd_al", f"{project_name}_al_retrain", mlff_dir=mlff_dir)

        job_id = self.sbatch(sub_sh, cwd=mlff_dir)
        if job_id:
            log.info("[h13_al] Submitted retrain job=%s", job_id)
            al_state = state.get_handler(self.name)
            state.set_handler(self.name, {
                "retrain_job": job_id,
                "n_frames_merged": int(n_total),
                "new_aimd_jobs": {},  # clear so check_progress won't loop
            })
            # Reset h04_mlip to RUNNING so the orchestrator waits for retrain
            state.set_stage("h04_mlip", "RUNNING", job=job_id,
                            note="AL retrain cycle")
        else:
            log.error("[h13_al] Retrain submission failed")


