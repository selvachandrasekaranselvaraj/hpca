"""
h04_mlip.py — MLIP training handler: DeepMD + MACE (SLURM gpu-h100).

Trains DeepMD or MACE model from AIMD-derived dataset.
HPC paths and account read from platform.yaml (hpc.* section).
Cross-ref: hpca/config/platform.yaml, hpca/orchestrator/handlers/base.py
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .base import SimulationHandler
from hpca.core.categories import is_sse as _cat_is_sse
from hpca.core.deepmd_job import (
    write_deepmd_input as _dj_write_input,
    write_mace_config as _dj_mace_cfg,
)
from hpca.registry.submission import write_submission as _write_sub

if TYPE_CHECKING:
    from hpca.orchestrator.state_tracker import ProjectState

log = logging.getLogger("hpca.orch")

# Layout: see hpca/core/paths.py
from hpca.core.paths import mlmd_mlff, load_platform_config as _lpc
from hpca.io.rmse import (
    parse_deepmd_lcurve as _parse_lcurve,
    parse_deepmd_lcurve_history as _parse_lcurve_history,
    converged as _rmse_converged,
)
from hpca.core.type_map import read_type_map as _read_type_map_file


def _mp(key: str, default=None):
    """Read a key from platform.yaml mlip_defaults section."""
    return _lpc().get("mlip_defaults", {}).get(key, default)


# All HPC paths read from platform.yaml via cls.hpc_path() / cls.platform_config()
# Cross-ref: hpca/config/platform.yaml hpc.* section


class MLIPHandler(SimulationHandler):
    """SLURM handler: submits DeepMD and/or MACE training job on 4xH100."""

    name = "h04_mlip"
    is_daemon = False

    @staticmethod
    def _default_backend(yaml_data: dict) -> str:
        """SSE projects train both backends by default; others use deepmd."""
        cat = yaml_data.get("category", "")
        if _cat_is_sse(cat):
            return "both"
        return "deepmd"

    def can_run(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True when an energy.npy DeepMD dataset exists in mlmd/mlff/dataset_data/."""
        return (
            mlmd_mlff(project_dir) / "dataset_data" / "set.000" / "energy.npy"
        ).exists()

    def is_complete(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True when the trained model file(s) required by the configured backend exist."""
        yaml_data = self.read_project_yaml(project_dir)
        backend = yaml_data.get("mlip_backend", self._default_backend(yaml_data)).lower()
        mlff_dir = mlmd_mlff(project_dir)
        deepmd_done = (mlff_dir / "pot_com.pb").exists()
        mace_done   = (mlff_dir / "MACE_model.pt").exists()
        if backend == "both":
            return deepmd_done and mace_done
        if backend == "mace":
            return mace_done
        return deepmd_done

    def submit(self, project_dir: Path, state: "ProjectState") -> str | None:
        """Prepare training inputs and submit DeepMD and/or MACE SLURM job(s); return first job ID."""
        yaml_data = self.read_project_yaml(project_dir)
        mlff_dir = mlmd_mlff(project_dir)
        mlff_dir.mkdir(parents=True, exist_ok=True)

        backend = yaml_data.get("mlip_backend", self._default_backend(yaml_data)).lower()

        if backend == "both":
            from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
            deepmd_done = (mlff_dir / "pot_com.pb").exists()
            mace_done   = (mlff_dir / "MACE_model.pt").exists()
            submit_fns: dict[str, callable] = {}
            if not deepmd_done:
                submit_fns["deepmd"] = lambda: self._submit_deepmd(project_dir, yaml_data, mlff_dir)
            if not mace_done:
                submit_fns["mace"] = lambda: self._submit_mace(project_dir, yaml_data, mlff_dir)
            jobs: dict[str, str] = {}
            if submit_fns:
                with ThreadPoolExecutor(max_workers=2) as _pool:
                    fut_to_be = {_pool.submit(fn): be for be, fn in submit_fns.items()}
                    for fut in _as_completed(fut_to_be):
                        be = fut_to_be[fut]
                        try:
                            jid = fut.result()
                        except Exception as exc:
                            log.error("[h04_mlip] %s submit failed: %s", be.upper(), exc)
                            jid = None
                        if jid:
                            jobs[be] = jid
                            log.info("[h04_mlip] Submitted %s training job=%s", be.upper(), jid)
            if jobs:
                first = next(iter(jobs.values()))
                state.set_stage("h04_mlip", "RUNNING", job=first, backend="both", jobs=jobs)
            return next(iter(jobs.values())) if jobs else None

        if backend == "mace":
            job_id = self._submit_mace(project_dir, yaml_data, mlff_dir)
        else:
            job_id = self._submit_deepmd(project_dir, yaml_data, mlff_dir)

        if job_id:
            state.set_stage("h04_mlip", "RUNNING", job=job_id, backend=backend)
            log.info("[h04_mlip] Submitted %s training job=%s", backend.upper(), job_id)
        return job_id

    def _prepare_deepmd_data(
        self, mlff_dir: Path, yaml_data: dict, numb_steps: int
    ) -> list[str]:
        """Split dataset_data into train/val, write 01.train/deepmd_input.json.

        Follows dp.py/dp_model.sh: uses dpdata to load existing npy data from
        dataset_data/, splits 95%/5% train/val, saves to 00.data/training_data
        and 00.data/validation_data, generates 01.train/deepmd_input.json with
        relative paths. Returns type_map list.
        """
        import numpy as np

        dataset_data = mlff_dir / "dataset_data"
        data_dir = mlff_dir / "00.data"
        data_dir.mkdir(exist_ok=True)

        try:
            import dpdata
            data = dpdata.LabeledSystem(str(dataset_data), fmt="deepmd/npy")
            n_frames = len(data)
            log.info("[h04_mlip] Loaded %d frames from %s", n_frames, dataset_data)

            n_val = max(1, int(n_frames * 0.05))
            idx_val = list(np.random.choice(n_frames, size=n_val, replace=False))
            idx_train = list(set(range(n_frames)) - set(idx_val))
            data_train = data.sub_system(idx_train)
            data_val = data.sub_system(idx_val)

            for d in [data_train, data_val]:
                for key in ["coords", "forces", "virials", "cells", "energies"]:
                    if key in d.data and d.data[key] is not None:
                        d.data[key] = d.data[key].astype(np.float32)

            data_train.to_deepmd_npy(str(data_dir / "training_data"))
            data_val.to_deepmd_npy(str(data_dir / "validation_data"))
            log.info("[h04_mlip] train=%d val=%d frames → 00.data/", len(idx_train), n_val)
        except Exception as exc:
            log.error("[h04_mlip] dpdata split failed: %s — using dataset_data directly", exc)
            import shutil as _sh
            for sub in ("training_data", "validation_data"):
                dst = data_dir / sub
                if not dst.exists():
                    _sh.copytree(str(dataset_data), str(dst))

        # type_map from dataset_data/type_map.raw
        type_map_src = dataset_data / "type_map.raw"
        type_map_dst = data_dir / "type_map.raw"
        type_map = _read_type_map_file(type_map_src) or yaml_data.get("type_map", ["Li", "C"])
        if type_map_src.exists():
            type_map_dst.write_text("\n".join(type_map) + "\n")

        # Build 01.train/deepmd_input.json with relative paths (../00.data/...)
        train_dir = mlff_dir / "01.train"
        train_dir.mkdir(exist_ok=True)

        input_json = _dj_write_input(train_dir, type_map, mlff_dir, numb_steps)
        log.info("[h04_mlip] Wrote %s (numb_steps=%d, type_map=%s)", input_json, numb_steps, type_map)
        return type_map

    def _submit_deepmd(self, project_dir: Path, yaml_data: dict, mlff_dir: Path) -> str | None:
        """Prepare DeepMD training data and submit the CPU training job; return job ID."""
        numb_steps = self.resolve(project_dir, "mlip_numb_steps", _mp("numb_steps", 500000))
        self._prepare_deepmd_data(mlff_dir, yaml_data, numb_steps)

        project_name = yaml_data.get("name", project_dir.name)
        sub_sh = mlff_dir / "sub_deepmd.sh"
        self._write_deepmd_sub(sub_sh, project_name, mlff_dir)

        return self.sbatch(sub_sh, cwd=mlff_dir)

    def _submit_mace(self, project_dir: Path, yaml_data: dict, mlff_dir: Path) -> str | None:
        """Submit MACE training job."""
        project_name = yaml_data.get("name", project_dir.name)

        # Write MACE config yaml
        type_map_file = mlff_dir / "00.data" / "type_map.raw"
        type_map = []
        if type_map_file.exists():
            type_map = [l.strip() for l in type_map_file.read_text().splitlines() if l.strip()]

        cfg_path = mlff_dir / "mace_config.yaml"
        _dj_mace_cfg(cfg_path, project_name, type_map, mlff_dir)

        sub_sh = mlff_dir / "sub_mace.sh"
        self._write_mace_sub(sub_sh, project_name, mlff_dir, cfg_path)

        return self.sbatch(sub_sh, cwd=mlff_dir)

    def check_progress(self, project_dir: Path, state: "ProjectState") -> None:
        """Parse lcurve.out RMSE and detect dead training jobs; log plateau warnings."""
        mlff_dir = mlmd_mlff(project_dir)
        handler_state = state.get_handler("h04_mlip")
        backend = handler_state.get("backend", "deepmd")

        if backend == "both":
            # Check each backend job independently; fail only if a job died with no output
            jobs: dict = handler_state.get("jobs", {})
            for be, jid in jobs.items():
                model = "pot_com.pb" if be == "deepmd" else "MACE_model.pt"
                if not (mlff_dir / model).exists() and jid and not self.job_alive(jid):
                    log.error("[h04_mlip] %s job %s dead, no model — FAILED", be.upper(), jid)
                    state.set_stage("h04_mlip", "FAILED",
                                    error=f"{be} job {jid} died without producing a model")
                    return
            deepmd_done = (mlff_dir / "pot_com.pb").exists()
            mace_done   = (mlff_dir / "MACE_model.pt").exists()
            log.info("[h04_mlip] dual-backend: deepmd=%s mace=%s",
                     "DONE" if deepmd_done else "training", "DONE" if mace_done else "training")
            return

        # Single-backend dead-job detection
        job_id = handler_state.get("job") or handler_state.get("job_id")
        pot_exists = (mlff_dir / "pot_com.pb").exists() or (mlff_dir / "MACE_model.pt").exists()
        if job_id and not self.job_alive(job_id) and not pot_exists:
            log.error("[h04_mlip] Training job %s dead and no output model — FAILED", job_id)
            state.set_stage("h04_mlip", "FAILED", error=f"job {job_id} died without producing a model")
            return

        lcurve_path = mlff_dir / "01.train" / "lcurve.out"
        if not lcurve_path.exists():
            lcurve_path = mlff_dir / "lcurve.out"  # legacy fallback
        if not lcurve_path.exists():
            return

        rmse = _parse_lcurve(lcurve_path)
        if not rmse:
            return

        accept = _rmse_converged(rmse)
        log.info("[h04_mlip] Step=%d  E_RMSE=%.4f eV/atom  F_RMSE=%.4f eV/Å  (%s)",
                 rmse["step"], rmse["e_rmse_eV"], rmse["f_rmse_eV_A"],
                 "ACCEPT" if accept else "still training")
        self._check_rmse_plateau(lcurve_path, rmse["step"])
        state.set_handler("h04_mlip", {
            "step":    rmse["step"],
            "e_rmse":  rmse["e_rmse_eV"],
            "f_rmse":  rmse["f_rmse_eV_A"],
            "accept":  accept,
        })

    @staticmethod
    def _check_rmse_plateau(lcurve_path: Path, current_step: int,
                            window: int = 5000, min_improvement: float = 0.01) -> None:
        """Warn if energy RMSE improvement over the last `window` steps is below `min_improvement`."""
        history = _parse_lcurve_history(lcurve_path)
        if len(history) < 2:
            return
        cutoff = current_step - window
        recent = [r for r in history if r["step"] >= cutoff]
        older  = [r for r in history if r["step"] < cutoff]
        if not recent or not older:
            return
        e_now  = recent[-1]["e_rmse_eV"]
        e_then = older[-1]["e_rmse_eV"]
        if e_then > 0 and (e_then - e_now) / e_then < min_improvement:
            log.warning(
                "[h04_mlip] RMSE plateau: E_RMSE %.4f → %.4f eV/atom over last %d steps "
                "(< %.0f%% improvement). Consider reducing learning rate.",
                e_then, e_now, window, min_improvement * 100,
            )

    def auto_fix(self, project_dir: Path, state: "ProjectState") -> bool:
        """Detect DeepMD training error, patch deepmd_input.json, and resubmit the job."""
        mlff_dir = mlmd_mlff(project_dir)
        train_log = mlff_dir / "train.log"

        try:
            from hpca.orchestrator.auto_fix import (
                detect_deepmd_error, fix_deepmd_input,
                within_fix_budget, increment_fix_count,
            )
        except ImportError:
            log.error("[h04_mlip] Cannot import auto_fix module")
            return False

        if not within_fix_budget(state, "h04_mlip"):
            log.warning("[h04_mlip] auto-fix budget exhausted — marking FAILED")
            state.set_stage("h04_mlip", "FAILED", error="FIX_BUDGET_EXHAUSTED")
            return False

        yaml_data = self.read_project_yaml(project_dir)
        handler_state = state.get_handler("h04_mlip")
        step_history = handler_state.get("step_history", [])

        err = detect_deepmd_error(train_log, step_history)
        if not err:
            return False

        input_json = mlff_dir / "01.train" / "deepmd_input.json"
        fixed = fix_deepmd_input(input_json, err)
        if not fixed:
            log.warning("[h04_mlip] No fix available for error: %s", err)
            return False

        # Resubmit
        project_name = yaml_data.get("name", project_dir.name)
        sub_sh = mlff_dir / "sub_deepmd.sh"
        self._write_deepmd_sub(sub_sh, project_name, mlff_dir)
        job_id = self.sbatch(sub_sh, cwd=mlff_dir)
        if job_id:
            increment_fix_count(state, "h04_mlip")
            state.set_stage("h04_mlip", "RUNNING", job=job_id, fixed=err)
            log.info("[h04_mlip] Fixed %s and resubmitted job=%s", err, job_id)
            return True
        return False

    # ── Script writers ──────────────────────────────────────────────────────────

    @classmethod
    def _write_deepmd_sub(cls, path: Path, project_name: str, mlff_dir: Path) -> None:
        """Write the SLURM submission script for DeepMD CPU training."""
        _write_sub(path, "deepmd_cpu", f"{project_name}_mlip", mlff_dir=mlff_dir)

    @classmethod
    def _write_mace_sub(cls, path: Path, project_name: str, mlff_dir: Path, cfg_path: Path) -> None:
        """Write the SLURM submission script for MACE GPU training."""
        _write_sub(path, "mace_gpu", f"{project_name}_mace", mlff_dir=mlff_dir, cfg_path=cfg_path)
