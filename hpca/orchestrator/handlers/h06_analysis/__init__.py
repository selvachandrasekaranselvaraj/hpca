"""
h06_analysis — Transport and structural analysis.

Submits one analysis_cpu SLURM job per variant (cmd / mlmd_dft / combined) —
see _worker.py for what each job runs and hpca.registry.submission for the
submission template. Moving this off the daemon process keeps trajectory
loading (MSD/RDF/Van Hove/coordination/ion-pairs/VACF) from competing with
the daemon's own memory budget.
CLAUDE.md rule: every figure saves both PNG and CSV.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..base import SimulationHandler
from ._sources import collect_sources
from ._submission import submit_variant_job

if TYPE_CHECKING:
    from hpca.orchestrator.state_tracker import ProjectState

log = logging.getLogger("hpca.orch")

VARIANTS = ("cmd", "mlmd_dft", "combined")


class AnalysisHandler(SimulationHandler):
    """Submits one analysis_cpu SLURM job per variant; completion is file-sentinel based."""

    name = "h06_analysis"
    is_daemon = False

    # ── Three-variant analysis: cmd / mlmd_dft / combined ────────────────────

    @staticmethod
    def _collect_sources(project_dir: Path, variant: str,
                         min_dump: int = 100_000) -> dict[int, Path]:
        """Return {T: traj_path} for the given analysis variant.

        Variants:
          cmd       — CMD NVT dumps only
          mlmd_dft  — MLMD dumps preferred over AIMD XDATCARs
          combined  — MLMD > AIMD > CMD per temperature
        """
        return collect_sources(project_dir, variant, min_dump)

    def can_run(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True when at least one variant has a sufficiently long trajectory."""
        min_dump = int(self.plat("analysis_defaults", "min_dump_bytes", 100_000))
        return any(bool(self._collect_sources(project_dir, v, min_dump)) for v in VARIANTS)

    def is_complete(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True when every variant with trajectories has an up-to-date arrhenius.csv."""
        min_dump = int(self.plat("analysis_defaults", "min_dump_bytes", 100_000))
        any_variant = False
        for variant in VARIANTS:
            sources = self._collect_sources(project_dir, variant, min_dump)
            if not sources:
                continue
            any_variant = True
            arr = project_dir / "Analysis" / variant / "arrhenius.csv"
            if not arr.exists():
                return False
            if any(p.stat().st_mtime > arr.stat().st_mtime for p in sources.values()):
                log.info("[h06_analysis] new data for variant=%s — re-running", variant)
                return False
        return any_variant

    def submit(self, project_dir: Path, state: "ProjectState") -> str | None:
        """Submit an analysis_cpu job for every variant that needs (re-)running."""
        return self._ensure_variant_jobs(project_dir, state)

    def check_progress(self, project_dir: Path, state: "ProjectState") -> None:
        """Submit jobs for any variant whose data went stale or became ready since the last poll."""
        self._ensure_variant_jobs(project_dir, state)

    def _ensure_variant_jobs(self, project_dir: Path, state: "ProjectState") -> str | None:
        """Submit one job per variant that is missing/stale and not already running.

        Mirrors ClassicalMDHandler's per-subtask "jobs" dict pattern: submit()
        and check_progress() share this so new variants (e.g. a molarity
        finishing later) get picked up on any poll, not only the first.
        """
        min_dump = int(self.plat("analysis_defaults", "min_dump_bytes", 100_000))
        project_name = self.read_project_yaml(project_dir).get("name", project_dir.name)
        handler_state = state.get_handler(self.name)
        submitted_jobs: dict = dict(handler_state.get("jobs", {}))
        first_job: str | None = None

        for variant in VARIANTS:
            sources = self._collect_sources(project_dir, variant, min_dump)
            if not sources:
                continue
            arr = project_dir / "Analysis" / variant / "arrhenius.csv"
            up_to_date = (arr.exists()
                         and not any(p.stat().st_mtime > arr.stat().st_mtime for p in sources.values()))
            if up_to_date or self.job_alive(submitted_jobs.get(variant)):
                continue
            job_id = submit_variant_job(project_dir, variant, f"{project_name}_h06_{variant}")
            if job_id:
                submitted_jobs[variant] = job_id
                first_job = first_job or job_id
                log.info("[h06_analysis] variant=%s submitted → job %s", variant, job_id)
            else:
                log.warning("[h06_analysis] variant=%s submission failed — retry next poll", variant)

        if submitted_jobs != handler_state.get("jobs", {}):
            state.set_stage(self.name, "RUNNING", jobs=submitted_jobs)
        return first_job

    def on_complete(self, project_dir: Path, state: "ProjectState") -> None:
        """Write the canonical transport.json/phase_transitions.json sentinels.

        Reads each variant's arrhenius.csv back from disk — each variant ran
        as its own SLURM job, so there is no in-memory result to reuse here.
        """
        results_dir = project_dir / "results" / "data"
        results_dir.mkdir(parents=True, exist_ok=True)
        transport: dict = {"variants": {}}
        n_temps: dict[str, int] = {}
        for variant in VARIANTS:
            arr_csv = project_dir / "Analysis" / variant / "arrhenius.csv"
            if not arr_csv.exists():
                continue
            D_per_T = self._read_D_per_T(arr_csv)
            entry: dict = {"D_per_T": D_per_T}
            Ea = self._read_Ea(arr_csv)
            if Ea is not None:
                entry["Ea_eV"] = Ea
            transport["variants"][variant] = entry
            n_temps[variant] = len(D_per_T)
        (results_dir / "transport.json").write_text(json.dumps(transport, indent=2))
        (results_dir / "phase_transitions.json").write_text("{}\n")
        state.set_handler(self.name, {"variants": list(transport["variants"]), "n_temps": n_temps})
        log.info("[h06_analysis] COMPLETE for %s (variants: %s)",
                 project_dir.name, list(transport["variants"]))

    @staticmethod
    def _read_D_per_T(arr_csv: Path) -> dict[str, float]:
        """Return {T_K: D_m2s} parsed back from a variant's arrhenius.csv."""
        try:
            import numpy as np
            arr = np.loadtxt(str(arr_csv), delimiter=",", skiprows=1)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return {str(int(row[0])): float(row[1]) for row in arr}
        except Exception:
            return {}

    @staticmethod
    def _read_Ea(arr_csv: Path) -> float | None:
        """Return the fitted activation energy (eV) from a variant's arrhenius.csv, if present."""
        try:
            import numpy as np
            arr = np.loadtxt(str(arr_csv), delimiter=",", skiprows=1)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] > 4:
                return float(arr[0, 4])
        except Exception:
            pass
        return None
