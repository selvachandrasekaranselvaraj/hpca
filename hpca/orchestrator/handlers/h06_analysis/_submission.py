"""
_submission.py — submit one h06_analysis variant as its own SLURM job via the
canonical submission registry (hpca.registry.submission).
"""
from __future__ import annotations

from pathlib import Path

from hpca.registry.submission import write_submission

from ..base import SimulationHandler


def submit_variant_job(project_dir: Path, variant: str, job_name: str) -> str | None:
    """Write the analysis_cpu submission script for one variant and sbatch it."""
    out_dir = project_dir / "Analysis" / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_path = out_dir / "sub.sh"
    write_submission(sub_path, "analysis_cpu", job_name,
                     project_dir=project_dir, variant=variant, time_key="analysis_cpu")
    return SimulationHandler.sbatch(sub_path, cwd=out_dir)
