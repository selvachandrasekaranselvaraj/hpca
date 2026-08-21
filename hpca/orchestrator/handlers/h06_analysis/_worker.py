"""
_worker.py — standalone entry point for one h06_analysis SLURM job.

Invoked by the analysis_cpu submission template (hpca/registry/submission.py)
as its own job, separate from the daemon, so trajectory loading and
per-temperature analysis no longer compete with the daemon's own memory
budget. Runs exactly one variant (cmd / mlmd_dft / combined) for one project;
AnalysisHandler.submit() in __init__.py submits one job per variant.

    python -m hpca.orchestrator.handlers.h06_analysis._worker \
        --project-dir /path/to/project --variant cmd
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from ._config import resolve_analysis_config
from ._sources import collect_sources
from ._variant import run_variant

log = logging.getLogger("hpca.orch")


def _read_project_yaml(project_dir: Path) -> dict:
    path = project_dir / "project.yaml"
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] hpca.orch (%(module)s): %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=("cmd", "mlmd_dft", "combined"))
    args = parser.parse_args(argv)

    project_dir = args.project_dir.resolve()
    yaml_data = _read_project_yaml(project_dir)
    mobile_ion = yaml_data.get("mobile_ion", "Li")
    anion = yaml_data.get("anion_ion", "")
    cfg = resolve_analysis_config(yaml_data)

    sources = collect_sources(project_dir, args.variant, cfg.min_dump_bytes)
    if not sources:
        log.error("[h06_analysis] no trajectories for variant=%s in %s", args.variant, project_dir)
        return 1

    out_dir = project_dir / "Analysis" / args.variant
    D_per_T = run_variant(project_dir, args.variant, sources, out_dir,
                          mobile_ion, anion, yaml_data, cfg)
    return 0 if D_per_T else 1


if __name__ == "__main__":
    sys.exit(main())
