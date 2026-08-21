"""
_config.py — Resolve h06_analysis settings once per run: platform.yaml
analysis_defaults, with project.yaml analysis: overrides where the original
handler allowed them (rdf_r_max, msd_skip_frac, parallel_temps).
"""
from __future__ import annotations

from dataclasses import dataclass

from hpca.core.paths import load_platform_config


@dataclass(frozen=True)
class AnalysisConfig:
    min_dump_bytes: int
    msd_skip_frac: float
    rdf_r_max: float
    rdf_n_bins: int
    van_hove_r_max: float
    van_hove_n_rbins: int
    van_hove_t_frames: list[int]
    coordination_r_max: float
    coordination_n_bins: int
    ion_pair_r_contact: float
    ion_pair_r_ssip: float
    vacf_max_lag: int
    parallel_temps: int


def plat(section: str, key: str, default=None):
    """Read any top-level section from platform.yaml (standalone — no handler instance needed)."""
    return load_platform_config().get(section, {}).get(key, default)


def resolve_analysis_config(project_yaml: dict) -> AnalysisConfig:
    """Merge platform.yaml's analysis_defaults with project.yaml analysis: overrides."""
    plat = load_platform_config().get("analysis_defaults", {})
    proj = project_yaml.get("analysis", {})

    return AnalysisConfig(
        min_dump_bytes=int(plat.get("min_dump_bytes", 100_000)),
        msd_skip_frac=float(proj.get("msd_skip_frac", plat.get("msd_skip_frac", 0.2))),
        rdf_r_max=float(proj.get("rdf_r_max", plat.get("rdf_r_max", 8.0))),
        rdf_n_bins=int(plat.get("rdf_n_bins", 200)),
        van_hove_r_max=float(plat.get("van_hove_r_max", 10.0)),
        van_hove_n_rbins=int(plat.get("van_hove_n_rbins", 100)),
        van_hove_t_frames=list(plat.get("van_hove_t_frames", [1, 5, 10, 20, 50, 100, 200])),
        coordination_r_max=float(plat.get("coordination_r_max", 8.0)),
        coordination_n_bins=int(plat.get("coordination_n_bins", 200)),
        ion_pair_r_contact=float(plat.get("ion_pair_r_contact", 4.5)),
        ion_pair_r_ssip=float(plat.get("ion_pair_r_ssip", 7.5)),
        vacf_max_lag=int(plat.get("vacf_max_lag", 500)),
        parallel_temps=int(proj.get("parallel_temps", plat.get("parallel_temps", 4))),
    )
