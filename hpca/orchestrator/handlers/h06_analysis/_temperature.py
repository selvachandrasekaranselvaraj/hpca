"""
_temperature.py — Run every per-temperature analysis (MSD, RDF, Van Hove,
coordination, ion pairs, transference, VACF/VDOS) for one trajectory.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ._config import AnalysisConfig
from ._dynamics import run_ion_pairs, run_transference, run_vacf_vdos
from ._parsers import parse_dump_lammps
from ._structure import run_coordination, run_rdf, run_van_hove
from ._transport import run_msd

log = logging.getLogger("hpca.orch")


def analyze_temperature(
    project_dir: Path,
    T: int,
    traj_file: Path,
    output_dir: Path,
    mobile_ion: str,
    anion: str,
    dt_frame_ps: float,
    cfg: AnalysisConfig,
) -> float | None:
    """Run all per-temperature analyses and return the self-diffusion coefficient D (m²/s)."""
    D = run_msd(project_dir, traj_file, T, output_dir, mobile_ion, dt_frame_ps,
               skip_frac=cfg.msd_skip_frac)

    try:
        run_rdf(traj_file, T, output_dir, mobile_ion, r_max=cfg.rdf_r_max, n_bins=cfg.rdf_n_bins)
    except Exception as exc:
        log.debug("[h06_analysis] RDF failed T=%d: %s", T, exc)

    try:
        positions = parse_dump_lammps(traj_file, target_element=mobile_ion)
        if positions is not None and len(positions) >= 10:
            run_van_hove(positions, T, output_dir, dt_frame_ps,
                        r_max=cfg.van_hove_r_max, n_rbins=cfg.van_hove_n_rbins,
                        t_frames_cfg=cfg.van_hove_t_frames)
    except Exception as exc:
        log.debug("[h06_analysis] Van Hove failed T=%d: %s", T, exc)

    try:
        run_coordination(traj_file, T, output_dir, mobile_ion,
                         r_max=cfg.coordination_r_max, n_bins=cfg.coordination_n_bins)
    except Exception as exc:
        log.debug("[h06_analysis] Coordination failed T=%d: %s", T, exc)

    if anion and anion != mobile_ion:
        try:
            run_ion_pairs(traj_file, T, output_dir, mobile_ion, anion,
                         r_contact=cfg.ion_pair_r_contact, r_ssip=cfg.ion_pair_r_ssip)
        except Exception as exc:
            log.debug("[h06_analysis] Ion pairs failed T=%d: %s", T, exc)
        try:
            run_transference(project_dir, traj_file, T, output_dir, mobile_ion, anion,
                            dt_frame_ps, skip_frac=cfg.msd_skip_frac)
        except Exception as exc:
            log.debug("[h06_analysis] Transference failed T=%d: %s", T, exc)

    try:
        run_vacf_vdos(traj_file, T, output_dir, mobile_ion, dt_frame_ps, max_lag=cfg.vacf_max_lag)
    except Exception as exc:
        log.debug("[h06_analysis] VACF/VDOS failed T=%d: %s", T, exc)

    return D
