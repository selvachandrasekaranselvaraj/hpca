"""
_variant.py — Run one analysis variant (cmd / mlmd_dft / combined) across all
its temperatures, then fit Arrhenius Ea and Haven ratio from the results.

This is the unit of work one analysis_cpu SLURM job performs — see _worker.py.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ._arrhenius import run_arrhenius, run_haven_ratio
from ._config import AnalysisConfig, plat
from ._sources import dt_frame_ps_for
from ._temperature import analyze_temperature
from ._transport import cached_D

log = logging.getLogger("hpca.orch")


def run_variant(
    project_dir: Path,
    variant: str,
    sources: dict[int, Path],
    out_dir: Path,
    mobile_ion: str,
    anion: str,
    yaml_data: dict,
    cfg: AnalysisConfig,
) -> dict[int, float]:
    """Analyze every temperature in `sources` for one variant; write arrhenius.csv + haven_ratio.csv.

    A prior (interrupted) attempt's msd_{T}K.csv is reused via cached_D()
    instead of re-parsing the trajectory, so a retried job doesn't redo
    already-completed temperatures.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("[h06_analysis] running variant=%s (%d temps: %s, workers=%d)",
             variant, len(sources), sorted(sources.keys()), cfg.parallel_temps)

    def _run_one(item: tuple[int, Path]) -> tuple[int, float | None]:
        T, traj = item
        D = cached_D(out_dir, T, traj)
        if D is not None:
            log.info("[h06_analysis] [%s] T=%d K: reusing cached MSD (D = %.3e m²/s)",
                     variant, T, D)
            return T, D
        dt = dt_frame_ps_for(traj, yaml_data, plat)
        return T, analyze_temperature(project_dir, T, traj, out_dir, mobile_ion, anion, dt, cfg)

    D_per_T: dict[int, float] = {}
    with ThreadPoolExecutor(max_workers=cfg.parallel_temps) as pool:
        futs = {pool.submit(_run_one, item): item[0] for item in sorted(sources.items())}
        for fut in as_completed(futs):
            T = futs[fut]
            try:
                _, D = fut.result()
                if D is not None:
                    D_per_T[T] = D
                    log.info("[h06_analysis] [%s] T=%d K: D = %.3e m²/s", variant, T, D)
            except Exception as exc:
                log.error("[h06_analysis] [%s] Error at T=%d: %s", variant, T, exc)

    if D_per_T:
        run_arrhenius(out_dir, D_per_T)
        run_haven_ratio(project_dir, out_dir, mobile_ion, D_per_T)
    else:
        log.warning("[h06_analysis] [%s] No valid D values", variant)
    return D_per_T
