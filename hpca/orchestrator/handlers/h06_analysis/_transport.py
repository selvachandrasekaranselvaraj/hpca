"""
_transport.py — MSD-derived self-diffusion coefficient: parse, compute, save.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ._msd import _compute_msd_numpy, _fit_D_from_msd
from ._parsers import find_mobile_type_id, parse_dump_lammps, parse_xdatcar

log = logging.getLogger("hpca.orch")

NREL_BLUE = "#0079C2"
MIN_FRAMES = 20


def cached_D(out_dir: Path, T: int, traj_file: Path) -> float | None:
    """Return D refit from an existing msd_{T}K.csv if it's newer than traj_file, else None."""
    csv_path = out_dir / f"msd_{T}K.csv"
    if not csv_path.exists():
        return None
    try:
        if csv_path.stat().st_mtime <= traj_file.stat().st_mtime:
            return None
        import numpy as np
        data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)
        if data.ndim != 2 or len(data) < 5:
            return None
        return _fit_D_from_msd(data[:, 0], data[:, 1])
    except Exception:
        return None


def _resolve_mobile_positions(project_dir: Path, traj_file: Path, mobile_ion: str, T: int):
    """Return mobile_ion positions (n_frames, n_target, 3) from a LAMMPS dump or XDATCAR.

    LAMMPS dumps are tried by element label first; if dump_modify wrote every
    atom as the same placeholder element, fall back to the atom type ID read
    from system.data (CMD) or type_map.raw (MLMD).
    """
    if traj_file.suffix != ".lmp" and "dump" not in traj_file.name:
        return parse_xdatcar(traj_file, mobile_ion_symbol=mobile_ion)

    positions = parse_dump_lammps(traj_file, target_element=mobile_ion)
    if positions is not None and len(positions) >= MIN_FRAMES:
        return positions

    # Legacy layout: system.data is 3 dirs up (nvt/ → T/ → comb/).
    # v2 layout:     cmd/nvt/{T}K/ → design/preopt/preopted_system_cmd.data
    system_data = traj_file.parent.parent.parent / "system.data"
    if not system_data.exists():
        candidate = (traj_file.parent.parent.parent.parent
                    / "design" / "preopt" / "preopted_system_cmd.data")
        if candidate.exists():
            system_data = candidate
    type_id = find_mobile_type_id(system_data, mobile_ion)
    if type_id is None:
        # MLMD: no system.data — read type_map.raw from the mlff directory
        from hpca.core.paths import mlmd_mlff
        for tm_path in (mlmd_mlff(project_dir) / "00.data" / "type_map.raw",
                        mlmd_mlff(project_dir) / "dataset_data" / "type_map.raw"):
            if tm_path.exists():
                tm = [l.strip() for l in tm_path.read_text().splitlines() if l.strip()]
                if mobile_ion in tm:
                    type_id = tm.index(mobile_ion) + 1  # 1-based
                    log.info("[h06_analysis] T=%d: type_map.raw → %s = type %d",
                             T, mobile_ion, type_id)
                break
    if type_id is not None:
        log.info("[h06_analysis] T=%d: element filter empty, retrying by type_id=%d", T, type_id)
        return parse_dump_lammps(traj_file, target_type_id=type_id)
    return positions


def run_msd(
    project_dir: Path,
    traj_file: Path,
    T: int,
    output_dir: Path,
    mobile_ion: str,
    dt_frame_ps: float,
    skip_frac: float,
    species_label: str = "",
) -> float | None:
    """Compute MSD for the mobile ion, save CSV + PNG, and return D (m²/s).

    species_label: optional filename prefix — used by transference-number
    analysis to save the anion MSD without overwriting msd_{T}K.csv.
    """
    try:
        import numpy as np
    except ImportError:
        log.error("[h06_analysis] numpy required")
        return None

    positions = _resolve_mobile_positions(project_dir, traj_file, mobile_ion, T)
    if positions is None or len(positions) < MIN_FRAMES:
        log.warning("[h06_analysis] T=%d: insufficient frames (%s, need %d)",
                    T, len(positions) if positions is not None else "None", MIN_FRAMES)
        return None

    times, msd, D_m2s, _slope = _compute_msd_numpy(positions, dt_frame_ps, skip_frac=skip_frac)

    fname = f"msd_{species_label}_{T}K.csv" if species_label else f"msd_{T}K.csv"
    csv_path = output_dir / fname
    try:
        data = np.column_stack([times, msd])
        np.savetxt(str(csv_path), data, delimiter=",", header="time_ps,msd_ang2", comments="")
        log.debug("[h06_analysis] Saved %s", csv_path)
    except Exception as exc:
        log.warning("[h06_analysis] MSD CSV save failed: %s", exc)

    png_name = f"msd_{species_label}_{T}K.png" if species_label else f"msd_{T}K.png"
    save_msd_png(times, msd, T, output_dir, png_name=png_name)
    return D_m2s


def save_msd_png(times, msd, T: int, output_dir: Path, png_name: str = "") -> None:
    """Save a MSD vs time PNG plot to output_dir/{png_name or msd_{T}K.png}."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(times, msd, color=NREL_BLUE, linewidth=1.5)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("MSD (Å²)")
        ax.set_title(f"MSD at {T} K")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(output_dir / (png_name or f"msd_{T}K.png")), dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        log.debug("[h06_analysis] MSD PNG save failed T=%d: %s", T, exc)
