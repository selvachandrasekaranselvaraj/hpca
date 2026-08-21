"""
_dynamics.py — Ion-pair classification, transference number, and VACF/VDOS.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from ._parsers import parse_dump_all
from ._transport import run_msd

log = logging.getLogger("hpca.orch")


def run_ion_pairs(traj_file: Path, T: int, output_dir: Path, cation: str, anion: str,
                  r_contact: float, r_ssip: float) -> dict[str, float] | None:
    """Classify cation-anion pairs as CIP / SSIP / aggregate by distance thresholds."""
    if "dump" not in traj_file.name and traj_file.suffix != ".lmp":
        return None
    positions_all, elements_all = parse_dump_all(traj_file)
    if positions_all is None:
        return None
    try:
        import numpy as np
    except ImportError:
        return None

    cat_idx = [i for i, e in enumerate(elements_all) if e == cation]
    ani_idx = [i for i, e in enumerate(elements_all) if e == anion]
    if not cat_idx or not ani_idx:
        return None
    n_frames = positions_all.shape[0]
    use = positions_all[n_frames // 2:]
    n_cip = n_ssip = n_agg = n_total = 0
    for frame in use:
        for ci in cat_idx:
            dists = np.linalg.norm(frame[ani_idx] - frame[ci], axis=1)
            min_d = dists.min()
            n_total += 1
            if min_d < r_contact:
                n_cip += 1
            elif min_d < r_ssip:
                n_ssip += 1
            else:
                n_agg += 1
    if n_total == 0:
        return None

    result = {
        "CIP": n_cip / n_total,
        "SSIP": n_ssip / n_total,
        "agg": n_agg / n_total,
    }
    csv_path = output_dir / f"ion_pairs_{T}K.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["T_K", "cation", "anion", "CIP_frac", "SSIP_frac", "agg_frac"])
        writer.writerow([T, cation, anion,
                         f"{result['CIP']:.4f}", f"{result['SSIP']:.4f}", f"{result['agg']:.4f}"])
    log.info("[h06_analysis] Ion pairs T=%d K: CIP=%.2f SSIP=%.2f agg=%.2f",
             T, result["CIP"], result["SSIP"], result["agg"])
    return result


def run_transference(project_dir: Path, traj_file: Path, T: int, output_dir: Path,
                     cation: str, anion: str, dt_frame_ps: float,
                     skip_frac: float) -> float | None:
    """Compute cationic transference t+ = D_cat / (D_cat + D_ani)."""
    D_cat = run_msd(project_dir, traj_file, T, output_dir, cation, dt_frame_ps,
                    skip_frac=skip_frac, species_label=cation)
    D_ani = run_msd(project_dir, traj_file, T, output_dir, anion, dt_frame_ps,
                    skip_frac=skip_frac, species_label=anion)
    if D_cat is None or D_ani is None or (D_cat + D_ani) < 1e-30:
        return None
    t_plus = D_cat / (D_cat + D_ani)
    csv_path = output_dir / f"transference_{T}K.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["T_K", "D_cation_m2s", "D_anion_m2s", "t_plus"])
        writer.writerow([T, f"{D_cat:.4e}", f"{D_ani:.4e}", f"{t_plus:.4f}"])
    log.info("[h06_analysis] Transference T=%d K: D_cat=%.3e D_ani=%.3e t+=%.3f",
             T, D_cat, D_ani, t_plus)
    return t_plus


def run_vacf_vdos(traj_file: Path, T: int, output_dir: Path, mobile_ion: str,
                  dt_frame_ps: float, max_lag: int) -> None:
    """VACF + power spectrum (VDOS) if velocity data is present in the dump."""
    if "dump" not in traj_file.name and traj_file.suffix != ".lmp":
        return
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    velocities: list = []
    in_atoms = False
    v_cols: tuple[int, int, int] | None = None
    elem_col = -1
    try:
        with open(str(traj_file)) as fh:
            frame_vels: list[list[float]] = []
            header_cols: list[str] = []
            for line in fh:
                line = line.rstrip()
                if line.startswith("ITEM: ATOMS"):
                    header_cols = line.split()[2:]
                    if "vx" not in header_cols:
                        return  # no velocity data
                    in_atoms = True
                    frame_vels = []
                    if v_cols is None:
                        vx = header_cols.index("vx")
                        vy = header_cols.index("vy")
                        vz = header_cols.index("vz")
                        elem_col = header_cols.index("element") if "element" in header_cols else -1
                        v_cols = (vx, vy, vz)
                    continue
                if line.startswith("ITEM:"):
                    if in_atoms and frame_vels:
                        velocities.append(np.array(frame_vels, dtype=np.float64))
                    in_atoms = False
                    continue
                if in_atoms:
                    parts = line.split()
                    if len(parts) > max(v_cols):
                        if elem_col >= 0 and elem_col < len(parts) and parts[elem_col] != mobile_ion:
                            continue
                        vx_v, vy_v, vz_v = (float(parts[v_cols[k]]) for k in range(3))
                        frame_vels.append([vx_v, vy_v, vz_v])
            if in_atoms and frame_vels:
                velocities.append(np.array(frame_vels, dtype=np.float64))
    except Exception as exc:
        log.debug("[h06_analysis] VACF parse failed T=%d: %s", T, exc)
        return
    if not velocities or v_cols is None:
        return

    vel_arr = np.stack(velocities, axis=0)  # (n_frames, n_atoms, 3)
    n_frames = vel_arr.shape[0]
    skip = n_frames // 5
    vel = vel_arr[skip:]
    lag_cap = min(max_lag, len(vel) // 2)
    vacf = np.zeros(lag_cap)
    v0 = vel[0]  # first frame as reference (simple estimate)
    for lag in range(lag_cap):
        vacf[lag] = np.mean(vel[lag] * v0)
    vacf /= vacf[0] + 1e-30  # normalize
    times_ps = np.arange(lag_cap) * dt_frame_ps
    freqs = np.fft.rfftfreq(lag_cap, d=dt_frame_ps)  # THz
    vdos = np.abs(np.fft.rfft(vacf))**2
    np.savetxt(str(output_dir / f"vacf_{T}K.csv"),
               np.column_stack([times_ps, vacf]),
               delimiter=",", header="time_ps,vacf_normalized", comments="")
    np.savetxt(str(output_dir / f"vdos_{T}K.csv"),
               np.column_stack([freqs, vdos]),
               delimiter=",", header="freq_THz,vdos", comments="")
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(times_ps, vacf)
        ax1.axhline(0, color="k", lw=0.5)
        ax1.set_xlabel("time (ps)")
        ax1.set_ylabel("VACF (normalized)")
        ax1.set_title(f"VACF at {T} K")
        ax1.grid(True, alpha=0.3)
        ax2.plot(freqs[freqs < 100], vdos[freqs < 100])
        ax2.set_xlabel("frequency (THz)")
        ax2.set_ylabel("VDOS (arb.)")
        ax2.set_title(f"VDOS at {T} K")
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(output_dir / f"vdos_{T}K.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass
    log.info("[h06_analysis] VACF+VDOS written for T=%d K", T)
