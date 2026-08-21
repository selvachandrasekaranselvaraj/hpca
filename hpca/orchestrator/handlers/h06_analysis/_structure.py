"""
_structure.py — Static structural analyses at one temperature: RDF, Van Hove,
partial RDF / running coordination number.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ._parsers import parse_dump_all

log = logging.getLogger("hpca.orch")

NREL_BLUE = "#0079C2"


def run_rdf(traj_file: Path, T: int, output_dir: Path, mobile_ion: str,
           r_max: float, n_bins: int) -> None:
    """Compute the mobile-ion-to-all radial distribution function, save CSV + PNG."""
    if traj_file.suffix != ".lmp" and "dump" not in traj_file.name:
        return  # XDATCAR RDF skipped (no element-resolved positions without OUTCAR)

    positions_all, elements_all = parse_dump_all(traj_file)
    if positions_all is None:
        return

    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    unique_elems = list(dict.fromkeys(elements_all))
    if mobile_ion not in unique_elems:
        return

    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    # Use only the last 50% of frames for RDF.
    n_frames = positions_all.shape[0]
    use_frames = positions_all[n_frames // 2:]
    n_use = len(use_frames)

    mi_idx = [i for i, e in enumerate(elements_all) if e == mobile_ion]
    if not mi_idx:
        return
    all_idx = list(range(len(elements_all)))

    hist = np.zeros(n_bins)
    for frame in use_frames:
        for i in mi_idx:
            diffs = frame[all_idx] - frame[i]
            dists = np.linalg.norm(diffs, axis=1)
            dists = dists[dists > 0.1]  # exclude self
            hist += np.histogram(dists, bins=r_edges)[0]

    n_mi = len(mi_idx)
    # Simple normalization (ideal gas density) — just normalize to integrate to 1.
    hist = hist.astype(float) / (n_use * n_mi)
    pair = f"{mobile_ion}-all"

    csv_path = output_dir / f"rdf_{pair}_{T}K.csv"
    data = np.column_stack([r_centers, hist])
    np.savetxt(str(csv_path), data, delimiter=",", header="r_ang,g_r", comments="")

    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(r_centers, hist, color=NREL_BLUE)
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("g(r) (unnormalized)")
        ax.set_title(f"RDF {pair} at {T} K")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(output_dir / f"rdf_{pair}_{T}K.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        log.debug("[h06_analysis] RDF PNG failed: %s", exc)


def run_van_hove(positions, T: int, output_dir: Path, dt_frame_ps: float,
                 r_max: float, n_rbins: int, t_frames_cfg: list[int]) -> None:
    """Self-part Van Hove correlation Gs(r,t) for mobile-ion positions."""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if positions is None or len(positions) < 10:
        return

    n_frames, n_atoms, _ = positions.shape
    t_frames = [t for t in t_frames_cfg if t < n_frames // 2]
    if not t_frames:
        return

    r_edges = np.linspace(0, r_max, n_rbins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = r_edges[1] - r_edges[0]
    skip = n_frames // 5
    pos = positions[skip:]
    n_use = len(pos)
    gs = np.zeros((len(t_frames), n_rbins))
    for ti, t_lag in enumerate(t_frames):
        n_orig = n_use - t_lag
        if n_orig <= 0:
            continue
        displ = pos[t_lag:] - pos[:n_orig]
        r = np.linalg.norm(displ, axis=-1).ravel()
        counts, _ = np.histogram(r, bins=r_edges)
        shell_vol = 4.0 * np.pi * r_centers**2 * dr
        gs[ti] = counts / (n_orig * n_atoms * shell_vol + 1e-30)

    header = "r_ang," + ",".join(f"Gs_{t * dt_frame_ps:.3f}ps" for t in t_frames)
    data = np.column_stack([r_centers, gs.T])
    np.savetxt(str(output_dir / f"van_hove_{T}K.csv"), data,
               delimiter=",", header=header, comments="")
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.viridis(np.linspace(0, 1, len(t_frames)))
        for ti, t_lag in enumerate(t_frames):
            ax.plot(r_centers, gs[ti], color=colors[ti],
                    label=f"t={t_lag * dt_frame_ps:.2f} ps")
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("Gs(r,t) (Å⁻³)")
        ax.set_title(f"Self Van Hove at {T} K")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(output_dir / f"van_hove_{T}K.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass


def run_coordination(traj_file: Path, T: int, output_dir: Path, mobile_ion: str,
                     r_max: float, n_bins: int) -> None:
    """Partial RDF and running coordination number CN(r) per species pair."""
    if "dump" not in traj_file.name and traj_file.suffix != ".lmp":
        return
    positions_all, elements_all = parse_dump_all(traj_file)
    if positions_all is None:
        return
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    unique_elems = list(dict.fromkeys(elements_all))
    if mobile_ion not in unique_elems:
        return

    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    n_frames = positions_all.shape[0]
    use = positions_all[n_frames // 2:]
    n_use = len(use)
    mi_idx = [i for i, e in enumerate(elements_all) if e == mobile_ion]
    n_mi = len(mi_idx)
    if n_mi == 0:
        return

    rows = []
    for partner in unique_elems:
        p_idx = [i for i, e in enumerate(elements_all) if e == partner]
        if not p_idx:
            continue
        hist = np.zeros(n_bins)
        for frame in use:
            for i in mi_idx:
                diffs = frame[p_idx] - frame[i]
                dists = np.linalg.norm(diffs, axis=1)
                if partner == mobile_ion:
                    dists = dists[dists > 0.1]
                hist += np.histogram(dists, bins=r_edges)[0]
        g_r = hist / (n_use * n_mi + 1e-30)
        cn_r = np.cumsum(g_r)
        rows.append((f"{mobile_ion}-{partner}", r_centers, g_r, cn_r))

    for pair, r_c, g_r, cn_r in rows:
        data = np.column_stack([r_c, g_r, cn_r])
        np.savetxt(str(output_dir / f"cn_{pair}_{T}K.csv"), data,
                   delimiter=",", header="r_ang,g_r,cn_running", comments="")
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        colors = plt.cm.tab10.colors
        for ci, (pair, r_c, g_r, cn_r) in enumerate(rows):
            c = colors[ci % len(colors)]
            ax1.plot(r_c, g_r, label=pair, color=c)
            ax2.plot(r_c, cn_r, label=pair, color=c)
        for ax in (ax1, ax2):
            ax.set_xlabel("r (Å)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        ax1.set_ylabel("g(r)")
        ax1.set_title(f"Partial RDF at {T} K")
        ax2.set_ylabel("CN(r)")
        ax2.set_title(f"Coordination Number at {T} K")
        fig.tight_layout()
        fig.savefig(str(output_dir / f"coordination_{T}K.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass
