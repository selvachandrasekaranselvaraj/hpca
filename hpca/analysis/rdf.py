"""
rdf.py — Radial Distribution Function (RDF) and coordination number analysis
for LAMMPS MD trajectories from battery materials simulations.

Trajectory dict schema (produced by core trajectory reader):
    traj = {
        "positions"  : np.ndarray  shape (n_frames, n_atoms, 3)   [Angstrom, unwrapped]
        "species"    : list[str]   length n_atoms
        "box"        : np.ndarray  shape (n_frames, 3, 2)  [[xlo,xhi],[ylo,yhi],[zlo,zhi]]
        "timesteps"  : np.ndarray  shape (n_frames,)       [LAMMPS integer steps]
        "dt_ps"      : float       timestep in ps (e.g. 0.001)
        "dump_freq"  : int         frames between dumps
        "energies"   : np.ndarray  shape (n_frames,)  optional (pot energy eV)
    }

Author: NREL battery-materials-ai pipeline
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dt_frame_ps(traj: dict) -> float:
    """Return the physical time interval between saved frames in picoseconds."""
    return float(traj.get("dt_ps", 0.001)) * float(traj.get("dump_freq", 1000))


def _species_mask(traj: dict, species: str) -> np.ndarray:
    """Return a boolean atom mask selecting atoms of element *species*."""
    return np.asarray(traj["species"]) == species


def _box_lengths(traj: dict) -> Optional[np.ndarray]:
    """Return box lengths, shape (n_frames, 3) or None."""
    box = traj.get("box")
    if box is None:
        return None
    box = np.asarray(box, dtype=np.float64)
    if box.ndim == 3:
        return box[:, :, 1] - box[:, :, 0]
    if box.ndim == 2 and box.shape[1] == 2:
        lengths = box[:, 1] - box[:, 0]
        n_frames = traj["positions"].shape[0]
        return np.tile(lengths, (n_frames, 1))
    return None


def _box_volume(traj: dict) -> Optional[np.ndarray]:
    """Volume per frame, shape (n_frames,)."""
    lengths = _box_lengths(traj)
    if lengths is None:
        return None
    return lengths[:, 0] * lengths[:, 1] * lengths[:, 2]


def _pbc_dist2_matrix(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    box_lengths: Optional[np.ndarray],
) -> np.ndarray:
    """
    Minimum-image squared distances between all pairs (a_i, b_j).

    pos_a : (n_a, 3)
    pos_b : (n_b, 3)
    box_lengths : (3,) or None

    Returns (n_a, n_b) array of squared distances.
    """
    diff = pos_a[:, None, :] - pos_b[None, :, :]               # (n_a, n_b, 3)
    if box_lengths is not None:
        L = box_lengths
        diff -= L * np.round(diff / L)
    return (diff ** 2).sum(axis=-1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rdf(
    traj: dict,
    species_a: str,
    species_b: str,
    r_max: float = 8.0,
    n_bins: int = 200,
    skip_frac: float = 0.2,
) -> dict:
    """
    Compute the radial distribution function g(r) between species_a and
    species_b, averaged over all usable trajectory frames.

    Normalisation: g(r) = (histogram count) / (n_a * n_b/V * 4πr²Δr)
    Running coordination number: CN(r) = ∫₀ʳ 4πr'² ρ_b g(r') dr'

    Returns dict with keys:
        r_centers    : np.ndarray (n_bins,)   bin centres in Å
        g_r          : np.ndarray (n_bins,)   time-averaged g(r)
        running_CN   : np.ndarray (n_bins,)   cumulative CN
        n_frames     : int
        species_a    : str
        species_b    : str
        r_max        : float
    """
    if not traj or "positions" not in traj:
        return {}

    try:
        positions = np.asarray(traj["positions"], dtype=np.float64)    # (F, N, 3)
        n_frames_total, n_atoms, _ = positions.shape

        mask_a = _species_mask(traj, species_a)                        # (N,)
        mask_b = _species_mask(traj, species_b)

        if not mask_a.any() or not mask_b.any():
            return {}

        n_a = int(mask_a.sum())
        n_b = int(mask_b.sum())
        same_species = species_a == species_b

        skip     = int(n_frames_total * skip_frac)
        pos_use  = positions[skip:]
        n_frames = pos_use.shape[0]

        if n_frames < 1:
            return {}

        box_l = _box_lengths(traj)
        vol   = _box_volume(traj)

        bin_edges   = np.linspace(0.0, r_max, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        dr          = bin_edges[1] - bin_edges[0]

        # Shell volumes for ideal-gas normalisation
        shell_vol = (4.0 / 3.0) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)

        hist_total = np.zeros(n_bins, dtype=np.float64)

        for fi in range(n_frames):
            gi   = fi + skip
            pa   = pos_use[fi][mask_a]                                 # (n_a, 3)
            pb   = pos_use[fi][mask_b]                                 # (n_b, 3)
            L    = box_l[gi] if box_l is not None else None

            d2 = _pbc_dist2_matrix(pa, pb, L)                         # (n_a, n_b)

            if same_species:
                # Exclude self-pairs on the diagonal (pa == pb, index-wise same atom)
                np.fill_diagonal(d2, r_max ** 2 + 1.0)

            d_flat = np.sqrt(d2.ravel())
            hist, _ = np.histogram(d_flat, bins=bin_edges)
            hist_total += hist

        # Mean number density available around one central a atom.  For the
        # same species, exclude the central atom itself (finite-size correction).
        if vol is not None:
            available_b = n_b - 1 if same_species else n_b
            rho_b = available_b / vol[skip:].mean()
        else:
            # Estimate from bounding box of last frame
            pos_b_last = positions[-1][mask_b]
            ext = pos_b_last.max(axis=0) - pos_b_last.min(axis=0)
            available_b = n_b - 1 if same_species else n_b
            rho_b = available_b / max(ext.prod(), 1.0)

        # Each of n_a central atoms expects rho_b * shell_vol neighbours in an
        # ideal fluid.  Do not multiply by n_b again: rho_b already contains
        # the neighbour population.
        denominator = n_frames * n_a * rho_b * shell_vol
        g_r = np.divide(hist_total, denominator, out=np.zeros_like(hist_total),
                        where=denominator > 0)

        # Running CN: integral of 4πr² ρ_b g(r) dr
        running_CN = np.cumsum(4.0 * np.pi * bin_centers ** 2 * rho_b * g_r * dr)

        return {
            "r_centers"  : bin_centers,
            "g_r"        : g_r,
            "running_CN" : running_CN,
            "n_frames"   : n_frames,
            "species_a"  : species_a,
            "species_b"  : species_b,
            "r_max"      : r_max,
        }

    except (KeyError, TypeError, ValueError, IndexError, FloatingPointError):
        return {}


# ---------------------------------------------------------------------------

def compute_rdf_evolution(
    traj: dict,
    species_a: str,
    species_b: str,
    n_windows: int = 10,
    r_max: float = 8.0,
    n_bins: int = 200,
    skip_frac: float = 0.2,
) -> dict:
    """
    Compute RDF in `n_windows` consecutive time windows to detect structural
    changes over the trajectory (e.g. crystallisation, melting, phase change).

    Returns dict with keys:
        window_times_ns  : np.ndarray (n_windows,)  midpoint time of each window
        r_centers        : np.ndarray (n_bins,)
        g_r_windows      : np.ndarray (n_windows, n_bins)
    """
    if not traj or "positions" not in traj:
        return {}

    try:
        positions = np.asarray(traj["positions"], dtype=np.float64)
        n_frames_total = positions.shape[0]

        skip   = int(n_frames_total * skip_frac)
        n_use  = n_frames_total - skip
        if n_use < n_windows:
            return {}

        dt_ns      = _dt_frame_ps(traj) * 1e-3
        win_size   = n_use // n_windows
        g_r_windows  = []
        window_times = []

        for w in range(n_windows):
            f0 = skip + w * win_size
            f1 = f0 + win_size

            # Build a sub-traj view for this window
            sub = {
                "positions" : positions[f0:f1],
                "species"   : traj["species"],
                "dt_ps"     : traj.get("dt_ps", 0.001),
                "dump_freq" : traj.get("dump_freq", 1000),
            }
            if "box" in traj:
                box = np.asarray(traj["box"])
                if box.ndim == 3:
                    sub["box"] = box[f0:f1]
                else:
                    sub["box"] = traj["box"]

            rdf_w = compute_rdf(
                sub, species_a, species_b,
                r_max=r_max, n_bins=n_bins, skip_frac=0.0,
            )
            if not rdf_w:
                g_r_windows.append(np.zeros(n_bins))
            else:
                g_r_windows.append(rdf_w["g_r"])

            mid_frame = f0 + win_size // 2
            window_times.append(mid_frame * dt_ns)

        r_ref = rdf_w.get("r_centers", np.linspace(0, r_max, n_bins)) if rdf_w else np.linspace(0, r_max, n_bins)

        return {
            "window_times_ns" : np.asarray(window_times),
            "r_centers"       : r_ref,
            "g_r_windows"     : np.array(g_r_windows),   # (n_windows, n_bins)
        }

    except Exception:
        return {}


# ---------------------------------------------------------------------------

def coordination_number(
    traj: dict,
    center: str,
    neighbor: str,
    r_cutoff: float,
    skip_frac: float = 0.2,
) -> dict:
    """
    Mean coordination number and distribution for center–neighbor pairs.

    Computes per-atom CN for every usable frame, then aggregates statistics.
    Vectorised: all pair distances computed via broadcasting per frame.

    Returns dict with keys:
        mean_CN          : float
        std_CN           : float
        CN_histogram     : dict {int_CN: fraction}
        per_frame_mean   : np.ndarray (n_use_frames,)
        r_cutoff         : float
    """
    if not traj or "positions" not in traj:
        return {}

    try:
        positions = np.asarray(traj["positions"], dtype=np.float64)
        n_frames_total, n_atoms, _ = positions.shape

        mask_c = _species_mask(traj, center)
        mask_n = _species_mask(traj, neighbor)

        if not mask_c.any() or not mask_n.any():
            return {}

        skip   = int(n_frames_total * skip_frac)
        pos    = positions[skip:]
        n_use  = pos.shape[0]
        box_l  = _box_lengths(traj)

        same_sp = center == neighbor
        r2_cut  = r_cutoff ** 2
        all_cn  = []
        frame_means = []

        for fi in range(n_use):
            gi = fi + skip
            pc = pos[fi][mask_c]                                       # (n_c, 3)
            pn = pos[fi][mask_n]                                       # (n_n, 3)
            L  = box_l[gi] if box_l is not None else None

            d2 = _pbc_dist2_matrix(pc, pn, L)                         # (n_c, n_n)

            if same_sp:
                np.fill_diagonal(d2, r2_cut + 1.0)

            cn = (d2 < r2_cut).sum(axis=1).astype(float)              # (n_c,)
            all_cn.extend(cn.tolist())
            frame_means.append(float(cn.mean()))

        cn_arr = np.asarray(all_cn, dtype=np.float64)
        cn_int = np.round(cn_arr).astype(int)
        uvals, ucounts = np.unique(cn_int, return_counts=True)
        total = ucounts.sum()

        return {
            "mean_CN"        : float(cn_arr.mean()),
            "std_CN"         : float(cn_arr.std()),
            "CN_histogram"   : {int(v): float(c / total) for v, c in zip(uvals, ucounts)},
            "per_frame_mean" : np.asarray(frame_means),
            "r_cutoff"       : float(r_cutoff),
        }

    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------

def plot_rdf(rdf_data: dict, project: str, T_K: int) -> go.Figure:
    """
    Plot g(r) on the primary y-axis and the running coordination number on
    a secondary y-axis.

    Args:
        rdf_data : output of compute_rdf()
        project  : project label for title
        T_K      : temperature in Kelvin for subtitle
    Returns:
        go.Figure
    """
    if not rdf_data:
        return go.Figure()

    r      = rdf_data.get("r_centers",  np.array([]))
    g      = rdf_data.get("g_r",        np.array([]))
    cn     = rdf_data.get("running_CN", np.array([]))
    sp_a   = rdf_data.get("species_a", "A")
    sp_b   = rdf_data.get("species_b", "B")
    r_max  = rdf_data.get("r_max",     8.0)

    if r.size == 0 or g.size == 0:
        return go.Figure()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=r, y=g,
            mode="lines",
            name=f"g(r) {sp_a}–{sp_b}",
            line=dict(color="#1f77b4", width=2),
        ),
        secondary_y=False,
    )

    if cn.size > 0:
        fig.add_trace(
            go.Scatter(
                x=r, y=cn,
                mode="lines",
                name="Running CN",
                line=dict(color="#ff7f0e", width=1.5, dash="dash"),
            ),
            secondary_y=True,
        )

    # Ideal-gas reference line
    fig.add_hline(y=1.0, line=dict(color="gray", dash="dot", width=1),
                  annotation_text="ideal gas", annotation_position="bottom right")

    fig.update_layout(
        title=dict(
            text=f"{project} — RDF {sp_a}–{sp_b} at {T_K} K",
            font_size=16,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=13),
        legend=dict(x=0.65, y=0.95),
        xaxis=dict(title="r (Å)", showgrid=True, gridcolor="#eee", range=[0, r_max]),
    )
    fig.update_yaxes(
        title_text="g(r)", secondary_y=False,
        showgrid=True, gridcolor="#eee",
    )
    fig.update_yaxes(
        title_text="Running CN", secondary_y=True,
        showgrid=False,
    )
    return fig


# ---------------------------------------------------------------------------

def plot_rdf_evolution(rdf_evo: dict, project: str) -> go.Figure:
    """
    Heatmap of g(r) over time windows (r on x-axis, time on y-axis,
    colour = g(r) value).  Structural evolution is immediately visible as
    changes in peak positions and heights.

    Args:
        rdf_evo : output of compute_rdf_evolution()
        project : project label for title
    Returns:
        go.Figure
    """
    if not rdf_evo:
        return go.Figure()

    r      = rdf_evo.get("r_centers",       np.array([]))
    t_ns   = rdf_evo.get("window_times_ns", np.array([]))
    g_mat  = rdf_evo.get("g_r_windows",     np.array([]))

    if r.size == 0 or t_ns.size == 0 or g_mat.size == 0:
        return go.Figure()

    # Clip colour scale to avoid single-frame artefacts
    vmax = np.percentile(g_mat, 98)

    fig = go.Figure(data=go.Heatmap(
        x=r,
        y=t_ns,
        z=g_mat,
        colorscale="RdBu_r",
        zmin=0.0,
        zmax=float(vmax),
        colorbar=dict(title="g(r)"),
        hoverongaps=False,
    ))

    fig.update_layout(
        title=dict(text=f"{project} — RDF Evolution", font_size=16),
        xaxis=dict(title="r (Å)", showgrid=False),
        yaxis=dict(title="Time (ns)", showgrid=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=13),
    )
    return fig


# ---------------------------------------------------------------------------

def plot_coordination_distribution(cn_data: dict, project: str) -> go.Figure:
    """
    Bar chart of the coordination number distribution (fraction of atoms
    vs integer CN).  Includes mean ± std annotation.

    Args:
        cn_data : output of coordination_number()
        project : project label for title
    Returns:
        go.Figure
    """
    if not cn_data:
        return go.Figure()

    hist     = cn_data.get("CN_histogram", {})
    mean_cn  = cn_data.get("mean_CN",      float("nan"))
    std_cn   = cn_data.get("std_CN",       float("nan"))
    r_cut    = cn_data.get("r_cutoff",     0.0)

    if not hist:
        return go.Figure()

    cn_vals  = sorted(hist.keys())
    fracs    = [hist[k] for k in cn_vals]

    fig = go.Figure(go.Bar(
        x=cn_vals,
        y=fracs,
        marker_color="#1f77b4",
        marker_line_color="white",
        marker_line_width=1,
        name="CN distribution",
    ))

    fig.add_annotation(
        x=0.98, y=0.95,
        xref="paper", yref="paper",
        text=f"Mean CN = {mean_cn:.2f} ± {std_cn:.2f}",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
        font=dict(size=13),
    )

    fig.update_layout(
        title=dict(
            text=f"{project} — Coordination Number Distribution (r < {r_cut:.2f} Å)",
            font_size=16,
        ),
        xaxis=dict(
            title="Coordination number",
            tickmode="array",
            tickvals=cn_vals,
            showgrid=False,
        ),
        yaxis=dict(title="Fraction", showgrid=True, gridcolor="#eee"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=13),
        bargap=0.3,
    )
    return fig
