"""
sei.py — SEI (Solid Electrolyte Interphase) interface analysis
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
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from typing import Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dt_frame_ps(traj: dict) -> float:
    """Return ps per saved frame."""
    return float(traj.get("dt_ps", 0.001)) * float(traj.get("dump_freq", 1000))


def _species_mask(traj: dict, species_list: list[str]) -> np.ndarray:
    """Boolean mask for atoms whose element is in species_list."""
    sp = np.asarray(traj["species"])
    return np.isin(sp, species_list)


def _box_lengths(traj: dict) -> np.ndarray:
    """
    Box lengths per frame, shape (n_frames, 3).
    Works for both orthorhombic box (n_frames, 3, 2) and legacy scalar box.
    """
    box = traj.get("box")
    if box is None:
        return None
    box = np.asarray(box, dtype=np.float64)
    if box.ndim == 3:                    # (n_frames, 3, 2)
        return box[:, :, 1] - box[:, :, 0]
    if box.ndim == 2 and box.shape[1] == 2:  # (3, 2) — single frame repeated
        lengths = box[:, 1] - box[:, 0]
        n_frames = traj["positions"].shape[0]
        return np.tile(lengths, (n_frames, 1))
    return None


def _cross_section_area(traj: dict, axis: int) -> np.ndarray:
    """
    Area of the face perpendicular to `axis`, per frame. Shape (n_frames,).
    """
    lengths = _box_lengths(traj)
    if lengths is None:
        # Fallback: unit area
        n_frames = traj["positions"].shape[0]
        return np.ones(n_frames)
    axes = [0, 1, 2]
    axes.remove(axis)
    return lengths[:, axes[0]] * lengths[:, axes[1]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def identify_sei_region(
    traj: dict,
    electrode_species: list[str],
    electrolyte_species: list[str],
    axis: int = 2,
    n_slabs: int = 50,
) -> dict:
    """
    Identify the SEI region from a trajectory with electrode + electrolyte atoms.

    Method:
      1. Bin atom positions along `axis` into `n_slabs` density slabs.
      2. Smooth each species density profile with a Gaussian kernel.
      3. SEI lo = position where electrode density falls to 10% of its peak.
         SEI hi = position where electrolyte density first rises to 10% of its bulk mean.
      4. If the two boundaries cannot be resolved, returns sei_lo == sei_hi.

    Returns dict with keys:
        slab_edges_A        : np.ndarray (n_slabs+1,)  slab bin edges in Angstrom
        slab_centers_A      : np.ndarray (n_slabs,)
        density_profile     : dict {species: (n_frames, n_slabs) number density in Å⁻³}
        density_mean        : dict {species: (n_slabs,) time-averaged}
        sei_lo_A            : float
        sei_hi_A            : float
        sei_thickness_A     : float  (mean across frames)
        sei_thickness_per_frame : np.ndarray (n_frames,)
    """
    if not traj or "positions" not in traj:
        return {}

    try:
        positions = np.asarray(traj["positions"], dtype=np.float64)   # (F, N, 3)
        n_frames, n_atoms, _ = positions.shape

        el_mask  = _species_mask(traj, electrode_species)
        sol_mask = _species_mask(traj, electrolyte_species)

        if not el_mask.any() or not sol_mask.any():
            return {}

        lengths = _box_lengths(traj)                                   # (F, 3) or None
        pos_axis = positions[:, :, axis]                               # (F, N)

        # Build slab edges using mean box extent
        if lengths is not None:
            box_lo  = np.mean(positions[:, :, axis].min(axis=1))
            box_hi  = np.mean(positions[:, :, axis].max(axis=1))
        else:
            box_lo  = pos_axis.min()
            box_hi  = pos_axis.max()

        slab_edges   = np.linspace(box_lo, box_hi, n_slabs + 1)
        slab_centers = 0.5 * (slab_edges[:-1] + slab_edges[1:])
        slab_width   = slab_edges[1] - slab_edges[0]                  # Å

        # Cross-section area per frame (Å²)
        area = _cross_section_area(traj, axis)                         # (F,)

        all_species = list(dict.fromkeys(
            electrode_species + electrolyte_species
        ))
        density_profile: dict[str, np.ndarray] = {}

        for sp in all_species:
            sp_mask = _species_mask(traj, [sp])
            if not sp_mask.any():
                density_profile[sp] = np.zeros((n_frames, n_slabs))
                continue

            sp_pos = pos_axis[:, sp_mask]                              # (F, n_sp)
            # Vectorised histogram over all frames
            counts = np.array([
                np.histogram(sp_pos[f], bins=slab_edges)[0]
                for f in range(n_frames)
            ], dtype=np.float64)                                       # (F, n_slabs)

            vol_slab = slab_width * area[:, None]                      # (F, n_slabs)
            density_profile[sp] = counts / np.where(vol_slab > 0, vol_slab, 1.0)

        density_mean = {
            sp: density_profile[sp].mean(axis=0) for sp in all_species
        }

        # SEI boundary detection on smoothed mean profiles
        sigma = max(1, n_slabs // 20)

        # Electrode: combine all electrode species
        el_mean  = sum(density_mean[s] for s in electrode_species)
        sol_mean = sum(density_mean[s] for s in electrolyte_species)
        el_smooth  = gaussian_filter1d(el_mean,  sigma=sigma)
        sol_smooth = gaussian_filter1d(sol_mean, sigma=sigma)

        el_peak   = el_smooth.max()
        sol_bulk  = np.percentile(sol_smooth, 75)                      # robust bulk mean

        sei_lo_A = slab_centers[0]
        sei_hi_A = slab_centers[-1]

        if el_peak > 0:
            # SEI lo: last slab (moving left→right past electrode peak) where
            # electrode density < 10% of peak
            peak_idx = int(el_smooth.argmax())
            for i in range(peak_idx, n_slabs):
                if el_smooth[i] < 0.10 * el_peak:
                    sei_lo_A = slab_centers[i]
                    break

        if sol_bulk > 0:
            # SEI hi: first slab right of sei_lo where electrolyte > 10% of bulk
            lo_idx = int(np.searchsorted(slab_centers, sei_lo_A))
            for i in range(lo_idx, n_slabs):
                if sol_smooth[i] >= 0.10 * sol_bulk:
                    sei_hi_A = slab_centers[i]
                    break

        # Per-frame thickness
        def _frame_thickness(f: int) -> float:
            """Estimate SEI thickness for frame *f* from density-profile crossings."""
            el_f  = sum(density_profile[s][f] for s in electrode_species)
            sol_f = sum(density_profile[s][f] for s in electrolyte_species)
            el_s  = gaussian_filter1d(el_f,  sigma=sigma)
            sol_s = gaussian_filter1d(sol_f, sigma=sigma)
            ep    = el_s.max()
            sb    = np.percentile(sol_s, 75)
            lo_f  = slab_centers[0]
            hi_f  = slab_centers[-1]
            if ep > 0:
                pk = int(el_s.argmax())
                for i in range(pk, n_slabs):
                    if el_s[i] < 0.10 * ep:
                        lo_f = slab_centers[i]
                        break
            if sb > 0:
                li = int(np.searchsorted(slab_centers, lo_f))
                for i in range(li, n_slabs):
                    if sol_s[i] >= 0.10 * sb:
                        hi_f = slab_centers[i]
                        break
            return max(0.0, hi_f - lo_f)

        thickness_per_frame = np.array([_frame_thickness(f) for f in range(n_frames)])

        return {
            "slab_edges_A"             : slab_edges,
            "slab_centers_A"           : slab_centers,
            "density_profile"          : density_profile,
            "density_mean"             : density_mean,
            "sei_lo_A"                 : float(sei_lo_A),
            "sei_hi_A"                 : float(sei_hi_A),
            "sei_thickness_A"          : float(sei_hi_A - sei_lo_A),
            "sei_thickness_per_frame"  : thickness_per_frame,
        }

    except Exception:
        return {}


# ---------------------------------------------------------------------------

def sei_composition_profile(
    traj: dict,
    sei_lo: float,
    sei_hi: float,
    axis: int = 2,
    n_bins: int = 30,
) -> dict:
    """
    Elemental composition within the SEI region as a function of position
    and time.

    Returns dict with keys:
        position_bins       : np.ndarray (n_bins,)  centres in Å
        species             : list[str]
        composition_mean    : dict {sp: (n_bins,) time-averaged mole fraction}
        composition_frames  : dict {sp: (n_frames, n_bins) raw counts normalised per bin}
        times_ns            : np.ndarray (n_frames,)
    """
    if not traj or "positions" not in traj:
        return {}
    if sei_hi <= sei_lo:
        return {}

    try:
        positions = np.asarray(traj["positions"], dtype=np.float64)
        n_frames, n_atoms, _ = positions.shape
        species_arr = np.asarray(traj["species"])
        unique_sp   = sorted(set(traj["species"]))

        bin_edges   = np.linspace(sei_lo, sei_hi, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        pos_axis = positions[:, :, axis]                               # (F, N)
        in_sei   = (pos_axis >= sei_lo) & (pos_axis <= sei_hi)        # (F, N)

        comp_frames: dict[str, np.ndarray] = {}
        for sp in unique_sp:
            sp_mask = species_arr == sp
            sp_pos  = pos_axis[:, sp_mask]                             # (F, n_sp)
            in_sei_sp = in_sei[:, sp_mask]                             # (F, n_sp)

            counts = np.array([
                np.histogram(
                    sp_pos[f][in_sei_sp[f]], bins=bin_edges
                )[0]
                for f in range(n_frames)
            ], dtype=np.float64)                                       # (F, n_bins)
            comp_frames[sp] = counts

        # Normalise to mole fractions per bin per frame
        total_counts = sum(comp_frames[s] for s in unique_sp)         # (F, n_bins)
        safe_total   = np.where(total_counts > 0, total_counts, 1.0)

        comp_frames_norm = {sp: comp_frames[sp] / safe_total for sp in unique_sp}
        comp_mean        = {sp: comp_frames_norm[sp].mean(axis=0) for sp in unique_sp}

        dt_ns = _dt_frame_ps(traj) * 1e-3
        times_ns = np.arange(n_frames) * dt_ns

        return {
            "position_bins"      : bin_centers,
            "species"            : unique_sp,
            "composition_mean"   : comp_mean,
            "composition_frames" : comp_frames_norm,
            "times_ns"           : times_ns,
        }

    except Exception:
        return {}


# ---------------------------------------------------------------------------

def sei_growth_kinetics(
    traj: dict,
    electrode_species: list[str],
    electrolyte_species: list[str],
    axis: int = 2,
    n_slabs: int = 50,
) -> dict:
    """
    SEI thickness vs time.  Fits:
      - Parabolic: L² = k_p * t + c   (diffusion-limited)
      - Linear:    L  = k_l * t + c   (reaction-limited)

    Chooses regime by comparing R² of the two fits.

    Returns dict with keys:
        times_ns            : np.ndarray
        thickness_A         : np.ndarray
        parabolic_fit       : {"k_p", "c", "r2", "L_fit"}
        linear_fit          : {"k_l", "c", "r2", "L_fit"}
        regime              : str  "parabolic" | "linear" | "undetermined"
    """
    if not traj or "positions" not in traj:
        return {}

    try:
        sei_data = identify_sei_region(
            traj, electrode_species, electrolyte_species,
            axis=axis, n_slabs=n_slabs
        )
        if not sei_data:
            return {}

        thickness = sei_data["sei_thickness_per_frame"]                # (F,)
        n_frames  = len(thickness)
        dt_ns     = _dt_frame_ps(traj) * 1e-3
        times_ns  = np.arange(n_frames) * dt_ns

        # Skip equilibration (first 20%)
        skip = max(1, int(n_frames * 0.2))
        t    = times_ns[skip:]
        L    = thickness[skip:]

        if len(t) < 4:
            return {
                "times_ns"      : times_ns,
                "thickness_A"   : thickness,
                "parabolic_fit" : {},
                "linear_fit"    : {},
                "regime"        : "undetermined",
            }

        # Parabolic fit:  L² = k_p * t + c
        L2 = L ** 2
        sl_p, ic_p, r_p, _, _ = stats.linregress(t, L2)
        r2_p = r_p ** 2
        L_para = np.sqrt(np.maximum(0.0, sl_p * times_ns + ic_p))

        # Linear fit:  L = k_l * t + c
        sl_l, ic_l, r_l, _, _ = stats.linregress(t, L)
        r2_l = r_l ** 2
        L_lin = sl_l * times_ns + ic_l

        if r2_p >= r2_l:
            regime = "parabolic"
        elif r2_l > r2_p:
            regime = "linear"
        else:
            regime = "undetermined"

        return {
            "times_ns"    : times_ns,
            "thickness_A" : thickness,
            "parabolic_fit": {
                "k_p"   : float(sl_p),
                "c"     : float(ic_p),
                "r2"    : float(r2_p),
                "L_fit" : L_para,
            },
            "linear_fit": {
                "k_l"   : float(sl_l),
                "c"     : float(ic_l),
                "r2"    : float(r2_l),
                "L_fit" : L_lin,
            },
            "regime": regime,
        }

    except Exception:
        return {}


# ---------------------------------------------------------------------------

def coordination_at_sei(
    traj: dict,
    ion: str,
    cutoff_A: float,
    sei_lo: float,
    sei_hi: float,
    axis: int = 2,
    skip_frac: float = 0.2,
) -> dict:
    """
    Coordination number of `ion` atoms in the SEI vs in the bulk electrolyte.

    Uses a distance-based count (pairs within `cutoff_A`) computed on a
    per-frame basis.  Vectorised over frames via broadcasting.

    Returns dict with keys:
        mean_CN_sei              : float
        mean_CN_bulk             : float
        std_CN_sei               : float
        std_CN_bulk              : float
        CN_distribution_sei      : dict {int_CN: fraction}
        CN_distribution_bulk     : dict {int_CN: fraction}
        n_frames_analysed        : int
    """
    if not traj or "positions" not in traj:
        return {}

    try:
        positions = np.asarray(traj["positions"], dtype=np.float64)    # (F, N, 3)
        n_frames, n_atoms, _ = positions.shape
        species_arr = np.asarray(traj["species"])

        ion_mask    = species_arr == ion
        other_mask  = ~ion_mask

        if not ion_mask.any():
            return {}

        skip = int(n_frames * skip_frac)
        pos  = positions[skip:]
        n_use = pos.shape[0]

        ion_pos   = pos[:, ion_mask, :]                                # (F, n_ion, 3)
        other_pos = pos[:, other_mask, :]                              # (F, n_other, 3)

        pos_axis  = pos[:, ion_mask, axis]                             # (F, n_ion)
        in_sei    = (pos_axis >= sei_lo) & (pos_axis <= sei_hi)
        in_bulk   = pos_axis < sei_lo                                  # left of SEI

        cn_sei_all: list[float] = []
        cn_bulk_all: list[float] = []

        box_l = _box_lengths(traj)

        for f in range(n_use):
            fi = f + skip
            ip  = ion_pos[f]                                           # (n_ion, 3)
            op  = other_pos[f]                                         # (n_other, 3)

            # PBC-aware minimum image distances (orthorhombic)
            if box_l is not None:
                L = box_l[fi]
                # diff: (n_ion, n_other, 3)
                diff = ip[:, None, :] - op[None, :, :]
                diff -= L * np.round(diff / L)
            else:
                diff = ip[:, None, :] - op[None, :, :]

            dist2 = (diff ** 2).sum(axis=-1)                           # (n_ion, n_other)
            cn    = (dist2 < cutoff_A ** 2).sum(axis=1).astype(float) # (n_ion,)

            sei_flag  = in_sei[f]
            bulk_flag = in_bulk[f]

            if sei_flag.any():
                cn_sei_all.extend(cn[sei_flag].tolist())
            if bulk_flag.any():
                cn_bulk_all.extend(cn[bulk_flag].tolist())

        def _cn_dist(vals: list[float]) -> dict[int, float]:
            """Return coordination-number histogram (integer CN → fraction)."""
            if not vals:
                return {}
            arr  = np.round(vals).astype(int)
            uval, ucnt = np.unique(arr, return_counts=True)
            total = ucnt.sum()
            return {int(v): float(c / total) for v, c in zip(uval, ucnt)}

        cn_sei_arr  = np.asarray(cn_sei_all)  if cn_sei_all  else np.array([])
        cn_bulk_arr = np.asarray(cn_bulk_all) if cn_bulk_all else np.array([])

        return {
            "mean_CN_sei"          : float(cn_sei_arr.mean())  if cn_sei_arr.size  else float("nan"),
            "mean_CN_bulk"         : float(cn_bulk_arr.mean()) if cn_bulk_arr.size else float("nan"),
            "std_CN_sei"           : float(cn_sei_arr.std())   if cn_sei_arr.size  else float("nan"),
            "std_CN_bulk"          : float(cn_bulk_arr.std())  if cn_bulk_arr.size else float("nan"),
            "CN_distribution_sei"  : _cn_dist(cn_sei_all),
            "CN_distribution_bulk" : _cn_dist(cn_bulk_all),
            "n_frames_analysed"    : n_use,
        }

    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------

def plot_sei_density_profile(density_data: dict, project: str) -> go.Figure:
    """
    Stacked density profiles (number density Å⁻³) along the simulation axis
    with the SEI region highlighted as a shaded band.

    Args:
        density_data : output of identify_sei_region()
        project      : project label for title
    Returns:
        go.Figure
    """
    if not density_data:
        return go.Figure()

    slab_c  = density_data.get("slab_centers_A", np.array([]))
    d_mean  = density_data.get("density_mean",   {})
    sei_lo  = density_data.get("sei_lo_A",        0.0)
    sei_hi  = density_data.get("sei_hi_A",        0.0)
    thick   = density_data.get("sei_thickness_A", 0.0)

    if slab_c.size == 0 or not d_mean:
        return go.Figure()

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]

    fig = go.Figure()

    for i, (sp, dens) in enumerate(d_mean.items()):
        fig.add_trace(go.Scatter(
            x=slab_c, y=dens,
            mode="lines",
            name=sp,
            line=dict(color=palette[i % len(palette)], width=2),
        ))

    # Shaded SEI band
    if sei_hi > sei_lo:
        fig.add_vrect(
            x0=sei_lo, x1=sei_hi,
            fillcolor="rgba(255,165,0,0.15)",
            line_width=0,
            annotation_text=f"SEI ({thick:.1f} Å)",
            annotation_position="top left",
        )
        fig.add_vline(x=sei_lo, line=dict(color="orange", dash="dash", width=1))
        fig.add_vline(x=sei_hi, line=dict(color="orange", dash="dash", width=1))

    fig.update_layout(
        title=dict(text=f"{project} — SEI Density Profile", font_size=16),
        xaxis=dict(title="Position along axis (Å)", showgrid=True, gridcolor="#eee"),
        yaxis=dict(title="Number density (Å⁻³)", showgrid=True, gridcolor="#eee"),
        legend=dict(title="Species"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=13),
    )
    return fig


# ---------------------------------------------------------------------------

def plot_sei_growth(kinetics: dict, project: str) -> go.Figure:
    """
    SEI thickness vs time with parabolic and linear fit curves.

    Args:
        kinetics : output of sei_growth_kinetics()
        project  : project label for title
    Returns:
        go.Figure
    """
    if not kinetics:
        return go.Figure()

    t   = kinetics.get("times_ns",    np.array([]))
    L   = kinetics.get("thickness_A", np.array([]))
    par = kinetics.get("parabolic_fit", {})
    lin = kinetics.get("linear_fit",   {})
    reg = kinetics.get("regime",       "undetermined")

    if t.size == 0 or L.size == 0:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t, y=L,
        mode="lines",
        name="SEI thickness",
        line=dict(color="#1f77b4", width=2),
    ))

    if par and "L_fit" in par:
        r2_p = par.get("r2", 0.0)
        fig.add_trace(go.Scatter(
            x=t, y=par["L_fit"],
            mode="lines",
            name=f"Parabolic fit (R²={r2_p:.3f})",
            line=dict(color="#ff7f0e", dash="dash", width=2),
        ))

    if lin and "L_fit" in lin:
        r2_l = lin.get("r2", 0.0)
        fig.add_trace(go.Scatter(
            x=t, y=lin["L_fit"],
            mode="lines",
            name=f"Linear fit (R²={r2_l:.3f})",
            line=dict(color="#2ca02c", dash="dot", width=2),
        ))

    fig.update_layout(
        title=dict(text=f"{project} — SEI Growth Kinetics ({reg})", font_size=16),
        xaxis=dict(title="Time (ns)", showgrid=True, gridcolor="#eee"),
        yaxis=dict(title="SEI thickness (Å)", showgrid=True, gridcolor="#eee"),
        legend=dict(title=""),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=13),
    )
    return fig


# ---------------------------------------------------------------------------

def plot_sei_composition(composition: dict, project: str) -> go.Figure:
    """
    Stacked area chart showing elemental composition (mole fraction) as a
    function of position within the SEI region.

    Args:
        composition : output of sei_composition_profile()
        project     : project label for title
    Returns:
        go.Figure
    """
    if not composition:
        return go.Figure()

    pos_bins = composition.get("position_bins", np.array([]))
    sp_list  = composition.get("species",       [])
    comp_m   = composition.get("composition_mean", {})

    if pos_bins.size == 0 or not sp_list:
        return go.Figure()

    # Palette as (r, g, b) tuples — used for rgba() fill strings
    palette_rgb = [
        (31, 119, 180), (255, 127, 14), (44, 160, 44),  (214, 39,  40),
        (148, 103, 189),(140,  86,  75),(227, 119, 194), (127,127, 127),
    ]

    fig = go.Figure()
    cumulative = np.zeros_like(pos_bins)

    for i, sp in enumerate(sp_list):
        y = comp_m.get(sp, np.zeros_like(pos_bins))
        r, g, b = palette_rgb[i % len(palette_rgb)]
        fill_col = f"rgba({r},{g},{b},0.7)"
        fig.add_trace(go.Scatter(
            x=np.concatenate([pos_bins, pos_bins[::-1]]),
            y=np.concatenate([cumulative + y, cumulative[::-1]]),
            fill="toself",
            mode="none",
            fillcolor=fill_col,
            name=sp,
            hoverinfo="x+name",
        ))
        cumulative = cumulative + y

    fig.update_layout(
        title=dict(text=f"{project} — SEI Composition Profile", font_size=16),
        xaxis=dict(title="Position in SEI (Å)", showgrid=True, gridcolor="#eee"),
        yaxis=dict(title="Mole fraction", range=[0, 1.05],
                   showgrid=True, gridcolor="#eee"),
        legend=dict(title="Species", traceorder="normal"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=13),
    )
    return fig
