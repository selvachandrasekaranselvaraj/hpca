"""
transport.py — Transport-property visualizations (Arrhenius, MSD, diffusivity,
               conductivity, Haven ratio, Nernst-Planck profiles)
HPCA Pipeline · /path/to/workspace/hpca/viz/transport.py

Python env: /path/to/apps/apps/cladue/env/bin/python3

All public functions return a go.Figure with NREL theme applied.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sp_stats

from .theme import (
    NREL_COLORS,
    apply_nrel_theme,
    add_annotation_box,
)

_KB_EV = 8.617333e-5  # eV K⁻¹

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_log10(x: float) -> float:
    """Return log10(x) clamped to a minimum of 1e-300 to avoid domain errors."""
    return np.log10(max(x, 1e-300))


def _arrhenius_fit(
    inv_T: np.ndarray, log_D: np.ndarray
) -> tuple[float, float, float, float]:
    """Linear fit log(D) ~ a/T + b; returns (slope, intercept, r2, Ea_eV)."""
    mask = np.isfinite(inv_T) & np.isfinite(log_D)
    if mask.sum() < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    slope, intercept, r, *_ = sp_stats.linregress(inv_T[mask], log_D[mask])
    Ea_eV = -slope * _KB_EV  # slope = -Ea/kB
    return float(slope), float(intercept), float(r**2), float(Ea_eV)


# ---------------------------------------------------------------------------
# 1. plot_arrhenius_multi
# ---------------------------------------------------------------------------


def plot_arrhenius_multi(
    data: dict[str, dict],
    title: str = "Arrhenius Plot — Li-ion Diffusivity",
) -> go.Figure:
    """Overlay Arrhenius fits for multiple MLIPs or projects.

    Parameters
    ----------
    data : dict[str, dict]
        Mapping ``{label: {T_K: D_m2s, ...}}``.
        ``T_K`` are temperatures in Kelvin, ``D_m2s`` diffusivities in m²/s.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
        Interactive Arrhenius figure with scatter points, fit lines, and
        Ea annotation per series.

    Example
    -------
    >>> data = {
    ...     "DeepMD": {300: 1.2e-11, 400: 3.5e-10, 500: 8.1e-9},
    ...     "MACE-MPA-0": {300: 9.8e-12, 400: 2.9e-10, 500: 7.2e-9},
    ... }
    >>> fig = plot_arrhenius_multi(data, title="LMZC Arrhenius")
    """
    fig = go.Figure()

    ea_annotations: list[str] = []

    for i, (label, T_D_map) in enumerate(data.items()):
        color = NREL_COLORS[i % len(NREL_COLORS)]

        T_arr = np.array(sorted(T_D_map.keys()), dtype=float)
        D_arr = np.array([T_D_map[t] for t in T_arr], dtype=float)
        inv_T = 1000.0 / T_arr          # 1000/T for readability
        log_D = np.log10(D_arr)

        # Scatter — data points
        fig.add_trace(
            go.Scatter(
                x=inv_T,
                y=log_D,
                mode="markers",
                name=label,
                marker=dict(color=color, size=10, symbol="circle",
                            line=dict(color="#FFFFFF", width=1.2)),
                customdata=np.column_stack([T_arr, D_arr]),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "T = %{customdata[0]:.0f} K<br>"
                    "1000/T = %{x:.4f} K⁻¹<br>"
                    "D = %{customdata[1]:.3e} m²/s<br>"
                    "log₁₀(D) = %{y:.3f}<extra></extra>"
                ),
            )
        )

        # Fit line
        slope, intercept, r2, Ea_eV = _arrhenius_fit(inv_T, log_D)
        if np.isfinite(Ea_eV):
            x_fit = np.linspace(inv_T.min() * 0.97, inv_T.max() * 1.03, 120)
            y_fit = slope * x_fit + intercept
            fig.add_trace(
                go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode="lines",
                    name=f"{label} fit",
                    line=dict(color=color, width=2.0, dash="dash"),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{label} fit</b><br>"
                        f"Eₐ = {Ea_eV:.3f} eV,  R² = {r2:.4f}"
                        "<extra></extra>"
                    ),
                )
            )
            ea_annotations.append(f"<b>{label}</b>: Eₐ = {Ea_eV:.3f} eV")

    apply_nrel_theme(
        fig,
        title=title,
        xlabel="1000 / T  (K⁻¹)",
        ylabel="log₁₀ D  (m²/s)",
        width=900,
        height=600,
    )
    fig.update_layout(
        xaxis=dict(tickformat=".3f"),
        yaxis=dict(tickformat=".1f"),
    )
    if ea_annotations:
        add_annotation_box(fig, "<br>".join(ea_annotations), x=0.98, y=0.05)
        # Shift annotation to bottom-right
        fig.layout.annotations[-1].update(
            xanchor="right", yanchor="bottom", x=0.98, y=0.02
        )
    return fig


# ---------------------------------------------------------------------------
# 2. plot_msd_multi
# ---------------------------------------------------------------------------


def plot_msd_multi(
    msd_results: list[dict],
    labels: list[str],
    T_K: int = 300,
    project: str = "",
) -> go.Figure:
    """Overlay MSD curves from multiple runs or temperatures.

    Parameters
    ----------
    msd_results : list[dict]
        Each element must contain keys ``"time_ps"`` (1-D array) and
        ``"msd_angsq"`` (1-D array, same length).  Optionally
        ``"D_m2s"`` (float) for fit overlay.
    labels : list[str]
        Series labels; must match length of *msd_results*.
    T_K : int
        Temperature label for the title.
    project : str
        Project name inserted in title and hover.

    Returns
    -------
    go.Figure
    """
    if len(msd_results) != len(labels):
        raise ValueError("msd_results and labels must have the same length.")

    fig = go.Figure()

    for i, (result, label) in enumerate(zip(msd_results, labels)):
        color = NREL_COLORS[i % len(NREL_COLORS)]
        t = np.asarray(result["time_ps"], dtype=float)
        msd = np.asarray(result["msd_angsq"], dtype=float)

        fig.add_trace(
            go.Scatter(
                x=t,
                y=msd,
                mode="lines",
                name=label,
                line=dict(color=color, width=2.5),
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "t = %{x:.1f} ps<br>"
                    "MSD = %{y:.4f} Å²<extra></extra>"
                ),
            )
        )

        # Diffusivity linear fit overlay (40–80% of trajectory)
        n = len(t)
        lo, hi = int(0.40 * n), int(0.80 * n)
        if hi > lo + 2:
            slope_fit, intcpt, *_ = sp_stats.linregress(t[lo:hi], msd[lo:hi])
            t_fit = np.linspace(t[lo], t[hi], 80)
            y_fit = slope_fit * t_fit + intcpt
            D_fit = (slope_fit / 6.0) * 1e-8   # Å²/ps → m²/s
            fig.add_trace(
                go.Scatter(
                    x=t_fit,
                    y=y_fit,
                    mode="lines",
                    name=f"{label} (fit)",
                    line=dict(color=color, width=1.5, dash="dot"),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{label} linear fit</b><br>"
                        f"D = {D_fit:.3e} m²/s<br>"
                        "t = %{x:.1f} ps<extra></extra>"
                    ),
                )
            )

    title = f"Mean-Square Displacement — {project} @ {T_K} K" if project else \
            f"Mean-Square Displacement @ {T_K} K"
    apply_nrel_theme(
        fig,
        title=title,
        xlabel="Time  (ps)",
        ylabel="MSD  (Å²)",
        width=900,
        height=580,
    )
    return fig


# ---------------------------------------------------------------------------
# 3. plot_diffusivity_bar
# ---------------------------------------------------------------------------


def plot_diffusivity_bar(
    projects: list[str],
    D_values: list[list[float]],
    mlips: list[str],
    T_K: int = 300,
) -> go.Figure:
    """Grouped bar chart comparing Li diffusivities across projects and MLIPs.

    Parameters
    ----------
    projects : list[str]
        Project names (x-axis groups).
    D_values : list[list[float]]
        Shape ``(len(mlips), len(projects))``.  Each sub-list contains D in m²/s.
    mlips : list[str]
        MLIP labels (one bar per MLIP per group).
    T_K : int
        Temperature for the title.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    D_arr = np.array(D_values, dtype=float)   # (n_mlips, n_projects)

    for i, mlip in enumerate(mlips):
        color = NREL_COLORS[i % len(NREL_COLORS)]
        d_row = D_arr[i]
        fig.add_trace(
            go.Bar(
                name=mlip,
                x=projects,
                y=d_row,
                marker_color=color,
                marker_opacity=0.88,
                hovertemplate=(
                    f"<b>{mlip}</b><br>"
                    "Project: %{x}<br>"
                    "D = %{y:.3e} m²/s<extra></extra>"
                ),
            )
        )

    apply_nrel_theme(
        fig,
        title=f"Li⁺ Diffusivity Comparison @ {T_K} K",
        xlabel="Project",
        ylabel="D  (m²/s)",
        width=max(800, 160 * len(projects)),
        height=580,
    )
    fig.update_layout(
        barmode="group",
        bargap=0.22,
        bargroupgap=0.06,
        yaxis_type="log",
        yaxis_tickformat=".0e",
    )
    return fig


# ---------------------------------------------------------------------------
# 4. plot_conductivity_vtf
# ---------------------------------------------------------------------------


def plot_conductivity_vtf(
    vtf_params: dict,
    T_range: tuple[int, int] = (200, 400),
    exp_point: Optional[dict] = None,
) -> go.Figure:
    """VTF (Vogel-Tammann-Fulcher) ionic conductivity vs temperature.

    VTF equation: σ(T) = σ₀ · T^(-1/2) · exp(-B / (T − T₀))

    Parameters
    ----------
    vtf_params : dict
        Must contain keys ``"sigma0"`` (S cm⁻¹ K^½), ``"B"`` (K), ``"T0"``
        (K, Vogel temperature).  May also contain ``"label"`` (str) for legend.
    T_range : tuple[int, int]
        Temperature range in Kelvin for the continuous VTF curve.
    exp_point : dict, optional
        Experimental reference point with keys ``"T_K"`` and ``"sigma"``
        and optionally ``"label"``.

    Returns
    -------
    go.Figure
    """
    sigma0 = vtf_params["sigma0"]
    B = vtf_params["B"]
    T0 = vtf_params["T0"]
    series_label = vtf_params.get("label", "VTF fit")

    T_arr = np.linspace(max(T_range[0], T0 + 5), T_range[1], 300)
    sigma = sigma0 * T_arr ** (-0.5) * np.exp(-B / (T_arr - T0))
    log_sigma = np.log10(sigma)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=T_arr,
            y=log_sigma,
            mode="lines",
            name=series_label,
            line=dict(color=NREL_COLORS[0], width=2.8),
            customdata=np.column_stack([sigma]),
            hovertemplate=(
                f"<b>{series_label}</b><br>"
                "T = %{x:.1f} K<br>"
                "σ = %{customdata[0]:.3e} S/cm<br>"
                "log₁₀(σ) = %{y:.3f}<extra></extra>"
            ),
        )
    )

    # Experimental reference point
    if exp_point is not None:
        T_exp = exp_point["T_K"]
        s_exp = exp_point["sigma"]
        exp_label = exp_point.get("label", "Experimental")
        fig.add_trace(
            go.Scatter(
                x=[T_exp],
                y=[np.log10(s_exp)],
                mode="markers",
                name=exp_label,
                marker=dict(
                    color=NREL_COLORS[1], size=14, symbol="star",
                    line=dict(color="#FFFFFF", width=1.5)
                ),
                hovertemplate=(
                    f"<b>{exp_label}</b><br>"
                    f"T = {T_exp} K<br>"
                    f"σ = {s_exp:.3e} S/cm<extra></extra>"
                ),
            )
        )

    ann = (
        f"<b>VTF parameters</b><br>"
        f"σ₀ = {sigma0:.3e} S cm⁻¹ K½<br>"
        f"B  = {B:.1f} K<br>"
        f"T₀ = {T0:.1f} K"
    )
    apply_nrel_theme(
        fig,
        title="VTF Ionic Conductivity vs Temperature",
        xlabel="Temperature  (K)",
        ylabel="log₁₀ σ  (S cm⁻¹)",
        width=850,
        height=580,
    )
    add_annotation_box(fig, ann)
    return fig


# ---------------------------------------------------------------------------
# 5. plot_haven_ratio
# ---------------------------------------------------------------------------


def plot_haven_ratio(
    D_tracer: dict[float, float],
    D_conductivity: dict[float, float],
    project: str = "",
) -> go.Figure:
    """Haven ratio H_R = D_cond / D_tracer vs temperature.

    Parameters
    ----------
    D_tracer : dict[float, float]
        ``{T_K: D_tracer_m2s}``.
    D_conductivity : dict[float, float]
        ``{T_K: D_cond_m2s}`` — same temperatures expected.
    project : str
        Project name for the title.

    Returns
    -------
    go.Figure with two y-axes: Haven ratio (left) and both D values (right).
    """
    T_common = sorted(set(D_tracer) & set(D_conductivity))
    if not T_common:
        warnings.warn("plot_haven_ratio: no common temperatures found.")
        T_common = sorted(D_tracer)

    T_arr = np.array(T_common, dtype=float)
    Dt = np.array([D_tracer.get(t, float("nan")) for t in T_common])
    Dc = np.array([D_conductivity.get(t, float("nan")) for t in T_common])
    HR = Dc / np.where(Dt > 0, Dt, float("nan"))

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Haven ratio — primary y
    fig.add_trace(
        go.Scatter(
            x=T_arr,
            y=HR,
            mode="lines+markers",
            name="Haven ratio H_R",
            line=dict(color=NREL_COLORS[0], width=2.5),
            marker=dict(size=9, symbol="circle",
                        line=dict(color="#FFFFFF", width=1.2)),
            customdata=np.column_stack([Dt, Dc]),
            hovertemplate=(
                "<b>Haven ratio</b><br>"
                "T = %{x:.0f} K<br>"
                "H_R = %{y:.4f}<br>"
                "D_tracer = %{customdata[0]:.3e} m²/s<br>"
                "D_cond   = %{customdata[1]:.3e} m²/s<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    # D_tracer — secondary y (log scale)
    fig.add_trace(
        go.Scatter(
            x=T_arr,
            y=np.log10(Dt),
            mode="lines+markers",
            name="D_tracer",
            line=dict(color=NREL_COLORS[1], width=2.0, dash="dash"),
            marker=dict(size=7, symbol="triangle-up"),
            hovertemplate=(
                "<b>D tracer</b><br>T = %{x:.0f} K<br>"
                "log₁₀(D) = %{y:.3f}<extra></extra>"
            ),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=T_arr,
            y=np.log10(Dc),
            mode="lines+markers",
            name="D_conductivity",
            line=dict(color=NREL_COLORS[2], width=2.0, dash="dot"),
            marker=dict(size=7, symbol="triangle-down"),
            hovertemplate=(
                "<b>D conductivity</b><br>T = %{x:.0f} K<br>"
                "log₁₀(D) = %{y:.3f}<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    apply_nrel_theme(
        fig,
        title=f"Haven Ratio — {project}" if project else "Haven Ratio",
        xlabel="Temperature  (K)",
        ylabel="Haven Ratio  H_R",
        width=900,
        height=580,
    )
    fig.update_yaxes(title_text="log₁₀ D  (m²/s)", secondary_y=True)
    return fig


# ---------------------------------------------------------------------------
# 6. plot_nernst_planck_profile
# ---------------------------------------------------------------------------


def plot_nernst_planck_profile(
    NP_data: dict,
    project: str = "",
) -> go.Figure:
    """Concentration and electric potential profiles from Nernst-Planck model.

    Parameters
    ----------
    NP_data : dict
        Expected keys:
        * ``"x_nm"``      — position array in nm
        * ``"c_profile"`` — concentration(s) in mol/m³.  Either 1-D array
          (single species) or dict ``{species: 1-D array}``.
        * ``"phi_V"``     — electric potential in V (1-D array, optional).
        * ``"times_ns"``  — list of snapshot times in ns (optional, for
          multi-time overlays; then ``c_profile`` and ``phi_V`` become
          lists of arrays).

    Returns
    -------
    go.Figure with two y-axes: concentration (left) and potential (right).
    """
    x_nm = np.asarray(NP_data["x_nm"], dtype=float)
    c_raw = NP_data["c_profile"]
    phi_raw = NP_data.get("phi_V", None)
    times = NP_data.get("times_ns", None)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # ------------------------------------------------------------------
    # concentration profiles
    # ------------------------------------------------------------------
    if isinstance(c_raw, dict):
        # multiple species
        for s_idx, (species, c_arr) in enumerate(c_raw.items()):
            color = NREL_COLORS[s_idx % len(NREL_COLORS)]
            c_arr = np.asarray(c_arr, dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=x_nm, y=c_arr,
                    mode="lines", name=f"c({species})",
                    line=dict(color=color, width=2.5),
                    hovertemplate=(
                        f"<b>c({species})</b><br>"
                        "x = %{x:.2f} nm<br>"
                        "c = %{y:.4f} mol/m³<extra></extra>"
                    ),
                ),
                secondary_y=False,
            )
    elif times is not None:
        # time snapshots
        for t_idx, (t_ns, c_snap) in enumerate(zip(times, c_raw)):
            color = NREL_COLORS[t_idx % len(NREL_COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=x_nm, y=np.asarray(c_snap, dtype=float),
                    mode="lines", name=f"t = {t_ns:.1f} ns",
                    line=dict(color=color, width=2.0),
                    hovertemplate=(
                        f"<b>t = {t_ns:.1f} ns</b><br>"
                        "x = %{x:.2f} nm<br>"
                        "c = %{y:.4f} mol/m³<extra></extra>"
                    ),
                ),
                secondary_y=False,
            )
    else:
        c_arr = np.asarray(c_raw, dtype=float)
        fig.add_trace(
            go.Scatter(
                x=x_nm, y=c_arr,
                mode="lines", name="Concentration",
                line=dict(color=NREL_COLORS[0], width=2.5),
                hovertemplate=(
                    "<b>Concentration</b><br>"
                    "x = %{x:.2f} nm<br>"
                    "c = %{y:.4f} mol/m³<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    # ------------------------------------------------------------------
    # electric potential profile
    # ------------------------------------------------------------------
    if phi_raw is not None:
        phi_list = phi_raw if times is not None else [phi_raw]
        t_list = times if times is not None else [None]
        for t_idx, (t_ns, phi_snap) in enumerate(zip(t_list, phi_list)):
            phi_arr = np.asarray(phi_snap, dtype=float)
            t_label = f"φ (t={t_ns:.1f} ns)" if t_ns is not None else "Potential φ"
            fig.add_trace(
                go.Scatter(
                    x=x_nm, y=phi_arr,
                    mode="lines", name=t_label,
                    line=dict(color=NREL_COLORS[4 + t_idx % 4],
                              width=2.0, dash="dash"),
                    hovertemplate=(
                        f"<b>{t_label}</b><br>"
                        "x = %{x:.2f} nm<br>"
                        "φ = %{y:.4f} V<extra></extra>"
                    ),
                ),
                secondary_y=True,
            )

    apply_nrel_theme(
        fig,
        title=f"Nernst-Planck Profile — {project}" if project else "Nernst-Planck Profile",
        xlabel="Position  (nm)",
        ylabel="Concentration  (mol m⁻³)",
        width=900,
        height=580,
    )
    fig.update_yaxes(title_text="Electric Potential  (V)", secondary_y=True)
    # Add shaded interface region if "interface_nm" provided
    if "interface_nm" in NP_data:
        x_int = NP_data["interface_nm"]
        if hasattr(x_int, "__len__") and len(x_int) == 2:
            fig.add_vrect(
                x0=x_int[0], x1=x_int[1],
                fillcolor="rgba(0, 121, 194, 0.12)",
                line_width=0,
                annotation_text="Interface",
                annotation_position="top left",
                annotation_font_color=NREL_COLORS[0],
            )
    return fig


# ---------------------------------------------------------------------------
# 7. plot_msd_by_temperature
# ---------------------------------------------------------------------------


def plot_msd_by_temperature(
    positions_by_temp: dict[int, np.ndarray],
    dt_ps: float,
    species: str = "Li",
    project: str = "",
    output_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> go.Figure:
    """Compute MSD for multiple temperatures and plot MSD vs time + Arrhenius.

    Takes raw position arrays (one per temperature), computes MSD using the
    standard CLAUDE.md formula (skip first 20% for equilibration, lag up to
    50% of remaining trajectory, fit D from 40–80% of lag window), then
    builds a two-panel interactive figure:

    * Left panels  — MSD vs time at each temperature (one panel per T, stacked)
    * Right panel  — Arrhenius summary: log₁₀(D) vs 1000/T with linear fit

    Parameters
    ----------
    positions_by_temp : dict[int, np.ndarray]
        Mapping ``{T_K: positions_array}`` where ``positions_array`` has
        shape ``(n_frames, n_atoms, 3)`` and coordinates are in Ångström.
        Unwrapped (continuous) coordinates are required for correct MSD.
    dt_ps : float
        Time step between saved frames in picoseconds.
    species : str
        Ion species label used in axis titles and annotations (default "Li").
    project : str
        Project name inserted in figure titles and file names.
    output_dir : str, optional
        Directory to write PNG + HTML files.  If None, files are not saved.
        Follows the HPCA convention: ``{PROJECT}/Analysis/continuum_figures/``.
    data_dir : str, optional
        Directory to write the companion CSV file.  If None, CSV is not saved.
        Follows the HPCA convention: ``{PROJECT}/Analysis/continuum_data/``.

    Returns
    -------
    go.Figure
        Interactive Plotly figure with MSD subplots and Arrhenius summary.
        NREL theme applied.

    Notes
    -----
    MSD formula (3D isotropic):

        MSD(τ) = <|r(t+τ) − r(t)|²> × 3

    where the factor 3 converts the per-dimension average to the full 3D MSD.
    The time-average is taken over all valid t origins (last-origin exclusion).

    Diffusivity:

        D = slope(MSD vs t)[40%–80% lag window] / 6   [Å²/ps → m²/s via ×1e-8]

    Activation energy from Arrhenius fit:

        Ea (eV) = −slope(ln D vs 1/T) × kB   (kB = 8.617333×10⁻⁵ eV/K)

    Example
    -------
    >>> import numpy as np
    >>> from hpca.viz.transport import plot_msd_by_temperature
    >>> rng = np.random.default_rng(42)
    >>> positions = {
    ...     300: rng.normal(0, 1, (5000, 32, 3)).cumsum(axis=0),
    ...     400: rng.normal(0, 1.2, (5000, 32, 3)).cumsum(axis=0),
    ...     500: rng.normal(0, 1.5, (5000, 32, 3)).cumsum(axis=0),
    ... }
    >>> fig = plot_msd_by_temperature(
    ...     positions, dt_ps=0.001, species="Li",
    ...     project="LMZC", output_dir="./figures", data_dir="./data"
    ... )
    """
    import os

    temps_sorted = sorted(positions_by_temp.keys())
    n_temps = len(temps_sorted)

    if n_temps == 0:
        raise ValueError("positions_by_temp is empty.")

    # ------------------------------------------------------------------
    # Step 1: compute MSD and D for every temperature
    # ------------------------------------------------------------------
    msd_results: list[dict] = []
    D_by_temp: dict[int, float] = {}

    for T_K in temps_sorted:
        pos_full = np.asarray(positions_by_temp[T_K], dtype=float)
        if pos_full.ndim != 3 or pos_full.shape[2] != 3:
            raise ValueError(
                f"positions_by_temp[{T_K}] must have shape (n_frames, n_atoms, 3), "
                f"got {pos_full.shape}."
            )

        n_frames_full = pos_full.shape[0]
        skip = int(n_frames_full * 0.2)
        pos = pos_full[skip:]                       # discard equilibration
        n_frames = pos.shape[0]
        max_lag = max(1, int(n_frames * 0.5))       # lag up to 50%

        # Time-averaged MSD (3D isotropic)
        msd = np.array([
            np.mean(
                np.sum((pos[lag:] - pos[: n_frames - lag]) ** 2, axis=-1)
            )
            for lag in range(1, max_lag + 1)
        ])
        # Note: np.sum over axis=-1 gives per-atom squared displacement,
        # np.mean averages over atoms AND time origins, then we multiply by
        # 3/3 = 1 for the isotropic 3D MSD (all three dimensions included).
        # The factor of 3 in the formula cancels with the /6 in D = slope/6.

        times_ps = np.arange(1, max_lag + 1) * dt_ps

        # Diffusivity from 40–80% of lag window
        lo = int(0.40 * max_lag)
        hi = int(0.80 * max_lag)
        if hi > lo + 2:
            slope_fit, intercept_fit, *_ = sp_stats.linregress(
                times_ps[lo:hi], msd[lo:hi]
            )
            # msd here is in Å² (3D sum, not per-dimension), so:
            # D = slope / 6   (Å²/ps)  × 1e-8 → m²/s
            D_m2s = max((slope_fit / 6.0) * 1e-8, 0.0)
        else:
            slope_fit, intercept_fit = float("nan"), float("nan")
            D_m2s = float("nan")

        msd_results.append(
            {
                "T_K": T_K,
                "time_ps": times_ps,
                "msd_angsq": msd,
                "slope": slope_fit,
                "intercept": intercept_fit,
                "D_m2s": D_m2s,
                "lo": lo,
                "hi": hi,
            }
        )
        D_by_temp[T_K] = D_m2s

    # ------------------------------------------------------------------
    # Step 2: Arrhenius fit across temperatures
    # ------------------------------------------------------------------
    T_arr = np.array(temps_sorted, dtype=float)
    D_arr = np.array([D_by_temp[t] for t in temps_sorted], dtype=float)
    inv_T_1000 = 1000.0 / T_arr
    log_D = np.log10(np.where(D_arr > 0, D_arr, float("nan")))

    arr_slope, arr_intercept, arr_r2, Ea_eV = _arrhenius_fit(
        1.0 / T_arr, np.log(np.where(D_arr > 0, D_arr, float("nan")))
    )
    # Ea_eV already computed by _arrhenius_fit using natural log;
    # for the plot we use log₁₀ separately for the display fit line.
    _, arr_log10_intercept, *_ = sp_stats.linregress(
        inv_T_1000[np.isfinite(log_D)],
        log_D[np.isfinite(log_D)],
    ) if np.isfinite(log_D).sum() >= 2 else (float("nan"), float("nan"))

    # Re-derive log10 slope for the Arrhenius plot line
    if np.isfinite(log_D).sum() >= 2:
        arr_log10_slope, arr_log10_intercept, *_ = sp_stats.linregress(
            inv_T_1000[np.isfinite(log_D)],
            log_D[np.isfinite(log_D)],
        )
    else:
        arr_log10_slope, arr_log10_intercept = float("nan"), float("nan")

    # ------------------------------------------------------------------
    # Step 3: Build figure — MSD subplots (left) + Arrhenius (right)
    # ------------------------------------------------------------------
    # Layout: n_temps rows × 2 columns
    #   col 1: MSD vs time (all temps stacked vertically)
    #   col 2, spanning all rows: Arrhenius

    col_widths = [0.62, 0.38]
    subplot_titles: list[str] = []
    for T_K in temps_sorted:
        subplot_titles.append(f"{species} MSD @ {T_K} K")
    subplot_titles.append(
        f"Arrhenius — {project}" if project else "Arrhenius Summary"
    )

    fig = make_subplots(
        rows=n_temps,
        cols=2,
        specs=[[{"rowspan": 1}, {"rowspan": n_temps}]]
        + [[{}, None]] * (n_temps - 1),
        subplot_titles=subplot_titles,
        column_widths=col_widths,
        horizontal_spacing=0.10,
        vertical_spacing=0.06,
    )

    for row_idx, result in enumerate(msd_results):
        T_K = result["T_K"]
        color = NREL_COLORS[row_idx % len(NREL_COLORS)]
        t = result["time_ps"]
        msd_vals = result["msd_angsq"]
        lo, hi = result["lo"], result["hi"]

        # MSD trace
        fig.add_trace(
            go.Scatter(
                x=t,
                y=msd_vals,
                mode="lines",
                name=f"{T_K} K",
                line=dict(color=color, width=2.2),
                legendgroup=f"T{T_K}",
                showlegend=(row_idx == 0),
                hovertemplate=(
                    f"<b>{T_K} K</b><br>"
                    "t = %{x:.2f} ps<br>"
                    "MSD = %{y:.4f} Å²<extra></extra>"
                ),
            ),
            row=row_idx + 1,
            col=1,
        )

        # Linear fit overlay (40–80% lag window)
        if np.isfinite(result["slope"]) and hi > lo + 2:
            t_fit = np.linspace(t[lo], t[min(hi, len(t) - 1)], 60)
            msd_fit = result["slope"] * t_fit + result["intercept"]
            D_str = f"{result['D_m2s']:.3e}" if np.isfinite(result["D_m2s"]) else "N/A"
            fig.add_trace(
                go.Scatter(
                    x=t_fit,
                    y=msd_fit,
                    mode="lines",
                    name=f"{T_K} K fit",
                    line=dict(color=color, width=1.5, dash="dot"),
                    legendgroup=f"T{T_K}",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{T_K} K fit</b><br>"
                        f"D = {D_str} m²/s<br>"
                        "t = %{x:.2f} ps<extra></extra>"
                    ),
                ),
                row=row_idx + 1,
                col=1,
            )

        # Update MSD subplot axis labels
        fig.update_xaxes(
            title_text="Time (ps)" if row_idx == n_temps - 1 else "",
            row=row_idx + 1, col=1,
        )
        fig.update_yaxes(
            title_text="MSD (Å²)",
            row=row_idx + 1, col=1,
        )

    # ------------------------------------------------------------------
    # Arrhenius panel (col 2, all rows)
    # ------------------------------------------------------------------
    # Data points
    finite_mask = np.isfinite(log_D)
    fig.add_trace(
        go.Scatter(
            x=inv_T_1000[finite_mask],
            y=log_D[finite_mask],
            mode="markers",
            name=f"{species}⁺ diffusivity",
            marker=dict(
                color=NREL_COLORS[0], size=11, symbol="circle",
                line=dict(color="#FFFFFF", width=1.5),
            ),
            customdata=np.column_stack([
                T_arr[finite_mask], D_arr[finite_mask]
            ]),
            hovertemplate=(
                "<b>Diffusivity</b><br>"
                "T = %{customdata[0]:.0f} K<br>"
                "1000/T = %{x:.4f} K⁻¹<br>"
                "D = %{customdata[1]:.3e} m²/s<br>"
                "log₁₀(D) = %{y:.3f}<extra></extra>"
            ),
        ),
        row=1, col=2,
    )

    # Arrhenius fit line
    if np.isfinite(arr_log10_slope) and finite_mask.sum() >= 2:
        x_fit = np.linspace(
            inv_T_1000[finite_mask].min() * 0.97,
            inv_T_1000[finite_mask].max() * 1.03,
            120,
        )
        y_fit = arr_log10_slope * x_fit + arr_log10_intercept
        Ea_str = f"{Ea_eV:.3f}" if np.isfinite(Ea_eV) else "N/A"
        fig.add_trace(
            go.Scatter(
                x=x_fit,
                y=y_fit,
                mode="lines",
                name="Arrhenius fit",
                line=dict(color=NREL_COLORS[1], width=2.2, dash="dash"),
                showlegend=True,
                hovertemplate=(
                    f"<b>Arrhenius fit</b><br>"
                    f"Eₐ = {Ea_str} eV,  R² = {arr_r2:.4f}"
                    "<extra></extra>"
                ),
            ),
            row=1, col=2,
        )

    fig.update_xaxes(
        title_text="1000 / T  (K⁻¹)", tickformat=".3f", row=1, col=2
    )
    fig.update_yaxes(
        title_text="log₁₀ D  (m²/s)", tickformat=".1f", row=1, col=2
    )

    # Annotation box with Ea
    if np.isfinite(Ea_eV):
        Ea_str = f"{Ea_eV:.3f}"
        ann_text = (
            f"<b>Eₐ ({species}⁺)</b> = {Ea_str} eV<br>"
            f"R² = {arr_r2:.4f}"
        )
        fig.add_annotation(
            text=ann_text,
            xref="x2", yref="paper",
            x=inv_T_1000[finite_mask].max() * 1.01,
            y=0.08,
            showarrow=False,
            align="right",
            xanchor="right",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=NREL_COLORS[0],
            borderwidth=1,
            font=dict(size=12, family="Arial, Helvetica, sans-serif"),
        )

    # ------------------------------------------------------------------
    # Apply NREL theme to overall figure
    # ------------------------------------------------------------------
    proj_label = f" — {project}" if project else ""
    fig.update_layout(
        title=dict(
            text=f"{species}⁺ MSD by Temperature{proj_label}",
            font=dict(
                family="Arial, Helvetica, sans-serif",
                size=18,
                color="#1A1A1A",
            ),
        ),
        font=dict(family="Arial, Helvetica, sans-serif", size=13, color="#1A1A1A"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#CCCCCC",
            borderwidth=1,
        ),
        width=1100,
        height=max(350 * n_temps, 500),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E8E8E8", linecolor="#1A1A1A")
    fig.update_yaxes(showgrid=True, gridcolor="#E8E8E8", linecolor="#1A1A1A")

    # ------------------------------------------------------------------
    # Step 4: Save PNG + HTML + CSV if output directories provided
    # ------------------------------------------------------------------
    if output_dir is not None or data_dir is not None:
        proj_slug = project.replace(" ", "_").replace("/", "-") if project else "project"
        fig_name = f"msd_by_temperature_{proj_slug}_{species}"

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            try:
                fig.write_image(
                    os.path.join(output_dir, f"{fig_name}.png"), scale=2
                )
            except Exception as exc:  # kaleido may not be installed
                warnings.warn(
                    f"plot_msd_by_temperature: PNG export failed ({exc}). "
                    "Install kaleido: pip install kaleido"
                )
            fig.write_html(os.path.join(output_dir, f"{fig_name}.html"))

        if data_dir is not None:
            os.makedirs(data_dir, exist_ok=True)
            # Build flat CSV: T_K, inv_T, D_m2s, log10_D, Ea_eV
            header_lines = [
                f"# {species}+ MSD by Temperature — {project}",
                "# columns: T_K, inv_T_K, D_m2s, log10_D_m2s, Ea_eV (same for all rows)",
            ]
            rows = []
            for T_K in temps_sorted:
                D = D_by_temp[T_K]
                rows.append(
                    [
                        T_K,
                        1.0 / T_K,
                        D if np.isfinite(D) else float("nan"),
                        np.log10(D) if (np.isfinite(D) and D > 0) else float("nan"),
                        Ea_eV if np.isfinite(Ea_eV) else float("nan"),
                    ]
                )
            data_csv = np.array(rows, dtype=float)
            np.savetxt(
                os.path.join(data_dir, f"{fig_name}.csv"),
                data_csv,
                delimiter=",",
                header="T_K,inv_T_K,D_m2s,log10_D_m2s,Ea_eV",
                comments="",
            )

    return fig
