#!/usr/bin/env python3
"""
msd.py — MSD, diffusivity, Van Hove, and Arrhenius transport analysis.

All plot functions return :class:`plotly.graph_objects.Figure` with a clean
dark-on-white theme.  Figures are saved as both HTML (interactive) and PNG
(kaleido, 900×600) in an ``output_dir/transport/`` sub-directory.

Physical conventions
--------------------
- Positions must be in Angstrom, unwrapped (no PBC jumps).
- Time steps are in picoseconds (ps).
- 1 Å²/ps = 1e-8 m²/s  (unit conversion for self-diffusion coefficient).
- MSD = <|r(t+τ) − r(t)|²> averaged over ions AND time origins.
- Fit range: 40–80 % of the usable lag window (avoids subdiffusive onset
  and noise at long lags).
- Directional MSD: MSD_α = <(r_α(t+τ) − r_α(t))²> averaged over origins
  and ions.  For the total isotropic MSD: MSD = MSD_x + MSD_y + MSD_z
  so D = slope/6 (or slope/2 per axis).

Arrhenius fit
-------------
  D(T) = D₀ · exp(−Ea / kB T)
  ln D = ln D₀ − Ea/(kB) · (1/T)
  slope = −Ea / kB   →   Ea = −slope × kB

References
----------
- Barai et al., J. Energy Storage 2026 (continuum model, NREL)
- sse_msd_diffusivity.py  (project reference implementation)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
kB = 8.617333e-5                    # eV/K  (Boltzmann constant)
ANG2_PER_PS_TO_M2_PER_S = 1e-8     # Å²/ps → m²/s
M2_PER_S_TO_CM2_PER_S   = 1e4      # m²/s → cm²/s

# NREL palette (primary project colours) + neutral fallbacks
COLORS = [
    "#0079C2",   # NREL blue
    "#F7A11A",   # NREL yellow
    "#5E9732",   # NREL green
    "#E31C3D",   # NREL red
    "#7A3988",   # purple
    "#D9531E",   # orange
    "#00846B",   # teal
    "#00A4E4",   # sky blue
]

# ---------------------------------------------------------------------------
# Plotly theme helpers
# ---------------------------------------------------------------------------

_LAYOUT_BASE = dict(
    template="plotly_white",
    font=dict(family="Arial, Helvetica, sans-serif", size=14, color="#1a1a1a"),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=75, r=45, t=65, b=65),
    legend=dict(
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="#cccccc",
        borderwidth=1,
        font=dict(size=12),
    ),
)

_AXIS_STYLE = dict(
    showline=True,
    linewidth=1.5,
    linecolor="#1a1a1a",
    mirror=True,
    ticks="inside",
    tickwidth=1.5,
    gridcolor="#e8e8e8",
    gridwidth=1,
    zeroline=False,
)


def _base_layout(**overrides) -> dict:
    """Return _LAYOUT_BASE merged with caller keyword overrides."""
    layout = dict(**_LAYOUT_BASE)
    layout.update(overrides)
    return layout


def _save_figure(fig: go.Figure, output_dir: Path, stem: str) -> tuple[Path, Optional[Path]]:
    """
    Save *fig* as HTML (always) and PNG (if kaleido is installed).

    Returns (html_path, png_path_or_None).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / f"{stem}.html"
    png_path  = output_dir / f"{stem}.png"

    fig.write_html(str(html_path))
    try:
        fig.write_image(str(png_path), width=900, height=600, scale=2)
    except Exception as exc:
        warnings.warn(
            f"PNG export failed for '{stem}' (install kaleido for PNG support): {exc}. "
            f"HTML saved at {html_path}.",
            stacklevel=3,
        )
        return html_path, None

    return html_path, png_path


# ---------------------------------------------------------------------------
# Core MSD computation
# ---------------------------------------------------------------------------

def compute_msd(
    positions: np.ndarray,
    dt_ps: float,
    skip_frac: float = 0.2,
    max_lag_frac: float = 0.5,
    directional: bool = False,
) -> dict:
    """
    Compute time-averaged mean-squared displacement for mobile ions.

    The MSD at lag τ is:

        MSD(τ) = (1/N_ions) · (1/N_origins) · Σ_i Σ_t |r_i(t+τ) − r_i(t)|²

    where t ranges over all valid time origins and i over all ions.

    Parameters
    ----------
    positions    : (n_frames, n_ions, 3) unwrapped positions in Angstrom
    dt_ps        : time between successive frames in picoseconds
    skip_frac    : fraction of trajectory to discard (equilibration); default 0.2
    max_lag_frac : maximum lag as fraction of usable frames; default 0.5
    directional  : if True, also compute MSD_x, MSD_y, MSD_z separately

    Returns
    -------
    dict with keys:
        lag_times_ps   : (n_lags,) lag times in ps
        msd_angsq      : (n_lags,) total MSD in Å²
        n_ions         : number of mobile ions
        n_frames_used  : frames after skip
        dt_ps          : frame timestep (echoed)
        [if directional=True]:
            msd_x, msd_y, msd_z  : (n_lags,) directional MSDs in Å²
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(
            f"positions must have shape (n_frames, n_ions, 3); got {positions.shape}"
        )

    n_frames, n_ions, _ = positions.shape
    skip    = max(1, int(n_frames * skip_frac))
    pos     = positions[skip:]           # (n_use, n_ions, 3)
    n_use   = pos.shape[0]
    max_lag = max(1, int(n_use * max_lag_frac))

    lags  = np.arange(1, max_lag + 1, dtype=np.int64)
    msd   = np.zeros(max_lag, dtype=np.float64)
    msd_x = np.zeros(max_lag, dtype=np.float64) if directional else None
    msd_y = np.zeros(max_lag, dtype=np.float64) if directional else None
    msd_z = np.zeros(max_lag, dtype=np.float64) if directional else None

    for lag in lags:
        disp = pos[lag:] - pos[:n_use - lag]      # (n_orig, n_ions, 3)
        d2   = disp ** 2                           # (n_orig, n_ions, 3)
        # Total MSD: mean over atoms and time-origins, summed over x,y,z
        msd[lag - 1] = float(d2.mean()) * 3       # ×3 converts per-dim to 3D
        if directional:
            msd_x[lag - 1] = float(d2[:, :, 0].mean())
            msd_y[lag - 1] = float(d2[:, :, 1].mean())
            msd_z[lag - 1] = float(d2[:, :, 2].mean())

    lag_times = lags.astype(np.float64) * dt_ps

    result: dict = {
        "lag_times_ps":  lag_times,
        "msd_angsq":     msd,
        "n_ions":        n_ions,
        "n_frames_used": n_use,
        "dt_ps":         float(dt_ps),
    }
    if directional:
        result["msd_x"] = msd_x
        result["msd_y"] = msd_y
        result["msd_z"] = msd_z

    return result


# ---------------------------------------------------------------------------
# Diffusivity fitting
# ---------------------------------------------------------------------------

def fit_diffusivity(
    lag_times_ps: np.ndarray,
    msd_angsq: np.ndarray,
    fit_frac: tuple[float, float] = (0.4, 0.8),
    msd_directional: Optional[dict] = None,
) -> dict:
    """
    Fit D = slope/6 from the linear (diffusive) regime of the MSD.

    Conversion:  D [Å²/ps] × 1e-8 = D [m²/s]

    Parameters
    ----------
    lag_times_ps      : (n_lags,) lag times in ps
    msd_angsq         : (n_lags,) total MSD in Å²
    fit_frac          : (lo_frac, hi_frac) fraction of lag range for linear fit
    msd_directional   : optional dict with keys msd_x, msd_y, msd_z for
                        directional diffusivities (D = slope/2 per axis)

    Returns
    -------
    dict with keys:
        D_m2s          : self-diffusion coefficient in m²/s
        D_cm2s         : self-diffusion coefficient in cm²/s
        slope          : linear fit slope in Å²/ps   (alias: slope_A2ps)
        slope_A2ps     : same as slope (backwards-compat)
        intercept      : fit intercept in Å²
        r2             : coefficient of determination
        fit_range_ps   : (t_lo, t_hi) fit window in ps
        se_slope       : standard error on slope
        [if msd_directional]:
            D_x, D_y, D_z : directional diffusivities (m²/s)
    """
    lag_times_ps = np.asarray(lag_times_ps, dtype=np.float64)
    msd_angsq    = np.asarray(msd_angsq,    dtype=np.float64)

    n = len(lag_times_ps)
    lo_frac, hi_frac = fit_frac
    lo = max(0, int(lo_frac * n))
    hi = min(n, int(hi_frac * n))
    if hi - lo < 3:
        warnings.warn(
            f"Fit window is very narrow ({hi - lo} points). "
            "Using full lag range as fallback.",
            stacklevel=2,
        )
        lo, hi = 0, n

    t_fit   = lag_times_ps[lo:hi]
    msd_fit = msd_angsq[lo:hi]

    slope, intercept, r, _p, se = stats.linregress(t_fit, msd_fit)

    # slope: Å²/ps  →  D = slope/6 × 1e-8 m²/s
    D_m2s  = max(0.0, (slope / 6.0) * ANG2_PER_PS_TO_M2_PER_S)
    D_cm2s = D_m2s * M2_PER_S_TO_CM2_PER_S

    result: dict = {
        "D_m2s":        float(D_m2s),
        "D_cm2s":       float(D_cm2s),
        "slope":        float(slope),
        "slope_A2ps":   float(slope),    # backwards-compat alias
        "intercept":    float(intercept),
        "r2":           float(r ** 2),
        "fit_range_ps": (float(t_fit[0]), float(t_fit[-1])),
        "se_slope":     float(se),
    }

    if msd_directional is not None:
        for key, out_key in [("msd_x", "D_x"), ("msd_y", "D_y"), ("msd_z", "D_z")]:
            if key in msd_directional:
                msd_d = np.asarray(msd_directional[key], dtype=np.float64)
                if len(msd_d) >= hi:
                    s_d, *_ = stats.linregress(t_fit, msd_d[lo:hi])
                    result[out_key] = float(max(0.0, (s_d / 2.0) * ANG2_PER_PS_TO_M2_PER_S))

    return result


# ---------------------------------------------------------------------------
# Arrhenius analysis
# ---------------------------------------------------------------------------

def arrhenius_fit(
    temps_K: list[Union[int, float]],
    D_m2s: list[float],
) -> dict:
    """
    Fit the Arrhenius relation  ln D = ln D₀ − Ea/(kB T).

    Invalid (None, NaN, ≤ 0) diffusivity values are filtered out silently.

    Parameters
    ----------
    temps_K : sequence of temperatures in Kelvin
    D_m2s   : sequence of diffusion coefficients in m²/s

    Returns
    -------
    dict with keys:
        Ea_eV        : activation energy in eV
        Ea_kJmol     : activation energy in kJ/mol
        D0_m2s       : pre-exponential factor (m²/s)
        r2           : coefficient of determination
        inv_T        : 1/T values used (K⁻¹)
        lnD          : ln(D) values used
        slope        : fit slope (= −Ea/kB)
        intercept    : fit intercept (= ln D₀)
        n_points     : number of temperature points used
        temps_used_K : temperatures that passed validity filter
    """
    T_full = np.asarray(temps_K,  dtype=np.float64)
    D_full = np.asarray(D_m2s,    dtype=np.float64)

    # Filter: must be finite, positive, non-None
    valid = np.isfinite(D_full) & (D_full > 0)
    if valid.sum() < 2:
        return {
            "Ea_eV": None, "Ea_kJmol": None,
            "D0_m2s": None, "r2": None, "n_points": int(valid.sum()),
        }

    T_arr = T_full[valid]
    D_arr = D_full[valid]
    inv_T = 1.0 / T_arr
    lnD   = np.log(D_arr)

    slope, intercept, r, _p, _se = stats.linregress(inv_T, lnD)

    Ea_eV    = -slope * kB
    Ea_kJmol = -slope * 8.314e-3     # kJ/mol  (R = 8.314e-3 kJ·mol⁻¹·K⁻¹)
    D0_m2s   = float(np.exp(intercept))

    return {
        "Ea_eV":        float(Ea_eV),
        "Ea_kJmol":     float(Ea_kJmol),
        "D0_m2s":       D0_m2s,
        "r2":           float(r ** 2),
        "slope":        float(slope),
        "intercept":    float(intercept),
        "inv_T":        inv_T.tolist(),
        "lnD":          lnD.tolist(),
        "n_points":     int(valid.sum()),
        "temps_used_K": T_arr.tolist(),
    }


# ---------------------------------------------------------------------------
# Van Hove self-correlation function
# ---------------------------------------------------------------------------

def van_hove_self(
    positions: np.ndarray,
    dt_ps: float,
    lag_times_ps: list[float],
    r_max: float = 10.0,
    n_bins: int = 100,
) -> dict:
    """
    Compute the self part of the Van Hove correlation function G_s(r, t).

        G_s(r, t) = (1/N) Σ_i ⟨δ(r − |r_i(t₀+t) − r_i(t₀)|)⟩_{t₀}

    Normalised so that ∫₀^∞ 4πr² G_s(r,t) dr = 1.

    Parameters
    ----------
    positions    : (n_frames, n_ions, 3) unwrapped positions in Å
    dt_ps        : time per frame in ps
    lag_times_ps : list of lag times (ps) to evaluate
    r_max        : maximum displacement to histogram (Å); default 10
    n_bins       : number of radial bins; default 100

    Returns
    -------
    dict with keys:
        r_bins       : (n_bins,) bin centers in Å   (also exposed as r_A for compat)
        r_A          : alias for r_bins
        G_s          : (n_valid_lags, n_bins) G_s values in Å⁻³
        lag_times_ps : lag times actually computed (ps)
        n_ions       : number of ions used
    """
    positions = np.asarray(positions, dtype=np.float64)
    n_frames, n_ions, _ = positions.shape

    r_edges  = np.linspace(0.0, r_max, n_bins + 1)
    r_bins   = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr       = r_edges[1] - r_edges[0]

    G_s_list   = []
    valid_lags = []

    for lag_t in lag_times_ps:
        lag_frames = max(1, int(round(lag_t / dt_ps)))
        if lag_frames >= n_frames:
            warnings.warn(
                f"Lag {lag_t} ps ({lag_frames} frames) >= n_frames ({n_frames}); skipping.",
                stacklevel=2,
            )
            continue

        disp  = positions[lag_frames:] - positions[:n_frames - lag_frames]
        r_mag = np.linalg.norm(disp, axis=-1).ravel()

        hist, _ = np.histogram(r_mag, bins=r_edges)
        n_origins = n_frames - lag_frames
        shell_vol = 4.0 * np.pi * r_bins ** 2 * dr
        shell_vol[shell_vol < 1e-30] = 1e-30
        G_s = hist.astype(np.float64) / (n_ions * n_origins * shell_vol)

        G_s_list.append(G_s)
        valid_lags.append(float(lag_t))

    if not G_s_list:
        raise ValueError("No valid lag times produced G_s data.")

    G_s_arr = np.array(G_s_list)   # (n_valid_lags, n_bins)

    return {
        "r_bins":       r_bins,
        "r_A":          r_bins,    # backwards-compat alias
        "G_s":          G_s_arr,
        "lag_times_ps": valid_lags,
        "n_ions":       n_ions,
    }


# ---------------------------------------------------------------------------
# Plot: MSD with diffusivity fit
# ---------------------------------------------------------------------------

def plot_msd(
    msd_data: dict,
    project: str,
    T_K: int,
    diffusivity: Optional[dict] = None,
) -> go.Figure:
    """
    Interactive Plotly MSD figure with optional fit line and diffusivity annotation.

    Parameters
    ----------
    msd_data    : dict returned by compute_msd
    project     : project name string for title/annotation
    T_K         : simulation temperature in K
    diffusivity : optional dict returned by fit_diffusivity

    Returns
    -------
    go.Figure
    """
    t      = msd_data["lag_times_ps"]
    msd    = msd_data["msd_angsq"]
    n_ions = msd_data.get("n_ions", "?")

    fig = go.Figure()

    # Total MSD
    fig.add_trace(go.Scatter(
        x=t.tolist(), y=msd.tolist(),
        mode="lines",
        name=f"MSD (total)",
        line=dict(color=COLORS[0], width=2.5),
        hovertemplate="τ = %{x:.1f} ps<br>MSD = %{y:.2f} Å²<extra></extra>",
    ))

    # Directional components
    dir_cfg = [
        ("msd_x", COLORS[1], "MSD<sub>x</sub>"),
        ("msd_y", COLORS[2], "MSD<sub>y</sub>"),
        ("msd_z", COLORS[3], "MSD<sub>z</sub>"),
    ]
    for key, color, label in dir_cfg:
        if key in msd_data:
            fig.add_trace(go.Scatter(
                x=t.tolist(), y=msd_data[key].tolist(),
                mode="lines",
                name=label,
                line=dict(color=color, width=1.5, dash="dot"),
                opacity=0.80,
                hovertemplate=f"{label}: %{{y:.2f}} Å²<extra></extra>",
            ))

    # Fit overlay
    if diffusivity is not None:
        t0, t1  = diffusivity["fit_range_ps"]
        slope   = diffusivity.get("slope", diffusivity.get("slope_A2ps", 0.0))
        b       = diffusivity["intercept"]
        t_fit   = np.linspace(t0, t1, 200)
        msd_fit = slope * t_fit + b

        D_m2s  = diffusivity["D_m2s"]
        D_cm2s = diffusivity["D_cm2s"]
        r2     = diffusivity["r2"]

        fig.add_trace(go.Scatter(
            x=t_fit.tolist(), y=msd_fit.tolist(),
            mode="lines",
            name=f"Fit (R²={r2:.3f})",
            line=dict(color="#d62728", width=2, dash="dash"),
            hovertemplate="τ = %{x:.1f} ps<br>Fit = %{y:.2f} Å²<extra></extra>",
        ))

        fig.add_annotation(
            x=0.97, y=0.06,
            xref="paper", yref="paper",
            text=(
                f"D = {D_m2s:.3e} m²/s<br>"
                f"D = {D_cm2s:.3e} cm²/s<br>"
                f"R² = {r2:.4f}"
            ),
            showarrow=False, align="right",
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#aaaaaa", borderwidth=1,
            font=dict(size=12),
        )

    layout = _base_layout(
        title=dict(
            text=f"{project} — MSD at {T_K} K ({n_ions} ions)",
            x=0.5, xanchor="center",
        ),
        xaxis=dict(title="Lag time (ps)", **_AXIS_STYLE),
        yaxis=dict(title="MSD (Å²)", **_AXIS_STYLE),
    )
    fig.update_layout(**layout)

    return fig


# ---------------------------------------------------------------------------
# Plot: Arrhenius
# ---------------------------------------------------------------------------

def plot_arrhenius(
    results: dict[int, dict],
    project: str,
    mlip: str = "deepmd",
) -> go.Figure:
    """
    Arrhenius plot: 1000/T (K⁻¹) vs ln D (m²/s) with linear fit and Ea annotation.

    Parameters
    ----------
    results : {T_K: diffusivity_dict} — keys are temperatures, values from
              fit_diffusivity or run_full_transport_analysis
    project : project name for title
    mlip    : MLIP model name for legend label

    Returns
    -------
    go.Figure
    """
    temps = sorted(results.keys())

    # Accept results that expose D_m2s either at top level or nested
    def _get_D(T):
        """Extract diffusivity D (m²/s) from a per-temperature result dict."""
        r = results[T]
        if "D_m2s" in r:
            return r["D_m2s"]
        if "diffusivity" in r and "D_m2s" in r["diffusivity"]:
            return r["diffusivity"]["D_m2s"]
        return None

    D_vals = [_get_D(T) for T in temps]

    valid_pairs = [(T, D) for T, D in zip(temps, D_vals)
                   if D is not None and np.isfinite(float(D)) and float(D) > 0]
    if len(valid_pairs) < 2:
        raise ValueError("Need ≥ 2 valid temperature points for Arrhenius plot.")

    T_arr = np.array([v[0] for v in valid_pairs])
    D_arr = np.array([v[1] for v in valid_pairs])
    inv_T_1000 = 1000.0 / T_arr
    lnD = np.log(D_arr)

    # Collect per-temperature R² for optional error bars
    def _get_r2(T):
        """Extract linear-fit R² from a per-temperature result dict."""
        r = results[T]
        if "r2" in r:
            return r["r2"]
        if "diffusivity" in r and "r2" in r["diffusivity"]:
            return r["diffusivity"]["r2"]
        return None

    r2_vals = [_get_r2(T) for T in T_arr]
    has_r2  = all(r is not None for r in r2_vals)

    fig = go.Figure()

    # Data points
    hover_texts = [
        f"T = {T} K<br>D = {D:.3e} m²/s<br>1000/T = {1000/T:.3f} K⁻¹"
        for T, D in zip(T_arr, D_arr)
    ]
    scatter_kwargs: dict = dict(
        x=inv_T_1000.tolist(), y=lnD.tolist(),
        mode="markers+text",
        name=f"{mlip.upper()} MD",
        marker=dict(size=11, color=COLORS[0], symbol="circle",
                    line=dict(width=1.5, color="#004a7c")),
        text=[f"  {T}K" for T in T_arr],
        textposition="middle right",
        textfont=dict(size=11, color="#444444"),
        customdata=hover_texts,
        hovertemplate="%{customdata}<extra></extra>",
    )
    if has_r2:
        sigma = np.array([(1.0 - r) * abs(ld) for r, ld in zip(r2_vals, lnD)])
        scatter_kwargs["error_y"] = dict(
            type="data", array=sigma.tolist(), visible=True,
            color=COLORS[0], thickness=1.5, width=6,
        )
    fig.add_trace(go.Scatter(**scatter_kwargs))

    # Arrhenius fit line
    try:
        arrh = arrhenius_fit(T_arr.tolist(), D_arr.tolist())
        if arrh["Ea_eV"] is not None:
            inv_T_fit  = np.linspace(inv_T_1000.min(), inv_T_1000.max(), 300) / 1000.0
            lnD_fit    = arrh["slope"] * inv_T_fit + arrh["intercept"]

            fig.add_trace(go.Scatter(
                x=(inv_T_fit * 1000).tolist(), y=lnD_fit.tolist(),
                mode="lines",
                name="Arrhenius fit",
                line=dict(color=COLORS[3], width=2, dash="dash"),
                hovertemplate="1000/T = %{x:.3f} K⁻¹<br>ln(D) = %{y:.3f}<extra></extra>",
            ))

            fig.add_annotation(
                x=0.97, y=0.97,
                xref="paper", yref="paper",
                text=(
                    f"Ea = {arrh['Ea_eV']:.3f} eV "
                    f"({arrh['Ea_kJmol']:.1f} kJ/mol)<br>"
                    f"D₀ = {arrh['D0_m2s']:.3e} m²/s<br>"
                    f"R² = {arrh['r2']:.4f}"
                ),
                showarrow=False, align="right",
                bgcolor="rgba(255,255,255,0.90)",
                bordercolor="#aaaaaa", borderwidth=1,
                font=dict(size=12),
            )
    except (ValueError, Exception) as exc:
        warnings.warn(f"Arrhenius fit annotation skipped: {exc}", stacklevel=2)

    # Secondary top-axis showing actual T in K
    T_ticks = sorted(
        {T for T in [250, 300, 320, 340, 360, 380, 400, 450, 500, 600, 700, 800]
         if T >= T_arr.min() * 0.98 and T <= T_arr.max() * 1.02},
        reverse=True,
    )
    tick_vals = [1000.0 / T for T in T_ticks]

    layout = _base_layout(
        title=dict(
            text=f"{project} — Arrhenius Plot ({mlip.upper()})",
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title="1000 / T (K⁻¹)",
            tickvals=tick_vals,
            ticktext=[str(T) for T in T_ticks],
            **_AXIS_STYLE,
        ),
        yaxis=dict(title="ln D (m²/s)", **_AXIS_STYLE),
    )
    fig.update_layout(**layout)

    return fig


# ---------------------------------------------------------------------------
# Plot: Van Hove G_s(r, t)
# ---------------------------------------------------------------------------

def plot_van_hove(
    vh_data: dict,
    project: str,
    T_K: int,
) -> go.Figure:
    """
    Van Hove self-correlation plot: r (Å) vs G_s(r,t) for multiple lag times.

    Curves are colour-coded from blue (short lag) to red (long lag) using the
    Plasma colorscale, with a continuous colorbar.

    Parameters
    ----------
    vh_data : dict returned by van_hove_self
    project : project name for title
    T_K     : temperature in K

    Returns
    -------
    go.Figure
    """
    r_bins    = vh_data.get("r_bins", vh_data.get("r_A"))
    G_s       = vh_data["G_s"]             # (n_lags, n_bins)
    lag_times = vh_data["lag_times_ps"]
    n_ions    = vh_data.get("n_ions", "?")

    n_lags = len(lag_times)
    colorscale = "Plasma"
    colors = px.colors.sample_colorscale(colorscale, np.linspace(0, 1, n_lags))

    fig = go.Figure()

    for i, (lag_t, color) in enumerate(zip(lag_times, colors)):
        fig.add_trace(go.Scatter(
            x=r_bins.tolist(), y=G_s[i].tolist(),
            mode="lines",
            name=f"τ = {lag_t:.1f} ps",
            line=dict(color=color, width=1.8),
            hovertemplate=(
                f"τ = {lag_t:.1f} ps<br>"
                "r = %{x:.2f} Å<br>"
                "G<sub>s</sub> = %{y:.5f} Å⁻³<extra></extra>"
            ),
        ))

    # Invisible proxy trace to render the colorbar
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        showlegend=False,
        hoverinfo="skip",
        marker=dict(
            colorscale=colorscale,
            cmin=float(min(lag_times)),
            cmax=float(max(lag_times)),
            color=[0],
            colorbar=dict(
                title=dict(text="Lag time (ps)", side="right"),
                thickness=16,
                len=0.80,
                outlinewidth=1,
                outlinecolor="#cccccc",
            ),
            showscale=True,
            size=0,
        ),
    ))

    layout = _base_layout(
        title=dict(
            text=(
                f"{project} — Van Hove G<sub>s</sub>(r, t) "
                f"at {T_K} K ({n_ions} ions)"
            ),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(title="Displacement r (Å)", range=[0.0, float(r_bins.max())],
                   **_AXIS_STYLE),
        yaxis=dict(title="G<sub>s</sub>(r, t) (Å⁻³)", **_AXIS_STYLE),
        showlegend=False,
    )
    fig.update_layout(**layout)

    return fig


# ---------------------------------------------------------------------------
# Bonus: multi-temperature MSD overlay
# ---------------------------------------------------------------------------

def plot_msd_multi(
    msd_list: list[dict],
    labels: list[str],
    project: str,
    title_suffix: str = "",
) -> go.Figure:
    """
    Overlay multiple MSD curves (e.g. different temperatures or MLIPs).

    Parameters
    ----------
    msd_list     : list of compute_msd output dicts
    labels       : matching list of label strings
    project      : project name for title
    title_suffix : appended to title if provided

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    for i, (m, label) in enumerate(zip(msd_list, labels)):
        t   = m["lag_times_ps"]
        msd = m["msd_angsq"]
        fig.add_trace(go.Scatter(
            x=t.tolist(), y=msd.tolist(),
            mode="lines",
            name=label,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            hovertemplate=f"{label}: %{{y:.2f}} Å²<extra></extra>",
        ))

    suffix = f" — {title_suffix}" if title_suffix else ""
    layout = _base_layout(
        title=dict(
            text=f"{project} — MSD Comparison{suffix}",
            x=0.5, xanchor="center",
        ),
        xaxis=dict(title="Lag time (ps)", **_AXIS_STYLE),
        yaxis=dict(title="MSD (Å²)", **_AXIS_STYLE),
    )
    fig.update_layout(**layout)

    return fig


# ---------------------------------------------------------------------------
# End-to-end single-temperature transport analysis
# ---------------------------------------------------------------------------

def run_full_transport_analysis(
    traj: dict,
    mobile_ion: str,
    dt_ps: float,
    T_K: int,
    output_dir: Path,
    fit_frac: tuple[float, float] = (0.4, 0.8),
    skip_frac: float = 0.2,
    max_lag_frac: float = 0.5,
    van_hove_lags_ps: Optional[list[float]] = None,
    van_hove_r_max: float = 10.0,
    project: str = "Project",
    mlip: str = "deepmd",
    directional: bool = True,
) -> dict:
    """
    End-to-end transport analysis for a single temperature trajectory.

    Steps
    -----
    1. Extract mobile-ion positions from *traj*.
    2. Compute time-averaged MSD (with optional directional components).
    3. Fit diffusivity D from the linear regime.
    4. Compute Van Hove G_s(r, t).
    5. Generate and save all figures as HTML + PNG in output_dir/transport/.
    6. Write MSD and Van Hove data as CSV files.

    Parameters
    ----------
    traj              : trajectory dict from trajectory.parse_trajectory
    mobile_ion        : element symbol for mobile species, e.g. "Li"
    dt_ps             : time between frames in ps
    T_K               : simulation temperature in K
    output_dir        : base output directory
    fit_frac          : (lo, hi) fractions of lag range for D fit
    skip_frac         : equilibration fraction to skip
    max_lag_frac      : max lag as fraction of usable frames
    van_hove_lags_ps  : lag times (ps) for Van Hove; auto-chosen if None
    van_hove_r_max    : max displacement for Van Hove histogram (Å)
    project           : project name string for plot titles
    mlip              : MLIP label for plot annotations
    directional       : if True, compute and plot directional MSDs

    Returns
    -------
    Summary dict::

        {
          "D_m2s":    float,
          "D_cm2s":   float,
          "r2":       float,          # MSD fit R²
          "Ea_eV":    None,           # placeholder (requires multi-T run)
          "T_K":      int,
          "n_ions":   int,
          "diffusivity": {...},       # full fit_diffusivity output
          "figures": {
              "msd_html": Path,
              "msd_png":  Path | None,
              "vh_html":  Path,
              "vh_png":   Path | None,
          },
          "data_files": {
              "msd_csv": Path,
              "vh_csv":  Path,
          },
        }
    """
    out = Path(output_dir) / "transport"
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 1. Mobile positions
    mask = traj.get("species_indices", {}).get(mobile_ion)
    if mask is None:
        available = list(traj.get("species_indices", {}).keys())
        raise KeyError(
            f"Mobile ion '{mobile_ion}' not found in trajectory. "
            f"Available: {available}"
        )
    idx = np.where(mask)[0] if mask.dtype == bool else mask
    positions = traj["positions"][:, idx, :]   # (n_frames, n_ions, 3)
    n_frames, n_ions, _ = positions.shape

    # ------------------------------------------------------------------ 2. MSD
    msd_data = compute_msd(
        positions, dt_ps,
        skip_frac=skip_frac,
        max_lag_frac=max_lag_frac,
        directional=directional,
    )

    # ------------------------------------------------------------------ 3. Fit D
    dir_kwarg = {}
    if directional:
        dir_kwarg["msd_directional"] = {
            k: msd_data[k] for k in ("msd_x", "msd_y", "msd_z") if k in msd_data
        }
    diff_result = fit_diffusivity(
        msd_data["lag_times_ps"], msd_data["msd_angsq"],
        fit_frac=fit_frac, **dir_kwarg,
    )

    # ------------------------------------------------------------------ 4. Van Hove
    if van_hove_lags_ps is None:
        total_lag = float(msd_data["lag_times_ps"][-1])
        vh_candidates = [
            dt_ps,
            round(total_lag * 0.01, 3),
            round(total_lag * 0.05, 2),
            round(total_lag * 0.20, 1),
            round(total_lag * 0.50, 1),
            round(total_lag * 0.80, 1),
        ]
        van_hove_lags_ps = sorted(set(l for l in vh_candidates if l >= dt_ps))

    vh_data = van_hove_self(
        positions, dt_ps,
        lag_times_ps=van_hove_lags_ps,
        r_max=van_hove_r_max,
    )

    # ------------------------------------------------------------------ 5. Figures
    fig_msd = plot_msd(msd_data, project, T_K, diffusivity=diff_result)
    fig_vh  = plot_van_hove(vh_data, project, T_K)

    stem_msd = f"{project}_{T_K}K_msd"
    stem_vh  = f"{project}_{T_K}K_van_hove"

    html_msd, png_msd = _save_figure(fig_msd, out, stem_msd)
    html_vh,  png_vh  = _save_figure(fig_vh,  out, stem_vh)

    # ------------------------------------------------------------------ 6. CSV data
    msd_csv = out / f"{stem_msd}.csv"
    msd_df: dict = {
        "lag_time_ps": msd_data["lag_times_ps"],
        "msd_angsq":   msd_data["msd_angsq"],
    }
    for k in ("msd_x", "msd_y", "msd_z"):
        if k in msd_data:
            msd_df[k] = msd_data[k]
    pd.DataFrame(msd_df).to_csv(msd_csv, index=False)

    vh_csv = out / f"{stem_vh}.csv"
    vh_rows = [
        {"lag_time_ps": lag_t, "r_ang": float(r), "G_s": float(G)}
        for i, lag_t in enumerate(vh_data["lag_times_ps"])
        for r, G in zip(vh_data["r_bins"], vh_data["G_s"][i])
    ]
    pd.DataFrame(vh_rows).to_csv(vh_csv, index=False)

    return {
        "D_m2s":       diff_result["D_m2s"],
        "D_cm2s":      diff_result["D_cm2s"],
        "r2":          diff_result["r2"],
        "Ea_eV":       None,    # placeholder — use run_multi_temperature_analysis
        "T_K":         T_K,
        "n_ions":      n_ions,
        "diffusivity": diff_result,
        "figures": {
            "msd_html": html_msd,
            "msd_png":  png_msd,
            "vh_html":  html_vh,
            "vh_png":   png_vh,
        },
        "data_files": {
            "msd_csv": msd_csv,
            "vh_csv":  vh_csv,
        },
    }


# ---------------------------------------------------------------------------
# Multi-temperature batch runner + Arrhenius
# ---------------------------------------------------------------------------

def run_multi_temperature_analysis(
    traj_map: dict[int, dict],
    mobile_ion: str,
    dt_ps: float,
    output_dir: Path,
    project: str = "Project",
    mlip: str = "deepmd",
    **kwargs,
) -> dict:
    """
    Run run_full_transport_analysis across multiple temperatures and
    produce the combined Arrhenius plot.

    Parameters
    ----------
    traj_map   : {T_K: traj_dict} mapping (from parse_trajectory)
    mobile_ion : mobile species symbol
    dt_ps      : timestep in ps (assumed same for all trajectories)
    output_dir : base output directory
    project    : project name
    mlip       : MLIP model label
    **kwargs   : forwarded to run_full_transport_analysis per temperature

    Returns
    -------
    dict with keys:
        per_T_results    : {T_K: summary_dict}
        arrhenius        : dict from arrhenius_fit
        arrhenius_figure : go.Figure
        arrhenius_html   : Path
        arrhenius_png    : Path | None
        arrhenius_csv    : Path
    """
    out = Path(output_dir) / "transport"
    out.mkdir(parents=True, exist_ok=True)

    per_T: dict[int, dict] = {}
    for T_K, traj in sorted(traj_map.items()):
        print(f"  [transport] T = {T_K} K ...", flush=True)
        try:
            result = run_full_transport_analysis(
                traj, mobile_ion, dt_ps, T_K, output_dir,
                project=project, mlip=mlip, **kwargs,
            )
            per_T[T_K] = result
            print(f"    D = {result['D_m2s']:.3e} m²/s  R² = {result['r2']:.4f}")
        except Exception as exc:
            warnings.warn(f"  T={T_K}K failed: {exc}", stacklevel=2)

    if not per_T:
        raise RuntimeError("All temperature runs failed.")

    temps  = sorted(per_T.keys())
    D_vals = [per_T[T]["D_m2s"] for T in temps]

    try:
        arrh = arrhenius_fit(temps, D_vals)
    except ValueError as exc:
        warnings.warn(f"Arrhenius fit failed: {exc}", stacklevel=2)
        arrh = {"Ea_eV": None, "D0_m2s": None, "r2": None}

    fig_arrh = plot_arrhenius(per_T, project, mlip=mlip)

    stem_arrh = f"{project}_arrhenius"
    html_arrh, png_arrh = _save_figure(fig_arrh, out, stem_arrh)

    arrh_csv = out / f"{stem_arrh}.csv"
    pd.DataFrame({
        "T_K":    temps,
        "D_m2s":  D_vals,
        "D_cm2s": [per_T[T]["D_cm2s"]  for T in temps],
        "r2_msd": [per_T[T]["r2"]       for T in temps],
    }).to_csv(arrh_csv, index=False)

    if arrh.get("Ea_eV") is not None:
        print(
            f"\n[arrhenius] Ea = {arrh['Ea_eV']:.3f} eV "
            f"({arrh['Ea_kJmol']:.1f} kJ/mol)  "
            f"D₀ = {arrh['D0_m2s']:.3e} m²/s  R² = {arrh['r2']:.4f}"
        )

    return {
        "per_T_results":    per_T,
        "arrhenius":        arrh,
        "arrhenius_figure": fig_arrh,
        "arrhenius_html":   html_arrh,
        "arrhenius_png":    png_arrh,
        "arrhenius_csv":    arrh_csv,
    }


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("msd.py — transport analysis module")
    print(f"  kB = {kB} eV/K")
    print(f"  1 Å²/ps = {ANG2_PER_PS_TO_M2_PER_S} m²/s")
    print()
    print("Quick-start example:")
    print("  from hpca.analysis.trajectory import parse_trajectory")
    print("  from hpca.analysis.msd import run_full_transport_analysis")
    print("  traj = parse_trajectory('dump_unwrapped.lmp')")
    print("  summary = run_full_transport_analysis(")
    print("      traj, 'Li', dt_ps=1.0, T_K=300,")
    print("      output_dir=Path('output'), project='LiTiCl')")
    print("  print(f\"D = {summary['D_m2s']:.3e} m²/s\")")
