"""
continuum_viz.py — Continuum interdiffusion / phase-field / SEI model plots.
HPCA Pipeline · /path/to/workspace/hpca/viz/continuum_viz.py

Python env: /path/to/apps/apps/cladue/env/bin/python3

All public functions return a go.Figure with NREL theme applied.
When output_path is provided the figure is saved as PNG and data as CSV alongside.

Reference: Ncube, Barai, Selvaraj et al., J. Energy Storage 2026
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go

from .theme import (
    NREL_COLORS,
    apply_nrel_theme,
    add_annotation_box,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_png_csv(
    fig: go.Figure,
    output_path: str | Path,
    data: np.ndarray,
    header: str,
) -> None:
    """Save figure PNG and companion CSV at the same path stem."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(p), scale=2)
    except Exception as exc:
        warnings.warn(
            f"[continuum_viz] PNG write failed ({exc}); saving HTML fallback.",
            RuntimeWarning,
            stacklevel=3,
        )
        fig.write_html(str(p.with_suffix(".html")), include_plotlyjs=True)
    np.savetxt(
        str(p.with_suffix(".csv")),
        data,
        delimiter=",",
        header=header,
        comments="",
    )


def _plasma_hex(t: float) -> str:
    """Interpolate plasma colorscale at position t ∈ [0,1] → hex color."""
    stops = [
        (0.000, (13, 8, 135)),
        (0.143, (84, 2, 163)),
        (0.286, (139, 10, 165)),
        (0.429, (185, 50, 137)),
        (0.571, (219, 92, 104)),
        (0.714, (244, 136, 73)),
        (0.857, (254, 188, 43)),
        (1.000, (252, 253, 191)),
    ]
    t = float(np.clip(t, 0.0, 1.0))
    for j in range(len(stops) - 1):
        t0, c0 = stops[j]
        t1, c1 = stops[j + 1]
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0)
            r = int(round(c0[0] + f * (c1[0] - c0[0])))
            g = int(round(c0[1] + f * (c1[1] - c0[1])))
            b = int(round(c0[2] + f * (c1[2] - c0[2])))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#fcfdbf"


# ---------------------------------------------------------------------------
# 1. plot_concentration_profile
# ---------------------------------------------------------------------------


def plot_concentration_profile(
    x_um: np.ndarray,
    c_t: np.ndarray,
    times_s: Sequence[float],
    title: Optional[str] = None,
    output_path: Optional[str | Path] = None,
) -> go.Figure:
    """Concentration profiles c(x,t) colored by time using plasma colorscale.

    c_t shape: (n_times, n_x).
    """
    x_um = np.asarray(x_um, dtype=float)
    c_t = np.asarray(c_t, dtype=float)
    times_s = list(times_s)
    n_times = len(times_s)

    fig = go.Figure()

    for i, (t_s, c_snap) in enumerate(zip(times_s, c_t)):
        color = _plasma_hex(i / max(n_times - 1, 1))
        t_label = _fmt_time(t_s)
        fig.add_trace(
            go.Scatter(
                x=x_um,
                y=c_snap,
                mode="lines",
                name=t_label,
                line=dict(color=color, width=2.0),
                hovertemplate=(
                    f"<b>t = {t_label}</b><br>"
                    "x = %{x:.3f} μm<br>"
                    "c = %{y:.6f}<extra></extra>"
                ),
            )
        )

    # Colorbar-like annotation
    add_annotation_box(
        fig,
        f"<b>Time range</b><br>{_fmt_time(times_s[0])} → {_fmt_time(times_s[-1])}",
        x=0.98,
        y=0.98,
    )
    fig.layout.annotations[-1].update(xanchor="right", yanchor="top")

    apply_nrel_theme(
        fig,
        title=title or "Concentration Profile c(x, t)",
        xlabel="Position  (μm)",
        ylabel="Concentration  (normalized)",
        width=960,
        height=600,
    )

    if output_path is not None:
        cols = [x_um] + [c_t[i] for i in range(n_times)]
        header_parts = ["x_um"] + [
            f"c_t{i}_{_fmt_time(t).replace(' ','')}" for i, t in enumerate(times_s)
        ]
        max_len = max(len(c) for c in cols)
        padded = [
            np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
            for c in cols
        ]
        _save_png_csv(fig, output_path, np.column_stack(padded), ",".join(header_parts))

    return fig


# ---------------------------------------------------------------------------
# 2. plot_phase_field
# ---------------------------------------------------------------------------


def plot_phase_field(
    x_um: np.ndarray,
    phi_t: np.ndarray,
    times_s: Sequence[float],
    output_path: Optional[str | Path] = None,
) -> go.Figure:
    """Phase-field order parameter φ(x,t) profiles colored by time.

    phi_t shape: (n_times, n_x).
    """
    x_um = np.asarray(x_um, dtype=float)
    phi_t = np.asarray(phi_t, dtype=float)
    times_s = list(times_s)
    n_times = len(times_s)

    fig = go.Figure()

    for i, (t_s, phi_snap) in enumerate(zip(times_s, phi_t)):
        color = _plasma_hex(i / max(n_times - 1, 1))
        t_label = _fmt_time(t_s)
        fig.add_trace(
            go.Scatter(
                x=x_um,
                y=phi_snap,
                mode="lines",
                name=t_label,
                line=dict(color=color, width=2.0),
                fill="tozeroy" if i == 0 else None,
                fillcolor="rgba(13, 8, 135, 0.06)" if i == 0 else None,
                hovertemplate=(
                    f"<b>t = {t_label}</b><br>"
                    "x = %{x:.3f} μm<br>"
                    "φ = %{y:.5f}<extra></extra>"
                ),
            )
        )

    # Guide lines at φ=0 and φ=1
    for yval, lbl in [(0.0, "φ=0"), (1.0, "φ=1")]:
        fig.add_hline(
            y=yval,
            line=dict(color="#CCCCCC", width=1.0, dash="dot"),
            annotation_text=lbl,
            annotation_position="right",
            annotation_font_size=9,
            annotation_font_color="#888888",
        )

    apply_nrel_theme(
        fig,
        title="Phase-Field Order Parameter φ(x, t)",
        xlabel="Position  (μm)",
        ylabel="Order Parameter  φ",
        width=960,
        height=580,
    )
    fig.update_layout(yaxis=dict(range=[-0.05, 1.10]))

    if output_path is not None:
        cols = [x_um] + [phi_t[i] for i in range(n_times)]
        header_parts = ["x_um"] + [f"phi_t{i}" for i in range(n_times)]
        max_len = max(len(c) for c in cols)
        padded = [
            np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
            for c in cols
        ]
        _save_png_csv(fig, output_path, np.column_stack(padded), ",".join(header_parts))

    return fig


# ---------------------------------------------------------------------------
# 3. plot_stress_profile
# ---------------------------------------------------------------------------


def plot_stress_profile(
    x_um: np.ndarray,
    sigma_MPa_t: np.ndarray,
    times_s: Sequence[float],
    output_path: Optional[str | Path] = None,
) -> go.Figure:
    """Vegard-law stress σ(x,t) profiles in MPa, colored by time.

    sigma_MPa_t shape: (n_times, n_x).
    """
    x_um = np.asarray(x_um, dtype=float)
    sigma_MPa_t = np.asarray(sigma_MPa_t, dtype=float)
    times_s = list(times_s)
    n_times = len(times_s)

    fig = go.Figure()

    for i, (t_s, sig_snap) in enumerate(zip(times_s, sigma_MPa_t)):
        color = _plasma_hex(i / max(n_times - 1, 1))
        t_label = _fmt_time(t_s)
        fig.add_trace(
            go.Scatter(
                x=x_um,
                y=sig_snap,
                mode="lines",
                name=t_label,
                line=dict(color=color, width=2.0),
                hovertemplate=(
                    f"<b>t = {t_label}</b><br>"
                    "x = %{x:.3f} μm<br>"
                    "σ = %{y:.3f} MPa<extra></extra>"
                ),
            )
        )

    # Zero-stress reference
    fig.add_hline(
        y=0.0,
        line=dict(color="#AAAAAA", width=1.2, dash="dash"),
        annotation_text="σ = 0",
        annotation_position="right",
        annotation_font_size=9,
        annotation_font_color="#888888",
    )

    apply_nrel_theme(
        fig,
        title="Vegard-Law Stress Profile σ(x, t)",
        xlabel="Position  (μm)",
        ylabel="Stress  (MPa)",
        width=960,
        height=580,
    )

    if output_path is not None:
        cols = [x_um] + [sigma_MPa_t[i] for i in range(n_times)]
        header_parts = ["x_um"] + [f"sigma_MPa_t{i}" for i in range(n_times)]
        max_len = max(len(c) for c in cols)
        padded = [
            np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
            for c in cols
        ]
        _save_png_csv(fig, output_path, np.column_stack(padded), ",".join(header_parts))

    return fig


# ---------------------------------------------------------------------------
# 4. plot_sei_growth
# ---------------------------------------------------------------------------


def plot_sei_growth(
    times_s: np.ndarray,
    thickness_nm: np.ndarray,
    A: Optional[float] = None,
    n: Optional[float] = None,
    output_path: Optional[str | Path] = None,
) -> go.Figure:
    """SEI layer thickness vs time with optional power-law overlay L = A·t^n."""
    times_s = np.asarray(times_s, dtype=float)
    thickness_nm = np.asarray(thickness_nm, dtype=float)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=times_s,
            y=thickness_nm,
            mode="lines+markers",
            name="SEI thickness",
            line=dict(color=NREL_COLORS[0], width=2.5),
            marker=dict(size=6, symbol="circle",
                        line=dict(color="#FFFFFF", width=1)),
            hovertemplate=(
                "t = %{x:.4g} s<br>"
                "L = %{y:.4f} nm<extra></extra>"
            ),
        )
    )

    # Power-law fit overlay
    fit_shown = False
    if A is not None and n is not None and np.isfinite(A) and np.isfinite(n):
        t_fit = np.linspace(times_s[times_s > 0].min() if (times_s > 0).any()
                            else times_s.min() + 1e-12,
                            times_s.max(), 300)
        L_fit = A * t_fit ** n
        fig.add_trace(
            go.Scatter(
                x=t_fit,
                y=L_fit,
                mode="lines",
                name=f"L = {A:.3f}·t^{n:.3f}",
                line=dict(color=NREL_COLORS[1], width=2.0, dash="dash"),
                hovertemplate=(
                    f"Power law  A={A:.3f}, n={n:.3f}<br>"
                    "t = %{x:.4g} s<br>"
                    "L = %{y:.4f} nm<extra></extra>"
                ),
            )
        )
        fit_shown = True

    if fit_shown:
        add_annotation_box(
            fig,
            f"<b>Power law</b><br>L = A·t<sup>n</sup><br>A = {A:.4f} nm<br>n = {n:.4f}",
            x=0.02,
            y=0.98,
        )

    apply_nrel_theme(
        fig,
        title="SEI Layer Growth  L(t)",
        xlabel="Time  (s)",
        ylabel="SEI Thickness  (nm)",
        width=880,
        height=560,
    )

    if output_path is not None:
        if fit_shown:
            t_grid = np.linspace(times_s.min(), times_s.max(), len(times_s))
            L_grid = A * np.clip(t_grid, 0, None) ** n
            data_arr = np.column_stack([times_s, thickness_nm,
                                        A * np.clip(times_s, 0, None) ** n])
            _save_png_csv(fig, output_path, data_arr,
                          "time_s,thickness_nm,thickness_fit_nm")
        else:
            data_arr = np.column_stack([times_s, thickness_nm])
            _save_png_csv(fig, output_path, data_arr, "time_s,thickness_nm")

    return fig


# ---------------------------------------------------------------------------
# 5. plot_kjma
# ---------------------------------------------------------------------------


def plot_kjma(
    times_s: np.ndarray,
    alpha_t: np.ndarray,
    k: Optional[float] = None,
    n: Optional[float] = None,
    output_path: Optional[str | Path] = None,
) -> go.Figure:
    """KJMA (Avrami) phase-transformation kinetics α(t) = 1 − exp(−k·t^n)."""
    times_s = np.asarray(times_s, dtype=float)
    alpha_t = np.asarray(alpha_t, dtype=float)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=times_s,
            y=alpha_t,
            mode="lines+markers",
            name="α(t) — simulated",
            line=dict(color=NREL_COLORS[0], width=2.5),
            marker=dict(size=6, symbol="circle",
                        line=dict(color="#FFFFFF", width=1)),
            hovertemplate=(
                "t = %{x:.4g} s<br>"
                "α = %{y:.5f}<extra></extra>"
            ),
        )
    )

    fit_shown = False
    if k is not None and n is not None and np.isfinite(k) and np.isfinite(n):
        t_fit = np.linspace(0.0, times_s.max(), 400)
        alpha_fit = 1.0 - np.exp(-k * t_fit ** n)
        fig.add_trace(
            go.Scatter(
                x=t_fit,
                y=alpha_fit,
                mode="lines",
                name=f"Avrami: k={k:.4g}, n={n:.3f}",
                line=dict(color=NREL_COLORS[1], width=2.0, dash="dash"),
                hovertemplate=(
                    f"Avrami  k={k:.4g}, n={n:.3f}<br>"
                    "t = %{x:.4g} s<br>"
                    "α_fit = %{y:.5f}<extra></extra>"
                ),
            )
        )
        fit_shown = True

    if fit_shown:
        add_annotation_box(
            fig,
            f"<b>Avrami fit</b><br>α = 1 − exp(−k·t<sup>n</sup>)<br>"
            f"k = {k:.4g}<br>n = {n:.3f}",
            x=0.02,
            y=0.98,
        )

    # Reference lines at α=0 and α=1
    for yval, lbl in [(0.0, "α=0"), (1.0, "α=1")]:
        fig.add_hline(
            y=yval,
            line=dict(color="#CCCCCC", width=1.0, dash="dot"),
            annotation_text=lbl,
            annotation_position="right",
            annotation_font_size=9,
            annotation_font_color="#888888",
        )

    apply_nrel_theme(
        fig,
        title="KJMA Phase-Transformation Kinetics  α(t)",
        xlabel="Time  (s)",
        ylabel="Transformed Fraction  α",
        width=880,
        height=560,
    )
    fig.update_layout(yaxis=dict(range=[-0.03, 1.08]))

    if output_path is not None:
        if fit_shown:
            alpha_fit_at_data = 1.0 - np.exp(-k * np.clip(times_s, 0, None) ** n)
            data_arr = np.column_stack([times_s, alpha_t, alpha_fit_at_data])
            _save_png_csv(fig, output_path, data_arr,
                          "time_s,alpha_simulated,alpha_avrami_fit")
        else:
            data_arr = np.column_stack([times_s, alpha_t])
            _save_png_csv(fig, output_path, data_arr, "time_s,alpha_simulated")

    return fig


# ---------------------------------------------------------------------------
# Internal: time formatter
# ---------------------------------------------------------------------------


def _fmt_time(t_s: float) -> str:
    """Format a time in seconds to a readable string with appropriate units."""
    if t_s == 0:
        return "0 s"
    abs_t = abs(t_s)
    if abs_t < 1e-9:
        return f"{t_s * 1e12:.3g} ps"
    if abs_t < 1e-6:
        return f"{t_s * 1e9:.3g} ns"
    if abs_t < 1e-3:
        return f"{t_s * 1e6:.3g} μs"
    if abs_t < 1.0:
        return f"{t_s * 1e3:.3g} ms"
    if abs_t < 3600:
        return f"{t_s:.3g} s"
    return f"{t_s / 3600:.3g} h"
