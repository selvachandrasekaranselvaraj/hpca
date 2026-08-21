"""
structure.py — RDF, coordination, Van Hove, and displacement visualizations.
HPCA Pipeline · /path/to/workspace/hpca/viz/structure.py

Python env: /path/to/apps/apps/cladue/env/bin/python3

All public functions return a go.Figure with NREL theme applied.
When output_path is provided the figure is saved as PNG and underlying data
as a CSV alongside it (same stem, different extension).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks
from scipy.stats import norm as sp_norm

from .theme import (
    NREL_COLORS,
    NREL_LIGHT_COLORS,
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
    """Save figure as PNG and data array as CSV alongside it."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(p), scale=2)
    except Exception as exc:
        warnings.warn(
            f"[structure] PNG write failed ({exc}); saving HTML fallback.",
            RuntimeWarning,
            stacklevel=3,
        )
        fig.write_html(str(p.with_suffix(".html")), include_plotlyjs=True)
    csv_path = p.with_suffix(".csv")
    np.savetxt(
        str(csv_path),
        data,
        delimiter=",",
        header=header,
        comments="",
    )


def _plasma_color(i: int, n: int) -> str:
    """Return a hex color sampled from the plasma colorscale for index i of n."""
    # Plasma: deep purple → blue → yellow
    _plasma = [
        "#0d0887", "#3e049c", "#6600a4", "#8b0aa5",
        "#ae2891", "#cc4778", "#e46e5a", "#f89441",
        "#fdc328", "#f0f921",
    ]
    if n <= 1:
        return _plasma[0]
    idx = int(round(i / (n - 1) * (len(_plasma) - 1)))
    return _plasma[max(0, min(idx, len(_plasma) - 1))]


def _blue_red_color(i: int, n: int) -> str:
    """Return a color interpolated from blue to red for index i of n."""
    t = i / max(n - 1, 1)
    r = int(round(t * 227 + (1 - t) * 0))
    g = int(round(t * 28 + (1 - t) * 121))
    b = int(round(t * 61 + (1 - t) * 194))
    return f"rgb({r},{g},{b})"


# ---------------------------------------------------------------------------
# 1. plot_rdf
# ---------------------------------------------------------------------------


def plot_rdf(
    r: np.ndarray,
    g_r_dict: dict[str, np.ndarray],
    title: Optional[str] = None,
    output_path: Optional[str | Path] = None,
) -> go.Figure:
    """Plot radial distribution functions for one or more atom-pair labels."""
    r = np.asarray(r, dtype=float)
    fig = go.Figure()

    all_curves: list[np.ndarray] = []
    col_headers: list[str] = ["r_angstrom"]

    for i, (label, g_r) in enumerate(g_r_dict.items()):
        g_r = np.asarray(g_r, dtype=float)
        color = NREL_COLORS[i % len(NREL_COLORS)]
        all_curves.append(g_r)
        col_headers.append(f"g_r_{label.replace('-','_')}")

        fig.add_trace(
            go.Scatter(
                x=r,
                y=g_r,
                mode="lines",
                name=label,
                line=dict(color=color, width=2.5),
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "r = %{x:.3f} Å<br>"
                    "g(r) = %{y:.4f}<extra></extra>"
                ),
            )
        )

        # Detect first peak and add vertical dashed line
        peaks, props = find_peaks(g_r, height=1.2, distance=5)
        if len(peaks) > 0:
            first_peak_idx = peaks[0]
            r_peak = float(r[first_peak_idx])
            g_peak = float(g_r[first_peak_idx])
            fig.add_vline(
                x=r_peak,
                line=dict(color=color, width=1.2, dash="dash"),
                annotation_text=f"{r_peak:.2f} Å",
                annotation_position="top",
                annotation_font_size=10,
                annotation_font_color=color,
            )

    # Reference line at g(r)=1 (uncorrelated)
    fig.add_hline(
        y=1.0,
        line=dict(color="#CCCCCC", width=1.0, dash="dot"),
        annotation_text="g(r)=1",
        annotation_position="right",
        annotation_font_size=9,
        annotation_font_color="#888888",
    )

    apply_nrel_theme(
        fig,
        title=title or "Radial Distribution Function",
        xlabel="r  (Å)",
        ylabel="g(r)",
        width=900,
        height=580,
    )
    fig.update_layout(
        yaxis=dict(rangemode="tozero"),
    )

    if output_path is not None:
        data_cols = [r] + all_curves
        data_arr = np.column_stack(data_cols)
        _save_png_csv(fig, output_path, data_arr, ",".join(col_headers))

    return fig


# ---------------------------------------------------------------------------
# 2. plot_coordination_histogram
# ---------------------------------------------------------------------------


def plot_coordination_histogram(
    cn_dist: dict[int, float],
    label: Optional[str] = None,
    output_path: Optional[str | Path] = None,
) -> go.Figure:
    """Bar chart of coordination number distribution."""
    cn_vals = sorted(cn_dist.keys())
    counts = [cn_dist[cn] for cn in cn_vals]
    total = sum(counts)
    fracs = [c / total if total > 0 else 0.0 for c in counts]

    fig = go.Figure(
        go.Bar(
            x=cn_vals,
            y=fracs,
            marker_color=NREL_COLORS[0],
            marker_opacity=0.88,
            hovertemplate=(
                "CN = %{x}<br>"
                "Fraction = %{y:.4f}<extra></extra>"
            ),
            name=label or "CN distribution",
        )
    )

    apply_nrel_theme(
        fig,
        title=f"Coordination Number Distribution{' — ' + label if label else ''}",
        xlabel="Coordination Number",
        ylabel="Fraction",
        width=700,
        height=500,
    )
    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=cn_vals),
        yaxis=dict(rangemode="tozero"),
    )

    if output_path is not None:
        data_arr = np.column_stack([cn_vals, fracs])
        _save_png_csv(fig, output_path, data_arr, "coordination_number,fraction")

    return fig


# ---------------------------------------------------------------------------
# 3. plot_bond_angle
# ---------------------------------------------------------------------------


def plot_bond_angle(
    angles_deg: np.ndarray,
    distribution: np.ndarray,
    label: Optional[str] = None,
    output_path: Optional[str | Path] = None,
) -> go.Figure:
    """Plot bond-angle distribution as a filled area trace."""
    angles_deg = np.asarray(angles_deg, dtype=float)
    distribution = np.asarray(distribution, dtype=float)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=angles_deg,
            y=distribution,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(0, 121, 194, 0.15)",
            line=dict(color=NREL_COLORS[0], width=2.5),
            name=label or "Bond angle",
            hovertemplate=(
                "θ = %{x:.1f}°<br>"
                "P(θ) = %{y:.4f}<extra></extra>"
            ),
        )
    )

    # Mark peak angle
    peaks, _ = find_peaks(distribution, height=distribution.max() * 0.3, distance=5)
    if len(peaks) > 0:
        peak_angle = float(angles_deg[peaks[0]])
        fig.add_vline(
            x=peak_angle,
            line=dict(color=NREL_COLORS[3], width=1.5, dash="dash"),
            annotation_text=f"{peak_angle:.1f}°",
            annotation_position="top",
            annotation_font_size=10,
            annotation_font_color=NREL_COLORS[3],
        )

    apply_nrel_theme(
        fig,
        title=f"Bond Angle Distribution{' — ' + label if label else ''}",
        xlabel="Bond Angle  (degrees)",
        ylabel="Probability Density",
        width=800,
        height=520,
    )

    if output_path is not None:
        data_arr = np.column_stack([angles_deg, distribution])
        _save_png_csv(fig, output_path, data_arr, "angle_deg,probability_density")

    return fig


# ---------------------------------------------------------------------------
# 4. plot_vanhove
# ---------------------------------------------------------------------------


def plot_vanhove(
    r: np.ndarray,
    G_dict: dict[str, np.ndarray],
    title: Optional[str] = None,
    output_path: Optional[str | Path] = None,
) -> go.Figure:
    """Van Hove self-correlation G_s(r,t) with blue→red colorscale by dt."""
    r = np.asarray(r, dtype=float)
    fig = go.Figure()
    n = len(G_dict)

    col_headers = ["r_angstrom"] + [
        f"G_s_{lbl.replace(' ', '_').replace('=', '').replace('.', 'p')}"
        for lbl in G_dict.keys()
    ]
    all_curves: list[np.ndarray] = []

    for i, (dt_label, G_s) in enumerate(G_dict.items()):
        G_s = np.asarray(G_s, dtype=float)
        color = _blue_red_color(i, n)
        all_curves.append(G_s)

        fig.add_trace(
            go.Scatter(
                x=r,
                y=G_s,
                mode="lines",
                name=dt_label,
                line=dict(color=color, width=2.0),
                hovertemplate=(
                    f"<b>t = {dt_label}</b><br>"
                    "r = %{x:.3f} Å<br>"
                    "G_s = %{y:.5f}<extra></extra>"
                ),
            )
        )

    apply_nrel_theme(
        fig,
        title=title or "Van Hove Self-Correlation G_s(r,t)",
        xlabel="r  (Å)",
        ylabel="G_s(r, t)  (Å⁻³)",
        width=900,
        height=600,
    )
    fig.update_layout(
        yaxis=dict(rangemode="tozero"),
        legend=dict(
            title=dict(text="Time lag", font=dict(size=11)),
        ),
    )

    if output_path is not None:
        data_arr = np.column_stack([r] + all_curves)
        _save_png_csv(fig, output_path, data_arr, ",".join(col_headers))

    return fig


# ---------------------------------------------------------------------------
# 5. plot_non_gaussian
# ---------------------------------------------------------------------------


def plot_non_gaussian(
    lags: np.ndarray,
    alpha2: np.ndarray,
    output_path: Optional[str | Path] = None,
) -> go.Figure:
    """Non-Gaussian parameter α₂(t) vs time lag."""
    lags = np.asarray(lags, dtype=float)
    alpha2 = np.asarray(alpha2, dtype=float)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=lags,
            y=alpha2,
            mode="lines+markers",
            name="α₂(t)",
            line=dict(color=NREL_COLORS[0], width=2.5),
            marker=dict(size=6, symbol="circle",
                        line=dict(color="#FFFFFF", width=1)),
            hovertemplate=(
                "t = %{x:.1f} ps<br>"
                "α₂ = %{y:.5f}<extra></extra>"
            ),
        )
    )

    # Zero reference
    fig.add_hline(
        y=0.0,
        line=dict(color="#AAAAAA", width=1.0, dash="dash"),
    )

    # Annotate peak α₂
    if len(alpha2) > 0:
        peak_idx = int(np.nanargmax(alpha2))
        peak_t = float(lags[peak_idx])
        peak_a = float(alpha2[peak_idx])
        add_annotation_box(
            fig,
            f"Peak α₂ = {peak_a:.4f}<br>at t = {peak_t:.1f} ps",
            x=0.98,
            y=0.98,
        )
        fig.layout.annotations[-1].update(xanchor="right", yanchor="top")

    apply_nrel_theme(
        fig,
        title="Non-Gaussian Parameter α₂(t)",
        xlabel="Time Lag  (ps)",
        ylabel="α₂(t)",
        width=850,
        height=520,
    )

    if output_path is not None:
        data_arr = np.column_stack([lags, alpha2])
        _save_png_csv(fig, output_path, data_arr, "lag_ps,alpha2")

    return fig


# ---------------------------------------------------------------------------
# 6. plot_displacement_distribution
# ---------------------------------------------------------------------------


def plot_displacement_distribution(
    disp_bins_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    dt_labels: list[str],
    output_path: Optional[str | Path] = None,
) -> go.Figure:
    """Displacement distribution P(|Δr|) with log y-axis and Gaussian reference.

    disp_bins_dict maps dt_label → (bin_centers, counts/density) array pair.
    dt_labels controls the legend order (must match disp_bins_dict keys).
    """
    fig = go.Figure()
    n = len(dt_labels)

    col_headers: list[str] = []
    all_data_cols: list[np.ndarray] = []

    # Collect a common r grid for Gaussian reference from first series
    ref_r: Optional[np.ndarray] = None
    ref_sigma: float = 1.0

    for i, lbl in enumerate(dt_labels):
        if lbl not in disp_bins_dict:
            continue
        bin_centers, density = disp_bins_dict[lbl]
        bin_centers = np.asarray(bin_centers, dtype=float)
        density = np.asarray(density, dtype=float)
        color = _blue_red_color(i, n)

        if ref_r is None:
            ref_r = bin_centers
            # Estimate sigma from the weighted mean-square displacement
            valid = density > 0
            if valid.any():
                mu2 = np.trapz(bin_centers[valid] ** 2 * density[valid],
                               bin_centers[valid])
                ref_sigma = float(np.sqrt(max(mu2, 1e-6)))

        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=density,
                mode="lines",
                name=lbl,
                line=dict(color=color, width=2.0),
                hovertemplate=(
                    f"<b>{lbl}</b><br>"
                    "|Δr| = %{x:.3f} Å<br>"
                    "P = %{y:.4e}<extra></extra>"
                ),
            )
        )
        col_headers += [
            f"r_{lbl.replace(' ', '_')}",
            f"P_{lbl.replace(' ', '_')}",
        ]
        all_data_cols += [bin_centers, density]

    # Gaussian reference overlay
    if ref_r is not None:
        gauss_r = np.linspace(ref_r[0], ref_r[-1], 300)
        # 3-D Maxwell-Boltzmann-like: 4π r² × Gaussian(σ)
        gauss_y = (
            (1.0 / (ref_sigma * np.sqrt(2 * np.pi))) ** 3
            * np.exp(-0.5 * (gauss_r / ref_sigma) ** 2)
            * 4 * np.pi * gauss_r ** 2
        )
        fig.add_trace(
            go.Scatter(
                x=gauss_r,
                y=gauss_y,
                mode="lines",
                name="Gaussian ref.",
                line=dict(color="#999999", width=1.5, dash="dash"),
                hovertemplate=(
                    "<b>Gaussian ref</b><br>"
                    "|Δr| = %{x:.3f} Å<br>"
                    "P = %{y:.4e}<extra></extra>"
                ),
            )
        )

    apply_nrel_theme(
        fig,
        title="Displacement Distribution P(|Δr|)",
        xlabel="|Δr|  (Å)",
        ylabel="Probability Density",
        width=900,
        height=580,
    )
    fig.update_layout(yaxis_type="log")

    if output_path is not None and all_data_cols:
        # Pad arrays to the same length before stacking
        max_len = max(len(c) for c in all_data_cols)
        padded = [
            np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
            for c in all_data_cols
        ]
        data_arr = np.column_stack(padded)
        _save_png_csv(fig, output_path, data_arr, ",".join(col_headers))

    return fig
