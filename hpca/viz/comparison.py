"""
comparison.py — Cross-project and cross-MLIP comparison dashboards
HPCA Pipeline · /path/to/workspace/hpca/viz/comparison.py

Python env: /path/to/apps/apps/cladue/env/bin/python3

All public functions return go.Figure (or write HTML for the report builder).
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

from .theme import (
    NREL_COLORS,
    NREL_SEQUENTIAL_COLORSCALE,
    NREL_DIVERGING_COLORSCALE,
    apply_nrel_theme,
    add_annotation_box,
    save_figure,
)

# ---------------------------------------------------------------------------
# Internal colour helpers
# ---------------------------------------------------------------------------


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert a 6-digit hex colour string to an ``rgba()`` CSS string.

    Parameters
    ----------
    hex_color : str
        Hex colour, e.g. ``"#0079C2"`` (with or without leading ``#``).
    alpha : float
        Opacity in [0, 1].

    Returns
    -------
    str
        E.g. ``"rgba(0, 121, 194, 0.15)"``.
    """
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha:.3f})"


# ---------------------------------------------------------------------------
# 1. plot_mlip_benchmark_heatmap
# ---------------------------------------------------------------------------


def plot_mlip_benchmark_heatmap(
    benchmark_df: pd.DataFrame,
    metric: str = "fmax",
) -> go.Figure:
    """Heatmap of MLIP × project benchmark values.

    Parameters
    ----------
    benchmark_df : pd.DataFrame
        Must contain columns ``"mlip"``, ``"project"``, and the *metric* column.
        May contain multiple rows per (mlip, project) — the mean is taken.
    metric : str
        Column name to visualise (e.g. ``"fmax"``, ``"energy_rmse"``,
        ``"force_rmse"``).

    Returns
    -------
    go.Figure
        Interactive annotated heatmap.
    """
    pivot = (
        benchmark_df.groupby(["mlip", "project"])[metric]
        .mean()
        .unstack("project")
    )
    mlips = list(pivot.index)
    projects = list(pivot.columns)
    z = pivot.values.astype(float)

    # Build annotation text matrix
    text_matrix = [
        [f"{v:.3g}" if np.isfinite(v) else "N/A" for v in row]
        for row in z
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=projects,
            y=mlips,
            text=text_matrix,
            texttemplate="%{text}",
            textfont={"size": 11, "color": "#1A1A1A"},
            colorscale=NREL_SEQUENTIAL_COLORSCALE,
            reversescale=False,
            showscale=True,
            xgap=2,
            ygap=2,
            hovertemplate=(
                "<b>MLIP:</b> %{y}<br>"
                "<b>Project:</b> %{x}<br>"
                f"<b>{metric}:</b> %{{z:.4g}}<extra></extra>"
            ),
            colorbar=dict(
                title=dict(text=metric, side="right", font=dict(size=13)),
                tickfont=dict(size=11),
                outlinecolor="#AAAAAA",
                outlinewidth=1,
            ),
        )
    )

    apply_nrel_theme(
        fig,
        title=f"MLIP Benchmark — {metric}",
        xlabel="Project",
        ylabel="MLIP",
        width=max(700, 140 * len(projects)),
        height=max(500, 80 * len(mlips) + 150),
    )
    fig.update_layout(
        xaxis_tickangle=-35,
        yaxis_autorange="reversed",  # top MLIP first
    )
    return fig


# ---------------------------------------------------------------------------
# 2. plot_benchmark_radar
# ---------------------------------------------------------------------------


def plot_benchmark_radar(
    project: str,
    mlip_data: dict[str, dict],
) -> go.Figure:
    """Spider/radar chart: energy RMSE, force RMSE, Fmax, wall-time per MLIP.

    Parameters
    ----------
    project : str
        Project name for the title.
    mlip_data : dict[str, dict]
        ``{mlip_name: {metric: value, ...}}``.
        Recognised metrics: ``"energy_rmse"`` (meV/atom), ``"force_rmse"``
        (meV/Å), ``"fmax"`` (eV/Å), ``"walltime_h"`` (hours), and any
        additional custom metrics.  Values are normalised per metric before
        plotting.

    Returns
    -------
    go.Figure
    """
    if not mlip_data:
        raise ValueError("mlip_data must not be empty.")

    # Collect all metrics present
    all_metrics: list[str] = []
    for d in mlip_data.values():
        for k in d:
            if k not in all_metrics:
                all_metrics.append(k)

    # Friendly display names
    _display = {
        "energy_rmse": "E RMSE<br>(meV/atom)",
        "force_rmse": "F RMSE<br>(meV/Å)",
        "fmax": "F_max<br>(eV/Å)",
        "walltime_h": "Wall-time<br>(h)",
    }
    theta_labels = [_display.get(m, m) for m in all_metrics]
    theta_labels.append(theta_labels[0])   # close the polygon

    # Normalise each metric to [0, 1] using max across MLIPs
    raw_values: dict[str, list[float]] = {
        m: [mlip_data[ml].get(m, float("nan")) for ml in mlip_data]
        for m in all_metrics
    }
    col_max = {
        m: np.nanmax(v) if any(np.isfinite(x) for x in v) else 1.0
        for m, v in raw_values.items()
    }

    fig = go.Figure()

    for i, (mlip_name, metrics_dict) in enumerate(mlip_data.items()):
        color = NREL_COLORS[i % len(NREL_COLORS)]
        r_vals = [
            metrics_dict.get(m, float("nan")) / col_max[m]
            if col_max[m] > 0 else 0.0
            for m in all_metrics
        ]
        r_vals.append(r_vals[0])   # close polygon

        raw_display = [
            f"{metrics_dict.get(m, float('nan')):.4g}" for m in all_metrics
        ]
        raw_display.append(raw_display[0])

        # Convert hex colour to rgba() for the translucent fill
        fill_color = _hex_to_rgba(color, alpha=0.15)

        fig.add_trace(
            go.Scatterpolar(
                r=r_vals,
                theta=theta_labels,
                fill="toself",
                fillcolor=fill_color,
                line=dict(color=color, width=2.5),
                name=mlip_name,
                customdata=np.array(raw_display),
                hovertemplate=(
                    f"<b>{mlip_name}</b><br>"
                    "Metric: %{theta}<br>"
                    "Raw value: %{customdata}<br>"
                    "Normalised: %{r:.3f}<extra></extra>"
                ),
            )
        )

    apply_nrel_theme(
        fig,
        title=f"MLIP Benchmark Radar — {project}",
        width=720,
        height=620,
    )
    fig.update_layout(
        polar=dict(
            bgcolor="#FAFAFA",
            radialaxis=dict(
                visible=True,
                range=[0, 1.05],
                tickfont=dict(size=10, color="#666666"),
                gridcolor="#E0E0E0",
                linecolor="#CCCCCC",
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color="#1A1A1A"),
                gridcolor="#E0E0E0",
                linecolor="#CCCCCC",
                direction="clockwise",
            ),
        ),
        legend=dict(
            x=1.12, y=1.0,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#CCCCCC",
            borderwidth=1,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. plot_crossproject_diffusivity
# ---------------------------------------------------------------------------


def plot_crossproject_diffusivity(
    projects: list[str],
    D_values: list[float],
    Ea_values: list[float],
    categories: list[str],
) -> go.Figure:
    """Scatter D_300K vs Ea_eV, colour-coded by material category.

    Parameters
    ----------
    projects : list[str]
        Project names (used in hover text and point labels).
    D_values : list[float]
        Li diffusivity at 300 K in m²/s — one per project.
    Ea_values : list[float]
        Activation energy in eV — one per project.
    categories : list[str]
        Material category labels (e.g. "halide", "oxide", "sulfide") — one
        per project.  Determines point colour.

    Returns
    -------
    go.Figure
    """
    unique_cats = list(dict.fromkeys(categories))   # preserves order
    cat_color = {c: NREL_COLORS[i % len(NREL_COLORS)] for i, c in enumerate(unique_cats)}

    fig = go.Figure()

    for cat in unique_cats:
        mask = [c == cat for c in categories]
        pr_cat = [p for p, m in zip(projects, mask) if m]
        D_cat = [d for d, m in zip(D_values, mask) if m]
        Ea_cat = [e for e, m in zip(Ea_values, mask) if m]

        log_D_cat = [np.log10(max(d, 1e-300)) for d in D_cat]

        fig.add_trace(
            go.Scatter(
                x=Ea_cat,
                y=log_D_cat,
                mode="markers+text",
                name=cat,
                text=pr_cat,
                textposition="top center",
                textfont=dict(size=10, color="#333333"),
                marker=dict(
                    color=cat_color[cat],
                    size=14,
                    opacity=0.85,
                    symbol="circle",
                    line=dict(color="#FFFFFF", width=1.5),
                ),
                customdata=np.column_stack([pr_cat, D_cat, Ea_cat]),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Category: " + cat + "<br>"
                    "Eₐ = %{customdata[2]:.3f} eV<br>"
                    "D₃₀₀ₖ = %{customdata[1]:.3e} m²/s<br>"
                    "log₁₀(D) = %{y:.3f}<extra></extra>"
                ),
            )
        )

    # Guide lines: high / moderate / low diffusivity bands
    Ea_range = [min(Ea_values) * 0.85, max(Ea_values) * 1.10]
    for log_d_ref, label, dash in [
        (-9, "10⁻⁹ m²/s", "dot"),
        (-11, "10⁻¹¹ m²/s", "dash"),
        (-13, "10⁻¹³ m²/s", "longdash"),
    ]:
        fig.add_hline(
            y=log_d_ref,
            line=dict(color="#CCCCCC", width=1.2, dash=dash),
            annotation_text=label,
            annotation_position="right",
            annotation_font_size=10,
            annotation_font_color="#888888",
        )

    apply_nrel_theme(
        fig,
        title="Cross-Project Li⁺ Diffusivity vs Activation Energy",
        xlabel="Activation Energy  Eₐ  (eV)",
        ylabel="log₁₀ D₃₀₀K  (m²/s)",
        width=900,
        height=620,
    )
    return fig


# ---------------------------------------------------------------------------
# 4. plot_sei_comparison
# ---------------------------------------------------------------------------


def plot_sei_comparison(
    sei_results: dict[str, dict],
) -> go.Figure:
    """Grouped bar chart: SEI thickness and growth rate per project.

    Parameters
    ----------
    sei_results : dict[str, dict]
        ``{project: {"thickness_nm": float, "growth_rate_nm_ns": float,
                     "error_thickness": float, "error_growth": float}}``.
        Error keys are optional (omit for bars without error bars).

    Returns
    -------
    go.Figure with two y-axes: thickness (left) and growth rate (right).
    """
    projects = list(sei_results.keys())
    thicknesses = [sei_results[p].get("thickness_nm", float("nan")) for p in projects]
    growth_rates = [sei_results[p].get("growth_rate_nm_ns", float("nan")) for p in projects]
    err_t = [sei_results[p].get("error_thickness", 0.0) for p in projects]
    err_g = [sei_results[p].get("error_growth", 0.0) for p in projects]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            name="SEI Thickness (nm)",
            x=projects,
            y=thicknesses,
            error_y=dict(type="data", array=err_t, visible=True,
                         thickness=1.5, width=5, color="#333333"),
            marker_color=NREL_COLORS[0],
            marker_opacity=0.88,
            offsetgroup="a",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "SEI thickness = %{y:.2f} nm<extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            name="Growth rate (nm/ns)",
            x=projects,
            y=growth_rates,
            error_y=dict(type="data", array=err_g, visible=True,
                         thickness=1.5, width=5, color="#333333"),
            marker_color=NREL_COLORS[1],
            marker_opacity=0.88,
            offsetgroup="b",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Growth rate = %{y:.4f} nm/ns<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    apply_nrel_theme(
        fig,
        title="SEI Layer Comparison Across Projects",
        xlabel="Project",
        ylabel="SEI Thickness  (nm)",
        width=max(700, 160 * len(projects)),
        height=560,
    )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    fig.update_yaxes(title_text="Growth Rate  (nm/ns)", secondary_y=True)
    return fig


# ---------------------------------------------------------------------------
# 5. plot_continuum_summary_dashboard
# ---------------------------------------------------------------------------


def plot_continuum_summary_dashboard(
    continuum_results: dict,
) -> go.Figure:
    """2×3 subplot grid summarising continuum model results.

    Subplots (left-to-right, top-to-bottom):
    1. Arrhenius (D vs 1000/T)
    2. SEI growth (L vs t^n power law)
    3. Phase-field order parameter profile
    4. Stress profile (Vegard law)
    5. Ionic conductivity vs T (VTF)
    6. Swelling / volume change vs concentration

    Parameters
    ----------
    continuum_results : dict
        Keys used (all optional; missing panels show a placeholder):
        - ``"arrhenius"``:   ``{label: {T_K: D_m2s, ...}}``
        - ``"sei_growth"``:  ``{"time_ns": arr, "L_nm": arr, "A": float, "n": float}``
        - ``"phase_field"``: ``{"x_nm": arr, "phi": arr}``
        - ``"stress"``:      ``{"x_nm": arr, "sigma_GPa": arr}``
        - ``"conductivity"``:``{"T_K": arr, "sigma": arr}``
        - ``"swelling"``:    ``{"c_mol_m3": arr, "dV_frac": arr}``

    Returns
    -------
    go.Figure
    """
    subplot_titles = [
        "Arrhenius", "SEI Growth", "Phase Field",
        "Stress (Vegard)", "Conductivity", "Swelling",
    ]
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.11,
        vertical_spacing=0.18,
    )

    # ---- helper: add placeholder ----
    def _placeholder(row: int, col: int, label: str) -> None:
        """Add a greyed-out italic placeholder annotation to subplot (row, col)."""
        fig.add_annotation(
            text=f"<i>{label}<br>data not provided</i>",
            xref=f"x{(row-1)*3+col} domain" if (row-1)*3+col > 1 else "x domain",
            yref=f"y{(row-1)*3+col} domain" if (row-1)*3+col > 1 else "y domain",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=11, color="#888888"),
            xanchor="center", yanchor="middle",
        )

    # ---- 1. Arrhenius ----
    ar = continuum_results.get("arrhenius")
    if ar:
        for i, (label, T_D_map) in enumerate(ar.items()):
            color = NREL_COLORS[i % len(NREL_COLORS)]
            T_arr = np.array(sorted(T_D_map), dtype=float)
            D_arr = np.array([T_D_map[t] for t in T_arr])
            inv_T = 1000.0 / T_arr
            log_D = np.log10(D_arr)
            fig.add_trace(
                go.Scatter(
                    x=inv_T, y=log_D, mode="lines+markers",
                    name=label,
                    line=dict(color=color, width=2.0),
                    marker=dict(size=7),
                    hovertemplate=(
                        f"<b>{label}</b><br>1000/T=%{{x:.3f}}<br>"
                        "log D=%{y:.2f}<extra></extra>"
                    ),
                    showlegend=True,
                ),
                row=1, col=1,
            )
    else:
        _placeholder(1, 1, "Arrhenius")

    # ---- 2. SEI growth ----
    sg = continuum_results.get("sei_growth")
    if sg:
        t_ns = np.asarray(sg["time_ns"], dtype=float)
        L_nm = np.asarray(sg["L_nm"], dtype=float)
        A = sg.get("A", float("nan"))
        n = sg.get("n", float("nan"))
        fig.add_trace(
            go.Scatter(
                x=t_ns, y=L_nm, mode="lines",
                name="SEI growth",
                line=dict(color=NREL_COLORS[1], width=2.5),
                hovertemplate=(
                    "t = %{x:.2f} ns<br>L = %{y:.3f} nm<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=2,
        )
        if np.isfinite(A) and np.isfinite(n):
            t_fit = np.linspace(t_ns.min(), t_ns.max(), 100)
            L_fit = A * t_fit ** n
            fig.add_trace(
                go.Scatter(
                    x=t_fit, y=L_fit, mode="lines",
                    name=f"L=At^n, A={A:.3f}, n={n:.3f}",
                    line=dict(color=NREL_COLORS[1], width=1.5, dash="dash"),
                    hovertemplate=(
                        f"Power law: A={A:.3f}, n={n:.3f}<br>"
                        "t = %{x:.2f} ns<br>L_fit = %{y:.3f} nm<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1, col=2,
            )
    else:
        _placeholder(1, 2, "SEI Growth")

    # ---- 3. Phase field ----
    pf = continuum_results.get("phase_field")
    if pf:
        x_pf = np.asarray(pf["x_nm"], dtype=float)
        phi_pf = np.asarray(pf["phi"], dtype=float)
        fig.add_trace(
            go.Scatter(
                x=x_pf, y=phi_pf, mode="lines",
                name="Phase field φ",
                line=dict(color=NREL_COLORS[2], width=2.5),
                fill="tozeroy",
                fillcolor="rgba(94, 151, 50, 0.12)",
                hovertemplate=(
                    "x = %{x:.2f} nm<br>φ = %{y:.4f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=3,
        )
    else:
        _placeholder(1, 3, "Phase Field")

    # ---- 4. Stress (Vegard) ----
    st = continuum_results.get("stress")
    if st:
        x_st = np.asarray(st["x_nm"], dtype=float)
        sig_st = np.asarray(st["sigma_GPa"], dtype=float)
        fig.add_trace(
            go.Scatter(
                x=x_st, y=sig_st, mode="lines",
                name="σ (GPa)",
                line=dict(color=NREL_COLORS[3], width=2.5),
                hovertemplate=(
                    "x = %{x:.2f} nm<br>σ = %{y:.4f} GPa<extra></extra>"
                ),
                showlegend=False,
            ),
            row=2, col=1,
        )
        # Zero stress reference
        fig.add_hline(
            y=0, line=dict(color="#AAAAAA", width=1.0, dash="dash"),
            row=2, col=1,
        )
    else:
        _placeholder(2, 1, "Stress")

    # ---- 5. Conductivity ----
    cond = continuum_results.get("conductivity")
    if cond:
        T_c = np.asarray(cond["T_K"], dtype=float)
        sig_c = np.asarray(cond["sigma"], dtype=float)
        fig.add_trace(
            go.Scatter(
                x=T_c, y=np.log10(sig_c), mode="lines+markers",
                name="σ ionic",
                line=dict(color=NREL_COLORS[0], width=2.5),
                marker=dict(size=7),
                hovertemplate=(
                    "T = %{x:.0f} K<br>log σ = %{y:.3f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=2, col=2,
        )
    else:
        _placeholder(2, 2, "Conductivity")

    # ---- 6. Swelling ----
    sw = continuum_results.get("swelling")
    if sw:
        c_sw = np.asarray(sw["c_mol_m3"], dtype=float)
        dV_sw = np.asarray(sw["dV_frac"], dtype=float)
        fig.add_trace(
            go.Scatter(
                x=c_sw, y=dV_sw * 100.0, mode="lines",
                name="ΔV (%)",
                line=dict(color=NREL_COLORS[7], width=2.5),
                fill="tozeroy",
                fillcolor="rgba(0, 132, 107, 0.12)",
                hovertemplate=(
                    "c = %{x:.2f} mol/m³<br>ΔV = %{y:.3f}%<extra></extra>"
                ),
                showlegend=False,
            ),
            row=2, col=3,
        )
    else:
        _placeholder(2, 3, "Swelling")

    # ---- Axis labels ----
    axis_map = {
        (1, 1): ("1000/T (K⁻¹)", "log₁₀ D (m²/s)"),
        (1, 2): ("Time (ns)", "L (nm)"),
        (1, 3): ("x (nm)", "φ"),
        (2, 1): ("x (nm)", "σ (GPa)"),
        (2, 2): ("T (K)", "log₁₀ σ (S/cm)"),
        (2, 3): ("c (mol/m³)", "ΔV (%)"),
    }
    for (r, c), (xl, yl) in axis_map.items():
        fig.update_xaxes(title_text=xl, row=r, col=c,
                         title_font_size=11, tickfont_size=9)
        fig.update_yaxes(title_text=yl, row=r, col=c,
                         title_font_size=11, tickfont_size=9)

    apply_nrel_theme(
        fig,
        title="Continuum Model Summary Dashboard",
        width=1350,
        height=900,
    )
    fig.update_layout(showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# 6. build_benchmark_html_report
# ---------------------------------------------------------------------------


def build_benchmark_html_report(
    benchmark_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Build a standalone HTML report with embedded interactive Plotly figures.

    The report contains:
    1. Summary statistics table
    2. MLIP × project heatmap (energy RMSE, force RMSE, Fmax)
    3. Per-metric box plots across all MLIPs
    4. Per-project radar charts (one per project, embedded as tab sections)

    Plotly.js is bundled inline (no CDN required).

    Parameters
    ----------
    benchmark_df : pd.DataFrame
        Must contain columns: ``"mlip"``, ``"project"``, and at least one of
        ``"energy_rmse"``, ``"force_rmse"``, ``"fmax"``.
    output_path : Path
        Destination HTML file path (parent dirs created if absent).

    Returns
    -------
    Path
        Absolute path to the written HTML file.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_present = [
        m for m in ("energy_rmse", "force_rmse", "fmax", "walltime_h")
        if m in benchmark_df.columns
    ]
    if not metrics_present:
        raise ValueError(
            "benchmark_df must contain at least one of: "
            "energy_rmse, force_rmse, fmax, walltime_h"
        )

    metric_labels = {
        "energy_rmse": "Energy RMSE (meV/atom)",
        "force_rmse": "Force RMSE (meV/Å)",
        "fmax": "F_max (eV/Å)",
        "walltime_h": "Wall-time (h)",
    }

    # ---- collect div HTML for each figure -----
    div_blocks: list[str] = []

    # 1. Heatmap per metric
    for met in metrics_present:
        hm_fig = plot_mlip_benchmark_heatmap(benchmark_df, metric=met)
        div_blocks.append(
            f'<h2 style="font-family:Arial;color:#0079C2;margin-top:2em;">'
            f'Heatmap — {metric_labels.get(met, met)}</h2>\n'
            + pio.to_html(hm_fig, full_html=False, include_plotlyjs=False,
                          config={"responsive": True})
        )

    # 2. Box plots: one box per MLIP, for each metric
    for met in metrics_present:
        box_fig = go.Figure()
        mlips_sorted = sorted(benchmark_df["mlip"].unique())
        for i, mlip in enumerate(mlips_sorted):
            vals = benchmark_df.loc[benchmark_df["mlip"] == mlip, met].dropna().values
            box_fig.add_trace(
                go.Box(
                    y=vals,
                    name=mlip,
                    marker_color=NREL_COLORS[i % len(NREL_COLORS)],
                    boxmean="sd",
                    hovertemplate=(
                        f"<b>{mlip}</b><br>"
                        f"{metric_labels.get(met, met)}: %{{y:.4g}}<extra></extra>"
                    ),
                )
            )
        apply_nrel_theme(
            box_fig,
            title=f"Distribution of {metric_labels.get(met, met)} by MLIP",
            xlabel="MLIP",
            ylabel=metric_labels.get(met, met),
            width=900, height=500,
        )
        div_blocks.append(
            f'<h2 style="font-family:Arial;color:#0079C2;margin-top:2em;">'
            f'Box Plot — {metric_labels.get(met, met)}</h2>\n'
            + pio.to_html(box_fig, full_html=False, include_plotlyjs=False,
                          config={"responsive": True})
        )

    # 3. Radar per project
    projects_all = sorted(benchmark_df["project"].unique())
    for proj in projects_all:
        proj_df = benchmark_df[benchmark_df["project"] == proj]
        mlip_data: dict[str, dict] = {}
        for mlip, grp in proj_df.groupby("mlip"):
            row_d: dict[str, float] = {}
            for met in metrics_present:
                v = grp[met].mean()
                if np.isfinite(v):
                    row_d[met] = float(v)
            if row_d:
                mlip_data[mlip] = row_d
        if mlip_data:
            radar_fig = plot_benchmark_radar(project=proj, mlip_data=mlip_data)
            div_blocks.append(
                f'<h2 style="font-family:Arial;color:#0079C2;margin-top:2em;">'
                f'Radar — {proj}</h2>\n'
                + pio.to_html(radar_fig, full_html=False, include_plotlyjs=False,
                              config={"responsive": True})
            )

    # 4. Summary table (HTML)
    summary = (
        benchmark_df.groupby(["mlip", "project"])[metrics_present]
        .mean()
        .round(4)
        .reset_index()
    )
    table_html = _df_to_html_table(summary)

    # ---- Assemble full HTML page ----
    # Embed plotly.js inline (avoids CDN)
    plotlyjs_str = pio.to_html(go.Figure(), full_html=True, include_plotlyjs=True)
    # Extract just the <script> block
    import re
    js_match = re.search(
        r'<script type="text/javascript">(.+?)</script>',
        plotlyjs_str, re.DOTALL
    )
    plotlyjs_inline = (
        f'<script type="text/javascript">{js_match.group(1)}</script>'
        if js_match else '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
    )

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>HPCA Pipeline — MLIP Benchmark Report</title>
  {plotlyjs_inline}
  <style>
    body {{
      font-family: Arial, Helvetica, sans-serif;
      background: #FFFFFF;
      color: #1A1A1A;
      max-width: 1400px;
      margin: 0 auto;
      padding: 2em 2em 4em;
    }}
    h1 {{ color: #0079C2; border-bottom: 3px solid #F7A11A; padding-bottom: 0.4em; }}
    h2 {{ color: #0079C2; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
      margin: 1em 0 2em;
    }}
    th {{
      background: #0079C2;
      color: #FFFFFF;
      padding: 8px 12px;
      text-align: left;
    }}
    td {{
      border: 1px solid #E5E5E5;
      padding: 6px 12px;
    }}
    tr:nth-child(even) {{ background: #F5F9FF; }}
    .section {{ margin-top: 3em; }}
  </style>
</head>
<body>
  <h1>HPCA Pipeline — MLIP Benchmark Report</h1>
  <p style="color:#555;">Generated by <code>comparison.build_benchmark_html_report()</code> &nbsp;|&nbsp;
     HPCA Pipeline</p>

  <div class="section">
    <h2>Summary Statistics</h2>
    {table_html}
  </div>

  {''.join(f'<div class="section">{blk}</div>' for blk in div_blocks)}
</body>
</html>
"""

    output_path.write_text(page_html, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Internal HTML table builder
# ---------------------------------------------------------------------------


def _df_to_html_table(df: pd.DataFrame) -> str:
    """Convert DataFrame to a styled HTML table string."""
    thead = "<tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"
    rows_html = []
    for _, row in df.iterrows():
        cells = "".join(
            f"<td>{v if not isinstance(v, float) else f'{v:.4g}'}</td>"
            for v in row
        )
        rows_html.append(f"<tr>{cells}</tr>")
    return f"<table><thead>{thead}</thead><tbody>{''.join(rows_html)}</tbody></table>"
