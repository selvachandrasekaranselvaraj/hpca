"""
dos_band.py — Electronic structure visualization: DOS/PDOS and Bader charges.
HPCA Pipeline · /path/to/workspace/hpca/viz/dos_band.py

Python env: /path/to/apps/apps/cladue/env/bin/python3

All public functions return a go.Figure with NREL theme applied.
When output_path is provided the figure is saved as PNG and data as CSV alongside.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    """Save figure PNG and companion CSV at the same path stem."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(p), scale=2)
    except Exception as exc:
        warnings.warn(
            f"[dos_band] PNG write failed ({exc}); saving HTML fallback.",
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


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert hex color string to rgba() CSS string."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha:.3f})"


# PDOS orbital colors — consistent across all projects
_ORBITAL_COLORS: dict[str, str] = {
    "s":   "#0079C2",   # NREL blue
    "p":   "#E31C3D",   # NREL red
    "d":   "#5E9732",   # NREL green
    "f":   "#7A3988",   # NREL purple
    "sp":  "#F7A11A",   # NREL gold
    "px":  "#E31C3D",
    "py":  "#D9531E",
    "pz":  "#F7A11A",
    "dxy": "#5E9732",
    "dyz": "#00846B",
    "dxz": "#00A4E4",
    "dx2": "#7A3988",
    "dz2": "#E31C3D",
}


def _pdos_color(label: str, index: int) -> str:
    """Pick a color for a PDOS series by label or fallback to palette index."""
    # Match orbital suffix: 'Li-s' → 's', 'Ni-3d' → 'd', 'O-2p' → 'p'
    for key in _ORBITAL_COLORS:
        if label.lower().endswith(key):
            return _ORBITAL_COLORS[key]
    return NREL_COLORS[index % len(NREL_COLORS)]


# ---------------------------------------------------------------------------
# 1. plot_dos
# ---------------------------------------------------------------------------


def plot_dos(
    energies: np.ndarray,
    total_dos: np.ndarray,
    pdos_dict: Optional[dict[str, np.ndarray]] = None,
    efermi: float = 0.0,
    erange: tuple[float, float] = (-6.0, 4.0),
    output_path: Optional[str | Path] = None,
) -> go.Figure:
    """Plot total DOS as filled area and PDOS species as colored lines.

    Energies are shifted so E_F = 0.  A vertical dashed line marks E_F.
    """
    energies = np.asarray(energies, dtype=float) - efermi
    total_dos = np.asarray(total_dos, dtype=float)

    # Energy window mask
    mask = (energies >= erange[0]) & (energies <= erange[1])
    e_win = energies[mask]
    td_win = total_dos[mask]

    fig = go.Figure()

    # Total DOS — filled area
    fig.add_trace(
        go.Scatter(
            x=e_win,
            y=td_win,
            mode="lines",
            fill="tozeroy",
            fillcolor=_hex_to_rgba(NREL_COLORS[0], alpha=0.20),
            line=dict(color=NREL_COLORS[0], width=2.5),
            name="Total DOS",
            hovertemplate=(
                "<b>Total DOS</b><br>"
                "E − E_F = %{x:.3f} eV<br>"
                "DOS = %{y:.4f}<extra></extra>"
            ),
        )
    )

    # PDOS — individual lines
    col_headers = ["energy_eV_minus_efermi", "total_dos"]
    all_cols = [e_win, td_win]

    if pdos_dict:
        for i, (label, pdos_arr) in enumerate(pdos_dict.items()):
            pdos_arr = np.asarray(pdos_arr, dtype=float)
            # Align to energy grid: trim or pad to match mask length
            pd_win = pdos_arr[mask] if len(pdos_arr) == len(energies) else pdos_arr[:len(e_win)]
            color = _pdos_color(label, i)

            fig.add_trace(
                go.Scatter(
                    x=e_win,
                    y=pd_win,
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=1.8),
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        "E − E_F = %{x:.3f} eV<br>"
                        "PDOS = %{y:.4f}<extra></extra>"
                    ),
                )
            )
            col_headers.append(f"pdos_{label.replace('-', '_').replace(' ', '_')}")
            all_cols.append(pd_win)

    # Fermi energy vertical line
    fig.add_vline(
        x=0.0,
        line=dict(color="#333333", width=1.5, dash="dash"),
        annotation_text="E_F",
        annotation_position="top",
        annotation_font_size=11,
        annotation_font_color="#333333",
    )

    apply_nrel_theme(
        fig,
        title="Density of States",
        xlabel="E − E_F  (eV)",
        ylabel="DOS  (states/eV)",
        width=900,
        height=580,
    )
    fig.update_layout(
        xaxis=dict(range=list(erange)),
        yaxis=dict(rangemode="tozero"),
    )

    if output_path is not None:
        # Pad to same length in case pdos arrays are shorter
        max_len = max(len(c) for c in all_cols)
        padded = [
            np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
            for c in all_cols
        ]
        _save_png_csv(
            fig, output_path,
            np.column_stack(padded),
            ",".join(col_headers),
        )

    return fig


# ---------------------------------------------------------------------------
# 2. plot_bader_charges
# ---------------------------------------------------------------------------


def plot_bader_charges(
    atom_labels: list[str],
    charges: np.ndarray,
    ref_charges: Optional[np.ndarray] = None,
    output_path: Optional[str | Path] = None,
) -> go.Figure:
    """Horizontal bar chart of Bader charges with optional reference overlay."""
    charges = np.asarray(charges, dtype=float)
    n = len(atom_labels)

    # Sort by charge (ascending) for readability
    order = np.argsort(charges)
    sorted_labels = [atom_labels[i] for i in order]
    sorted_charges = charges[order]

    fig = go.Figure()

    # Determine bar colors: positive → blue, negative → red
    bar_colors = [
        NREL_COLORS[0] if c >= 0 else NREL_COLORS[3]
        for c in sorted_charges
    ]

    fig.add_trace(
        go.Bar(
            y=sorted_labels,
            x=sorted_charges,
            orientation="h",
            name="Bader charge",
            marker=dict(
                color=bar_colors,
                opacity=0.85,
                line=dict(color="#FFFFFF", width=0.8),
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Δq = %{x:.4f} e<extra></extra>"
            ),
        )
    )

    # Reference charges overlay (scatter markers)
    if ref_charges is not None:
        ref_charges = np.asarray(ref_charges, dtype=float)
        sorted_ref = ref_charges[order]
        fig.add_trace(
            go.Scatter(
                y=sorted_labels,
                x=sorted_ref,
                mode="markers",
                name="Reference",
                marker=dict(
                    symbol="line-ns",
                    size=12,
                    color=NREL_COLORS[1],
                    line=dict(color=NREL_COLORS[1], width=2.5),
                ),
                hovertemplate=(
                    "<b>%{y} — Ref</b><br>"
                    "Δq_ref = %{x:.4f} e<extra></extra>"
                ),
            )
        )

    # Zero-charge reference line
    fig.add_vline(
        x=0.0,
        line=dict(color="#999999", width=1.0, dash="dash"),
    )

    apply_nrel_theme(
        fig,
        title="Bader Charge Transfer",
        xlabel="Charge Transfer  Δq  (e)",
        ylabel="Atom",
        width=max(700, 30 * n + 200),
        height=max(500, 22 * n + 150),
    )
    fig.update_layout(
        yaxis=dict(autorange=True),
        xaxis=dict(zeroline=False),
        barmode="overlay",
    )

    if output_path is not None:
        if ref_charges is not None:
            data_arr = np.column_stack([sorted_charges, ref_charges[order]])
            hdr = "bader_charge_e,reference_charge_e"
        else:
            data_arr = sorted_charges.reshape(-1, 1)
            hdr = "bader_charge_e"
        # Also save atom labels as an index column
        combined = np.column_stack(
            [np.arange(n).reshape(-1, 1), data_arr]
        )
        _save_png_csv(fig, output_path, combined, f"atom_index,{hdr}")

    return fig
