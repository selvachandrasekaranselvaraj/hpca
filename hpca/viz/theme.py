"""
theme.py — NREL-inspired Plotly theme and figure utilities
HPCA Pipeline · /path/to/workspace/hpca/viz/theme.py

Python env: /path/to/apps/apps/cladue/env/bin/python3
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

NREL_COLORS = [
    "#0079C2",  # NREL blue
    "#F7A11A",  # NREL gold
    "#5E9732",  # green
    "#E31C3D",  # red
    "#00A4E4",  # light blue
    "#7A3988",  # purple
    "#D9531E",  # orange
    "#00846B",  # teal
]

NREL_LIGHT_COLORS = [
    "#CCE5F5",  # light blue
    "#FDE9B5",  # light gold
    "#D6EAC2",  # light green
    "#FAC8D1",  # light red
    "#CCF0FF",  # lightest blue
    "#DECCEC",  # light purple
    "#F5D4C4",  # light orange
    "#C2E8E0",  # light teal
]

# ---------------------------------------------------------------------------
# Full Plotly template dict
# ---------------------------------------------------------------------------

NREL_TEMPLATE: dict = {
    "layout": {
        "font": {
            "family": "Arial, Helvetica, sans-serif",
            "size": 14,
            "color": "#1A1A1A",
        },
        "title": {
            "font": {
                "family": "Arial, Helvetica, sans-serif",
                "size": 18,
                "color": "#1A1A1A",
            },
            "x": 0.5,
            "xanchor": "center",
        },
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#FAFAFA",
        "colorway": NREL_COLORS,
        "xaxis": {
            "showgrid": True,
            "gridcolor": "#E5E5E5",
            "gridwidth": 1,
            "linecolor": "#AAAAAA",
            "linewidth": 1.5,
            "ticks": "outside",
            "ticklen": 6,
            "tickwidth": 1.5,
            "tickcolor": "#AAAAAA",
            "showline": True,
            "zeroline": False,
            "tickfont": {"size": 12, "color": "#1A1A1A"},
            "title": {"font": {"size": 14, "color": "#1A1A1A"}},
            "mirror": False,
        },
        "yaxis": {
            "showgrid": True,
            "gridcolor": "#E5E5E5",
            "gridwidth": 1,
            "linecolor": "#AAAAAA",
            "linewidth": 1.5,
            "ticks": "outside",
            "ticklen": 6,
            "tickwidth": 1.5,
            "tickcolor": "#AAAAAA",
            "showline": True,
            "zeroline": False,
            "tickfont": {"size": 12, "color": "#1A1A1A"},
            "title": {"font": {"size": 14, "color": "#1A1A1A"}},
            "mirror": False,
        },
        "legend": {
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "#CCCCCC",
            "borderwidth": 1,
            "font": {"size": 12, "color": "#1A1A1A"},
            "orientation": "v",
            "x": 1.01,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
        },
        "margin": {"l": 70, "r": 40, "t": 80, "b": 70},
        "hoverlabel": {
            "bgcolor": "#FFFFFF",
            "bordercolor": "#0079C2",
            "font": {"size": 12, "family": "Arial, Helvetica, sans-serif"},
        },
        "hovermode": "closest",
        "colorscale": {
            "sequential": "Blues",
            "diverging": "RdBu",
        },
    },
    "data": {
        "scatter": [
            {
                "type": "scatter",
                "marker": {"size": 8, "line": {"width": 1.2, "color": "#FFFFFF"}},
                "line": {"width": 2.5},
            }
        ],
        "bar": [
            {
                "type": "bar",
                "marker": {
                    "line": {"width": 1.0, "color": "#FFFFFF"},
                    "opacity": 0.90,
                },
                "error_y": {"color": "#333333", "thickness": 1.5, "width": 4},
            }
        ],
        "heatmap": [
            {
                "type": "heatmap",
                "colorscale": "Blues",
                "showscale": True,
                "xgap": 1,
                "ygap": 1,
            }
        ],
        "violin": [
            {
                "type": "violin",
                "box_visible": True,
                "meanline_visible": True,
                "opacity": 0.75,
            }
        ],
        "histogram": [
            {
                "type": "histogram",
                "marker": {"opacity": 0.80},
            }
        ],
    },
}

# Register as a named Plotly template so callers can use template="nrel"
pio.templates["nrel"] = go.layout.Template(
    layout=go.Layout(**{
        k: v for k, v in NREL_TEMPLATE["layout"].items()
        if k not in ("colorway",)   # colorway set separately below
    }),
)
pio.templates["nrel"].layout.colorway = NREL_COLORS
pio.templates.default = "nrel"


# ---------------------------------------------------------------------------
# apply_nrel_theme
# ---------------------------------------------------------------------------

def apply_nrel_theme(
    fig: go.Figure,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    width: int = 900,
    height: int = 600,
) -> go.Figure:
    """Apply NREL template, axis labels, and dimensions; return the figure.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure to update.
    title : str
        Main figure title.
    xlabel : str
        X-axis label (all subplots share if only one axis).
    ylabel : str
        Y-axis label.
    width : int
        Figure width in pixels.
    height : int
        Figure height in pixels.

    Returns
    -------
    go.Figure
        The updated figure (same object, mutated in-place and returned).
    """
    fig.update_layout(
        template="nrel",
        title=title if title else None,
        width=width,
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFAFA",
        font_family="Arial, Helvetica, sans-serif",
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#0079C2",
            font_size=12,
        ),
        margin=dict(l=75, r=50, t=90 if title else 50, b=75),
    )
    if xlabel:
        fig.update_xaxes(title_text=xlabel)
    if ylabel:
        fig.update_yaxes(title_text=ylabel)
    return fig


# ---------------------------------------------------------------------------
# save_figure
# ---------------------------------------------------------------------------

def save_figure(
    fig: go.Figure,
    output_dir: Path,
    name: str,
    formats: tuple | list = ("html", "png"),
) -> dict[str, Path]:
    """Save figure as HTML and/or PNG; return {format: absolute Path}.

    PNG export requires kaleido. If kaleido is missing the function falls
    back to HTML-only and emits a warning rather than raising.

    Parameters
    ----------
    fig : go.Figure
    output_dir : Path
        Directory where files are written (created if absent).
    name : str
        Base filename without extension.
    formats : tuple or list
        Any subset of ``("html", "png", "svg", "pdf")``.

    Returns
    -------
    dict[str, Path]
        Mapping from format string to the saved file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}

    # Sanitise name for filesystem
    safe_name = name.replace(" ", "_").replace("/", "-")

    for fmt in formats:
        fmt = fmt.lower()
        out_path = output_dir / f"{safe_name}.{fmt}"
        try:
            if fmt == "html":
                fig.write_html(
                    str(out_path),
                    include_plotlyjs=True,
                    full_html=True,
                    config={"responsive": True, "displayModeBar": True},
                )
                saved["html"] = out_path
            elif fmt in ("png", "svg", "pdf", "webp"):
                try:
                    fig.write_image(str(out_path), scale=2)
                    saved[fmt] = out_path
                except Exception as img_err:
                    import warnings
                    warnings.warn(
                        f"[theme.save_figure] Could not write {fmt} "
                        f"(kaleido issue?): {img_err}. "
                        "Falling back to HTML-only.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    if "html" not in saved:
                        html_path = output_dir / f"{safe_name}.html"
                        fig.write_html(str(html_path), include_plotlyjs=True)
                        saved["html"] = html_path
            else:
                import warnings
                warnings.warn(f"[theme.save_figure] Unknown format '{fmt}' — skipped.")
        except Exception as exc:
            import warnings
            warnings.warn(f"[theme.save_figure] Failed to save {fmt}: {exc}")

    return saved


# ---------------------------------------------------------------------------
# multi_panel
# ---------------------------------------------------------------------------

def multi_panel(
    figs: list[go.Figure],
    rows: int,
    cols: int,
    shared_x: bool = False,
    shared_y: bool = False,
    titles: Optional[list[str]] = None,
) -> go.Figure:
    """Combine multiple standalone figures into a subplot grid.

    Each figure in *figs* may contain one or more traces. All traces are
    copied into the corresponding subplot cell (left-to-right, top-to-bottom).

    Parameters
    ----------
    figs : list[go.Figure]
        Source figures. Length must equal rows × cols.
    rows, cols : int
        Grid dimensions.
    shared_x, shared_y : bool
        Whether subplot axes share ranges.
    titles : list[str], optional
        Per-subplot titles; falls back to each source figure's layout title.

    Returns
    -------
    go.Figure
        Combined subplot figure with NREL theme applied.
    """
    n_panels = rows * cols
    if len(figs) > n_panels:
        raise ValueError(
            f"multi_panel: {len(figs)} figures provided but grid is "
            f"{rows}×{cols} = {n_panels} cells."
        )

    if titles is None:
        titles = []
        for f in figs:
            t = f.layout.title.text if f.layout.title and f.layout.title.text else ""
            titles.append(t)
        # Pad to n_panels
        titles += [""] * (n_panels - len(titles))

    combined = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=shared_x,
        shared_yaxes=shared_y,
        subplot_titles=titles[:n_panels],
        horizontal_spacing=0.10,
        vertical_spacing=0.14,
    )

    for idx, src_fig in enumerate(figs):
        row = idx // cols + 1
        col = idx % cols + 1
        for trace in src_fig.data:
            # Avoid duplicate legend entries across panels
            trace_copy = trace
            if hasattr(trace_copy, "showlegend"):
                trace_copy = trace_copy.update(showlegend=False)
            combined.add_trace(trace_copy, row=row, col=col)

        # Propagate axis labels from source figure
        src_xaxis = src_fig.layout.xaxis
        src_yaxis = src_fig.layout.yaxis
        axis_idx = "" if idx == 0 else str(idx + 1)
        if src_xaxis and src_xaxis.title and src_xaxis.title.text:
            combined.update_xaxes(
                title_text=src_xaxis.title.text, row=row, col=col
            )
        if src_yaxis and src_yaxis.title and src_yaxis.title.text:
            combined.update_yaxes(
                title_text=src_yaxis.title.text, row=row, col=col
            )

    apply_nrel_theme(
        combined,
        width=cols * 480,
        height=rows * 380,
    )
    combined.update_layout(showlegend=False)
    return combined


# ---------------------------------------------------------------------------
# add_annotation_box
# ---------------------------------------------------------------------------

def add_annotation_box(
    fig: go.Figure,
    text: str,
    x: float = 0.02,
    y: float = 0.98,
) -> go.Figure:
    """Add a styled text annotation box to a figure.

    The annotation is placed in paper (0–1) coordinates. Default position is
    top-left. Useful for adding inset statistics, model parameters, or labels.

    Parameters
    ----------
    fig : go.Figure
    text : str
        Annotation text; HTML tags (``<b>``, ``<br>``) are supported.
    x, y : float
        Paper-coordinate anchor position (default: top-left corner).

    Returns
    -------
    go.Figure
        The updated figure.
    """
    fig.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=x,
        y=y,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        bordercolor="#0079C2",
        borderwidth=1.5,
        borderpad=8,
        bgcolor="rgba(255, 255, 255, 0.88)",
        font=dict(
            family="Arial, Helvetica, sans-serif",
            size=12,
            color="#1A1A1A",
        ),
        align="left",
    )
    return fig


# ---------------------------------------------------------------------------
# Convenience: NREL-styled colour scale for continuous data
# ---------------------------------------------------------------------------

NREL_SEQUENTIAL_COLORSCALE = [
    [0.000, "#CCE5F5"],
    [0.250, "#66B2E0"],
    [0.500, "#0079C2"],
    [0.750, "#004F80"],
    [1.000, "#002840"],
]

NREL_DIVERGING_COLORSCALE = [
    [0.000, "#E31C3D"],
    [0.250, "#F4A0AF"],
    [0.500, "#F5F5F5"],
    [0.750, "#8BBFD8"],
    [1.000, "#0079C2"],
]
