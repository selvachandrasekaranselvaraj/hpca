"""
publication.py — Publication-ready matplotlib figures for journal submission.
HPCA Pipeline · /path/to/workspace/hpca/viz/publication.py

Python env: /path/to/apps/apps/cladue/env/bin/python3

Uses matplotlib only (no Plotly). Every figure saves PDF + PNG and a companion
CSV with the underlying data.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Journal column widths in inches
# ---------------------------------------------------------------------------

JOURNAL_WIDTHS: dict[str, tuple[float, float]] = {
    "nature": (3.50, 7.20),
    "acs":    (3.33, 7.00),
    "rsc":    (3.46, 7.17),
    "report": (6.69, 6.69),
}

# ---------------------------------------------------------------------------
# Element color palette (consistent with NREL scheme)
# ---------------------------------------------------------------------------

ELEMENT_COLORS: dict[str, str] = {
    "Li": "#2ecc71",  "Na": "#f1c40f",  "K":  "#9b59b6",
    "O":  "#e74c3c",  "F":  "#1abc9c",  "Cl": "#27ae60",
    "Zr": "#95a5a6",  "Y":  "#3498db",  "Ti": "#2980b9",
    "Ni": "#e67e22",  "Mn": "#d35400",  "Co": "#c0392b",
    "Sr": "#8e44ad",  "V":  "#16a085",  "Sn": "#7f8c8d",
    "C":  "#2c3e50",  "H":  "#ecf0f1",  "N":  "#3498db",
}

# Publication matplotlib rcParams
_PUB_RCPARAMS: dict = {
    "axes.linewidth": 0.8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": 7,
    "legend.frameon": True,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "#CCCCCC",
    "legend.handlelength": 1.6,
    "lines.linewidth": 1.2,
    "lines.markersize": 4.0,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,   # TrueType in PDF (editable in Illustrator)
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

# NREL-inspired color cycle for matplotlib
_NREL_CYCLE = [
    "#0079C2", "#F7A11A", "#5E9732", "#E31C3D",
    "#00A4E4", "#7A3988", "#D9531E", "#00846B",
]

_KB_EV = 8.617333e-5  # eV K⁻¹


# ---------------------------------------------------------------------------
# PublicationFigure
# ---------------------------------------------------------------------------


class PublicationFigure:
    """Publication-ready figure with journal-appropriate sizing and styling."""

    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        style: str = "nature",
        double_col: bool = False,
        dpi: int = 300,
    ) -> None:
        """Create publication-ready figure with journal-appropriate sizing."""
        self.style = style
        self.dpi = dpi
        self.nrows = nrows
        self.ncols = ncols

        widths = JOURNAL_WIDTHS.get(style, JOURNAL_WIDTHS["nature"])
        col_width = widths[1] if double_col else widths[0]
        fig_width = col_width * ncols
        # Golden-ratio height per panel
        panel_h = col_width * 0.75
        fig_height = panel_h * nrows

        # Apply rcParams
        mpl.rcParams.update(_PUB_RCPARAMS)
        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=_NREL_CYCLE)

        self.fig, raw_axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(fig_width, fig_height),
            dpi=dpi,
            constrained_layout=True,
        )

        # Normalise axes to 2-D array for uniform indexing
        if nrows == 1 and ncols == 1:
            self.axes: np.ndarray = np.array([[raw_axes]])
        elif nrows == 1:
            self.axes = np.array([raw_axes])
        elif ncols == 1:
            self.axes = np.array([[ax] for ax in raw_axes])
        else:
            self.axes = np.array(raw_axes)

        # Storage for underlying data for CSV export
        self._data_records: list[dict] = []

    # ------------------------------------------------------------------
    # Internal axis resolver
    # ------------------------------------------------------------------

    def _get_ax(self, ax_idx: Union[int, tuple]) -> plt.Axes:
        """Return an Axes object from an integer or (row, col) index."""
        if isinstance(ax_idx, int):
            row = ax_idx // self.ncols
            col = ax_idx % self.ncols
        else:
            row, col = ax_idx
        return self.axes[row, col]

    # ------------------------------------------------------------------
    # plot
    # ------------------------------------------------------------------

    def plot(
        self,
        ax_idx: Union[int, tuple],
        x: np.ndarray,
        y: np.ndarray,
        label: Optional[str] = None,
        color: Optional[str] = None,
        marker: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Plot to panel ax_idx (int or (row, col) tuple)."""
        ax = self._get_ax(ax_idx)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        kw = {}
        if color:
            kw["color"] = color
        if marker:
            kw["marker"] = marker
        kw.update(kwargs)
        ax.plot(x, y, label=label, **kw)
        if label:
            ax.legend()
        self._data_records.append(
            {"panel": ax_idx, "label": label, "x": x, "y": y}
        )

    # ------------------------------------------------------------------
    # arrhenius
    # ------------------------------------------------------------------

    def arrhenius(
        self,
        ax_idx: Union[int, tuple],
        temps_K: np.ndarray,
        D_vals: np.ndarray,
        label: Optional[str] = None,
        color: Optional[str] = None,
    ) -> None:
        """Arrhenius panel: ln(D) vs 1000/T with linear fit and Ea annotation."""
        ax = self._get_ax(ax_idx)
        temps_K = np.asarray(temps_K, dtype=float)
        D_vals = np.asarray(D_vals, dtype=float)
        inv_T = 1000.0 / temps_K
        log_D = np.log10(D_vals)

        c = color or _NREL_CYCLE[len(ax.lines) % len(_NREL_CYCLE)]

        ax.scatter(inv_T, log_D, color=c, s=30, zorder=5,
                   label=label, linewidths=0.5, edgecolors="white")

        # Linear fit
        mask = np.isfinite(inv_T) & np.isfinite(log_D)
        if mask.sum() >= 2:
            slope, intercept, r2, *_ = sp_stats.linregress(
                inv_T[mask], log_D[mask]
            )
            # slope in log10-space → Ea = -slope * kB * ln(10)
            Ea_eV = -slope * _KB_EV * np.log(10)
            x_fit = np.linspace(inv_T[mask].min() * 0.97,
                                inv_T[mask].max() * 1.03, 120)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, "--", color=c, linewidth=0.9, zorder=4)
            # Annotate Ea
            ax.text(
                0.97, 0.05,
                f"$E_a$ = {Ea_eV:.3f} eV",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#CCCCCC", alpha=0.85),
            )

        ax.set_xlabel("1000 / T  (K⁻¹)", fontsize=8)
        ax.set_ylabel("log₁₀ D  (m²/s)", fontsize=8)
        if label:
            ax.legend(fontsize=7)

        self._data_records.append(
            {"panel": ax_idx, "label": label,
             "x": inv_T, "y": log_D,
             "x_name": "inv_T_1000_K", "y_name": "log10_D_m2s"}
        )

    # ------------------------------------------------------------------
    # msd
    # ------------------------------------------------------------------

    def msd(
        self,
        ax_idx: Union[int, tuple],
        times_ps: np.ndarray,
        msd_angsq: np.ndarray,
        D_m2s: Optional[float] = None,
        label: Optional[str] = None,
        color: Optional[str] = None,
    ) -> None:
        """MSD panel with optional diffusivity slope line."""
        ax = self._get_ax(ax_idx)
        times_ps = np.asarray(times_ps, dtype=float)
        msd_angsq = np.asarray(msd_angsq, dtype=float)

        c = color or _NREL_CYCLE[len(ax.lines) % len(_NREL_CYCLE)]
        ax.plot(times_ps, msd_angsq, color=c, linewidth=1.2, label=label)

        # Linear fit in 40–80% region
        n = len(times_ps)
        lo, hi = int(0.40 * n), int(0.80 * n)
        if hi > lo + 2:
            slope_fit, intcpt, *_ = sp_stats.linregress(
                times_ps[lo:hi], msd_angsq[lo:hi]
            )
            t_fit = np.linspace(times_ps[lo], times_ps[hi - 1], 60)
            y_fit = slope_fit * t_fit + intcpt
            ax.plot(t_fit, y_fit, ":", color=c, linewidth=0.9)
            D_fit = D_m2s if D_m2s is not None else (slope_fit / 6.0) * 1e-8
            ax.text(
                0.97, 0.05,
                f"D = {D_fit:.2e} m²/s",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#CCCCCC", alpha=0.85),
            )

        ax.set_xlabel("Time  (ps)", fontsize=8)
        ax.set_ylabel("MSD  (Å²)", fontsize=8)
        if label:
            ax.legend(fontsize=7)

        self._data_records.append(
            {"panel": ax_idx, "label": label,
             "x": times_ps, "y": msd_angsq,
             "x_name": "time_ps", "y_name": "msd_angsq"}
        )

    # ------------------------------------------------------------------
    # rdf
    # ------------------------------------------------------------------

    def rdf(
        self,
        ax_idx: Union[int, tuple],
        r: np.ndarray,
        g_r: np.ndarray,
        label: Optional[str] = None,
        color: Optional[str] = None,
        peaks: bool = True,
    ) -> None:
        """RDF panel with automatic first-peak detection markers."""
        ax = self._get_ax(ax_idx)
        r = np.asarray(r, dtype=float)
        g_r = np.asarray(g_r, dtype=float)

        c = color or _NREL_CYCLE[len(ax.lines) % len(_NREL_CYCLE)]
        ax.plot(r, g_r, color=c, linewidth=1.2, label=label)

        if peaks:
            peak_idxs, _ = find_peaks(g_r, height=1.2, distance=5)
            if len(peak_idxs) > 0:
                r_p = float(r[peak_idxs[0]])
                g_p = float(g_r[peak_idxs[0]])
                ax.axvline(r_p, color=c, linewidth=0.7, linestyle="--", alpha=0.7)
                ax.annotate(
                    f"{r_p:.2f} Å",
                    xy=(r_p, g_p),
                    xytext=(r_p + 0.15, g_p),
                    fontsize=6,
                    color=c,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.5),
                )

        ax.axhline(1.0, color="#AAAAAA", linewidth=0.6, linestyle=":")
        ax.set_xlabel("r  (Å)", fontsize=8)
        ax.set_ylabel("g(r)", fontsize=8)
        if label:
            ax.legend(fontsize=7)

        self._data_records.append(
            {"panel": ax_idx, "label": label, "x": r, "y": g_r,
             "x_name": "r_angstrom", "y_name": "g_r"}
        )

    # ------------------------------------------------------------------
    # vanhove
    # ------------------------------------------------------------------

    def vanhove(
        self,
        ax_idx: Union[int, tuple],
        r: np.ndarray,
        G_dict: dict[str, np.ndarray],
        colormap: str = "coolwarm",
    ) -> None:
        """Van Hove G_s(r,t) panel with time-colored lines."""
        ax = self._get_ax(ax_idx)
        r = np.asarray(r, dtype=float)
        n = len(G_dict)
        cmap = plt.get_cmap(colormap)

        for i, (dt_label, G_s) in enumerate(G_dict.items()):
            G_s = np.asarray(G_s, dtype=float)
            c = cmap(i / max(n - 1, 1))
            ax.plot(r, G_s, color=c, linewidth=1.0, label=dt_label)
            self._data_records.append(
                {"panel": ax_idx, "label": dt_label, "x": r, "y": G_s,
                 "x_name": "r_angstrom", "y_name": f"G_s_{dt_label}"}
            )

        ax.set_xlabel("r  (Å)", fontsize=8)
        ax.set_ylabel("G_s(r, t)  (Å⁻³)", fontsize=8)
        # Compact legend
        if n <= 8:
            ax.legend(fontsize=6, loc="upper right",
                      handlelength=1.2, ncol=1)

    # ------------------------------------------------------------------
    # label_panels
    # ------------------------------------------------------------------

    def label_panels(
        self,
        labels: Optional[list[str]] = None,
        x: float = -0.15,
        y: float = 1.05,
        fontsize: int = 9,
        bold: bool = True,
    ) -> None:
        """Add (a), (b), (c)... panel labels to each subplot."""
        total = self.nrows * self.ncols
        if labels is None:
            labels = [f"({chr(97 + i)})" for i in range(total)]
        weight = "bold" if bold else "normal"
        for idx in range(min(len(labels), total)):
            row = idx // self.ncols
            col = idx % self.ncols
            ax = self.axes[row, col]
            ax.text(
                x, y, labels[idx],
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight=weight,
                va="top",
                ha="right",
            )

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------

    def save(
        self,
        path_stem: Union[str, Path],
        formats: tuple[str, ...] | list[str] = ("pdf", "png"),
    ) -> dict[str, Path]:
        """Save figure to path_stem.pdf and path_stem.png, plus data CSV."""
        path_stem = Path(path_stem)
        path_stem.parent.mkdir(parents=True, exist_ok=True)
        saved: dict[str, Path] = {}

        for fmt in formats:
            out = path_stem.with_suffix(f".{fmt}")
            try:
                self.fig.savefig(
                    str(out),
                    dpi=self.dpi,
                    bbox_inches="tight",
                    pad_inches=0.02,
                )
                saved[fmt] = out
            except Exception as exc:
                warnings.warn(
                    f"[publication] Could not save {fmt}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Save underlying data as CSV
        csv_path = path_stem.with_name(path_stem.stem + "_data.csv")
        _write_data_csv(self._data_records, csv_path)
        saved["csv"] = csv_path

        return saved


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------


def quick_transport_figure(
    projects_dict: dict,
    style: str = "nature",
    output_path: Optional[Union[str, Path]] = None,
) -> "PublicationFigure":
    """2-panel: Arrhenius (left) + D comparison bar (right).

    projects_dict: {name: MaterialProject} where project.D_mlmd and
    project.Ea_best are populated.
    """
    pf = PublicationFigure(nrows=1, ncols=2, style=style, double_col=True)
    ax_arr = pf._get_ax(0)
    ax_bar = pf._get_ax(1)

    project_names: list[str] = []
    D_300_values: list[float] = []

    for i, (name, proj) in enumerate(projects_dict.items()):
        color = _NREL_CYCLE[i % len(_NREL_CYCLE)]
        # Arrhenius panel
        D_map: dict = getattr(proj, "D_mlmd", {})
        # Build per-temperature D map from mlmd_dirs keys if possible
        T_D: dict[int, float] = {}
        if hasattr(proj, "D_mlmd") and proj.D_mlmd:
            # Assume D_mlmd might contain {mlip_name: D_300} or {T: D}
            for k, v in proj.D_mlmd.items():
                try:
                    T_D[int(k)] = float(v)
                except (ValueError, TypeError):
                    pass
        if len(T_D) >= 2:
            pf.arrhenius(0, list(T_D.keys()), list(T_D.values()),
                         label=name, color=color)

        D_best = getattr(proj, "D_best", None)
        if D_best and D_best > 0:
            project_names.append(name)
            D_300_values.append(float(D_best))

    # D comparison bar
    if project_names:
        bar_colors = [_NREL_CYCLE[i % len(_NREL_CYCLE)]
                      for i in range(len(project_names))]
        ax_bar.bar(project_names, D_300_values, color=bar_colors,
                   edgecolor="white", linewidth=0.5, alpha=0.88)
        ax_bar.set_yscale("log")
        ax_bar.set_xlabel("Project", fontsize=8)
        ax_bar.set_ylabel("D  (m²/s)", fontsize=8)
        ax_bar.tick_params(axis="x", rotation=30, labelsize=7)

    pf.label_panels()
    ax_arr.set_title("Arrhenius", fontsize=9)
    ax_bar.set_title("Diffusivity at T_ref", fontsize=9)

    if output_path is not None:
        pf.save(Path(output_path))

    return pf


def quick_characterization_figure(
    result: dict,
    style: str = "nature",
    output_path: Optional[Union[str, Path]] = None,
) -> "PublicationFigure":
    """4-panel: MSD, Arrhenius, RDF, Van Hove for one analysis result dict."""
    pf = PublicationFigure(nrows=2, ncols=2, style=style, double_col=True)

    # ── MSD panel (0,0) ───────────────────────────────────────────────────────
    for T, T_res in (result.get("by_temperature") or {}).items():
        msd_data = (T_res.get("msd") or {})
        t = np.asarray(msd_data.get("time_ps", []), dtype=float)
        msd = np.asarray(msd_data.get("msd_angsq", []), dtype=float)
        D = (T_res.get("diffusivity") or {}).get("D_m2s")
        if len(t) > 1 and len(msd) > 1:
            pf.msd(0, t, msd, D_m2s=D, label=f"{T} K")

    ax_msd = pf._get_ax(0)
    ax_msd.set_title("MSD", fontsize=9)

    # ── Arrhenius panel (1) ───────────────────────────────────────────────────
    D_by_T: dict[int, float] = {}
    for T, T_res in (result.get("by_temperature") or {}).items():
        D = (T_res.get("diffusivity") or {}).get("D_m2s")
        if D and D > 0:
            D_by_T[int(T)] = float(D)
    if len(D_by_T) >= 2:
        pf.arrhenius(1, list(D_by_T.keys()), list(D_by_T.values()))
    pf._get_ax(1).set_title("Arrhenius", fontsize=9)

    # ── RDF panel (2) ─────────────────────────────────────────────────────────
    # Use first available temperature and first available RDF pair
    for T, T_res in (result.get("by_temperature") or {}).items():
        rdf_block = T_res.get("rdf") or {}
        for pair_label, rdf_data in rdf_block.items():
            if isinstance(rdf_data, dict) and "r" in rdf_data and "g_r" in rdf_data:
                pf.rdf(2,
                       np.asarray(rdf_data["r"], dtype=float),
                       np.asarray(rdf_data["g_r"], dtype=float),
                       label=pair_label)
                break
        else:
            continue
        break
    pf._get_ax(2).set_title("RDF", fontsize=9)

    # ── Van Hove panel (3) ────────────────────────────────────────────────────
    for T, T_res in (result.get("by_temperature") or {}).items():
        vh = T_res.get("van_hove")
        if vh and "curves" in vh:
            G_dict = {}
            for lag_label, (r_arr, gs_arr) in vh["curves"].items():
                G_dict[str(lag_label)] = np.asarray(gs_arr, dtype=float)
            if G_dict:
                r_arr = np.asarray(list(vh["curves"].values())[0][0], dtype=float)
                pf.vanhove(3, r_arr, G_dict)
                break
    pf._get_ax(3).set_title("Van Hove G_s(r,t)", fontsize=9)

    pf.label_panels()

    if output_path is not None:
        pf.save(Path(output_path))

    return pf


# ---------------------------------------------------------------------------
# Internal: data CSV writer
# ---------------------------------------------------------------------------


def _write_data_csv(records: list[dict], path: Path) -> None:
    """Write a flat CSV with one column pair per data record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("")
        return

    # Build uniform column arrays — pad with NaN to common max length
    columns: dict[str, np.ndarray] = {}
    for rec in records:
        x = np.asarray(rec.get("x", []), dtype=float)
        y = np.asarray(rec.get("y", []), dtype=float)
        lbl = (rec.get("label") or "series").replace(" ", "_")
        panel = rec.get("panel", 0)
        xname = rec.get("x_name", f"x_p{panel}_{lbl}")
        yname = rec.get("y_name", f"y_p{panel}_{lbl}")
        # Ensure uniqueness
        xk = xname
        yk = yname
        suffix = 0
        while xk in columns or yk in columns:
            suffix += 1
            xk = f"{xname}_{suffix}"
            yk = f"{yname}_{suffix}"
        columns[xk] = x
        columns[yk] = y

    max_len = max((len(v) for v in columns.values()), default=0)
    if max_len == 0:
        path.write_text(",".join(columns.keys()) + "\n")
        return

    padded = {
        k: np.pad(v, (0, max_len - len(v)), constant_values=np.nan)
        for k, v in columns.items()
    }

    header = ",".join(padded.keys())
    data = np.column_stack(list(padded.values()))
    np.savetxt(str(path), data, delimiter=",", header=header, comments="")
