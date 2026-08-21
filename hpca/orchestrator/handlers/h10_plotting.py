"""
h10_plotting.py — Common figure generation handler (daemon-local, Plotly-based).

All scientific figures follow the autopsy Plotly style:
  - AUTOPSY_COLORS palette ['#008000','#FF0000','#0000FF',...]
  - Transparent background  paper_bgcolor/plot_bgcolor = "rgba(0,0,0,0)"
  - Inside ticks, mirrored black borders, linewidth=2
  - font size 18, black
  - Horizontal legend at top centre (y=1.01, xanchor="center")
  - Every PNG saved with write_image() + companion HTML with write_html()
  - Every PNG has a companion CSV (CLAUDE.md rule, enforced)

Ball-and-stick atomic structures use ASE plot_atoms (matplotlib on white bg)
because Plotly has no equivalent 3-D molecular renderer.

Works for ANY project type: SSE, NMC, LiPS, LYC, DMB/electrolyte, Na-air, etc.
"""
from __future__ import annotations

import csv as _csv
import json
import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

from .base import SimulationHandler
from hpca.core.paths import dft_opt, results_figures, results_data

if TYPE_CHECKING:
    from hpca.orchestrator.state_tracker import ProjectState

log = logging.getLogger("hpca.orch")
# Layout: see hpca/core/paths.py

# ── Autopsy colour palette (matches ~/myopt/autopsy1/autopsy/util/plot_a*) ──
AUTOPSY_COLORS = [
    "#008000", "#FF0000", "#0000FF", "#FFA500",
    "#800080", "#00FFFF", "#FF00FF", "#FFFF00",
    "#000000", "#A52A2A",
]

# ── CPK colours for ball-and-stick (publication white background) ────────────
CPK_COLORS = {
    "H":  "#CCCCCC", "C":  "#404040", "O":  "#CC2929", "N":  "#1F77B4",
    "S":  "#FFD700", "F":  "#17BECF", "Li": "#9467BD", "P":  "#FF7F0E",
    "Cl": "#2CA02C", "Na": "#1166AA", "Mg": "#E377C2", "Zn": "#8C564B",
    "K":  "#7F7F00", "Ca": "#17BECF", "Ti": "#AEC7E8", "Mn": "#FFBB78",
}

_FONT_SIZE = 18
_DPI       = 300

def _axis_style(title: str = "", log_scale: bool = False, exp_fmt: bool = False) -> dict:
    """Return a Plotly axis dict with autopsy style (inside ticks, mirrored black border)."""
    # Note: title font size is inherited from layout font (size=18).
    # titlefont was deprecated in Plotly 4+; use global font setting instead.
    d = dict(
        showline=True, linecolor="black", linewidth=2, mirror=True,
        ticks="inside", tickwidth=2, ticklen=10,
        minor=dict(ticks="inside", ticklen=5, tickwidth=1,
                   tickcolor="black", showgrid=False),
    )
    if title:
        d["title_text"] = title
    if log_scale:
        d["type"] = "log"
    if exp_fmt:
        d["tickformat"] = ".1e"
    return d


def _layout_base(height: int = 500, width: int = 700, **kw) -> dict:
    """Return a Plotly layout dict with transparent background, font size 18, and horizontal legend."""
    d = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=_FONT_SIZE, color="black"),
        legend=dict(
            orientation="h", xref="paper", yref="paper",
            xanchor="center", yanchor="bottom",
            x=0.5, y=1.01,
            font=dict(size=_FONT_SIZE),
            itemsizing="constant",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=height, width=width,
    )
    d.update(kw)
    return d


def _save_fig(fig, stem: str, fig_dir: Path) -> str:
    """Save Plotly figure as PNG (scale=2) + HTML. Returns PNG path."""
    png = str(fig_dir / f"{stem}.png")
    html = str(fig_dir / f"{stem}.html")
    try:
        fig.write_image(png, format="png", scale=2)
    except Exception as exc:
        log.warning("[h10_plotting] write_image failed for %s: %s", stem, exc)
    try:
        fig.write_html(html, include_plotlyjs=True)
    except Exception as exc:
        log.debug("[h10_plotting] write_html failed for %s: %s", stem, exc)
    return png


def _write_csv(rows: list, path: Path) -> None:
    """Write a list of rows (including header) to a CSV file."""
    with open(str(path), "w", newline="") as fh:
        _csv.writer(fh).writerows(rows)


def _extract_T(filename: str) -> int:
    """Extract temperature integer from a filename like 'msd_600K.csv'; returns 0 if not found."""
    m = re.search(r"(\d+)K", filename)
    return int(m.group(1)) if m else 0


class PlottingHandler(SimulationHandler):
    """Daemon-local handler: generates publication-ready figures from CSV/JSON data."""

    name = "h10_plotting"
    is_daemon = True

    def can_run(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True when at least one analysis variant or legacy Analysis/ directory has CSV data."""
        # Ready when any analysis variant has at least one CSV
        for variant in ("cmd", "mlmd_dft", "combined"):
            if (project_dir / "Analysis" / variant / "arrhenius.csv").exists():
                return True
        # Legacy fallback: flat Analysis/ directory
        legacy = project_dir / "Analysis"
        canonical = results_data(project_dir)
        return (canonical.is_dir() and any(canonical.glob("*.csv"))) or \
               (legacy.is_dir() and any(legacy.glob("*.csv")))

    def is_complete(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True when all variant plot_manifest.json files are newer than their arrhenius.csv."""
        any_variant = False
        for variant in ("cmd", "mlmd_dft", "combined"):
            arr = project_dir / "Analysis" / variant / "arrhenius.csv"
            if not arr.exists():
                continue
            any_variant = True
            manifest = results_figures(project_dir) / variant / "plot_manifest.json"
            if not manifest.exists():
                return False
            if manifest.stat().st_mtime < arr.stat().st_mtime:
                return False
        if any_variant:
            return True
        # Legacy path fallback
        return (results_figures(project_dir) / "plot_manifest.json").exists()

    def _plot_analysis_csvs(self, analysis_dir: Path, fig_dir: Path) -> list[str]:
        """Plot variant-specific transport figures (MSD, Arrhenius, RDF) from analysis_dir."""
        manifest: list[str] = []
        if any(analysis_dir.glob("msd_*K.csv")):
            manifest.extend(self._plot_msd_curves(analysis_dir, fig_dir))
        if (analysis_dir / "arrhenius.csv").exists():
            png = self._plot_arrhenius(analysis_dir, fig_dir)
            if png:
                manifest.append(png)
        if any(analysis_dir.glob("rdf_*.csv")):
            manifest.extend(self._plot_rdf(analysis_dir, fig_dir))
        return manifest

    def submit(self, project_dir: Path, state: "ProjectState") -> str | None:
        """Generate all publication figures (shared and per-variant) and write plot_manifest.json files."""
        shared_fig_dir = results_figures(project_dir)
        shared_fig_dir.mkdir(parents=True, exist_ok=True)
        shared_manifest: list[str] = []

        # ── Shared (non-variant) plots — run once into results/figures/ ──────
        opt_dir = dft_opt(project_dir)
        if opt_dir.is_dir() and any(opt_dir.rglob("CONTCAR")):
            shared_manifest.extend(self._plot_molecular_structures(project_dir, shared_fig_dir))

        homo_lumo_json = project_dir / "results" / "electronic" / "homo_lumo.json"
        if homo_lumo_json.exists():
            png = self._plot_homo_lumo_orbital(homo_lumo_json, shared_fig_dir)
            if png:
                shared_manifest.append(png)

        # For SSE: shared echem folder lives at parent level (one for the whole project)
        try:
            _is_sse = self.read_project_yaml(project_dir).get("category", "") == "inorganic_sse"
        except Exception:
            _is_sse = False
        echem_json = (
            project_dir.parent / "echem" / "echem_summary.json"
            if _is_sse
            else project_dir / "results" / "echem" / "echem_summary.json"
        )
        if echem_json.exists():
            echem_fig_dir = (
                project_dir.parent / "echem" / "figures" if _is_sse else shared_fig_dir
            )
            echem_fig_dir.mkdir(parents=True, exist_ok=True)
            png = self._plot_echem_window(echem_json, echem_fig_dir, project_dir)
            if png:
                shared_manifest.append(png)

        dos_csv = project_dir / "results" / "electronic" / "dos_total.csv"
        if dos_csv.exists():
            png = self._plot_dos(project_dir, shared_fig_dir)
            if png:
                shared_manifest.append(png)

        neb_json = project_dir / "results" / "neb_barriers.json"
        if neb_json.exists():
            for fn in (self._plot_neb_profile_spline, self._plot_neb_barriers):
                png = fn(project_dir, shared_fig_dir)
                if png:
                    shared_manifest.append(png)

        from hpca.core.paths import continuum_base
        cont_dir = continuum_base(project_dir)
        if cont_dir.is_dir() and any(cont_dir.glob("*.csv")):
            shared_manifest.extend(self._plot_continuum(project_dir, shared_fig_dir))

        bader_csv = project_dir / "results" / "electronic" / "bader_charges.csv"
        if bader_csv.exists():
            png = self._plot_bader_replot(bader_csv, shared_fig_dir)
            if png:
                shared_manifest.append(png)

        # ── Variant-specific transport plots ─────────────────────────────────
        total_figs = len(shared_manifest)
        any_variant = False
        for variant in ("cmd", "mlmd_dft", "combined"):
            analysis_dir = project_dir / "Analysis" / variant
            arr = analysis_dir / "arrhenius.csv"
            if not arr.exists():
                continue
            fig_dir = results_figures(project_dir) / variant
            manifest_path = fig_dir / "plot_manifest.json"
            if manifest_path.exists() and manifest_path.stat().st_mtime >= arr.stat().st_mtime:
                log.info("[h10_plotting] variant=%s already up-to-date", variant)
                any_variant = True
                continue
            fig_dir.mkdir(parents=True, exist_ok=True)
            variant_figs = self._plot_analysis_csvs(analysis_dir, fig_dir)
            manifest_path.write_text(
                json.dumps({"variant": variant, "figures": variant_figs,
                            "shared_figures": shared_manifest}, indent=2))
            log.info("[h10_plotting] variant=%s: %d figures", variant, len(variant_figs))
            total_figs += len(variant_figs)
            any_variant = True

        if not any_variant:
            # Legacy fallback: flat Analysis/ with no variant subdirs
            analysis_dir = results_data(project_dir)
            if not analysis_dir.is_dir() or not any(analysis_dir.glob("*.csv")):
                analysis_dir = project_dir / "Analysis"
            shared_manifest.extend(self._plot_analysis_csvs(analysis_dir, shared_fig_dir))
            (shared_fig_dir / "plot_manifest.json").write_text(
                json.dumps({"figures": shared_manifest}, indent=2))
            total_figs = len(shared_manifest)

        state.set_stage("h10_plotting", "COMPLETE", n_figures=total_figs)
        log.info("[h10_plotting] Generated %d figures for %s", total_figs, project_dir.name)
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # MSD — 4-panel autopsy style (all temperatures in one figure)
    # ═══════════════════════════════════════════════════════════════════════

    def _plot_msd_curves(self, analysis_dir: Path, fig_dir: Path) -> list[str]:
        """Generate the 4-panel MSD figure (raw, log-log, β exponent, D(t)) for all temperatures."""
        try:
            import numpy as np
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            from scipy.signal import savgol_filter
        except ImportError as e:
            log.warning("[h10_plotting] MSD skipped: %s", e)
            return []

        csv_files = sorted(analysis_dir.glob("msd_*K.csv"), key=lambda p: _extract_T(p.name))
        if not csv_files:
            return []

        fig = make_subplots(
            rows=2, cols=2,
            shared_xaxes=False,
            vertical_spacing=0.15, horizontal_spacing=0.15,
        )

        csv_rows = [["temperature_K", "time_ps", "msd_ang2", "msd_smooth_ang2",
                     "log_time", "log_msd", "beta", "D_ang2_ps"]]

        for csv_path, c in zip(csv_files, AUTOPSY_COLORS):
            try:
                data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                T = _extract_T(csv_path.name)
                time_ps = data[:, 0]
                msd_a2  = data[:, 1]

                # Savitzky-Golay smooth
                wlen = max(5, int(len(msd_a2) * 0.01) | 1)  # must be odd
                if wlen % 2 == 0:
                    wlen += 1
                smooth = savgol_filter(msd_a2, wlen, 1)

                label = f"{T} K"
                c_rgba = _hex_to_rgba(c, 0.35)

                # (1,1) raw (transparent) + smoothed
                fig.add_trace(go.Scatter(x=time_ps, y=msd_a2,
                                         line=dict(color=c_rgba),
                                         name=label, showlegend=False),
                              row=1, col=1)
                fig.add_trace(go.Scatter(x=time_ps, y=smooth,
                                         line=dict(color=c, width=2),
                                         name=label, showlegend=True),
                              row=1, col=1)

                # (1,2) log-log
                valid = (time_ps > 0) & (smooth > 0)
                logx = np.log(time_ps[valid])
                logy = np.log(smooth[valid])
                fig.add_trace(go.Scatter(x=logx, y=logy,
                                         line=dict(color=c, width=2),
                                         showlegend=False),
                              row=1, col=2)

                # (2,1) β exponent = d(ln MSD)/d(ln t)
                beta = np.gradient(logy, logx)
                fig.add_trace(go.Scatter(x=time_ps[valid], y=beta,
                                         line=dict(color=c, width=2),
                                         showlegend=False),
                              row=2, col=1)

                # (2,2) D(t) = dMSD/dt / 6
                d_t = np.gradient(smooth, time_ps) / 6.0
                fig.add_trace(go.Scatter(x=time_ps, y=d_t,
                                         line=dict(color=c, width=2),
                                         showlegend=False),
                              row=2, col=2)

                # CSV rows
                for i in range(len(time_ps)):
                    lx = np.log(time_ps[i]) if time_ps[i] > 0 else float("nan")
                    ly = np.log(smooth[i]) if smooth[i] > 0 else float("nan")
                    b  = float(beta[i]) if valid[i] else float("nan")
                    csv_rows.append([T, round(time_ps[i], 6), round(msd_a2[i], 6),
                                     round(smooth[i], 6), round(lx, 6), round(ly, 6),
                                     round(b, 6), round(d_t[i], 9)])

            except Exception as exc:
                log.debug("[h10_plotting] MSD csv %s failed: %s", csv_path, exc)

        ax = _axis_style()
        fig.update_layout(**_layout_base(height=600, width=900))
        fig.update_xaxes(title_text="Time (ps)",       **ax, row=1, col=1)
        fig.update_yaxes(title_text="MSD (Å²)",        **ax, row=1, col=1)
        fig.update_xaxes(title_text="ln(t / ps)",      **ax, row=1, col=2)
        fig.update_yaxes(title_text="ln(MSD / Å²)",   **ax, row=1, col=2)
        fig.update_xaxes(title_text="Time (ps)",       **ax, row=2, col=1)
        fig.update_yaxes(title_text="β = d ln(MSD)/d ln(t)", **ax, row=2, col=1)
        fig.update_xaxes(title_text="Time (ps)",       **ax, row=2, col=2)
        fig.update_yaxes(title_text="D(t) (Å²/ps)",   **ax, row=2, col=2)

        stem = "msd_4panel"
        png = _save_fig(fig, stem, fig_dir)
        _write_csv(csv_rows, fig_dir / f"{stem}.csv")
        log.info("[h10_plotting] Saved %s", stem)
        return [png]

    # ═══════════════════════════════════════════════════════════════════════
    # Arrhenius — open circles + dashed fit line (autopsy style)
    # ═══════════════════════════════════════════════════════════════════════

    def _plot_arrhenius(self, analysis_dir: Path, fig_dir: Path) -> str | None:
        """Generate the Arrhenius ln(D) vs 1000/T plot with open-circle data and dashed fit line."""
        try:
            import numpy as np
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            from scipy import stats
        except ImportError as e:
            log.warning("[h10_plotting] Arrhenius skipped: %s", e)
            return None

        csv_path = analysis_dir / "arrhenius.csv"
        try:
            data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)
            if data.size == 0:
                log.warning("[h10_plotting] arrhenius.csv is empty — skipping plot")
                return None
            if data.ndim == 1:
                data = data.reshape(1, -1)
        except Exception as exc:
            log.warning("[h10_plotting] arrhenius.csv read failed: %s", exc)
            return None

        T_K   = data[:, 0]
        D_m2s = data[:, 1]
        ln_D  = data[:, 2] if data.shape[1] > 2 else np.log(D_m2s)
        inv_T = 1.0 / T_K
        Ea_eV_col = data[0, 4] if data.shape[1] > 4 else None

        c = AUTOPSY_COLORS[0]
        x_plot = 1000.0 / T_K

        fig = make_subplots(rows=1, cols=1)

        # Data points (open circles, no legend)
        fig.add_trace(go.Scatter(
            x=x_plot, y=ln_D,
            mode="markers",
            marker=dict(size=10, color=c, symbol="circle-open", line=dict(width=2, color=c)),
            name="MLMD data",
            showlegend=False,
        ), row=1, col=1)

        # Fit line (dashed, with legend showing Ea)
        Ea_eV = Ea_eV_col
        if len(T_K) >= 2:
            slope, intercept, r_val, _, _ = stats.linregress(inv_T, ln_D)
            Ea_eV = Ea_eV or -slope * 8.617333e-5
            x_fit = np.linspace(inv_T.min(), inv_T.max(), 120)
            y_fit = slope * x_fit + intercept
            fig.add_trace(go.Scatter(
                x=1000.0 * x_fit, y=y_fit,
                mode="lines",
                line=dict(color=c, dash="dash", width=2),
                name=f"Fit  E<sub>a</sub> = {Ea_eV:.3f} eV  (R²={r_val**2:.4f})",
                showlegend=True,
            ), row=1, col=1)

        ax = _axis_style()
        fig.update_layout(**_layout_base(height=500, width=450))
        fig.update_xaxes(title_text="1000/T (K<sup>−1</sup>)", **ax, row=1, col=1)
        fig.update_yaxes(title_text="ln(D)  [D in m²/s]",     **ax, row=1, col=1)

        stem = "arrhenius"
        png = _save_fig(fig, stem, fig_dir)
        _write_csv(
            [["T_K", "D_m2s", "ln_D", "inv_T_1000", "Ea_eV"]] +
            [[round(t, 1), float(d), round(float(ld), 6),
              round(1000 / t, 6), round(Ea_eV, 6) if Ea_eV else ""]
             for t, d, ld in zip(T_K, D_m2s, ln_D)],
            fig_dir / f"{stem}.csv",
        )
        log.info("[h10_plotting] Saved arrhenius (Ea=%.3f eV)", Ea_eV or -1)
        return png

    # ═══════════════════════════════════════════════════════════════════════
    # RDF
    # ═══════════════════════════════════════════════════════════════════════

    def _plot_rdf(self, analysis_dir: Path, fig_dir: Path) -> list[str]:
        """Generate one RDF figure per pair type, overlaying all temperatures on a single axis."""
        try:
            import numpy as np
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
        except ImportError:
            return []

        csv_files = list(analysis_dir.glob("rdf_*.csv"))
        if not csv_files:
            return []

        pairs: dict[str, list[Path]] = {}
        for f in csv_files:
            parts = f.stem.split("_")
            # Collect all tokens between "rdf" and the temperature token (e.g. "300K")
            # so "rdf_Li-O_total_300K" → pair="Li-O_total"
            pair_parts = []
            for p in parts[1:]:
                if re.match(r"^\d+K$", p):
                    break
                pair_parts.append(p)
            pair = "_".join(pair_parts) if pair_parts else "pair"
            pairs.setdefault(pair, []).append(f)

        out_pngs: list[str] = []
        for pair, files in pairs.items():
            files_sorted = sorted(files, key=lambda p: _extract_T(p.name))
            fig = make_subplots(rows=1, cols=1)
            csv_rows = [["pair", "T_K", "r_ang", "g_r"]]

            for f, c in zip(files_sorted, AUTOPSY_COLORS):
                try:
                    data = np.loadtxt(str(f), delimiter=",", skiprows=1)
                    if data.ndim == 1:
                        data = data.reshape(1, -1)
                    T = _extract_T(f.name)
                    fig.add_trace(go.Scatter(
                        x=data[:, 0], y=data[:, 1],
                        mode="lines", line=dict(color=c, width=2),
                        name=f"{T} K", showlegend=True,
                    ), row=1, col=1)
                    for r, g in zip(data[:, 0], data[:, 1]):
                        csv_rows.append([pair, T, round(float(r), 4), round(float(g), 6)])
                except Exception:
                    continue

            ax = _axis_style()
            fig.update_layout(**_layout_base(height=450, width=500))
            fig.update_xaxes(title_text="r (Å)",  **ax, row=1, col=1)
            fig.update_yaxes(title_text="g(r)",   **ax, row=1, col=1)

            stem = f"rdf_{pair}"
            png = _save_fig(fig, stem, fig_dir)
            _write_csv(csv_rows, fig_dir / f"{stem}.csv")
            out_pngs.append(png)

        log.info("[h10_plotting] Saved %d RDF figures", len(out_pngs))
        return out_pngs

    # ═══════════════════════════════════════════════════════════════════════
    # DOS — filled area, spin-up/down
    # ═══════════════════════════════════════════════════════════════════════

    def _plot_dos(self, project_dir: Path, fig_dir: Path) -> str | None:
        """Plot total DOS as filled area traces (spin-up above, spin-down mirrored below)."""
        csv_path = project_dir / "results" / "electronic" / "dos_total.csv"
        try:
            import numpy as np
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)
            e       = data[:, 0]
            dos_up  = data[:, 1]
            dos_dn  = data[:, 2] if data.shape[1] > 2 else None

            fig = make_subplots(rows=1, cols=1)
            c_up = AUTOPSY_COLORS[0]
            c_dn = AUTOPSY_COLORS[1]

            fig.add_trace(go.Scatter(
                x=e, y=dos_up, mode="lines",
                line=dict(color=c_up, width=1.5),
                fill="tozeroy", fillcolor=_hex_to_rgba(c_up, 0.4),
                name="DOS ↑", showlegend=True,
            ))
            if dos_dn is not None:
                fig.add_trace(go.Scatter(
                    x=e, y=-dos_dn, mode="lines",
                    line=dict(color=c_dn, width=1.5),
                    fill="tozeroy", fillcolor=_hex_to_rgba(c_dn, 0.4),
                    name="DOS ↓", showlegend=True,
                ))
            fig.add_vline(x=0, line=dict(color="gray", dash="dash", width=1))

            ax = _axis_style()
            fig.update_layout(**_layout_base(height=450, width=550))
            fig.update_xaxes(title_text="E − E<sub>F</sub> (eV)", **ax)
            fig.update_yaxes(title_text="DOS (states/eV)",        **ax)

            stem = "dos_total"
            png = _save_fig(fig, stem, fig_dir)
            _write_csv([["E_eV", "DOS_up", "DOS_dn"]] +
                       [[round(float(ee), 4), round(float(u), 4),
                         round(float(dos_dn[i]), 4) if dos_dn is not None else ""]
                        for i, (ee, u) in enumerate(zip(e, dos_up))],
                       fig_dir / f"{stem}.csv")
            log.info("[h10_plotting] Saved dos_total")
            return png
        except Exception as exc:
            log.warning("[h10_plotting] DOS plot failed: %s", exc)
            return None

    # ═══════════════════════════════════════════════════════════════════════
    # NEB barriers — bar chart
    # ═══════════════════════════════════════════════════════════════════════

    def _plot_neb_barriers(self, project_dir: Path, fig_dir: Path) -> str | None:
        """Generate a bar chart of NEB migration barriers (meV) coloured by mechanism."""
        neb_json = project_dir / "results" / "neb_barriers.json"
        try:
            neb_data = json.loads(neb_json.read_text())
        except Exception:
            return None
        if not neb_data:
            return None

        try:
            import plotly.graph_objects as go

            path_names = list(neb_data.keys())
            Ea_vals    = [neb_data[p].get("Ea_meV", 0)   for p in path_names]
            mechs      = [neb_data[p].get("mechanism", "?") for p in path_names]

            colors = [AUTOPSY_COLORS[0] if m == "vacancy" else AUTOPSY_COLORS[1]
                      for m in mechs]

            fig = go.Figure(go.Bar(
                x=path_names, y=Ea_vals, marker_color=colors,
                text=[f"{v:.0f}" for v in Ea_vals], textposition="outside",
            ))
            ax = _axis_style()
            fig.update_layout(**_layout_base(height=450, width=max(500, len(path_names) * 80)))
            fig.update_xaxes(title_text="Migration path",       **ax, tickangle=45)
            fig.update_yaxes(title_text="Migration barrier (meV)", **ax)

            stem = "neb_barriers"
            png = _save_fig(fig, stem, fig_dir)
            csv_path = project_dir / "results" / "neb_barriers.csv"
            _write_csv(
                [["path_name", "Ea_meV", "mechanism", "direction"]] +
                [[pn, neb_data[pn].get("Ea_meV", ""), neb_data[pn].get("mechanism", ""),
                  neb_data[pn].get("direction", "")]
                 for pn in path_names],
                csv_path,
            )
            log.info("[h10_plotting] Saved neb_barriers (%d paths)", len(path_names))
            return png
        except Exception as exc:
            log.warning("[h10_plotting] NEB bars failed: %s", exc)
            return None

    # ═══════════════════════════════════════════════════════════════════════
    # NEB profile — cubic spline, Ea / ΔE annotations
    # ═══════════════════════════════════════════════════════════════════════

    def _plot_neb_profile_spline(self, project_dir: Path, fig_dir: Path) -> str | None:
        """Plot NEB energy profiles as cubic splines with TS/IS/FS annotations for each path."""
        neb_json = project_dir / "results" / "neb_barriers.json"
        try:
            neb_data = json.loads(neb_json.read_text())
        except Exception:
            return None
        if not neb_data:
            return None

        paths_with_images = {
            k: v for k, v in neb_data.items()
            if "image_energies_eV" in v and len(v["image_energies_eV"]) >= 3
        }
        if not paths_with_images:
            return None

        try:
            import numpy as np
            from scipy.interpolate import CubicSpline
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            n_paths = len(paths_with_images)
            fig = make_subplots(rows=1, cols=n_paths,
                                subplot_titles=list(paths_with_images.keys()))

            csv_rows = [["path", "image", "rc", "E_meV", "E_spline_meV"]]

            for col, (path_name, pdata) in enumerate(paths_with_images.items(), start=1):
                c = AUTOPSY_COLORS[(col - 1) % len(AUTOPSY_COLORS)]
                energies = np.array(pdata["image_energies_eV"])
                n_img = len(energies)
                rc = np.linspace(0, 1, n_img)
                E_rel = (energies - energies[0]) * 1000  # meV

                cs = CubicSpline(rc, E_rel, bc_type="not-a-knot")
                rc_fine = np.linspace(0, 1, 300)
                E_fine  = cs(rc_fine)

                # Spline line
                fig.add_trace(go.Scatter(
                    x=rc_fine, y=E_fine,
                    mode="lines", line=dict(color=c, width=2.5),
                    showlegend=False,
                ), row=1, col=col)

                # Image points (open circles)
                fig.add_trace(go.Scatter(
                    x=rc, y=E_rel,
                    mode="markers",
                    marker=dict(size=10, color=c, symbol="circle-open",
                                line=dict(width=2, color=c)),
                    showlegend=False,
                ), row=1, col=col)

                # TS annotation
                ts_idx = int(np.argmax(E_fine))
                Ea = E_fine[ts_idx]
                dE = E_rel[-1]

                fig.add_annotation(
                    x=rc_fine[ts_idx], y=Ea,
                    ax=rc_fine[ts_idx] + 0.08, ay=Ea + 12,
                    text=f"E<sub>a</sub> = {Ea:.1f} meV",
                    font=dict(size=14, color=c),
                    arrowcolor=c, arrowhead=2, arrowwidth=1.5,
                    showarrow=True,
                    xref=f"x{col}", yref=f"y{col}",
                )
                fig.add_annotation(
                    x=0, y=E_rel[0] + 6, text="IS",
                    font=dict(size=13, color="black"),
                    showarrow=False,
                    xref=f"x{col}", yref=f"y{col}",
                )
                fig.add_annotation(
                    x=1.0, y=E_rel[-1] + 6,
                    text=f"FS  ΔE={dE:+.0f} meV",
                    font=dict(size=12, color="black"),
                    showarrow=False,
                    xref=f"x{col}", yref=f"y{col}",
                )

                for j, (r, e) in enumerate(zip(rc, E_rel)):
                    csv_rows.append([path_name, j, round(r, 4), round(e, 3), ""])
                for k in range(0, 300, 10):
                    csv_rows.append([path_name, f"spl{k}", round(rc_fine[k], 4),
                                     "", round(E_fine[k], 3)])

            ax = _axis_style()
            fig.update_layout(**_layout_base(height=480, width=520 * n_paths))
            for col in range(1, n_paths + 1):
                fig.update_xaxes(title_text="Reaction coordinate", **ax, row=1, col=col)
                fig.update_yaxes(title_text="Relative energy (meV)", **ax, row=1, col=col)

            stem = "neb_profile_spline"
            png = _save_fig(fig, stem, fig_dir)
            _write_csv(csv_rows, fig_dir / f"{stem}.csv")
            log.info("[h10_plotting] Saved neb_profile_spline (%d paths)", n_paths)
            return png
        except Exception as exc:
            log.warning("[h10_plotting] NEB spline profile failed: %s", exc)
            return None

    # ═══════════════════════════════════════════════════════════════════════
    # Continuum model results
    # ═══════════════════════════════════════════════════════════════════════

    def _plot_continuum(self, project_dir: Path, fig_dir: Path) -> list[str]:
        """Replot each continuum model CSV from h09_continuum as an individual Plotly figure."""
        cont_dir = project_dir / "results" / "continuum"
        try:
            import numpy as np
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
        except ImportError:
            return []

        csv_map = {
            "conductivity_T.csv":      ("T (K)",     "σ (S/m)",       False),
            "diffusivity_T.csv":       ("T (K)",     "D (m²/s)",      True),
            "sei_growth.csv":          ("Time (s)",  "Thickness (nm)", False),
            "fick_profile.csv":        ("x (nm)",    "c (norm)",      False),
            "kjma_crystallization.csv":("Time (s)",  "X (fraction)",  False),
            "vegard_stress.csv":       ("Li frac.",  "Stress (MPa)",  False),
            "vtf_conductivity.csv":    ("T (K)",     "σ (S/m)",       False),
            "nernst_planck_flux.csv":  ("c₀ (mol/m³)", "J (mol/m²s)", False),
        }

        out_pngs: list[str] = []
        for csv_name, (xlabel, ylabel, log_y) in csv_map.items():
            csv_path = cont_dir / csv_name
            if not csv_path.exists():
                continue
            try:
                data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                if data.shape[1] < 2:
                    continue

                c = AUTOPSY_COLORS[0]
                fig = make_subplots(rows=1, cols=1)
                fig.add_trace(go.Scatter(
                    x=data[:, 0], y=data[:, 1],
                    mode="lines", line=dict(color=c, width=2),
                    showlegend=False,
                ))
                ax = _axis_style(log_scale=log_y and False)  # apply log axis separately
                fig.update_layout(**_layout_base(height=430, width=480))
                fig.update_xaxes(title_text=xlabel, **ax)
                fig.update_yaxes(title_text=ylabel,
                                 type="log" if log_y else "linear",
                                 **_axis_style())

                stem = csv_name.replace(".csv", "")
                png = _save_fig(fig, stem, fig_dir)
                out_pngs.append(png)
            except Exception as exc:
                log.debug("[h10_plotting] Continuum %s failed: %s", csv_name, exc)

        log.info("[h10_plotting] Saved %d continuum figures", len(out_pngs))
        return out_pngs

    # ═══════════════════════════════════════════════════════════════════════
    # Bader charges
    # ═══════════════════════════════════════════════════════════════════════

    def _plot_bader_replot(self, csv_path: Path, fig_dir: Path) -> str | None:
        """Generate a bar chart of Bader charge transfer per atom, coloured by element."""
        try:
            import plotly.graph_objects as go

            rows: list[dict] = []
            for line in csv_path.read_text().splitlines()[1:]:
                parts = line.split(",")
                if len(parts) >= 5:
                    try:
                        rows.append({"idx": int(parts[0]), "element": parts[1],
                                     "ct": float(parts[4])})
                    except ValueError:
                        continue
            if not rows:
                return None

            unique_el = list(dict.fromkeys(r["element"] for r in rows))
            cmap = {el: AUTOPSY_COLORS[i % len(AUTOPSY_COLORS)]
                    for i, el in enumerate(unique_el)}

            fig = go.Figure()
            for el in unique_el:
                el_rows = [r for r in rows if r["element"] == el]
                fig.add_trace(go.Bar(
                    x=[r["idx"] for r in el_rows],
                    y=[r["ct"]  for r in el_rows],
                    name=el, marker_color=cmap[el],
                ))
            fig.add_hline(y=0, line=dict(color="black", width=1))

            ax = _axis_style()
            fig.update_layout(**_layout_base(height=430, width=max(500, len(rows) * 15 + 100)),
                              barmode="overlay")
            fig.update_xaxes(title_text="Atom index",            **ax)
            fig.update_yaxes(title_text="Charge transfer (e⁻)", **ax)

            stem = "bader_charges"
            png = _save_fig(fig, stem, fig_dir)
            _write_csv(
                [["atom_idx", "element", "charge_transfer_e"]] +
                [[r["idx"], r["element"], r["ct"]] for r in rows],
                fig_dir / f"{stem}.csv",
            )
            return png
        except Exception as exc:
            log.debug("[h10_plotting] Bader failed: %s", exc)
            return None

    # ═══════════════════════════════════════════════════════════════════════
    # HOMO/LUMO energy level diagram (Plotly)
    # ═══════════════════════════════════════════════════════════════════════

    def _plot_homo_lumo_orbital(self, homo_lumo_json: Path, fig_dir: Path) -> str | None:
        """Generate a HOMO/LUMO energy level diagram with gap rectangles and Koopmans annotations."""
        try:
            data = json.loads(homo_lumo_json.read_text())
        except Exception:
            return None

        try:
            import numpy as np
            import plotly.graph_objects as go

            LI_REF = -1.37
            HOMO_COL = "#2166AC"
            LUMO_COL = "#D6604D"
            GAP_COL  = "#92C5DE"
            GAP_RAD  = "#FDDBC7"

            mols      = data if isinstance(data, list) else list(data.values())
            mol_names = (list(data.keys()) if isinstance(data, dict)
                         else [m.get("name", f"mol{i}") for i, m in enumerate(mols)])
            n = len(mols)
            if n == 0:
                return None

            fig = go.Figure()
            bw = 0.5  # bar half-width in x-axis units

            csv_rows = [["molecule", "channel", "HOMO_eV", "LUMO_eV",
                          "gap_eV", "IP_eV", "EA_eV", "V_ox_V", "V_red_V"]]

            def _add_level(x, homo, lumo, col_h, col_l, gap_col, bw=bw, suffix=""):
                """Draw HOMO/LUMO bars, gap rectangle, electron arrows, and energy annotations at x."""
                gap = lumo - homo
                # Gap rectangle
                fig.add_shape(type="rect",
                              x0=x - bw/2, x1=x + bw/2, y0=homo, y1=lumo,
                              fillcolor=gap_col, opacity=0.65, line_width=0)
                # HOMO line
                fig.add_shape(type="line",
                              x0=x - bw/2, x1=x + bw/2, y0=homo, y1=homo,
                              line=dict(color=col_h, width=3))
                # LUMO line (dashed)
                fig.add_shape(type="line",
                              x0=x - bw/2, x1=x + bw/2, y0=lumo, y1=lumo,
                              line=dict(color=col_l, width=3, dash="dash"))
                # Electron arrows on HOMO (↑↓)
                for dx, dy_start, dy_end in [(-bw*0.18, homo+0.12, homo+0.50),
                                              (+bw*0.18, homo+0.50, homo+0.12)]:
                    fig.add_annotation(
                        x=x + dx, y=dy_end, ax=x + dx, ay=dy_start,
                        arrowhead=2, arrowcolor=col_h, arrowwidth=1.8,
                        showarrow=True, text="", xref="x", yref="y",
                    )
                # Gap label
                mid = (homo + lumo) / 2
                fig.add_annotation(
                    x=x, y=mid,
                    text=f"<b>{gap:.2f} eV</b>",
                    showarrow=False, font=dict(size=11, color="#1a1a1a"),
                    xref="x", yref="y",
                )
                # IP annotation
                ip  = -homo
                vox = ip + LI_REF
                fig.add_annotation(
                    x=x + bw/2 + 0.05, y=homo,
                    text=f"IP={ip:.2f} eV<br>V<sub>ox</sub>={vox:.2f} V",
                    showarrow=False, font=dict(size=9, color=col_h),
                    xanchor="left", yanchor="middle",
                    xref="x", yref="y",
                )
                ea   = -lumo
                vred = ea + LI_REF
                return ip, ea, vox, vred

            for i, (name, mol) in enumerate(zip(mol_names, mols)):
                os_flag = mol.get("open_shell", False)
                x = float(i)

                if not os_flag:
                    homo = mol.get("HOMO", -5.5)
                    lumo = mol.get("LUMO", -0.5)
                    ip, ea, vox, vred = _add_level(x, homo, lumo, HOMO_COL, LUMO_COL, GAP_COL)
                    csv_rows.append([name, "closed-shell", homo, lumo,
                                     lumo - homo, ip, ea, vox, vred])
                else:
                    ha = mol.get("HOMO_a", -5.5); la = mol.get("LUMO_a", -0.5)
                    hb = mol.get("HOMO_b", -5.5); lb = mol.get("LUMO_b", -0.5)
                    bw_h = bw * 0.44
                    ip_a, ea_a, vox_a, vred_a = _add_level(
                        x - 0.14, ha, la, "#4DAF4A", "#4DAF4A", GAP_COL, bw=bw_h)
                    ip_b, ea_b, vox_b, vred_b = _add_level(
                        x + 0.14, hb, lb, "#FF7F00", "#FF7F00", GAP_RAD, bw=bw_h)
                    csv_rows.extend([
                        [name, "alpha", ha, la, la-ha, ip_a, ea_a, vox_a, vred_a],
                        [name, "beta",  hb, lb, lb-hb, ip_b, ea_b, vox_b, vred_b],
                    ])
                    for lbl, xx in [("α", x - 0.14), ("β", x + 0.14)]:
                        fig.add_annotation(
                            x=xx, y=min(ha, hb) - 0.3, text=f"<b>{lbl}</b>",
                            showarrow=False, font=dict(size=13),
                            xref="x", yref="y",
                        )

            # Reference lines
            fig.add_hline(y=0,       line=dict(color="gray", dash="dash", width=1))
            fig.add_hline(y=LI_REF,  line=dict(color="#8B0000", dash="dot", width=1.5))
            fig.add_annotation(
                x=n - 0.3, y=LI_REF + 0.18,
                text=f"Li/Li⁺ ({LI_REF} eV)",
                showarrow=False, font=dict(size=11, color="#8B0000"),
                xanchor="right", xref="x", yref="y",
            )

            # Koopmans annotation box (top-left)
            fig.add_annotation(
                x=0.02, y=0.98, xref="paper", yref="paper",
                text=("IP ≈ −ε<sub>HOMO</sub>,  EA ≈ −ε<sub>LUMO</sub><br>"
                      "V<sub>ox</sub> = IP − 1.37 V (vs Li/Li⁺)<br>"
                      "V<sub>red</sub> = EA − 1.37 V (vs Li/Li⁺)"),
                showarrow=False, align="left",
                xanchor="left", yanchor="top",
                font=dict(size=12, color="black"),
                bgcolor="lightyellow", bordercolor="goldenrod", borderwidth=1.5,
            )

            # Invisible traces for legend
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                     line=dict(color=HOMO_COL, width=3),
                                     name="HOMO (filled)"))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                     line=dict(color=LUMO_COL, width=3, dash="dash"),
                                     name="LUMO (empty)"))

            ax = _axis_style()
            fig.update_layout(**_layout_base(height=600, width=max(500, n * 140)))
            fig.update_xaxes(
                tickvals=list(range(n)), ticktext=mol_names,
                tickangle=20, **ax,
            )
            fig.update_yaxes(title_text="Energy vs Vacuum (eV)",
                             range=[-10, 3], **ax)

            stem = "homo_lumo_orbital"
            png = _save_fig(fig, stem, fig_dir)
            _write_csv(csv_rows, fig_dir / f"{stem}.csv")
            log.info("[h10_plotting] Saved homo_lumo_orbital")
            return png
        except Exception as exc:
            log.warning("[h10_plotting] HOMO/LUMO plot failed: %s", exc)
            return None

    # ═══════════════════════════════════════════════════════════════════════
    # Electrochemical stability window — horizontal bars
    # ═══════════════════════════════════════════════════════════════════════

    def _plot_echem_window(self, echem_json: Path, fig_dir: Path,
                           project_dir: Path) -> str | None:
        """Generate a horizontal-bar electrochemical stability window plot (V_red to V_ox)."""
        try:
            data = json.loads(echem_json.read_text())
        except Exception:
            return None

        try:
            import plotly.graph_objects as go

            if "species" in data and isinstance(data["species"], list):
                species_list = data["species"]
            elif "V_ox" in data and "V_red" in data:
                species_list = [{"name": data.get("name", project_dir.name),
                                 "V_ox": data["V_ox"], "V_red": data["V_red"]}]
            else:
                return None

            n = len(species_list)
            fig = go.Figure()

            # Li-ion typical window shading
            fig.add_vrect(x0=0, x1=4.2, fillcolor="steelblue", opacity=0.08,
                          layer="below", line_width=0)
            fig.add_vline(x=0, line=dict(color="rgba(128,128,128,0.5)", dash="dash", width=1))

            csv_rows = [["molecule", "V_red_V", "V_ox_V", "window_V"]]

            for i, sp in enumerate(species_list):
                v_red = sp.get("V_red", -1.0)
                v_ox  = sp.get("V_ox",   4.0)
                win   = v_ox - v_red
                name  = sp.get("name", f"species_{i}")
                c = AUTOPSY_COLORS[i % len(AUTOPSY_COLORS)]

                fig.add_trace(go.Bar(
                    y=[name], x=[win], base=[v_red],
                    orientation="h",
                    marker=dict(color=c, opacity=0.72),
                    name=name,
                    width=0.4,
                    showlegend=False,
                ))
                # V_red marker
                fig.add_trace(go.Scatter(
                    x=[v_red], y=[name], mode="markers",
                    marker=dict(size=12, color=c, symbol="circle",
                                line=dict(color="white", width=2)),
                    showlegend=False,
                ))
                # V_ox marker
                fig.add_trace(go.Scatter(
                    x=[v_ox], y=[name], mode="markers",
                    marker=dict(size=12, color=c, symbol="square",
                                line=dict(color="white", width=2)),
                    showlegend=False,
                ))
                # Text annotations
                fig.add_annotation(x=v_red - 0.1, y=name,
                                   text=f"<b>{v_red:.2f} V</b>",
                                   showarrow=False, xanchor="right",
                                   font=dict(size=11, color=c))
                fig.add_annotation(x=v_ox  + 0.1, y=name,
                                   text=f"<b>{v_ox:.2f} V</b>",
                                   showarrow=False, xanchor="left",
                                   font=dict(size=11, color=c))
                fig.add_annotation(x=(v_red + v_ox) / 2, y=name,
                                   text=f"Δ = {win:.2f} V",
                                   showarrow=False, yshift=16,
                                   font=dict(size=10, color="black"))
                csv_rows.append([name, v_red, v_ox, win])

            # Math annotation box
            fig.add_annotation(
                x=0.98, y=0.02, xref="paper", yref="paper",
                text=("V<sub>ox</sub> = IP − 1.37 V (Trasatti ref.)<br>"
                      "V<sub>red</sub> = EA − 1.37 V<br>"
                      "IP = −ε<sub>HOMO</sub>,  EA = −ε<sub>LUMO</sub>"),
                showarrow=False, align="right",
                xanchor="right", yanchor="bottom",
                font=dict(size=12), bgcolor="lightyellow",
                bordercolor="goldenrod", borderwidth=1.5,
            )

            ax = _axis_style()
            v_min = min(sp.get("V_red", 0) for sp in species_list) - 0.8
            v_max = max(sp.get("V_ox",  0) for sp in species_list) + 1.0
            fig.update_layout(**_layout_base(height=max(350, n * 90 + 150), width=650),
                              barmode="overlay")
            fig.update_xaxes(title_text="Potential vs Li/Li⁺ (V)",
                             range=[min(-2.0, v_min), max(7.5, v_max)], **ax)
            fig.update_yaxes(title_text="", **ax)

            stem = "echem_window"
            png = _save_fig(fig, stem, fig_dir)
            _write_csv(csv_rows, fig_dir / f"{stem}.csv")
            log.info("[h10_plotting] Saved echem_window (%d species)", n)
            return png
        except Exception as exc:
            log.warning("[h10_plotting] Echem window failed: %s", exc)
            return None

    # ═══════════════════════════════════════════════════════════════════════
    # Ball-and-stick molecular/atomic structures (ASE plot_atoms, white bg)
    # Works for any project: molecules, crystal slabs, periodic cells
    # ═══════════════════════════════════════════════════════════════════════

    def _plot_molecular_structures(self, project_dir: Path, fig_dir: Path) -> list[str]:
        """Render ball-and-stick atomic structures from DFT-optimised CONTCAR/POSCAR files using ASE."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from ase.io.vasp import read_vasp
            from ase.visualize.plot import plot_atoms
        except ImportError as e:
            log.warning("[h10_plotting] Ball-stick skipped: %s", e)
            return []

        opt_dir    = dft_opt(project_dir)
        struct_dir = fig_dir / "structures"
        struct_dir.mkdir(parents=True, exist_ok=True)

        mol_dirs: list[tuple[str, Path]] = []
        for d in sorted(opt_dir.iterdir()):
            if not d.is_dir():
                continue
            for name in ("CONTCAR", "POSCAR"):
                src = d / name
                if src.exists():
                    mol_dirs.append((d.name, src))
                    break

        if not mol_dirs:
            return []

        # Rotation defaults — simple heuristic per atom count
        def _auto_rot(atoms) -> str:
            """Return an ASE rotation string heuristically chosen by atom count."""
            n = len(atoms)
            if n < 30:
                return "20x,10y,0z"   # molecule
            if n < 100:
                return "10x,5y,0z"    # small cell
            return "0x,0y,0z"         # large slab / crystal — keep z-projection

        loaded: list[tuple[str, object]] = []
        individual_pngs: list[str] = []

        for name, src in mol_dirs:
            try:
                atoms = read_vasp(str(src))
                # Centre molecule (remove PBC offset for non-periodic)
                pbc = atoms.get_pbc()
                if not any(pbc):
                    pos = atoms.get_positions()
                    atoms.set_positions(pos - pos.mean(axis=0))

                colors = [CPK_COLORS.get(s, "#AAAAAA")
                          for s in atoms.get_chemical_symbols()]
                rot = _auto_rot(atoms)

                fig_i, ax_i = plt.subplots(figsize=(3.5, 3.5))
                fig_i.patch.set_facecolor("white")
                ax_i.set_facecolor("white")
                plot_atoms(atoms, ax_i, radii=0.40, colors=colors,
                           rotation=rot, show_unit_cell=0)
                ax_i.set_axis_off()
                ax_i.set_title(name, fontsize=10, fontweight="bold", pad=4)
                out_i = str(struct_dir / f"struct_{name}.png")
                fig_i.savefig(out_i, dpi=_DPI, bbox_inches="tight",
                              facecolor="white", edgecolor="none")
                plt.close(fig_i)
                individual_pngs.append(out_i)
                loaded.append((name, atoms))
                log.info("[h10_plotting] Struct: %s", out_i)
            except Exception as exc:
                log.warning("[h10_plotting] Struct failed for %s: %s", name, exc)

        if not loaded:
            return []

        # Panel figure
        n = len(loaded)
        ncols = min(3, n)
        nrows = math.ceil(n / ncols)
        fig_p, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.8 * nrows))
        fig_p.patch.set_facecolor("white")
        # Normalise axes array
        if n == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [list(axes)]
        else:
            axes = [list(row) for row in axes]

        for idx, (name, atoms) in enumerate(loaded):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]
            ax.set_facecolor("white")
            colors = [CPK_COLORS.get(s, "#AAAAAA")
                      for s in atoms.get_chemical_symbols()]
            rot = _auto_rot(atoms)
            try:
                plot_atoms(atoms, ax, radii=0.40, colors=colors,
                           rotation=rot, show_unit_cell=0)
            except Exception:
                pass
            ax.set_axis_off()
            ax.set_title(name, fontsize=10, fontweight="bold", pad=4)

        for idx in range(n, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r][c].set_visible(False)

        fig_p.suptitle("Atomic Structures — DFT-optimised (CPK colours)",
                       fontsize=13, fontweight="bold", y=1.01)
        fig_p.tight_layout()
        panel_png = str(fig_dir / "atomic_structures_panel.png")
        fig_p.savefig(panel_png, dpi=_DPI, bbox_inches="tight",
                      facecolor="white", edgecolor="none")
        plt.close(fig_p)
        log.info("[h10_plotting] Struct panel: %s (%d)", panel_png, n)

        # Companion CSV
        _write_csv(
            [["name", "n_atoms", "formula", "source_file"]] +
            [[nm, len(at), at.get_chemical_formula(), str(opt_dir / nm / "CONTCAR")]
             for nm, at in loaded],
            fig_dir / "atomic_structures_panel.csv",
        )

        return [panel_png] + individual_pngs


# ── Module-level helpers ──────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float = 0.5) -> str:
    """Convert '#RRGGBB' to 'rgba(R,G,B,alpha)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
