"""
report.py — Automated HTML project report generator.
HPCA Pipeline · /path/to/workspace/hpca/viz/report.py

Python env: /path/to/apps/apps/cladue/env/bin/python3

Generates a self-contained, single-file HTML report with inline CSS, base64
figure images, and academic styling.  All Plotly figures are rendered as
base64 PNG images for maximum portability (no JavaScript runtime required).

Usage
-----
from hpca.viz.report import ProjectReport, generate_full_report

report = ProjectReport(project, output_dir="/path/to/workspace/LMZC/Analysis")
report.add_status(status_dict)
report.add_transport(D_dict, Ea_dict)
report.add_trajectory(char_result)
report.add_continuum(continuum_results)
report.add_electronic(dos_data)
out_path = report.render_html("LMZC_report.html")
"""

from __future__ import annotations

import base64
import io
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from .theme import NREL_COLORS, apply_nrel_theme
from .structure import plot_rdf, plot_vanhove, plot_non_gaussian
from .transport import plot_arrhenius_multi, plot_msd_multi
from .continuum_viz import (
    plot_concentration_profile,
    plot_phase_field,
    plot_stress_profile,
    plot_sei_growth,
    plot_kjma,
)
from .dos_band import plot_dos, plot_bader_charges

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fig_to_base64(fig: go.Figure, width: int = 900, height: int = 520) -> str:
    """Render a Plotly figure to a base64-encoded PNG string."""
    try:
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=1.5)
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception:
        # Fallback: blank 1×1 transparent PNG
        _blank = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
            "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )
        return _blank


def _b64_img_tag(b64: str, alt: str = "", width: str = "100%") -> str:
    """Wrap a base64 PNG string in an HTML <img> tag."""
    return (
        f'<img src="data:image/png;base64,{b64}" '
        f'alt="{alt}" style="width:{width};max-width:960px;'
        f'display:block;margin:1em auto;" />'
    )


def _fmt_value(v: Any) -> str:
    """Format a value for display in an HTML table cell."""
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) < 1e-3 or abs(v) >= 1e4:
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)


def _html_table(rows: list[dict], caption: str = "") -> str:
    """Build a styled HTML table from a list of row dicts."""
    if not rows:
        return "<p><em>No data available.</em></p>"
    headers = list(rows[0].keys())
    thead = "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
    tbody_rows = []
    for i, row in enumerate(rows):
        cells = "".join(f"<td>{_fmt_value(row.get(h))}</td>" for h in headers)
        tbody_rows.append(f"<tr>{cells}</tr>")
    cap = f"<caption>{caption}</caption>" if caption else ""
    return (
        f"<table>{cap}<thead>{thead}</thead>"
        f"<tbody>{''.join(tbody_rows)}</tbody></table>"
    )


# ---------------------------------------------------------------------------
# CSS stylesheet
# ---------------------------------------------------------------------------

_CSS = """
  body {
    font-family: Georgia, 'Times New Roman', serif;
    background: #FFFFFF;
    color: #1A1A1A;
    max-width: 1100px;
    margin: 0 auto;
    padding: 2.5em 2em 5em;
    line-height: 1.65;
  }
  h1 {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 1.7em;
    color: #0079C2;
    border-bottom: 3px solid #F7A11A;
    padding-bottom: 0.4em;
    margin-bottom: 0.5em;
  }
  h2 {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 1.2em;
    color: #0079C2;
    margin-top: 2.5em;
    border-left: 4px solid #F7A11A;
    padding-left: 0.6em;
  }
  h3 {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 1.0em;
    color: #333333;
    margin-top: 1.5em;
  }
  p.meta {
    font-family: Arial, sans-serif;
    color: #666666;
    font-size: 0.85em;
    margin-bottom: 2em;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    font-size: 0.90em;
    margin: 1em 0 2em;
    font-family: Arial, sans-serif;
  }
  caption {
    font-family: Arial, sans-serif;
    font-size: 0.85em;
    color: #444444;
    text-align: left;
    margin-bottom: 0.3em;
  }
  th {
    background: #0079C2;
    color: #FFFFFF;
    padding: 7px 12px;
    text-align: left;
    font-weight: 600;
    font-size: 0.88em;
  }
  td {
    border: 1px solid #E5E5E5;
    padding: 5px 10px;
    vertical-align: middle;
  }
  tr:nth-child(even) td { background: #F5F9FF; }
  tr:hover td { background: #EEF5FC; }
  .section {
    margin-top: 3em;
    padding-top: 0.5em;
  }
  .exec-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1em;
    margin: 1.5em 0;
  }
  .kpi-card {
    background: #F5F9FF;
    border: 1px solid #CCE5F5;
    border-radius: 6px;
    padding: 1em 1.2em;
    text-align: center;
  }
  .kpi-value {
    font-size: 1.4em;
    font-weight: bold;
    color: #0079C2;
    font-family: Arial, sans-serif;
  }
  .kpi-label {
    font-size: 0.78em;
    color: #555555;
    font-family: Arial, sans-serif;
    margin-top: 0.2em;
  }
  .status-ok   { color: #5E9732; font-weight: bold; }
  .status-warn { color: #F7A11A; font-weight: bold; }
  .status-err  { color: #E31C3D; font-weight: bold; }
  .fig-caption {
    font-family: Arial, sans-serif;
    font-size: 0.80em;
    color: #555555;
    text-align: center;
    margin-top: 0.3em;
    margin-bottom: 1.5em;
  }
  .toc {
    font-family: Arial, sans-serif;
    font-size: 0.88em;
    background: #F5F9FF;
    border: 1px solid #CCE5F5;
    border-radius: 4px;
    padding: 1em 1.5em;
    display: inline-block;
    margin-bottom: 2em;
  }
  .toc a { color: #0079C2; text-decoration: none; }
  .toc a:hover { text-decoration: underline; }
  .toc ul { margin: 0.4em 0 0 1em; padding: 0; }
  .toc li { margin: 0.25em 0; }
  hr.section-divider {
    border: none;
    border-top: 1px solid #E5E5E5;
    margin: 2em 0;
  }
"""


# ---------------------------------------------------------------------------
# CharacterizationResult — lightweight named container
# ---------------------------------------------------------------------------


class CharacterizationResult:
    """Container for trajectory characterization results.

    Attributes
    ----------
    project_name : str
    by_temperature : dict[int, dict]
        Keyed by temperature (K); each dict may contain:
        - msd: {time_ps, msd_angsq}
        - diffusivity: {D_m2s, R2}
        - rdf: {pair_label: {r, g_r}}
        - van_hove: {curves: {lag_label: (r_arr, G_s_arr)}}
        - alpha2: {lags_ps, alpha2}
    arrhenius : dict, optional
        Keys: Ea_eV, D0_m2s, R2.
    electronic : dict, optional
        Keys: energies, total_dos, pdos_dict, efermi.
    """

    def __init__(
        self,
        project_name: str,
        by_temperature: Optional[dict] = None,
        arrhenius: Optional[dict] = None,
        electronic: Optional[dict] = None,
    ) -> None:
        """Initialise with optional per-temperature analysis, Arrhenius, and electronic dicts."""
        self.project_name = project_name
        self.by_temperature: dict[int, dict] = by_temperature or {}
        self.arrhenius: dict = arrhenius or {}
        self.electronic: dict = electronic or {}


# ---------------------------------------------------------------------------
# ProjectReport
# ---------------------------------------------------------------------------


class ProjectReport:
    """Automated HTML report generator for a single HPCA material project."""

    def __init__(
        self,
        project,
        output_dir: Union[str, Path],
    ) -> None:
        """Initialise with a MaterialProject object and output directory path."""
        self._project = project
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._status: dict = {}
        self._D_dict: dict[str, dict] = {}
        self._Ea_dict: dict[str, float] = {}
        self._char_result: Optional[CharacterizationResult] = None
        self._continuum: dict = {}
        self._dos_data: dict = {}

        # HTML section fragments collected in order
        self._sections: list[str] = []

    # ------------------------------------------------------------------
    # Data registration
    # ------------------------------------------------------------------

    def add_status(self, status_dict: dict) -> None:
        """Register simulation status dict {stage: status_string}."""
        self._status = dict(status_dict)

    def add_transport(
        self,
        D_dict: dict[str, dict],
        Ea_dict: dict[str, float],
    ) -> None:
        """Register transport properties.

        D_dict: {label: {T_K: D_m2s}} for multi-temperature Arrhenius.
        Ea_dict: {label: Ea_eV}.
        """
        self._D_dict = D_dict
        self._Ea_dict = Ea_dict

    def add_trajectory(
        self,
        char_result: Union[CharacterizationResult, dict],
    ) -> None:
        """Register a CharacterizationResult or raw analysis result dict."""
        if isinstance(char_result, dict):
            # Wrap raw dict from s04_analysis.run()
            cr = CharacterizationResult(
                project_name=char_result.get("project", "unknown"),
                by_temperature=char_result.get("by_temperature", {}),
                arrhenius=char_result.get("arrhenius", {}),
                electronic=char_result.get("electronic", {}),
            )
            self._char_result = cr
        else:
            self._char_result = char_result

    def add_continuum(self, continuum_results: dict) -> None:
        """Register continuum model results dict."""
        self._continuum = dict(continuum_results)

    def add_electronic(self, dos_data: dict) -> None:
        """Register electronic structure data {energies, total_dos, pdos_dict, efermi}."""
        self._dos_data = dict(dos_data)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render_html(self, filename: Optional[str] = None) -> str:
        """Build and write the full HTML report; return the output file path."""
        proj_name = getattr(self._project, "name", str(self._project))
        full_name = getattr(self._project, "full_name", proj_name)
        timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M")

        # Build sections
        sections: list[str] = []
        toc_items: list[str] = []

        def _add_section(anchor: str, title: str, content: str) -> None:
            """Append a TOC entry and a section div to the in-progress report."""
            toc_items.append(f'<li><a href="#{anchor}">{title}</a></li>')
            sections.append(
                f'<div class="section" id="{anchor}">\n'
                f"  <h2>{title}</h2>\n{content}\n"
                f'  <hr class="section-divider"/>\n</div>'
            )

        # 1. Executive Summary
        _add_section(
            "exec-summary",
            "1. Executive Summary",
            self._build_exec_summary(proj_name, full_name),
        )

        # 2. Transport Properties
        _add_section(
            "transport",
            "2. Transport Properties",
            self._build_transport_section(),
        )

        # 3. Trajectory Analysis
        _add_section(
            "trajectory",
            "3. Trajectory Analysis",
            self._build_trajectory_section(),
        )

        # 4. Continuum Model
        _add_section(
            "continuum",
            "4. Continuum Model",
            self._build_continuum_section(),
        )

        # 5. Electronic Structure
        _add_section(
            "electronic",
            "5. Electronic Structure",
            self._build_electronic_section(),
        )

        # Table of contents
        toc_html = (
            '<div class="toc">\n'
            "  <strong>Contents</strong>\n"
            "  <ul>\n"
            + "".join(f"    {item}\n" for item in toc_items)
            + "  </ul>\n</div>"
        )

        page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>HPCA Report — {proj_name}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>HPCA Battery Materials Report</h1>
  <p class="meta">
    <strong>Project:</strong> {full_name} ({proj_name}) &nbsp;|&nbsp;
    <strong>Generated:</strong> {timestamp} &nbsp;|&nbsp;
    HPCA &nbsp;·&nbsp; <code>{proj_root}</code>
  </p>
  {toc_html}
  {''.join(sections)}
</body>
</html>
"""

        fname = filename or f"{proj_name}_report.html"
        out_path = self._output_dir / fname
        out_path.write_text(page_html, encoding="utf-8")
        return str(out_path)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_exec_summary(self, proj_name: str, full_name: str) -> str:
        """Build the executive summary HTML with stage status table and KPI cards."""
        html = ""

        # Status table
        if self._status:
            rows = [{"Stage": k, "Status": v} for k, v in self._status.items()]
            # Color-code the status cells
            status_table = _html_table(rows, caption="Simulation Stage Status")
            # Colour via CSS classes
            status_table = (
                status_table
                .replace(">complete<", ' class="status-ok">complete<')
                .replace(">done<", ' class="status-ok">done<')
                .replace(">running<", ' class="status-warn">running<')
                .replace(">failed<", ' class="status-err">failed<')
                .replace(">error<", ' class="status-err">error<')
                .replace(">pending<", ' class="status-warn">pending<')
            )
            html += status_table

        # KPI cards
        kpis: list[tuple[str, str]] = []
        # Best D
        best_D: Optional[float] = None
        best_D_label = ""
        for label, T_D_map in self._D_dict.items():
            for T_val, D_val in T_D_map.items():
                if best_D is None or D_val > best_D:
                    best_D = D_val
                    best_D_label = f"@ {T_val} K ({label})"
        if best_D is not None:
            kpis.append(
                (f"{best_D:.2e} m²/s", f"Best D_Li {best_D_label}")
            )
        # Best Ea
        if self._Ea_dict:
            Ea_str = "; ".join(
                f"{v:.3f} eV ({k})" for k, v in self._Ea_dict.items()
            )
            kpis.append((f"{list(self._Ea_dict.values())[0]:.3f} eV", "Activation Energy Eₐ"))
        # Char result summary
        if self._char_result and self._char_result.arrhenius:
            arr = self._char_result.arrhenius
            if "Ea_eV" in arr:
                kpis.append((f"{arr['Ea_eV']:.3f} eV", "Eₐ (Arrhenius fit)"))
            if "R2" in arr:
                kpis.append((f"{arr['R2']:.4f}", "Arrhenius R²"))

        if kpis:
            cards = "".join(
                f'<div class="kpi-card">'
                f'<div class="kpi-value">{val}</div>'
                f'<div class="kpi-label">{lbl}</div>'
                f"</div>"
                for val, lbl in kpis
            )
            html += f'<div class="exec-grid">{cards}</div>\n'

        return html or "<p>No summary data registered.</p>"

    def _build_transport_section(self) -> str:
        """Build the transport properties HTML section with Arrhenius plot and summary table."""
        if not self._D_dict:
            return "<p>No transport data registered via <code>add_transport()</code>.</p>"

        html = ""

        # Arrhenius plot
        if self._D_dict:
            try:
                arr_fig = plot_arrhenius_multi(self._D_dict)
                b64 = _fig_to_base64(arr_fig)
                html += _b64_img_tag(b64, "Arrhenius plot")
                html += '<p class="fig-caption">Fig. T-1. Arrhenius plot of Li⁺ diffusivity. Dashed lines are linear fits.</p>\n'
            except Exception as exc:
                html += f"<p><em>Arrhenius plot error: {exc}</em></p>\n"

        # Summary table
        rows = []
        for label, T_D_map in self._D_dict.items():
            Ea = self._Ea_dict.get(label)
            for T_val, D_val in sorted(T_D_map.items()):
                rows.append({
                    "Label": label,
                    "T (K)": int(T_val),
                    "D (m²/s)": float(D_val),
                    "Eₐ (eV)": Ea if Ea is not None else "—",
                })
        html += _html_table(rows, caption="Table T-1. Transport properties summary.")

        return html

    def _build_trajectory_section(self) -> str:
        """Build the trajectory analysis HTML section with MSD, RDF, Van Hove, and Arrhenius tables."""
        if self._char_result is None:
            return "<p>No trajectory data registered via <code>add_trajectory()</code>.</p>"

        cr = self._char_result
        html = ""

        # MSD multi-plot
        msd_inputs = []
        msd_labels = []
        for T_val, T_res in sorted(cr.by_temperature.items()):
            msd_d = T_res.get("msd") or {}
            if "time_ps" in msd_d and "msd_angsq" in msd_d:
                D_val = (T_res.get("diffusivity") or {}).get("D_m2s")
                msd_inputs.append({
                    "time_ps": msd_d["time_ps"],
                    "msd_angsq": msd_d["msd_angsq"],
                    "D_m2s": D_val,
                })
                msd_labels.append(f"{T_val} K")

        if msd_inputs:
            try:
                msd_fig = plot_msd_multi(
                    msd_inputs, msd_labels,
                    project=cr.project_name,
                )
                b64 = _fig_to_base64(msd_fig)
                html += _b64_img_tag(b64, "MSD plot")
                html += '<p class="fig-caption">Fig. Tr-1. Mean-squared displacement of mobile ions across temperatures.</p>\n'
            except Exception as exc:
                html += f"<p><em>MSD plot error: {exc}</em></p>\n"

        # RDF from first available temperature
        for T_val, T_res in sorted(cr.by_temperature.items()):
            rdf_block = T_res.get("rdf") or {}
            if rdf_block:
                r_dict = {}
                for pair, rdf_data in rdf_block.items():
                    if isinstance(rdf_data, dict) and "r" in rdf_data and "g_r" in rdf_data:
                        r_arr = np.asarray(rdf_data["r"], dtype=float)
                        g_arr = np.asarray(rdf_data["g_r"], dtype=float)
                        r_dict[pair] = g_arr
                        r_ref = r_arr
                if r_dict:
                    try:
                        rdf_fig = plot_rdf(r_ref, r_dict,
                                           title=f"RDF — {cr.project_name} @ {T_val} K")
                        b64 = _fig_to_base64(rdf_fig)
                        html += _b64_img_tag(b64, "RDF plot")
                        html += (
                            f'<p class="fig-caption">'
                            f"Fig. Tr-2. Radial distribution functions @ {T_val} K. "
                            f"Dashed verticals mark first-peak positions.</p>\n"
                        )
                    except Exception as exc:
                        html += f"<p><em>RDF plot error: {exc}</em></p>\n"
                break

        # Van Hove from first available temperature
        for T_val, T_res in sorted(cr.by_temperature.items()):
            vh = T_res.get("van_hove")
            if vh and "curves" in vh:
                G_dict = {}
                r_arr_vh = None
                for lag_label, (r_a, G_s_a) in vh["curves"].items():
                    G_dict[str(lag_label)] = np.asarray(G_s_a, dtype=float)
                    if r_arr_vh is None:
                        r_arr_vh = np.asarray(r_a, dtype=float)
                if G_dict and r_arr_vh is not None:
                    try:
                        vh_fig = plot_vanhove(
                            r_arr_vh, G_dict,
                            title=f"Van Hove G_s(r,t) — {cr.project_name} @ {T_val} K"
                        )
                        b64 = _fig_to_base64(vh_fig)
                        html += _b64_img_tag(b64, "Van Hove plot")
                        html += (
                            f'<p class="fig-caption">'
                            f"Fig. Tr-3. Van Hove self-correlation function @ {T_val} K. "
                            f"Color gradient blue→red indicates increasing time lag.</p>\n"
                        )
                    except Exception as exc:
                        html += f"<p><em>Van Hove plot error: {exc}</em></p>\n"
                break

        # Diffusivity summary table
        rows = []
        for T_val, T_res in sorted(cr.by_temperature.items()):
            diff = T_res.get("diffusivity") or {}
            row: dict[str, Any] = {"T (K)": int(T_val)}
            row["D (m²/s)"] = diff.get("D_m2s")
            row["R²"] = diff.get("R2")
            rows.append(row)
        if rows:
            html += _html_table(
                rows,
                caption="Table Tr-1. Diffusivity extracted from MSD linear fits.",
            )

        # Arrhenius result block
        if cr.arrhenius:
            arr = cr.arrhenius
            items = [
                ("Eₐ (eV)", arr.get("Ea_eV")),
                ("D₀ (m²/s)", arr.get("D0_m2s")),
                ("R²", arr.get("R2")),
            ]
            html += _html_table(
                [{k: v for k, v in items}],
                caption="Table Tr-2. Arrhenius fit parameters.",
            )

        return html or "<p>No trajectory data to display.</p>"

    def _build_continuum_section(self) -> str:
        """Build the continuum model HTML section with concentration, phase-field, SEI, and KJMA plots."""
        if not self._continuum:
            return "<p>No continuum results registered via <code>add_continuum()</code>.</p>"

        html = ""

        # Concentration profile
        if "x_um" in self._continuum and "c_t" in self._continuum:
            try:
                c_fig = plot_concentration_profile(
                    self._continuum["x_um"],
                    self._continuum["c_t"],
                    self._continuum.get("times_s", [0.0]),
                )
                b64 = _fig_to_base64(c_fig)
                html += _b64_img_tag(b64, "Concentration profile")
                html += '<p class="fig-caption">Fig. C-1. Concentration profile c(x,t) — plasma colorscale by time.</p>\n'
            except Exception as exc:
                html += f"<p><em>Concentration profile error: {exc}</em></p>\n"

        # Phase field
        if "phi_t" in self._continuum:
            try:
                pf_fig = plot_phase_field(
                    self._continuum.get("x_um", np.linspace(0, 1, 100)),
                    self._continuum["phi_t"],
                    self._continuum.get("times_s", [0.0]),
                )
                b64 = _fig_to_base64(pf_fig)
                html += _b64_img_tag(b64, "Phase field")
                html += '<p class="fig-caption">Fig. C-2. Phase-field order parameter φ(x,t).</p>\n'
            except Exception as exc:
                html += f"<p><em>Phase field plot error: {exc}</em></p>\n"

        # Stress profile
        if "sigma_MPa_t" in self._continuum:
            try:
                st_fig = plot_stress_profile(
                    self._continuum.get("x_um", np.linspace(0, 1, 100)),
                    self._continuum["sigma_MPa_t"],
                    self._continuum.get("times_s", [0.0]),
                )
                b64 = _fig_to_base64(st_fig)
                html += _b64_img_tag(b64, "Stress profile")
                html += '<p class="fig-caption">Fig. C-3. Vegard-law stress profile σ(x,t).</p>\n'
            except Exception as exc:
                html += f"<p><em>Stress plot error: {exc}</em></p>\n"

        # SEI growth
        if "sei_thickness_nm" in self._continuum:
            try:
                sei_fig = plot_sei_growth(
                    self._continuum.get("sei_times_s",
                                        np.linspace(0, 1, len(
                                            self._continuum["sei_thickness_nm"]))),
                    self._continuum["sei_thickness_nm"],
                    A=self._continuum.get("sei_A"),
                    n=self._continuum.get("sei_n"),
                )
                b64 = _fig_to_base64(sei_fig)
                html += _b64_img_tag(b64, "SEI growth")
                html += '<p class="fig-caption">Fig. C-4. SEI layer growth L(t) with power-law overlay.</p>\n'
            except Exception as exc:
                html += f"<p><em>SEI growth plot error: {exc}</em></p>\n"

        # KJMA kinetics
        if "alpha_t" in self._continuum:
            try:
                kj_fig = plot_kjma(
                    self._continuum.get("kjma_times_s",
                                        np.linspace(0, 1, len(
                                            self._continuum["alpha_t"]))),
                    self._continuum["alpha_t"],
                    k=self._continuum.get("kjma_k"),
                    n=self._continuum.get("kjma_n"),
                )
                b64 = _fig_to_base64(kj_fig)
                html += _b64_img_tag(b64, "KJMA kinetics")
                html += '<p class="fig-caption">Fig. C-5. KJMA (Avrami) transformation kinetics α(t).</p>\n'
            except Exception as exc:
                html += f"<p><em>KJMA plot error: {exc}</em></p>\n"

        # Scalar parameter table
        scalar_rows = []
        scalar_keys = [
            ("sei_A", "SEI power-law A (nm)"),
            ("sei_n", "SEI power-law n"),
            ("kjma_k", "Avrami k"),
            ("kjma_n", "Avrami n"),
            ("D_cat", "D_cathode (m²/s)"),
            ("D_sse", "D_SSE (m²/s)"),
        ]
        for key, disp in scalar_keys:
            if key in self._continuum:
                scalar_rows.append({"Parameter": disp, "Value": self._continuum[key]})
        if scalar_rows:
            html += _html_table(
                scalar_rows,
                caption="Table C-1. Continuum model parameters.",
            )

        return html or "<p>No continuum plots generated.</p>"

    def _build_electronic_section(self) -> str:
        """Build the electronic structure HTML section with DOS/PDOS and Bader charge plots."""
        if not self._dos_data:
            return "<p>No electronic structure data registered via <code>add_electronic()</code>.</p>"

        html = ""
        d = self._dos_data

        energies = d.get("energies") or d.get("energy")
        total_dos = d.get("total_dos") or d.get("dos_total")
        pdos_dict = d.get("pdos_dict") or d.get("pdos")
        efermi = float(d.get("efermi", 0.0))
        erange = d.get("erange", (-6.0, 4.0))

        if energies is not None and total_dos is not None:
            try:
                dos_fig = plot_dos(
                    energies=np.asarray(energies, dtype=float),
                    total_dos=np.asarray(total_dos, dtype=float),
                    pdos_dict=pdos_dict,
                    efermi=efermi,
                    erange=tuple(erange),
                )
                b64 = _fig_to_base64(dos_fig)
                html += _b64_img_tag(b64, "DOS/PDOS")
                html += '<p class="fig-caption">Fig. E-1. Density of states (total: filled area; PDOS: colored lines). Vertical dashed line marks E_F.</p>\n'
            except Exception as exc:
                html += f"<p><em>DOS plot error: {exc}</em></p>\n"

        # Bader charges
        bader = d.get("bader")
        if bader and "atom_labels" in bader and "charges" in bader:
            try:
                bader_fig = plot_bader_charges(
                    atom_labels=bader["atom_labels"],
                    charges=np.asarray(bader["charges"], dtype=float),
                    ref_charges=(
                        np.asarray(bader["ref_charges"], dtype=float)
                        if "ref_charges" in bader else None
                    ),
                )
                b64 = _fig_to_base64(bader_fig)
                html += _b64_img_tag(b64, "Bader charges")
                html += '<p class="fig-caption">Fig. E-2. Bader charge transfer per atom. Blue = positive transfer, red = negative.</p>\n'
            except Exception as exc:
                html += f"<p><em>Bader plot error: {exc}</em></p>\n"

        # Band gap
        if "band_gap" in d:
            bg = d["band_gap"]
            html += _html_table(
                [{"Band Gap (eV)": bg, "E_F (eV)": efermi}],
                caption="Table E-1. Electronic structure summary.",
            )

        return html or "<p>Electronic structure data present but no plots generated.</p>"


# ---------------------------------------------------------------------------
# Convenience: generate_full_report
# ---------------------------------------------------------------------------


def generate_full_report(
    project,
    output_dir: Union[str, Path],
    filename: Optional[str] = None,
) -> str:
    """Load all available results from project.root/results/ and render a report.

    Looks for standard CSV/JSON outputs from s04_analysis and s05_characterization
    in ``project.root/results/analysis/{project.name}/``.  Gracefully skips
    missing data.

    Parameters
    ----------
    project : MaterialProject
        Loaded project from hpca.core.project.
    output_dir : str or Path
        Destination directory for the HTML report.
    filename : str, optional
        Output filename.  Defaults to ``{project.name}_report.html``.

    Returns
    -------
    str
        Absolute path to the generated HTML file.
    """
    from pathlib import Path as _Path
    import json as _json

    proj_name = getattr(project, "name", str(project))
    from hpca.core.config import Config as _Cfg
    _base = _Cfg.get().hpc("project_base", ".")
    proj_root = _Path(getattr(project, "root", f"{_base}/{proj_name}"))

    report = ProjectReport(project, output_dir)

    # ── Status ────────────────────────────────────────────────────────────────
    status: dict[str, str] = {}
    for stage_dir in ["dft/vc", "dft/opt", "aimd", "mlmd/mlff", "mlmd/nvt", "cmd/nvt", "results"]:
        sd = proj_root / stage_dir
        status[stage_dir] = "present" if sd.exists() else "absent"
    report.add_status(status)

    # ── Transport ─────────────────────────────────────────────────────────────
    D_dict: dict[str, dict] = {}
    Ea_dict: dict[str, float] = {}

    if hasattr(project, "D_mlmd") and project.D_mlmd:
        for mlip, D_val in project.D_mlmd.items():
            D_dict[mlip] = {getattr(project, "T_ref", 300): float(D_val)}
    if hasattr(project, "Ea_mlmd") and project.Ea_mlmd:
        for mlip, Ea_val in project.Ea_mlmd.items():
            Ea_dict[mlip] = float(Ea_val)

    # Try to load multi-T from summary CSV
    summary_csv = proj_root / "results" / "analysis" / proj_name / \
        f"{proj_name}_diffusivity_summary.csv"
    if summary_csv.exists():
        try:
            data = np.loadtxt(str(summary_csv), delimiter=",", skiprows=1)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            T_col = data[:, 0]
            D_col = data[:, 1]
            T_D = {int(T): float(D) for T, D in zip(T_col, D_col)
                   if np.isfinite(D) and D > 0}
            if T_D:
                D_dict["MLMD"] = T_D
        except Exception:
            pass

    report.add_transport(D_dict, Ea_dict)

    # ── Trajectory (CharacterizationResult) ───────────────────────────────────
    # Build CharacterizationResult from project transport data if available
    by_T: dict[int, dict] = {}
    for T_int in getattr(project, "mlmd_temperatures", []):
        T_dict: dict = {}
        # Diffusivity
        D_best = None
        if hasattr(project, "D_mlmd"):
            for _, D_val in project.D_mlmd.items():
                D_best = float(D_val)
                break
        if D_best:
            T_dict["diffusivity"] = {"D_m2s": D_best}
        by_T[T_int] = T_dict

    # Load detailed per-T CSVs if present
    analysis_dir = proj_root / "results" / "analysis" / proj_name
    if analysis_dir.exists():
        for T_subdir in sorted(analysis_dir.iterdir()):
            if not T_subdir.is_dir():
                continue
            try:
                T_val = int(T_subdir.name)
            except ValueError:
                continue
            T_res = by_T.setdefault(T_val, {})
            # MSD CSV
            msd_csv = T_subdir / f"msd_{T_val}K.csv"
            if msd_csv.exists():
                try:
                    msd_data = np.loadtxt(str(msd_csv), delimiter=",", skiprows=1)
                    if msd_data.ndim == 2 and msd_data.shape[1] >= 2:
                        T_res["msd"] = {
                            "time_ps": msd_data[:, 0].tolist(),
                            "msd_angsq": msd_data[:, 1].tolist(),
                        }
                except Exception:
                    pass
            # RDF CSVs
            rdf_data: dict = {}
            for rdf_csv in sorted(T_subdir.glob(f"rdf_*_{T_val}K.csv")):
                try:
                    arr = np.loadtxt(str(rdf_csv), delimiter=",", skiprows=1)
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        pair = rdf_csv.stem.replace(f"rdf_", "").replace(f"_{T_val}K", "")
                        rdf_data[pair] = {"r": arr[:, 0].tolist(),
                                          "g_r": arr[:, 1].tolist()}
                except Exception:
                    pass
            if rdf_data:
                T_res["rdf"] = rdf_data

    cr = CharacterizationResult(
        project_name=proj_name,
        by_temperature=by_T,
        arrhenius={
            "Ea_eV": getattr(project, "Ea_best", None),
        },
    )
    report.add_trajectory(cr)

    # ── Continuum ─────────────────────────────────────────────────────────────
    continuum: dict = {}
    # Check for known continuum output files
    cont_dir = proj_root / "Analysis" / "continuum_data"
    if cont_dir.exists():
        for csv_file in sorted(cont_dir.glob("*.csv")):
            try:
                arr = np.loadtxt(str(csv_file), delimiter=",", skiprows=1)
                stem = csv_file.stem
                if "concentration" in stem and arr.ndim == 2:
                    continuum["x_um"] = arr[:, 0]
                    continuum["c_t"] = arr[:, 1:].T
                    continuum["times_s"] = list(range(arr.shape[1] - 1))
                elif "sei_growth" in stem and arr.ndim == 2:
                    continuum["sei_times_s"] = arr[:, 0]
                    continuum["sei_thickness_nm"] = arr[:, 1]
            except Exception:
                pass
    report.add_continuum(continuum)

    # ── Electronic structure ───────────────────────────────────────────────────
    elec: dict = {}
    dos_dir = proj_root / "dos" / "nonscf"
    doscar = dos_dir / "DOSCAR"
    if doscar.exists():
        try:
            from hpca.analysis.electronic import parse_doscar
            dos_parsed = parse_doscar(doscar)
            elec = {
                "energies": dos_parsed.get("energies"),
                "total_dos": dos_parsed.get("dos_total"),
                "efermi": dos_parsed.get("efermi", 0.0),
            }
        except Exception:
            pass
    report.add_electronic(elec)

    return report.render_html(filename)
