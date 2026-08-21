"""
Stage 06 — Continuum Model Runner.
Dispatches to hpca.continuum.models based on project.category.

Physics models implemented
--------------------------
ion transport   : arrhenius, vtf, nernst_planck, fick_1d, effective_medium
interface       : power_law, parabolic_sei, kjma, phase_field, sei_reactive
electrochemical : butler_volmer, tafel, dfn_simple
mechanical      : vegard_stress, swelling, fracture_criterion

Usage (from pipeline.py):
    from hpca.stages.s06_continuum import run
    results = run(project, output_base=project.root / "results",
                  models=["ion", "interface"], overwrite=False)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional


from hpca.continuum.models import (
    arrhenius,
    vtf_conductivity,
    fick_1d,
    nernst_planck_1d,
    effective_medium_theory,
    sei_parabolic_growth,
    sei_reactive_diffusion,
    phase_field_allen_cahn,
    power_law_growth,
    kjma_crystallization,
    butler_volmer,
    tafel_kinetics,
    vegard_stress,
    swelling_strain,
    fracture_criterion,
    run_all_models,
)

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ---------------------------------------------------------------------------
# Category → applicable model sets
# ---------------------------------------------------------------------------

CATEGORY_MODELS: Dict[str, List[str]] = {
    "polymer": [
        "arrhenius", "vtf", "nernst_planck",
        "power_law", "parabolic_sei", "kjma",
        "butler_volmer", "vegard_stress", "swelling",
    ],
    "inorganic_sse": [
        "arrhenius", "fick_1d", "effective_medium",
        "power_law", "parabolic_sei", "kjma",
        "butler_volmer", "vegard_stress", "swelling",
    ],
    "inorganic": [
        "arrhenius", "fick_1d", "effective_medium",
        "power_law", "parabolic_sei", "kjma",
        "butler_volmer", "vegard_stress", "swelling", "fracture",
    ],
    "liquid_electrolyte": [
        "arrhenius", "vtf", "nernst_planck", "dfn_simple",
        "butler_volmer", "sei_reactive", "parabolic_sei",
    ],
}

# Group aliases used in the ``models`` argument
_GROUP_MAP: Dict[str, List[str]] = {
    "ion":             ["arrhenius", "fick_1d", "vtf",
                        "nernst_planck", "effective_medium"],
    "interface":       ["power_law", "parabolic_sei", "kjma",
                        "phase_field", "sei_reactive"],
    "electrochemical": ["butler_volmer", "tafel", "dfn_simple"],
    "mechanical":      ["vegard_stress", "swelling", "fracture"],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    project,
    output_base: Path,
    models: Optional[List[str]] = None,
    overwrite: bool = False,
) -> dict:
    """
    Run the continuum model suite for a project.

    Parameters
    ----------
    project : MaterialProject
        Loaded project object (from hpca.core.project).  The
        ``project.category`` attribute determines which models are applicable:

        * ``polymer``           — arrhenius, vtf, nernst_planck, power_law,
                                  parabolic_sei, kjma, butler_volmer,
                                  vegard_stress, swelling
        * ``inorganic_sse``     — arrhenius, fick_1d, effective_medium,
                                  power_law, parabolic_sei, kjma,
                                  butler_volmer, vegard_stress, swelling
        * ``inorganic``         — same as inorganic_sse + fracture_criterion
        * ``liquid_electrolyte``— arrhenius, vtf, nernst_planck, dfn_simple,
                                  butler_volmer, sei_reactive_diffusion,
                                  parabolic_sei

    output_base : Path
        Root output directory.  Results are written to
        ``output_base/continuum/{project.name}/``.
    models : list of str, optional
        Which models (or groups) to run.  If ``None`` or ``["all"]``, every
        model applicable to the project category is run.

        Group names expand as follows:

        * ``ion``           → arrhenius, fick_1d, vtf, nernst_planck,
                              effective_medium
        * ``interface``     → power_law, parabolic_sei, kjma, phase_field,
                              sei_reactive
        * ``electrochemical``→ butler_volmer, tafel, dfn_simple
        * ``mechanical``    → vegard_stress, swelling, fracture_criterion

        Individual model names are also accepted (e.g. ``["arrhenius",
        "fick_1d"]``).
    overwrite : bool
        Re-run models even when their CSV/figure outputs already exist.

    Returns
    -------
    dict
        Keys per model name → sub-dict of scalar results.  Additional keys:

        * ``summary_csv``   — path to ``{project.name}_continuum_summary.csv``
        * ``figure_html``   — dict of model → HTML figure path
        * ``figure_png``    — dict of model → PNG figure path
    """
    out_dir = Path(output_base) / "continuum" / project.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine applicable models ───────────────────────────────────────────
    applicable = list(CATEGORY_MODELS.get(project.category,
                                           CATEGORY_MODELS["inorganic"]))
    if models and "all" not in models:
        selected: set = set()
        for m in models:
            if m in _GROUP_MAP:
                selected.update(_GROUP_MAP[m])
            else:
                selected.add(m)
        applicable = [m for m in applicable if m in selected]

    print(f"\n{'='*60}")
    print(f"Continuum stage: {project.name}  (category={project.category})")
    print(f"Models selected: {applicable}")
    print(f"Output dir:      {out_dir}")
    print(f"{'='*60}")

    D    = project.D_best
    Ea   = project.Ea_best
    T_ref = project.T_ref

    all_results: dict = {}
    summary_rows: list = []
    fig_html: dict = {}
    fig_png:  dict = {}

    for model_name in applicable:
        print(f"  {model_name:25s} ... ", end="", flush=True)

        # Skip if output already exists and overwrite=False
        csv_out = out_dir / f"{project.name}_{model_name}.csv"
        if not overwrite and csv_out.exists():
            print("cached")
            all_results[model_name] = {"cached": True, "csv": str(csv_out)}
            continue

        try:
            result = _run_model(model_name, project, D, Ea, T_ref, out_dir)
            if result:
                all_results[model_name] = result
                scalars = {k: v for k, v in result.items()
                           if isinstance(v, (int, float, str))}
                scalars["model"] = model_name
                summary_rows.append(scalars)
                # Collect figure paths
                if "figure_html" in result:
                    fig_html[model_name] = result["figure_html"]
                if "figure_png" in result:
                    fig_png[model_name] = result["figure_png"]
                print("OK")
            else:
                print("skipped (insufficient input data)")
        except Exception as exc:
            print(f"FAILED — {exc}")
            all_results[model_name] = {"error": str(exc)}

    # ── Summary CSV ───────────────────────────────────────────────────────────
    if summary_rows:
        all_keys = sorted({k for row in summary_rows for k in row})
        csv_path = out_dir / f"{project.name}_continuum_summary.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=all_keys)
            w.writeheader()
            for row in summary_rows:
                w.writerow({k: row.get(k, "") for k in all_keys})
        all_results["summary_csv"] = str(csv_path)
        print(f"\nSummary CSV → {csv_path}")

    all_results["figure_html"] = fig_html
    all_results["figure_png"]  = fig_png
    return all_results


# ---------------------------------------------------------------------------
# Per-model dispatch
# ---------------------------------------------------------------------------


def _run_model(
    name: str, project, D: Optional[float], Ea: Optional[float],
    T_ref: int, out_dir: Path
) -> dict:
    """
    Dispatch to the appropriate model function and return a scalar results dict
    with optional ``figure_html`` / ``figure_png`` keys.
    """
    p = project

    # ── Ion transport models ──────────────────────────────────────────────────
    if name == "arrhenius":
        if D is None or Ea is None:
            return {}
        res = arrhenius(D, Ea, T_ref)
        html, png = _save_fig(res, out_dir, f"{p.name}_arrhenius")
        out = {k: v for k, v in res.items() if isinstance(v, (int, float, str))}
        out.update({"figure_html": html, "figure_png": png})
        return out

    if name == "vtf":
        A  = p.param("A_VTF",  None)
        B  = p.param("B_VTF",  None)
        T0 = p.param("T0_VTF", None)
        if None in (A, B, T0):
            return {}
        res = vtf_conductivity(A, B, T0,
                                sigma_exp=p.param("sigma_exp_Scm", None))
        html, png = _save_fig(res, out_dir, f"{p.name}_vtf")
        out = {k: v for k, v in res.items() if isinstance(v, (int, float))}
        out.update({"figure_html": html, "figure_png": png})
        return out

    if name == "fick_1d":
        if D is None:
            return {}
        res = fick_1d(D)
        html, png = _save_fig(res, out_dir, f"{p.name}_fick_1d")
        return {"D_m2s": D,
                "L_steady_nm": res.get("L_steady_nm", 20.0),
                "figure_html": html, "figure_png": png}

    if name == "nernst_planck":
        if D is None:
            return {}
        res = nernst_planck_1d(D, L_nm=10.0, V_V=0.1, c0_mM=1.0)
        html, png = _save_fig(res, out_dir, f"{p.name}_nernst_planck")
        out = {k: v for k, v in res.items() if isinstance(v, (int, float))}
        out.update({"figure_html": html, "figure_png": png})
        return out

    if name == "effective_medium":
        D_cat = p.param("D_cat", None)
        D_SSE = p.param("D_SSE", None)
        if D_cat is None or D_SSE is None:
            return {}
        res = effective_medium_theory(D_cat, D_SSE, phi1=0.3)
        return res.get("params", {})

    # ── Interface / growth models ─────────────────────────────────────────────
    if name == "power_law":
        A = p.param("A_powerlaw", 0.265)
        n = p.param("n_powerlaw", 0.155)
        res = power_law_growth(A, n)
        html, png = _save_fig(res, out_dir, f"{p.name}_power_law")
        L_100h = None
        if res.get("data") and "L_A" in res["data"]:
            L_100h = float(res["data"]["L_A"][-1])
        return {"A_um": A, "n": n, "L_100h_um": L_100h,
                "figure_html": html, "figure_png": png}

    if name == "parabolic_sei":
        k = p.param("k_SEI", 1e-18)
        res = sei_parabolic_growth(k)
        html, png = _save_fig(res, out_dir, f"{p.name}_parabolic_sei")
        delta_1h = None
        if res.get("data") and "delta_nm" in res["data"]:
            delta_1h = float(res["data"]["delta_nm"][0])
        return {"k_SEI_m2s": k, "delta_1h_nm": delta_1h,
                "figure_html": html, "figure_png": png}

    if name == "sei_reactive":
        D_SEI = p.param("D_SEI",  1e-20)
        k_rxn = p.param("k_rxn",  1e-8)
        res   = sei_reactive_diffusion(D_SEI, k_rxn, L0_nm=1.0)
        html, png = _save_fig(res, out_dir, f"{p.name}_sei_reactive")
        out = res.get("params", {})
        out.update({"figure_html": html, "figure_png": png})
        return out

    if name == "phase_field":
        res = phase_field_allen_cahn(phi0=0.0, W=1.0, kappa=0.5, M=0.1)
        html, png = _save_fig(res, out_dir, f"{p.name}_phase_field")
        out = res.get("params", {})
        out.update({"figure_html": html, "figure_png": png})
        return out

    if name == "kjma":
        k = p.param("kjma_k", 0.001)
        n = p.param("kjma_n", 2.5)
        res = kjma_crystallization(k, n)
        html, png = _save_fig(res, out_dir, f"{p.name}_kjma")
        t_half = None
        if res.get("data"):
            t_half = res["data"].get("t_half")
        return {"k": k, "n_avrami": n, "t_half": t_half,
                "figure_html": html, "figure_png": png}

    # ── Electrochemical models ────────────────────────────────────────────────
    if name == "butler_volmer":
        j0    = p.param("j0_mA_cm2", 0.5)
        alpha = p.param("alpha",     0.5)
        res   = butler_volmer(j0, alpha, alpha)
        html, png = _save_fig(res, out_dir, f"{p.name}_butler_volmer")
        j_100mV = None
        if res.get("data"):
            j_100mV = res["data"].get("j_100mV")
        return {"j0_mA_cm2": j0, "alpha": alpha, "j_100mV_mA_cm2": j_100mV,
                "figure_html": html, "figure_png": png}

    if name == "tafel":
        j0    = p.param("j0_mA_cm2", 0.5)
        alpha = p.param("alpha",     0.5)
        res   = tafel_kinetics(j0, alpha)
        html, png = _save_fig(res, out_dir, f"{p.name}_tafel")
        return {"j0_mA_cm2": j0, "figure_html": html, "figure_png": png}

    if name == "dfn_simple":
        if D is None:
            return {}
        from hpca.continuum.models import dfn_simple as _dfn
        res   = _dfn(D)
        html, png = _save_fig(res, out_dir, f"{p.name}_dfn")
        return {"D_m2s": D, "figure_html": html, "figure_png": png}

    # ── Mechanical models ─────────────────────────────────────────────────────
    if name == "vegard_stress":
        E_GPa = p.param("E_GPa",   100.0)
        nu    = p.param("nu",       0.25)
        Omega = p.param("Omega_A3", 20.0)
        res   = vegard_stress(E_GPa, nu, Omega)
        html, png = _save_fig(res, out_dir, f"{p.name}_vegard_stress")
        sigma_max = None
        if res.get("data"):
            sigma_max = res["data"].get("sigma_max_GPa")
        return {"E_GPa": E_GPa, "nu": nu, "Omega_A3": Omega,
                "sigma_max_GPa": sigma_max,
                "figure_html": html, "figure_png": png}

    if name == "swelling":
        Omega = p.param("Omega_A3", 20.0)
        rho   = p.param("rho_gcm3",  3.0)
        MW    = p.param("MW_mobile", 6.941)
        res   = swelling_strain(Omega, rho, MW)
        html, png = _save_fig(res, out_dir, f"{p.name}_swelling")
        dVV = None
        if res.get("data"):
            dVV = res["data"].get("dV_V_max_pct")
        return {"Omega_A3": Omega, "dV_V_max_pct": dVV,
                "figure_html": html, "figure_png": png}

    if name == "fracture":
        E_GPa  = p.param("E_GPa",            100.0)
        KIC    = p.param("KIC_MPa_sqrtm",       1.0)
        sigma  = p.param("sigma_max_GPa",       1.0)
        res    = fracture_criterion(E_GPa, KIC, sigma)
        return res.get("params", {})

    # Unknown model
    return {}


# ---------------------------------------------------------------------------
# Figure I/O helper
# ---------------------------------------------------------------------------


def _save_fig(res: dict, out_dir: Path, stem: str):
    """
    Extract a Plotly figure from a model result dict and save as HTML + PNG.
    Returns (html_path_str_or_None, png_path_str_or_None).
    """
    if not HAS_PLOTLY:
        return None, None
    fig = res.get("figure")
    if fig is None:
        return None, None
    html_path = out_dir / f"{stem}.html"
    png_path  = out_dir / f"{stem}.png"
    html_str  = str(html_path)
    png_str   = str(png_path)
    try:
        fig.write_html(html_str)
    except Exception:
        html_str = None
    try:
        fig.write_image(png_str, width=900, height=520)
    except Exception:
        png_str = None
    return html_str, png_str
