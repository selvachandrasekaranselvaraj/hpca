"""
Stage 04 — Property Analysis orchestrator.
Runs MSD, RDF, SEI, Van Hove, phase-transition, and transport analyses for a
project across all available MLMD and VASP AIMD trajectories.

Usage (from pipeline.py):
    from hpca.stages.s04_analysis import run
    results = run(project, output_base=project.root / "results",
                  temps_K=[300, 400, 500], analyses=["msd", "rdf", "transport"])
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


from hpca.analysis.trajectory import parse_lammps_dump, parse_xdatcar
from hpca.analysis.msd import (
    compute_msd,
    fit_diffusivity,
    van_hove_self,
    arrhenius_fit,
    run_full_transport_analysis,
    plot_arrhenius,
)
from hpca.analysis.rdf import compute_rdf, plot_rdf
from hpca.analysis.electronic import run_electronic_analysis

try:
    from hpca.analysis.sei import sei_growth_kinetics, plot_sei_growth
    from hpca.analysis.phase import lindemann_criterion, plot_lindemann
    HAS_SEI_PHASE = True
except ImportError:
    HAS_SEI_PHASE = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

VALID_ANALYSES = frozenset(
    ["msd", "rdf", "sei", "van_hove", "phase", "transport", "electronic", "all"]
)


def run(
    project,
    output_base: Path,
    temps_K: Optional[List[int]] = None,
    analyses: Optional[List[str]] = None,
    overwrite: bool = False,
) -> dict:
    """
    Run analysis suite for a project.

    Parameters
    ----------
    project : MaterialProject
        Loaded project object from hpca.core.project.
    output_base : Path
        Root output directory.  Results land in
        ``output_base/analysis/{project.name}/``.
    temps_K : list of int, optional
        Temperatures to analyse.  If None, all temperatures available in
        ``project.mlmd_dirs`` and ``project.aimd_dirs`` are used.
    analyses : list of str, optional
        Analyses to run.  Valid tokens:

        * ``msd``       — mean-squared displacement + diffusivity
        * ``rdf``       — radial distribution functions
        * ``sei``       — SEI growth kinetics from interface trajectory
        * ``van_hove``  — Van Hove self-correlation function (jump vs diffuse)
        * ``phase``     — Lindemann criterion, phase-transition detection
        * ``transport`` — full transport analysis (MSD + Green-Kubo + VTF fit)
        * ``electronic``— DOS/PDOS from vasprun.xml (static DFT sub-dir)
        * ``all``       — all of the above

        Default when *None*: ``["msd", "rdf", "transport", "electronic"]``
        plus ``sei`` and ``phase`` when their modules are available.
    overwrite : bool
        Re-run even when output files already exist.

    Returns
    -------
    dict
        ``{
            "project": str,
            "analyses": list,
            "by_temperature": {T_K: {analysis: result, ...}, ...},
            "arrhenius": {Ea_eV, D0_m2s, R2, ...},  # if >= 2 temps
            "electronic": {...},                      # if requested
            "summary_csv": str,                       # path
        }``
    """
    if analyses is None or "all" in (analyses or []):
        analyses = ["msd", "rdf", "transport", "electronic"]
        if HAS_SEI_PHASE:
            analyses += ["sei", "phase", "van_hove"]
    else:
        unknown = set(analyses) - VALID_ANALYSES
        if unknown:
            print(f"  [s04] WARNING: unrecognised analysis tokens: {unknown}")

    out_base = Path(output_base) / "analysis" / project.name
    out_base.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "project": project.name,
        "analyses": analyses,
        "by_temperature": {},
    }

    # ── MLMD (LAMMPS) trajectories ────────────────────────────────────────────
    mlmd_temps = temps_K or list(project.mlmd_temperatures)
    for T in sorted(mlmd_temps):
        dump_path = project.get_mlmd_dump(T)
        if dump_path is None or not dump_path.exists():
            print(f"  [{project.name}] T={T}K  MLMD dump not found — skipping")
            continue

        print(f"  [{project.name}] MLMD T={T}K  dump: {dump_path}")
        T_res = _analyze_lammps_trajectory(
            dump_path, project, T, out_base, analyses, overwrite
        )
        results["by_temperature"][T] = T_res

    # ── VASP AIMD trajectories ────────────────────────────────────────────────
    aimd_temps = temps_K or list(project.aimd_temperatures)
    for T in sorted(aimd_temps):
        aimd_dir = project.get_aimd_dir(T)
        if aimd_dir is None or not aimd_dir.exists():
            continue
        xdatcar = aimd_dir / "XDATCAR"
        if xdatcar.exists():
            print(f"  [{project.name}] AIMD T={T}K  XDATCAR: {xdatcar}")
            aimd_res = _analyze_vasp_aimd(
                aimd_dir, project, T, out_base, analyses, overwrite
            )
            results["by_temperature"].setdefault(T, {})["aimd"] = aimd_res

    # ── Electronic structure (DOS/PDOS) ───────────────────────────────────────
    if "electronic" in analyses:
        elec_out = out_base / "electronic"
        elec_out.mkdir(parents=True, exist_ok=True)
        try:
            elec = run_electronic_analysis(project.root, elec_out, project.name)
            results["electronic"] = elec
        except Exception as exc:
            results["electronic"] = {"error": str(exc)}

    # ── Multi-temperature Arrhenius (over all MLMD results) ───────────────────
    D_by_T: Dict[int, float] = {}
    for T, T_res in results["by_temperature"].items():
        D = (T_res.get("diffusivity") or {}).get("D_m2s")
        if D and D > 0:
            D_by_T[T] = D

    if len(D_by_T) >= 2:
        arr_fit = arrhenius_fit(list(D_by_T.keys()), list(D_by_T.values()))
        results["arrhenius"] = arr_fit
        if HAS_PLOTLY:
            D_dict = {T: {"D_m2s": D} for T, D in D_by_T.items()}
            fig = plot_arrhenius(D_dict, project.name, arr_fit)
            html = out_base / f"{project.name}_arrhenius.html"
            png  = out_base / f"{project.name}_arrhenius.png"
            fig.write_html(str(html))
            try:
                fig.write_image(str(png), width=900, height=520)
            except Exception:
                pass
            results["arrhenius"]["figure_html"] = str(html)
            results["arrhenius"]["figure_png"]  = str(png)

    # ── Summary CSV ───────────────────────────────────────────────────────────
    csv_rows = []
    for T, T_res in results["by_temperature"].items():
        row = {"temperature_K": T}
        diff = T_res.get("diffusivity", {})
        if diff:
            row["D_m2s"]   = diff.get("D_m2s", "")
            row["R2_msd"]  = diff.get("R2",    "")
        csv_rows.append(row)

    if csv_rows:
        csv_path = out_base / f"{project.name}_diffusivity_summary.csv"
        all_keys = ["temperature_K", "D_m2s", "R2_msd"]
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=all_keys)
            w.writeheader()
            for row in csv_rows:
                w.writerow({k: row.get(k, "") for k in all_keys})
        results["summary_csv"] = str(csv_path)

    return results


# ---------------------------------------------------------------------------
# Per-trajectory helpers
# ---------------------------------------------------------------------------


def _analyze_lammps_trajectory(
    traj_path: Path,
    project,
    T_K: int,
    out_dir: Path,
    analyses: List[str],
    overwrite: bool = False,
) -> dict:
    """
    Parse a LAMMPS dump_unwrapped.lmp and run the requested analyses.

    Parameters
    ----------
    traj_path : Path
        Path to ``dump_unwrapped.lmp``.
    project : MaterialProject
        Project metadata (mobile ion, species list, etc.).
    T_K : int
        Simulation temperature in K (used for labelling).
    out_dir : Path
        Base output directory; per-temperature sub-directory is created
        automatically as ``out_dir/{T_K}/``.
    analyses : list of str
        Subset of VALID_ANALYSES tokens.
    overwrite : bool
        If False and output files already exist, skip recomputation.

    Returns
    -------
    dict
        Keys depend on analyses requested:
        ``diffusivity``, ``rdf``, ``van_hove``, ``sei``, ``lindemann``.
        On parse failure returns ``{"error": str}``.
    """
    out_T = out_dir / str(T_K)
    out_T.mkdir(parents=True, exist_ok=True)

    stat       = traj_path.stat()
    size_mb    = stat.st_size / 1e6
    # Subsample very large trajectories for RDF/phase (> 500 MB)
    every_n    = max(1, int(stat.st_size // (500 * 1024 * 1024)))

    print(f"    Parsing LAMMPS dump ({size_mb:.0f} MB, every_n={every_n}) ...")
    traj = parse_lammps_dump(str(traj_path), stride=every_n)
    if not traj or "positions" not in traj:
        return {"error": "parse_lammps_dump returned empty trajectory"}

    # Frame timestep
    dt_frame_ps = 0.001 * every_n  # default 1 fs, corrected for stride
    if "timestep_fs" in traj:
        dt_frame_ps = float(traj["timestep_fs"]) * every_n / 1000.0

    n_frames = traj["positions"].shape[0]
    print(f"    {n_frames} frames loaded, dt_frame={dt_frame_ps:.3f} ps")

    results: dict = {}

    # ── MSD / transport ───────────────────────────────────────────────────────
    if "msd" in analyses or "transport" in analyses:
        msd_csv = out_T / f"msd_{T_K}K.csv"
        if overwrite or not msd_csv.exists():
            transport = run_full_transport_analysis(
                traj, project.mobile_ion, dt_frame_ps, T_K, out_T
            )
            results.update(transport)
        else:
            # Load cached D value
            try:
                data = np.loadtxt(str(msd_csv), delimiter=",", skiprows=1)
                results["diffusivity"] = {"D_m2s": float(data[-1, 2]),
                                           "cached": True}
            except Exception:
                pass

    # ── Van Hove self-correlation ─────────────────────────────────────────────
    if "van_hove" in analyses:
        try:
            mobile_idx = traj.get("species_indices", {}).get(project.mobile_ion)
            if mobile_idx is not None:
                pos_m = traj["positions"][:, mobile_idx, :]
                vh = van_hove_self(pos_m, dt_frame_ps, r_max=8.0, n_r=200,
                                    lag_steps=[1, 5, 10, 50, 100])
                results["van_hove"] = vh
                if HAS_PLOTLY:
                    _save_van_hove_fig(vh, project.name, T_K, out_T)
        except Exception as exc:
            results["van_hove_error"] = str(exc)

    # ── RDF ───────────────────────────────────────────────────────────────────
    if "rdf" in analyses:
        pairs       = _get_rdf_pairs(project)
        rdf_results = {}
        for sp_a, sp_b in pairs[:4]:
            idx_a = traj["species_indices"].get(sp_a)
            idx_b = traj["species_indices"].get(sp_b)
            if idx_a is None or idx_b is None:
                continue
            try:
                rdf_data = compute_rdf(traj, sp_a, sp_b)
                rdf_results[f"{sp_a}-{sp_b}"] = rdf_data
                if HAS_PLOTLY:
                    fig = plot_rdf(rdf_data, project.name, T_K)
                    html = out_T / f"rdf_{sp_a}_{sp_b}_{T_K}K.html"
                    fig.write_html(str(html))
                    try:
                        fig.write_image(
                            str(out_T / f"rdf_{sp_a}_{sp_b}_{T_K}K.png"),
                            width=900, height=520,
                        )
                    except Exception:
                        pass
                # Save CSV
                rdf_csv = out_T / f"rdf_{sp_a}_{sp_b}_{T_K}K.csv"
                np.savetxt(
                    str(rdf_csv),
                    np.column_stack([rdf_data["r"], rdf_data["g_r"]]),
                    delimiter=",",
                    header="r_angstrom,g_r",
                    comments="",
                )
            except Exception as exc:
                rdf_results[f"{sp_a}-{sp_b}_error"] = str(exc)
        results["rdf"] = rdf_results

    # ── SEI growth kinetics ───────────────────────────────────────────────────
    if "sei" in analyses and HAS_SEI_PHASE:
        try:
            host_species = [
                s for s in (traj.get("atom_types") or [])
                if s != project.mobile_ion
            ][:3]
            kinetics = sei_growth_kinetics(traj, host_species,
                                            [project.mobile_ion])
            results["sei"] = kinetics
            if kinetics and HAS_PLOTLY:
                fig = plot_sei_growth(kinetics, project.name)
                html = out_T / f"sei_growth_{T_K}K.html"
                fig.write_html(str(html))
                try:
                    fig.write_image(str(out_T / f"sei_growth_{T_K}K.png"),
                                    width=900, height=520)
                except Exception:
                    pass
        except Exception as exc:
            results["sei_error"] = str(exc)

    # ── Lindemann / phase detection ───────────────────────────────────────────
    if "phase" in analyses and HAS_SEI_PHASE:
        try:
            lind = lindemann_criterion(traj)
            results["lindemann"] = lind
            if lind and HAS_PLOTLY:
                fig = plot_lindemann(lind, project.name, T_K)
                html = out_T / f"lindemann_{T_K}K.html"
                fig.write_html(str(html))
                try:
                    fig.write_image(str(out_T / f"lindemann_{T_K}K.png"),
                                    width=900, height=520)
                except Exception:
                    pass
        except Exception as exc:
            results["phase_error"] = str(exc)

    return results


def _analyze_vasp_aimd(
    aimd_dir: Path,
    project,
    T_K: int,
    out_dir: Path,
    analyses: List[str],
    overwrite: bool = False,
) -> dict:
    """
    Parse a VASP AIMD directory (XDATCAR + optionally vasprun.xml) and run
    MSD analysis on the mobile ion.

    Parameters
    ----------
    aimd_dir : Path
        Directory containing XDATCAR (and optionally vasprun.xml).
    project : MaterialProject
        Project metadata.
    T_K : int
        Temperature label.
    out_dir : Path
        Output base directory.
    analyses : list of str
        Requested analysis tokens.
    overwrite : bool
        Re-run even if outputs exist.

    Returns
    -------
    dict
        ``{"source": "vasp_aimd", "diffusivity": {...}, ...}``
    """
    out_T = out_dir / f"aimd_{T_K}"
    out_T.mkdir(parents=True, exist_ok=True)

    results: dict = {"source": "vasp_aimd"}

    xdatcar = aimd_dir / "XDATCAR"
    if not xdatcar.exists():
        return {"error": f"XDATCAR not found in {aimd_dir}"}

    print(f"    Parsing XDATCAR ({xdatcar.stat().st_size / 1e6:.0f} MB) ...")
    traj = parse_xdatcar(str(xdatcar))
    if not traj or "positions" not in traj:
        return {"error": "parse_xdatcar returned empty trajectory"}

    n_frames = traj["positions"].shape[0]
    # VASP AIMD default: POTIM = 1 fs, but check INCAR if present
    dt_ps = 0.001  # 1 fs default
    incar_path = aimd_dir / "INCAR"
    if incar_path.exists():
        try:
            for line in incar_path.read_text().splitlines():
                if "POTIM" in line:
                    val = line.split("=")[-1].strip().split()[0]
                    dt_ps = float(val) / 1000.0  # fs → ps
                    break
        except Exception:
            pass

    print(f"    {n_frames} frames, dt={dt_ps*1000:.1f} fs")

    if "msd" in analyses or "transport" in analyses:
        transport = run_full_transport_analysis(
            traj, project.mobile_ion, dt_ps, T_K, out_T
        )
        results.update(transport)

    if "rdf" in analyses:
        pairs = _get_rdf_pairs(project)
        rdf_out = {}
        for sp_a, sp_b in pairs[:3]:
            idx_a = traj["species_indices"].get(sp_a)
            idx_b = traj["species_indices"].get(sp_b)
            if idx_a is None or idx_b is None:
                continue
            try:
                rdf_data = compute_rdf(traj, sp_a, sp_b)
                rdf_out[f"{sp_a}-{sp_b}"] = rdf_data
                csv_p = out_T / f"rdf_{sp_a}_{sp_b}_aimd_{T_K}K.csv"
                np.savetxt(str(csv_p),
                           np.column_stack([rdf_data["r"], rdf_data["g_r"]]),
                           delimiter=",", header="r_angstrom,g_r", comments="")
            except Exception as exc:
                rdf_out[f"{sp_a}-{sp_b}_error"] = str(exc)
        results["rdf"] = rdf_out

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_rdf_pairs(project) -> List[tuple]:
    """Return physically relevant (species_A, species_B) pairs for RDF."""
    ion = project.mobile_ion
    anion_map = {
        "Na": ["O", "Cl", "F", "S"],
        "Li": ["O", "Cl", "F", "S"],
        "F":  ["Sr", "Li", "Ca"],
    }
    cation_map = {
        "Na": ["V", "Mn", "Fe", "Co", "Ni", "Zr"],
        "Li": ["Ni", "Co", "Mn", "Zr", "Y", "Al", "Mg", "Zn", "Sb", "Sr"],
    }
    pairs = [(ion, ion)]
    for anion in anion_map.get(ion, ["O"]):
        pairs.append((ion, anion))
    for cat in cation_map.get(ion, []):
        pairs.append((ion, cat))
    return pairs[:6]


def _save_van_hove_fig(vh: dict, project_name: str, T_K: int, out_dir: Path):
    """Save Van Hove figure as HTML + PNG."""
    if not HAS_PLOTLY or vh is None:
        return
    try:
        import plotly.graph_objects as go_inner
        fig = go_inner.Figure()
        for lag_label, (r_arr, gs_arr) in vh.get("curves", {}).items():
            fig.add_trace(go_inner.Scatter(x=r_arr, y=gs_arr,
                                           mode="lines",
                                           name=f"t={lag_label}"))
        fig.update_layout(
            title=f"{project_name} Van Hove G_s(r,t) — {T_K} K",
            xaxis_title="r (Å)",
            yaxis_title="G_s(r,t)",
            template="plotly_white",
        )
        html = out_dir / f"van_hove_{T_K}K.html"
        png  = out_dir / f"van_hove_{T_K}K.png"
        fig.write_html(str(html))
        try:
            fig.write_image(str(png), width=900, height=520)
        except Exception:
            pass
    except Exception:
        pass
