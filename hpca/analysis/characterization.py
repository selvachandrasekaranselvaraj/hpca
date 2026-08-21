#!/usr/bin/env python3
"""
characterization.py — Unified characterization dataclass and full-analysis
runner for battery materials MD trajectories.

Ties together: MSD/diffusivity, RDF, coordination, Van Hove, alpha2.

Usage
-----
from hpca.analysis.characterization import run_full_characterization, save_characterization
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CharacterizationResult:
    """Container for all trajectory characterization outputs for one project/temperature."""

    project_name: str
    temperature_K: int
    n_frames: int
    n_atoms: int

    # Transport
    D_m2s: Optional[float] = None
    Ea_eV: Optional[float] = None

    # Structure
    rdf: Dict[str, Any] = field(default_factory=dict)           # {pair: (r, g_r)}
    coordination: Dict[str, Any] = field(default_factory=dict)  # {pair: mean_CN}

    # Dynamics
    msd: Optional[Tuple[np.ndarray, np.ndarray]] = None         # (times_ps, msd_angsq)
    vanhove: Dict[int, Any] = field(default_factory=dict)       # {dt_frame: (r, G_s)}
    alpha2: Optional[Tuple[np.ndarray, np.ndarray]] = None      # (lags, alpha2)
    hop_rate: Optional[float] = None


# ---------------------------------------------------------------------------
# Analysis runner
# ---------------------------------------------------------------------------

def run_full_characterization(
    project,
    T: Optional[int] = None,
    mobile_ion: Optional[str] = None,
    dt_ps: float = 1.0,
    rdf_pairs: Optional[list] = None,
    coord_pairs: Optional[list] = None,
    coord_cutoffs: Optional[list] = None,
    van_hove_dt_frames: Optional[list] = None,
    max_lag_frac: float = 0.5,
    skip_frac: float = 0.2,
) -> CharacterizationResult:
    """Run all trajectory analyses for one project/temperature and return CharacterizationResult.

    Parameters
    ----------
    project          : object with get_mlmd_dump(T) -> traj dict, or a pre-loaded traj dict.
    T                : temperature in K (used for metadata and dump retrieval).
    mobile_ion       : element symbol for diffusing species, e.g. 'Li', 'Na'.
    dt_ps            : time between frames in ps.
    rdf_pairs        : list of (type_A, type_B) pairs for RDF; default = [(mobile_ion, '*')].
    coord_pairs      : list of (central, neighbor) pairs for CN analysis.
    coord_cutoffs    : matching cutoff distances (Ang) for coord_pairs.
    van_hove_dt_frames : list of integer frame lags for Van Hove; default [1,10,100,1000].
    max_lag_frac     : max MSD lag as fraction of usable frames.
    skip_frac        : fraction of trajectory to skip as equilibration.
    """
    from hpca.analysis.msd import compute_msd, fit_diffusivity
    from hpca.analysis.vanhove import self_van_hove, non_gaussian_parameter

    # ------------------------------------------------------------------ load trajectory
    if hasattr(project, "get_mlmd_dump") and callable(project.get_mlmd_dump):
        traj = project.get_mlmd_dump(T)
    elif isinstance(project, dict) and "positions" in project:
        traj = project
    else:
        raise TypeError(
            "project must be either a dict trajectory or an object with get_mlmd_dump(T)."
        )

    positions  = np.asarray(traj["positions"], dtype=np.float64)
    box        = np.asarray(traj["box"],       dtype=np.float64)
    n_frames, n_atoms, _ = positions.shape

    # Resolve temperature
    if T is None:
        T = 0

    # Resolve mobile ion
    sp_indices = traj.get("species_indices", {})
    if mobile_ion is None:
        # Guess: first element not a framework species (heuristic: largest count is framework)
        counts = {k: int(v.sum()) for k, v in sp_indices.items()}
        if counts:
            mobile_ion = min(counts, key=counts.get)
            warnings.warn(
                f"mobile_ion not specified; guessing '{mobile_ion}' (lowest atom count).",
                stacklevel=2,
            )
        else:
            mobile_ion = "Li"

    project_name = project if isinstance(project, str) else str(T)

    # ------------------------------------------------------------------ mobile positions
    mask = sp_indices.get(mobile_ion)
    if mask is None:
        raise KeyError(
            f"Mobile ion '{mobile_ion}' not found. Available: {list(sp_indices.keys())}"
        )
    idx         = np.where(np.asarray(mask, dtype=bool))[0]
    mobile_pos  = positions[:, idx, :]              # (n_frames, n_mobile, 3)

    # ------------------------------------------------------------------ MSD + diffusivity
    msd_result = compute_msd(
        mobile_pos, dt_ps,
        skip_frac=skip_frac,
        max_lag_frac=max_lag_frac,
    )
    diff_result = fit_diffusivity(
        msd_result["lag_times_ps"], msd_result["msd_angsq"]
    )
    D_m2s = diff_result.get("D_m2s")
    msd_tuple = (msd_result["lag_times_ps"], msd_result["msd_angsq"])

    # ------------------------------------------------------------------ RDF
    rdf_data: Dict[str, Any] = {}
    if rdf_pairs is not None:
        try:
            from hpca.analysis.rdf import compute_rdf
            for pA, pB in rdf_pairs:
                try:
                    r, g = compute_rdf(traj, pA, pB)
                    rdf_data[f"{pA}-{pB}"] = (r, g)
                except Exception as exc:
                    warnings.warn(f"RDF {pA}-{pB} failed: {exc}", stacklevel=2)
        except ImportError:
            warnings.warn("hpca.analysis.rdf not available; skipping RDF.", stacklevel=2)

    # ------------------------------------------------------------------ Coordination
    coord_data: Dict[str, Any] = {}
    if coord_pairs is not None and coord_cutoffs is not None:
        try:
            from hpca.analysis.coordination import compute_coordination_number
            for (cA, cB), r_cut in zip(coord_pairs, coord_cutoffs):
                try:
                    cn_arr = compute_coordination_number(
                        positions, cA, cB, r_cut, box, sp_indices
                    )
                    coord_data[f"{cA}-{cB}"] = float(cn_arr.mean())
                except Exception as exc:
                    warnings.warn(f"CN {cA}-{cB} failed: {exc}", stacklevel=2)
        except ImportError:
            warnings.warn("coordination module not available; skipping.", stacklevel=2)

    # ------------------------------------------------------------------ Van Hove
    if van_hove_dt_frames is None:
        van_hove_dt_frames = [1, 10, 100, 1000]
    # Filter lags that are within trajectory length
    valid_vh_dts = [d for d in van_hove_dt_frames if d < n_frames]
    vh_data: Dict[int, Any] = {}
    try:
        vh_result = self_van_hove(
            mobile_pos,
            dt_frames=valid_vh_dts,
        )
        vh_data = vh_result
    except Exception as exc:
        warnings.warn(f"Van Hove computation failed: {exc}", stacklevel=2)

    # ------------------------------------------------------------------ Non-Gaussian parameter alpha2
    alpha2_tuple = None
    try:
        max_lag_frames = max(1, int(n_frames * max_lag_frac))
        lags, a2 = non_gaussian_parameter(mobile_pos, max_lag_frames=max_lag_frames)
        alpha2_tuple = (lags, a2)
    except Exception as exc:
        warnings.warn(f"Alpha2 computation failed: {exc}", stacklevel=2)

    return CharacterizationResult(
        project_name=str(project_name),
        temperature_K=int(T),
        n_frames=n_frames,
        n_atoms=n_atoms,
        D_m2s=D_m2s,
        Ea_eV=None,
        rdf=rdf_data,
        coordination=coord_data,
        msd=msd_tuple,
        vanhove=vh_data,
        alpha2=alpha2_tuple,
        hop_rate=None,
    )


# ---------------------------------------------------------------------------
# Save all arrays as CSV per project figure rule
# ---------------------------------------------------------------------------

def save_characterization(result: CharacterizationResult, output_dir) -> None:
    """Save all arrays in result as CSV files to output_dir/continuum_data/."""
    output_dir = Path(output_dir) / "continuum_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{result.project_name}_{result.temperature_K}K"

    # MSD
    if result.msd is not None:
        times_ps, msd_angsq = result.msd
        data = np.column_stack([times_ps, msd_angsq])
        np.savetxt(
            output_dir / f"{stem}_msd.csv",
            data,
            delimiter=",",
            header="lag_time_ps,msd_angsq",
            comments="",
        )

    # RDF
    for pair, (r, g_r) in result.rdf.items():
        data = np.column_stack([r, g_r])
        np.savetxt(
            output_dir / f"{stem}_rdf_{pair.replace('-','_')}.csv",
            data,
            delimiter=",",
            header="r_ang,g_r",
            comments="",
        )

    # Coordination
    if result.coordination:
        rows = [[k, v] for k, v in result.coordination.items()]
        with open(output_dir / f"{stem}_coordination.csv", "w") as fh:
            fh.write("pair,mean_CN\n")
            for k, v in result.coordination.items():
                fh.write(f"{k},{v:.6f}\n")

    # Van Hove
    for dt_frame, (r_centers, G_s) in result.vanhove.items():
        data = np.column_stack([r_centers, G_s])
        np.savetxt(
            output_dir / f"{stem}_vanhove_dt{dt_frame}.csv",
            data,
            delimiter=",",
            header="r_ang,G_s",
            comments="",
        )

    # Alpha2
    if result.alpha2 is not None:
        lags, a2 = result.alpha2
        data = np.column_stack([lags, a2])
        np.savetxt(
            output_dir / f"{stem}_alpha2.csv",
            data,
            delimiter=",",
            header="lag_frames,alpha2",
            comments="",
        )

    # Summary scalar CSV
    with open(output_dir / f"{stem}_summary.csv", "w") as fh:
        fh.write("property,value\n")
        fh.write(f"project_name,{result.project_name}\n")
        fh.write(f"temperature_K,{result.temperature_K}\n")
        fh.write(f"n_frames,{result.n_frames}\n")
        fh.write(f"n_atoms,{result.n_atoms}\n")
        fh.write(f"D_m2s,{result.D_m2s if result.D_m2s is not None else ''}\n")
        fh.write(f"Ea_eV,{result.Ea_eV if result.Ea_eV is not None else ''}\n")
        fh.write(f"hop_rate,{result.hop_rate if result.hop_rate is not None else ''}\n")


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------

def print_summary(result: CharacterizationResult) -> None:
    """Print formatted summary table to stdout."""
    sep = "-" * 55
    print(sep)
    print(f"  Characterization Summary: {result.project_name}")
    print(sep)
    print(f"  Temperature  : {result.temperature_K} K")
    print(f"  Frames       : {result.n_frames}")
    print(f"  Atoms        : {result.n_atoms}")
    print()

    # Transport
    print("  Transport:")
    if result.D_m2s is not None:
        print(f"    D (m^2/s)    = {result.D_m2s:.4e}")
        print(f"    D (cm^2/s)   = {result.D_m2s * 1e4:.4e}")
    else:
        print("    D            = N/A")
    if result.Ea_eV is not None:
        print(f"    Ea (eV)      = {result.Ea_eV:.4f}")
    if result.hop_rate is not None:
        print(f"    Hop rate     = {result.hop_rate:.4e} hops/atom/ps")

    # Structure
    if result.coordination:
        print()
        print("  Coordination:")
        for pair, val in result.coordination.items():
            print(f"    CN ({pair:12s}) = {val:.3f}")

    # Dynamics
    print()
    print("  Dynamics:")
    if result.msd is not None:
        t, m = result.msd
        print(f"    MSD frames   : {len(t)}")
        print(f"    MSD max      : {float(m[-1]):.2f} Ang^2 at {float(t[-1]):.1f} ps")
    if result.vanhove:
        print(f"    Van Hove dts : {sorted(result.vanhove.keys())}")
    if result.alpha2 is not None:
        lags, a2 = result.alpha2
        peak_idx = int(np.argmax(a2))
        print(f"    alpha2 peak  : {float(a2[peak_idx]):.4f} at lag {int(lags[peak_idx])}")

    print(sep)
