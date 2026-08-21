"""
h09_continuum.py — Continuum physics models (daemon-local).
Applies all relevant physics models based on project category.
Uses hpca.continuum.models with numpy/scipy fallback implementations.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .base import SimulationHandler
from hpca.core.categories import is_molecular as _cat_is_molecular

if TYPE_CHECKING:
    from hpca.orchestrator.state_tracker import ProjectState

log = logging.getLogger("hpca.orch")

# Layout: see hpca/core/paths.py
# Cross-ref: hpca/core/paths.py continuum_base(), results_data()
from hpca.core.paths import continuum_base, results_data

NREL_COLORS = ["#0079C2", "#F7A11A", "#5E9732", "#D1495B", "#6A0572",
               "#00B4D8", "#F4A261", "#6B6B6B"]


# ── Inline physics models (fallback when hpca.continuum not importable) ─────────

def _arrhenius_conductivity(T_arr, D0: float, Ea_eV: float,
                             q: float = 1.602e-19, c: float = 1e27,
                             kB: float = 8.617e-5):
    """σ(T) = D(T) * q² * c / (kB * T * e)  [S/m, approximate]"""
    import numpy as np
    D = D0 * np.exp(-Ea_eV / (kB * T_arr))
    sigma = D * q * c / (kB * T_arr)
    return D, sigma


def _power_law_sei(t_arr, A: float = 0.265e-6, n: float = 0.155):
    """L(t) = A * t^n  [m], LCO|LGPS reference: A=0.265 μm, n=0.155"""
    import numpy as np
    return A * np.power(t_arr, n)


def _fick_1d_steady(x_arr, D: float, c0: float = 1.0, L: float = 1e-6):
    """Steady-state Fick 1D: c(x) = c0 * (1 - x/L)"""
    import numpy as np
    return c0 * (1.0 - x_arr / L)


def _vtf_conductivity(T_arr, A: float, B: float, T0: float):
    """VTF (Vogel-Tammann-Fulcher): σ = A * exp(-B / (T - T0))"""
    import numpy as np
    return A * np.exp(-B / (T_arr - T0))


def _kjma_fraction(t_arr, k: float, n: float):
    """KJMA crystallization: X(t) = 1 - exp(-(k*t)^n)"""
    import numpy as np
    return 1.0 - np.exp(-np.power(k * t_arr, n))


def _vegard_stress(c_arr, E: float = 150e9, Omega: float = 1e-29,
                   c0: float = 0.5, nu: float = 0.3):
    """Vegard law stress: σ = E * Ω * (c - c₀) / (1 - ν)"""
    import numpy as np
    return E * Omega * (c_arr - c0) / (1.0 - nu)


def _nernst_planck_flux(D: float, c0: float, c1: float, L: float = 1e-6,
                         phi0: float = 0.0, phi1: float = 0.1,
                         z: float = 1.0, F: float = 96485.0,
                         R: float = 8.314, T: float = 300.0):
    """Simplified Nernst-Planck: J = -D*(dc/dx) - D*z*F*c̄/(RT)*(dφ/dx)"""
    c_avg = 0.5 * (c0 + c1)
    dc_dx = (c1 - c0) / L
    dphi_dx = (phi1 - phi0) / L
    J = -D * dc_dx - D * z * F * c_avg / (R * T) * dphi_dx
    return J


class ContinuumHandler(SimulationHandler):
    """Daemon-local handler: runs continuum physics models from MLMD results."""

    name = "h09_continuum"
    is_daemon = True

    def can_run(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True when arrhenius.csv exists in either canonical or any Analysis/{variant}/ location."""
        # Cross-ref: hpca/core/paths.py results_data()
        canonical = results_data(project_dir) / "arrhenius.csv"
        legacy_candidates = sorted(project_dir.glob("Analysis/*/arrhenius.csv"))
        return canonical.exists() or bool(legacy_candidates)

    def is_complete(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True when the continuum/ output directory contains at least 3 CSV files."""
        # Complete when continuum/ has at least 3 csv files
        # Cross-ref: hpca/core/paths.py continuum_base()
        cont_dir = continuum_base(project_dir)
        if not cont_dir.is_dir():
            return False
        return len(list(cont_dir.glob("*.csv"))) >= 3

    def submit(self, project_dir: Path, state: "ProjectState") -> str | None:
        """Run all continuum physics models in-process and write CSV/PNG outputs."""
        try:
            import numpy as np
        except ImportError:
            log.error("[h09_continuum] numpy required")
            state.set_stage("h09_continuum", "FAILED", error="numpy missing")
            return None

        yaml = self.read_project_yaml(project_dir)
        category = yaml.get("category", "inorganic_sse")
        # Cross-ref: hpca/core/paths.py continuum_base()
        output_dir = continuum_base(project_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Read D0 and Ea from arrhenius.csv
        D0, Ea_eV = self._read_arrhenius_params(project_dir)
        if D0 is None or Ea_eV is None:
            log.warning("[h09_continuum] Could not read D0/Ea from arrhenius.csv")
            state.set_stage("h09_continuum", "FAILED", error="Missing D0/Ea")
            return None

        log.info("[h09_continuum] D0=%.3e m²/s  Ea=%.3f eV  category=%s",
                 D0, Ea_eV, category)

        # Try to import hpca.continuum; fall back to inline models
        try:
            from hpca.continuum import models as hpca_models
            use_hpca = True
        except ImportError:
            use_hpca = False
            log.debug("[h09_continuum] hpca.continuum not available — using inline models")

        summary: dict = {"D0_m2s": D0, "Ea_eV": Ea_eV, "category": category}

        if _cat_is_molecular(category):
            n_models = self._run_polymer_models(output_dir, D0, Ea_eV, yaml, summary)
        else:
            n_models = self._run_inorganic_models(output_dir, D0, Ea_eV, yaml, summary)

        # Write summary
        summary["n_models_run"] = n_models
        (continuum_base(project_dir) / "continuum_summary.json").write_text(json.dumps(summary, indent=2))

        state.set_stage("h09_continuum", "COMPLETE",
                        n_models=n_models, Ea_eV=Ea_eV, D0=D0)
        log.info("[h09_continuum] COMPLETE — %d models for %s", n_models, project_dir.name)
        return None

    # ── Inorganic / SSE models ─────────────────────────────────────────────

    def _run_inorganic_models(
        self, output_dir: Path, D0: float, Ea_eV: float,
        yaml: dict, summary: dict
    ) -> int:
        """Run Arrhenius conductivity, SEI growth, Fick, KJMA, and Vegard models for inorganic SSEs."""
        import numpy as np
        n = 0

        # 1. Arrhenius conductivity over temperature range
        T_arr = np.linspace(250, 900, 200)
        D_arr, sigma_arr = _arrhenius_conductivity(T_arr, D0, Ea_eV)
        data = np.column_stack([T_arr, D_arr, sigma_arr])
        _save_csv(output_dir / "conductivity_T.csv", data,
                  "T_K,D_m2s,sigma_S_m")
        _save_png(output_dir / "conductivity_T.png",
                  T_arr, sigma_arr, xlabel="T (K)", ylabel="σ (S/m)",
                  title="Arrhenius Conductivity", color=NREL_COLORS[0])
        summary["sigma_300K_S_m"] = float(sigma_arr[np.argmin(np.abs(T_arr - 300))])
        n += 1

        # 2. SEI power-law growth
        t_arr = np.logspace(-3, 6, 300)  # 0.001 s to 1e6 s
        A = yaml.get("sei_A", 0.265e-6)
        n_exp = yaml.get("sei_n", 0.155)
        L_sei = _power_law_sei(t_arr, A=A, n=n_exp)
        data = np.column_stack([t_arr, L_sei * 1e9])  # nm
        _save_csv(output_dir / "sei_growth.csv", data, "time_s,thickness_nm")
        _save_png(output_dir / "sei_growth.png",
                  t_arr, L_sei * 1e9, xlabel="Time (s)", ylabel="SEI thickness (nm)",
                  title="SEI Growth (Power Law)", color=NREL_COLORS[1], xlog=True)
        n += 1

        # 3. Fick 1D steady-state profile
        x_arr = np.linspace(0, 1e-6, 100)  # 0 to 1 μm
        D_300 = D0 * np.exp(-Ea_eV / (8.617e-5 * 300.0))
        c_fick = _fick_1d_steady(x_arr, D=D_300, c0=yaml.get("c0", 1.0), L=1e-6)
        data = np.column_stack([x_arr * 1e9, c_fick])  # nm, normalized
        _save_csv(output_dir / "fick_profile.csv", data, "x_nm,c_norm")
        _save_png(output_dir / "fick_profile.png",
                  x_arr * 1e9, c_fick, xlabel="x (nm)", ylabel="c (normalized)",
                  title="Fick Steady-State Profile", color=NREL_COLORS[2])
        n += 1

        # 4. KJMA crystallization
        k_kjma = yaml.get("kjma_k", 1e-4)
        n_kjma = yaml.get("kjma_n", 2.5)
        t_kjma = np.linspace(0, 2e4, 300)
        X_kjma = _kjma_fraction(t_kjma, k=k_kjma, n=n_kjma)
        data = np.column_stack([t_kjma, X_kjma])
        _save_csv(output_dir / "kjma_crystallization.csv", data, "time_s,X_fraction")
        _save_png(output_dir / "kjma_crystallization.png",
                  t_kjma, X_kjma, xlabel="Time (s)", ylabel="X (fraction)",
                  title="KJMA Crystallization", color=NREL_COLORS[3])
        n += 1

        # 5. Vegard stress profile
        c_vegard = np.linspace(0.0, 1.0, 100)
        E = yaml.get("elastic_modulus_Pa", 150e9)
        sigma_vegard = _vegard_stress(c_vegard, E=E)
        data = np.column_stack([c_vegard, sigma_vegard / 1e6])  # MPa
        _save_csv(output_dir / "vegard_stress.csv", data, "c_fraction,stress_MPa")
        _save_png(output_dir / "vegard_stress.png",
                  c_vegard, sigma_vegard / 1e6,
                  xlabel="Li fraction", ylabel="Stress (MPa)",
                  title="Vegard Stress", color=NREL_COLORS[4])
        n += 1

        return n

    # ── Polymer / liquid models ─────────────────────────────────────────────

    def _run_polymer_models(
        self, output_dir: Path, D0: float, Ea_eV: float,
        yaml: dict, summary: dict
    ) -> int:
        """Run VTF conductivity, Arrhenius D(T), SEI growth, and Nernst-Planck models for polymer/liquid systems."""
        import numpy as np
        n = 0

        # 1. VTF conductivity
        A_vtf = yaml.get("vtf_A", 1e3)   # S/m
        B_vtf = yaml.get("vtf_B", 800.0)  # K
        T0_vtf = yaml.get("vtf_T0", 180.0)  # K (Tg - 50)
        T_arr = np.linspace(T0_vtf + 10, 400, 200)
        sigma_vtf = _vtf_conductivity(T_arr, A=A_vtf, B=B_vtf, T0=T0_vtf)
        data = np.column_stack([T_arr, sigma_vtf])
        _save_csv(output_dir / "vtf_conductivity.csv", data, "T_K,sigma_S_m")
        _save_png(output_dir / "vtf_conductivity.png",
                  T_arr, sigma_vtf, xlabel="T (K)", ylabel="σ (S/m)",
                  title="VTF Conductivity", color=NREL_COLORS[0])
        summary["vtf_sigma_300K_S_m"] = float(
            _vtf_conductivity(np.array([300.0]), A_vtf, B_vtf, T0_vtf)[0]
        )
        n += 1

        # 2. Arrhenius D vs T (for comparison)
        T_arr2 = np.linspace(250, 450, 100)
        D_arr, _ = _arrhenius_conductivity(T_arr2, D0, Ea_eV)
        data = np.column_stack([T_arr2, D_arr])
        _save_csv(output_dir / "diffusivity_T.csv", data, "T_K,D_m2s")
        _save_png(output_dir / "diffusivity_T.png",
                  T_arr2, D_arr, xlabel="T (K)", ylabel="D (m²/s)",
                  title="Diffusivity vs Temperature", color=NREL_COLORS[1])
        n += 1

        # 3. SEI power-law growth
        t_arr = np.logspace(-3, 6, 300)
        L_sei = _power_law_sei(t_arr)
        data = np.column_stack([t_arr, L_sei * 1e9])
        _save_csv(output_dir / "sei_growth.csv", data, "time_s,thickness_nm")
        _save_png(output_dir / "sei_growth.png",
                  t_arr, L_sei * 1e9,
                  xlabel="Time (s)", ylabel="SEI thickness (nm)",
                  title="SEI Growth", color=NREL_COLORS[2], xlog=True)
        n += 1

        # 4. Nernst-Planck flux as function of concentration gradient
        c_vals = np.linspace(0.1, 2.0, 50)
        D_300 = D0 * np.exp(-Ea_eV / (8.617e-5 * 300.0))
        J_vals = np.array([
            _nernst_planck_flux(D_300, c0=c, c1=0.1) for c in c_vals
        ])
        data = np.column_stack([c_vals, J_vals])
        _save_csv(output_dir / "nernst_planck_flux.csv", data, "c0_mol_m3,J_mol_m2s")
        _save_png(output_dir / "nernst_planck_flux.png",
                  c_vals, J_vals, xlabel="c₀ (mol/m³)", ylabel="J (mol/m²s)",
                  title="Nernst-Planck Flux", color=NREL_COLORS[3])
        n += 1

        return n

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _read_arrhenius_params(project_dir: Path) -> tuple[float | None, float | None]:
        """Read D0 (pre-exponential) and Ea from arrhenius.csv, preferring aimd variant."""
        _aimd_csv = project_dir / "Analysis" / "aimd" / "arrhenius.csv"
        _candidates = sorted(project_dir.glob("Analysis/*/arrhenius.csv"))
        csv_path = _aimd_csv if _aimd_csv.exists() else (_candidates[0] if _candidates else None)
        if csv_path is None:
            return None, None
        try:
            import numpy as np
            data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)
            if data.size == 0:
                return None, None
            if data.ndim == 1:
                data = data.reshape(1, -1)
            T_arr = data[:, 0]
            D_arr = data[:, 1]
            Ea_eV = float(data[0, 4]) if data.shape[1] > 4 else None

            if Ea_eV is None:
                # Recompute from data
                from scipy import stats
                mask = D_arr > 0
                if mask.sum() < 2:
                    return None, None
                slope, intercept, *_ = stats.linregress(1.0 / T_arr[mask],
                                                         np.log(D_arr[mask]))
                Ea_eV = -slope * 8.617333e-5
                D0 = float(np.exp(intercept))
            else:
                # D0 from Arrhenius: ln(D0) = intercept
                from scipy import stats
                mask = D_arr > 0
                if mask.sum() >= 2:
                    slope, intercept, *_ = stats.linregress(1.0 / T_arr[mask],
                                                             np.log(D_arr[mask]))
                    D0 = float(np.exp(intercept))
                else:
                    D0 = float(D_arr[-1])

            return D0, Ea_eV
        except Exception as exc:
            log.warning("[h09_continuum] arrhenius.csv read failed: %s", exc)
            return None, None


# ── Plot/CSV helpers ─────────────────────────────────────────────────────────────

def _save_csv(path: Path, data, header: str) -> None:
    """Write a numpy array to a CSV file with a header row; silently ignores errors."""
    try:
        import numpy as np
        np.savetxt(str(path), data, delimiter=",", header=header, comments="")
    except Exception as exc:
        log.debug("CSV save failed %s: %s", path, exc)


def _save_png(path: Path, x, y, xlabel: str = "x", ylabel: str = "y",
              title: str = "", color: str = "#0079C2", xlog: bool = False) -> None:
    """Render a simple line plot with matplotlib and save to PNG at 300 DPI; silently ignores errors."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, y, color=color, linewidth=1.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if xlog:
            ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(str(path), dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        log.debug("PNG save failed %s: %s", path, exc)
