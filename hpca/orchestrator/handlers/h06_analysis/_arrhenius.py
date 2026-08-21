"""
_arrhenius.py — Arrhenius Ea fit and Haven ratio from a variant's D(T) values.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("hpca.orch")

NREL_BLUE = "#0079C2"
BOLTZMANN_EV_K = 8.617333e-5


def run_arrhenius(output_dir: Path, D_per_T: dict[int, float]) -> None:
    """Fit Arrhenius ln(D) vs 1/T, save arrhenius.csv and arrhenius.png to output_dir."""
    csv_path = output_dir / "arrhenius.csv"
    header = "T_K,D_m2s,ln_D,inv_T,Ea_eV"
    if not D_per_T:
        csv_path.write_text(header + "\n")
        log.warning("[h06_analysis] No D values computed; writing header-only arrhenius.csv")
        return

    try:
        import numpy as np
        from scipy import stats
    except ImportError:
        csv_path.write_text(header + "\n")
        return

    T_arr = np.array(sorted(D_per_T.keys()), dtype=float)
    D_arr = np.array([D_per_T[int(T)] for T in T_arr])

    # Filter out non-positive D for the Arrhenius fit, but always write a CSV.
    mask = D_arr > 0
    T_valid, D_valid = T_arr[mask], D_arr[mask]
    if len(T_valid) == 0:
        log.warning("[h06_analysis] No valid (positive) D values; writing raw CSV")
        inv_T_raw = np.where(T_arr > 0, 1.0 / T_arr, np.nan)
        data = np.column_stack([T_arr, D_arr, np.full_like(D_arr, np.nan),
                                inv_T_raw, np.zeros_like(T_arr)])
        np.savetxt(str(csv_path), data, delimiter=",", header=header, comments="")
        return

    T_arr, D_arr = T_valid, D_valid
    if len(T_arr) < 2:
        # Single temperature: write CSV with D value, no Ea fit.
        inv_T = 1.0 / T_arr
        ln_D = np.log(D_arr)
        data = np.column_stack([T_arr, D_arr, ln_D, inv_T, np.zeros_like(T_arr)])
        np.savetxt(str(csv_path), data, delimiter=",", header=header, comments="")
        log.info("[h06_analysis] Single-T: D(%.0fK) = %.3e m²/s  (no Arrhenius fit)",
                 T_arr[0], D_arr[0])
        return

    inv_T = 1.0 / T_arr
    ln_D = np.log(D_arr)
    slope, intercept, r_val, _p_val, _se = stats.linregress(inv_T, ln_D)
    Ea_eV = -slope * BOLTZMANN_EV_K

    log.info("[h06_analysis] Arrhenius: Ea = %.3f eV  (R²=%.4f)", Ea_eV, r_val**2)

    Ea_col = np.full_like(T_arr, Ea_eV)
    data = np.column_stack([T_arr, D_arr, ln_D, inv_T, Ea_col])
    np.savetxt(str(csv_path), data, delimiter=",", header=header, comments="")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        inv_T_fit = np.linspace(inv_T.min(), inv_T.max(), 100)
        ln_D_fit = slope * inv_T_fit + intercept
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(1000 / T_arr, ln_D, color=NREL_BLUE, s=80, zorder=5, label="MD data")
        ax.plot(1000 * inv_T_fit, ln_D_fit, color="#D1495B", linewidth=1.5,
                label=f"Fit: Ea = {Ea_eV:.3f} eV")
        ax.set_xlabel("1000/T (K⁻¹)")
        ax.set_ylabel("ln(D)")
        ax.set_title("Arrhenius Plot")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.annotate(f"Ea = {Ea_eV:.3f} eV", xy=(0.05, 0.85), xycoords="axes fraction",
                    fontsize=12, color="#D1495B")
        fig.tight_layout()
        fig.savefig(str(output_dir / "arrhenius.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        log.debug("[h06_analysis] Arrhenius PNG failed: %s", exc)


def run_haven_ratio(project_dir: Path, output_dir: Path, mobile_ion: str,
                    D_per_T: dict[int, float]) -> None:
    """Compute Haven ratio H_R = D_sigma / D_tracer for SSE/inorganic materials.

    D_sigma (charge diffusivity) is estimated from ionic conductivity via
    Nernst-Einstein: sigma = n_c q^2 D_sigma / (kB T)  ->  D_sigma = sigma kB T / (n_c q^2)
    where n_c is the mobile-ion number density (from the DFT POSCAR). Writes
    nothing if conductivity_summary.csv or n_c cannot be determined.
    """
    try:
        import numpy as np
    except ImportError:
        return

    cond_csv = output_dir / "conductivity_summary.csv"
    if not cond_csv.exists():
        return

    try:
        data = np.genfromtxt(str(cond_csv), delimiter=",", names=True)
        if data.ndim == 0:
            data = data.reshape(1)
    except Exception:
        return

    kB = 1.380649e-23  # J/K
    q = 1.602176634e-19  # C

    from hpca.core.poscar_source import find_poscar
    from hpca.core.vasp_job import poscar_element_counts
    n_c: float | None = None
    try:
        poscar = find_poscar(project_dir, "dft")
        counts = poscar_element_counts(poscar)
        lines = poscar.read_text().splitlines()
        if len(lines) >= 5:
            scale = abs(float(lines[1].split()[0]))
            a = [float(x) * scale for x in lines[2].split()]
            b = [float(x) * scale for x in lines[3].split()]
            c = [float(x) * scale for x in lines[4].split()]
            vol_A3 = abs(
                a[0] * (b[1] * c[2] - b[2] * c[1])
                - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0])
            )
            n_mobile = counts.get(mobile_ion, 0)
            if vol_A3 > 0 and n_mobile > 0:
                n_c = n_mobile / (vol_A3 * 1e-30)  # m^-3
    except Exception:
        pass

    if n_c is None:
        log.debug("[h06_analysis] Haven ratio: cannot determine n_c — skipping")
        return

    rows: list[str] = ["T_K,D_tracer_m2s,sigma_Sm,D_sigma_m2s,haven_ratio"]
    for T in sorted(D_per_T.keys()):
        D_tr = D_per_T[T]
        try:
            T_col = data["T_K"] if "T_K" in data.dtype.names else data[data.dtype.names[0]]
            s_col = data["sigma_Sm"] if "sigma_Sm" in data.dtype.names else data[data.dtype.names[1]]
            idx = np.argmin(np.abs(T_col - T))
            sigma = float(s_col[idx])
            D_sigma = sigma * kB * T / (n_c * q**2)
            Hr = D_sigma / D_tr if D_tr > 0 else float("nan")
            rows.append(f"{T},{D_tr:.6e},{sigma:.6e},{D_sigma:.6e},{Hr:.4f}")
            log.info("[h06_analysis] Haven ratio T=%d K: H_R = %.4f", T, Hr)
        except Exception:
            rows.append(f"{T},{D_tr:.6e},nan,nan,nan")

    csv_path = output_dir / "haven_ratio.csv"
    csv_path.write_text("\n".join(rows) + "\n")
    log.info("[h06_analysis] Haven ratio CSV: %s", csv_path)
