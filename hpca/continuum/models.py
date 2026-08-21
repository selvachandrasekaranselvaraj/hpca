"""
continuum/models.py
===================
Comprehensive continuum physics models for battery-materials simulation pipeline.

Covers all material categories:
  - Inorganic SSE (Li6PS5Cl, LiPSCl, LMZC, LYC, SrF2 …)
  - Polymer electrolytes
  - Liquid electrolytes

All plot_* helpers return plotly.graph_objects.Figure.
All model functions return {"params": ..., "data": ..., "figure": go.Figure}.

Physical constants
------------------
  kB   = 8.617333e-5 eV/K
  F    = 96485 C/mol
  R    = 8.314  J/(mol·K)
  NA   = 6.022e23
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy import stats, optimize
import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

KB_EV  = 8.617333e-5    # eV / K
FARADAY = 96485.0        # C / mol
R_JMK   = 8.314          # J / (mol K)
NA      = 6.022e23

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _default_T(T_range: Optional[np.ndarray]) -> np.ndarray:
    """Return T_range if provided, else a default 200–800 K array."""
    return T_range if T_range is not None else np.linspace(200, 800, 200)


def _default_eta(eta_range_V: Optional[np.ndarray]) -> np.ndarray:
    """Return eta_range_V if provided, else a default ±0.5 V array."""
    return eta_range_V if eta_range_V is not None else np.linspace(-0.5, 0.5, 300)


def _save_figure(fig: go.Figure, output_dir: Path, name: str) -> None:
    """Save Plotly figure as HTML and attempt kaleido PNG export."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{name}.html"
    fig.write_html(str(html_path))
    try:
        png_path = output_dir / f"{name}.png"
        fig.write_image(str(png_path), width=900, height=600, scale=2)
    except Exception:
        pass  # kaleido may not be installed


# ===========================================================================
# ION TRANSPORT MODELS
# ===========================================================================


def arrhenius(
    D_ref: float,
    Ea_eV: float,
    T_ref_K: float,
    T_range: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Arrhenius diffusivity model.

    D(T) = D_ref * exp(-Ea/kB * (1/T - 1/T_ref))

    Parameters
    ----------
    D_ref   : reference diffusivity at T_ref [m²/s]
    Ea_eV   : activation energy [eV]
    T_ref_K : reference temperature [K]
    T_range : temperature array [K]; default 200-800 K

    Returns
    -------
    dict with keys: params, data, figure
    """
    T = _default_T(T_range)
    D_arr = D_ref * np.exp(-Ea_eV / KB_EV * (1.0 / T - 1.0 / T_ref_K))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=1000.0 / T, y=np.log10(D_arr),
        mode="lines", name="D(T)",
        line=dict(color="#1f77b4", width=2.5),
    ))
    fig.update_layout(
        title=f"Arrhenius Diffusivity (Ea = {Ea_eV:.3f} eV)",
        xaxis_title="1000/T (K⁻¹)",
        yaxis_title="log₁₀ D (m²/s)",
        template="plotly_white",
        font=dict(size=14),
    )

    return {
        "params": {"D_ref": D_ref, "Ea_eV": Ea_eV, "T_ref_K": T_ref_K},
        "data":   {"T_K": T, "D_m2s": D_arr, "inv_T": 1.0 / T, "log10_D": np.log10(D_arr)},
        "figure": fig,
    }


def vtf_conductivity(
    A: float,
    B: float,
    T0: float,
    T_range: Optional[np.ndarray] = None,
    sigma_exp: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Vogel-Tammann-Fulcher (VTF) ionic conductivity model for polymer electrolytes.

    sigma(T) = A * exp(-B / (T - T0))

    Parameters
    ----------
    A         : pre-exponential factor [S/cm]
    B         : pseudo-activation parameter [K]
    T0        : ideal glass transition temperature [K]
    T_range   : temperature array [K]; default 250-400 K
    sigma_exp : optional experimental conductivities for overlay

    Returns
    -------
    dict with params, data, figure
    """
    T = T_range if T_range is not None else np.linspace(250.0, 400.0, 200)
    # Only compute where T > T0
    valid = T > T0 + 1.0
    sigma = np.full_like(T, np.nan)
    sigma[valid] = A * np.exp(-B / (T[valid] - T0))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=T, y=np.log10(sigma),
        mode="lines", name="VTF fit",
        line=dict(color="#d62728", width=2.5),
    ))
    if sigma_exp is not None:
        T_exp = T[:len(sigma_exp)]
        fig.add_trace(go.Scatter(
            x=T_exp, y=np.log10(sigma_exp),
            mode="markers", name="Experiment",
            marker=dict(color="black", size=8, symbol="circle-open"),
        ))
    fig.update_layout(
        title=f"VTF Conductivity (A={A:.2e}, B={B:.1f} K, T₀={T0:.1f} K)",
        xaxis_title="Temperature (K)",
        yaxis_title="log₁₀ σ (S/cm)",
        template="plotly_white",
        font=dict(size=14),
    )

    return {
        "params": {"A": A, "B": B, "T0": T0},
        "data":   {"T_K": T, "sigma_S_cm": sigma},
        "figure": fig,
    }


def nernst_planck_1d(
    D: float,
    L_nm: float,
    V_V: float,
    c0_mM: float,
    z: int = 1,
    T_K: float = 300.0,
) -> Dict[str, Any]:
    """
    Steady-state 1D Nernst-Planck flux through a membrane.

    J = -D * (dc/dx + z*e*F/(RT) * c * dV/dx)

    Assumes linear potential profile V(x) = V * x/L,
    solves the steady-state concentration profile analytically.

    Returns flux J [mol/(m²·s)] and c(x) profile.
    """
    L  = L_nm * 1.0e-9          # m
    c0 = c0_mM * 1.0e-3         # mol/m³  (mM → mol/m³)
    beta = z * FARADAY * V_V / (R_JMK * T_K)   # dimensionless Peclet-like number

    nx = 300
    x  = np.linspace(0.0, L, nx)
    xi = x / L                  # normalised coordinate [0,1]

    # Analytical solution for c(xi):
    # c(xi) = c0 * [exp(beta*xi) - 1] / [exp(beta) - 1]  (V != 0)
    # c(xi) = c0 * xi  (V = 0)
    if abs(beta) > 1.0e-6:
        exp_beta = np.exp(beta)
        c_profile = c0 * (np.exp(beta * xi) - 1.0) / (exp_beta - 1.0)
        # Flux (from Planck equation, constant in SS)
        J = -D * c0 * beta / L * np.exp(beta * xi) / (exp_beta - 1.0)
        J_mean = float(np.mean(J))
    else:
        c_profile = c0 * xi
        J_mean = -D * c0 / L

    V_profile = V_V * xi

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x * 1e9, y=c_profile * 1e3,
        mode="lines", name="c(x)",
        line=dict(color="#2ca02c", width=2.5),
    ))
    fig.update_layout(
        title=f"Nernst-Planck Concentration Profile (V={V_V:.2f} V, D={D:.2e})",
        xaxis_title="x (nm)",
        yaxis_title="Concentration (mM)",
        template="plotly_white",
        font=dict(size=14),
    )

    return {
        "params": {"D": D, "L_nm": L_nm, "V_V": V_V, "c0_mM": c0_mM, "z": z, "T_K": T_K},
        "data":   {
            "x_nm": x * 1e9,
            "c_mM": c_profile * 1e3,
            "V_profile_V": V_profile,
            "J_mol_m2s": J_mean,
        },
        "figure": fig,
    }


def effective_medium_theory(
    D_phase1: float,
    D_phase2: float,
    phi1: float,
    topology: str = "series",
) -> Dict[str, Any]:
    """
    Effective medium theory for composite electrolyte diffusivity.

    topology options:
      "series"          : 1/D_eff = phi1/D1 + phi2/D2
      "parallel"        : D_eff = phi1*D1 + phi2*D2
      "bruggeman"       : D_eff using Bruggeman mixing rule
      "maxwell_garnett" : Maxwell-Garnett approximation (inclusions in matrix)
    """
    phi2 = 1.0 - phi1
    topo = topology.lower().replace("-", "_")

    results: Dict[str, float] = {}

    if topo == "series":
        D_eff = 1.0 / (phi1 / D_phase1 + phi2 / D_phase2)
        results["series"] = D_eff
    elif topo == "parallel":
        D_eff = phi1 * D_phase1 + phi2 * D_phase2
        results["parallel"] = D_eff
    elif topo == "bruggeman":
        # Solve quadratic: sum_i phi_i*(D_i - D_eff)/(D_i + 2*D_eff) = 0
        def _bruggeman_residual(D_e):
            """Bruggeman implicit equation; root gives the effective diffusivity."""
            return (
                phi1 * (D_phase1 - D_e) / (D_phase1 + 2.0 * D_e)
                + phi2 * (D_phase2 - D_e) / (D_phase2 + 2.0 * D_e)
            )
        D_guess = phi1 * D_phase1 + phi2 * D_phase2
        sol = optimize.brentq(
            _bruggeman_residual,
            min(D_phase1, D_phase2) * 1e-3,
            max(D_phase1, D_phase2) * 1e3,
        )
        D_eff = sol
        results["bruggeman"] = D_eff
    elif topo in ("maxwell_garnett", "mg"):
        # Phase1 = inclusions, phase2 = matrix
        D_m = D_phase2
        D_i = D_phase1
        D_eff = D_m * (D_i + 2.0 * D_m + 2.0 * phi1 * (D_i - D_m)) / \
                      (D_i + 2.0 * D_m - phi1 * (D_i - D_m))
        results["maxwell_garnett"] = D_eff
    else:
        raise ValueError(f"Unknown topology: {topology!r}")

    # Sweep phi1 for plot
    phi1_arr = np.linspace(0.0, 1.0, 200)
    phi2_arr = 1.0 - phi1_arr
    D_series   = 1.0 / (phi1_arr / D_phase1 + phi2_arr / D_phase2)
    D_parallel = phi1_arr * D_phase1 + phi2_arr * D_phase2

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=phi1_arr, y=np.log10(D_series),   mode="lines",
                             name="Series",   line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=phi1_arr, y=np.log10(D_parallel), mode="lines",
                             name="Parallel", line=dict(color="#d62728")))
    fig.add_trace(go.Scatter(x=[phi1], y=[np.log10(D_eff)],
                             mode="markers", name=f"{topology} @ φ₁={phi1:.2f}",
                             marker=dict(color="black", size=12, symbol="star")))
    fig.update_layout(
        title="Effective Medium Theory — Composite Diffusivity",
        xaxis_title="Volume fraction φ₁ (phase 1)",
        yaxis_title="log₁₀ D_eff (m²/s)",
        template="plotly_white",
        font=dict(size=14),
    )

    return {
        "params": {"D_phase1": D_phase1, "D_phase2": D_phase2, "phi1": phi1, "topology": topology},
        "data":   {"D_eff": D_eff, "phi1_sweep": phi1_arr, "D_series": D_series, "D_parallel": D_parallel},
        "figure": fig,
    }


def fick_1d(
    D: float,
    L_nm: float = 20.0,
    t_max_s: Optional[float] = None,
    nx: int = 200,
    max_steps: int = 50_000,
) -> Dict[str, Any]:
    """
    1D Fick diffusion via explicit Euler on a finite slab.

    Initial condition: c(x,0) = 1 for x < L/2, c = 0 for x > L/2 (step profile).
    Boundary conditions: c(0,t) = 1, c(L,t) = 0.

    t_max is capped at min(10 * L²/D, 1e4) seconds; n_steps capped at max_steps.
    """
    L = L_nm * 1.0e-9
    dx = L / (nx - 1)
    # CFL-stable time step (r = D*dt/dx² ≤ 0.4)
    dt_stable = 0.4 * dx**2 / D

    if t_max_s is None:
        t_max_s = min(10.0 * L**2 / D, 1.0e4)

    n_steps = min(int(t_max_s / dt_stable) + 1, max_steps)
    dt = t_max_s / n_steps

    # Clamp dt to CFL limit; adjust t_max to match actual steps
    if D * dt / dx**2 > 0.5:
        dt = dt_stable
        n_steps = min(int(t_max_s / dt) + 1, max_steps)
        t_max_s = n_steps * dt

    r = D * dt / dx**2    # recompute after dt is finalised (r ≤ 0.5 guaranteed)

    x = np.linspace(0.0, L, nx)
    c = np.where(x <= L / 2.0, 1.0, 0.0).astype(float)
    c[0] = 1.0
    c[-1] = 0.0

    snapshots = {}
    save_every = max(1, n_steps // 10)
    t = 0.0

    for step in range(n_steps):
        c[1:-1] += r * (c[2:] - 2.0 * c[1:-1] + c[:-2])
        c[0] = 1.0
        c[-1] = 0.0
        t += dt
        if step % save_every == 0:
            snapshots[f"t={t*1e9:.2f}ns"] = c.copy()

    snapshots["final"] = c.copy()

    fig = go.Figure()
    cmap = [f"rgb({int(255*i/(len(snapshots)-1))},0,{int(255*(1-i/(len(snapshots)-1)))})"
            for i in range(len(snapshots))]
    for idx, (label, c_snap) in enumerate(snapshots.items()):
        fig.add_trace(go.Scatter(
            x=x * 1e9, y=c_snap,
            mode="lines", name=label,
            line=dict(color=cmap[idx % len(cmap)], width=1.8),
        ))
    fig.update_layout(
        title=f"1D Fick Diffusion (D={D:.2e} m²/s, L={L_nm} nm)",
        xaxis_title="x (nm)", yaxis_title="Normalised concentration",
        template="plotly_white", font=dict(size=14),
    )

    return {
        "params": {"D": D, "L_nm": L_nm, "t_max_s": t_max_s, "nx": nx, "n_steps": n_steps},
        "data":   {"x_nm": x * 1e9, "c_final": c, "t_s": t, "snapshots": snapshots},
        "figure": fig,
    }


# ===========================================================================
# INTERFACE MODELS
# ===========================================================================


def sei_parabolic_growth(
    k_SEI_m2s: float,
    t_max_h: float = 100.0,
) -> Dict[str, Any]:
    """
    Parabolic SEI growth: delta(t) = sqrt(2 * k_SEI * t).

    Parameters
    ----------
    k_SEI_m2s : parabolic rate constant [m²/s]
    t_max_h   : maximum time [hours]
    """
    t_h = np.linspace(0.0, t_max_h, 1000)
    t_s = t_h * 3600.0
    delta_nm = np.sqrt(2.0 * k_SEI_m2s * t_s) * 1.0e9   # m → nm

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_h, y=delta_nm,
        mode="lines", name="SEI thickness",
        line=dict(color="#9467bd", width=2.5),
    ))
    fig.update_layout(
        title=f"Parabolic SEI Growth (k_SEI = {k_SEI_m2s:.2e} m²/s)",
        xaxis_title="Time (h)", yaxis_title="SEI thickness δ (nm)",
        template="plotly_white", font=dict(size=14),
    )

    return {
        "params": {"k_SEI_m2s": k_SEI_m2s, "t_max_h": t_max_h},
        "data":   {"t_h": t_h, "delta_nm": delta_nm},
        "figure": fig,
    }


def sei_reactive_diffusion(
    D_SEI: float,
    k_rxn: float,
    L0_nm: float,
    t_max_h: float = 100.0,
) -> Dict[str, Any]:
    """
    Coupled diffusion + reaction SEI growth model.

    dL/dt = D_SEI / (k_rxn * L)
    => L(t)^2 - L0^2 = 2 * D_SEI/k_rxn * t

    Parameters
    ----------
    D_SEI   : diffusivity through SEI [m²/s]
    k_rxn   : dimensionless reaction resistance parameter
    L0_nm   : initial SEI thickness [nm]
    t_max_h : max time [hours]
    """
    t_h  = np.linspace(0.0, t_max_h, 1000)
    t_s  = t_h * 3600.0
    L0_m = L0_nm * 1.0e-9   # m
    # dL/dt = D_SEI / (k_rxn * L)  =>  L² = L0² + 2*D_SEI/k_rxn * t
    # k_rxn has units of [1/m] so that D_SEI/k_rxn has units [m³/s]·[m]⁻¹ = [m²/s]
    # For physically meaningful defaults: k_rxn ~ 1e9 /m gives D/k ~ 1e-20 m³/s
    k_rxn_safe = max(abs(k_rxn), 1.0e-30)
    L_m  = np.sqrt(np.maximum(L0_m**2 + 2.0 * D_SEI / k_rxn_safe * t_s, 0.0))
    L_nm_arr = L_m * 1.0e9
    L_nm = L_nm_arr  # reassign for rest of function

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_h, y=L_nm_arr,
        mode="lines", name="L(t) reactive-diffusion",
        line=dict(color="#8c564b", width=2.5),
    ))
    fig.update_layout(
        title=f"SEI Reactive-Diffusion Growth (D_SEI={D_SEI:.2e}, k_rxn={k_rxn:.2e})",
        xaxis_title="Time (h)", yaxis_title="SEI thickness (nm)",
        template="plotly_white", font=dict(size=14),
    )

    return {
        "params": {"D_SEI": D_SEI, "k_rxn": k_rxn, "L0_nm": L0_nm},
        "data":   {"t_h": t_h, "L_nm": L_nm_arr},
        "figure": fig,
    }


def phase_field_allen_cahn(
    phi0: np.ndarray,
    W: float,
    kappa: float,
    M: float,
    dx: float = 0.5,
    nx: int = 100,
    dt: float = 0.01,
    n_steps: int = 500,
) -> Dict[str, Any]:
    """
    Allen-Cahn phase-field equation:

    dphi/dt = M * (W * dg/dphi - kappa * lap(phi))

    g(phi) = phi^2 * (1 - phi)^2   (double-well)
    dg/dphi = 2*phi*(1-phi)^2 - 2*phi^2*(1-phi) = 2*phi*(1-phi)*(1-2*phi)

    Parameters
    ----------
    phi0    : initial phase-field profile (1D array of length nx, or None for step)
    W       : barrier height (double-well amplitude)
    kappa   : gradient energy coefficient
    M       : phase-field mobility
    dx      : spatial step size (dimensionless units)
    nx      : grid points (used only if phi0 is None)
    dt      : time step
    n_steps : number of evolution steps
    """
    if phi0 is None:
        phi = np.zeros(nx)
        phi[: nx // 2] = 1.0
    else:
        phi = np.array(phi0, dtype=float)
        nx = len(phi)

    x = np.arange(nx) * dx
    snapshots = {"t=0": phi.copy()}
    save_every = max(1, n_steps // 5)

    for step in range(n_steps):
        # Laplacian (periodic BC)
        lap = (np.roll(phi, -1) - 2.0 * phi + np.roll(phi, 1)) / dx**2
        dg  = 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)
        phi += dt * M * (W * dg - kappa * lap)
        phi = np.clip(phi, 0.0, 1.0)
        if (step + 1) % save_every == 0:
            snapshots[f"t={dt*(step+1):.2f}"] = phi.copy()

    snapshots["final"] = phi.copy()

    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    for idx, (label, ph) in enumerate(snapshots.items()):
        fig.add_trace(go.Scatter(
            x=x, y=ph, mode="lines", name=label,
            line=dict(color=colors[idx % len(colors)], width=2.0),
        ))
    fig.update_layout(
        title=f"Allen-Cahn Phase Field (W={W}, κ={kappa}, M={M})",
        xaxis_title="x (a.u.)", yaxis_title="φ",
        template="plotly_white", font=dict(size=14),
    )

    return {
        "params": {"W": W, "kappa": kappa, "M": M, "dx": dx, "dt": dt, "n_steps": n_steps},
        "data":   {"x": x, "phi_final": phi, "snapshots": snapshots},
        "figure": fig,
    }


def power_law_growth(
    A_um: float,
    n: float,
    t_max_h: float = 1000.0,
) -> Dict[str, Any]:
    """
    Power-law interphase growth: L(t) = A * t^n  [µm, t in hours].

    Reference: NMC622|LiPSCl: A=0.265 µm, n=0.155 (Ncube, Barai, Selvaraj et al., 2026).
    """
    t_h = np.linspace(0.0, t_max_h, 1000)
    L_um = A_um * np.power(np.maximum(t_h, 1.0e-10), n)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_h, y=L_um,
        mode="lines", name=f"L = {A_um:.3f}·t^{n:.3f}",
        line=dict(color="#17becf", width=2.5),
    ))
    fig.update_layout(
        title=f"Power-Law Interphase Growth (A={A_um:.3f} µm, n={n:.3f})",
        xaxis_title="Time (h)", yaxis_title="Interphase thickness (µm)",
        template="plotly_white", font=dict(size=14),
    )

    return {
        "params": {"A_um": A_um, "n": n, "t_max_h": t_max_h},
        "data":   {"t_h": t_h, "L_um": L_um},
        "figure": fig,
    }


def kjma_crystallization(
    k: float,
    n_avrami: float,
    t_max: float = 100.0,
) -> Dict[str, Any]:
    """
    Johnson-Mehl-Avrami-Kolmogorov (KJMA) crystallization kinetics.

    X(t) = 1 - exp(-k * t^n)

    Parameters
    ----------
    k         : rate constant [time units^(-n)]
    n_avrami  : Avrami exponent (nucleation/growth mechanism)
    t_max     : maximum time
    """
    t = np.linspace(0.0, t_max, 1000)
    X = 1.0 - np.exp(-k * np.power(t, n_avrami))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=X * 100.0,
        mode="lines", name=f"k={k:.3e}, n={n_avrami:.2f}",
        line=dict(color="#bcbd22", width=2.5),
    ))
    fig.update_layout(
        title=f"KJMA Crystallization (k={k:.3e}, n={n_avrami:.2f})",
        xaxis_title="Time (a.u.)", yaxis_title="Crystalline fraction X (%)",
        template="plotly_white", font=dict(size=14),
    )

    return {
        "params": {"k": k, "n_avrami": n_avrami, "t_max": t_max},
        "data":   {"t": t, "X_fraction": X},
        "figure": fig,
    }


# ===========================================================================
# ELECTROCHEMICAL MODELS
# ===========================================================================


def butler_volmer(
    j0_mA_cm2: float,
    alpha_a: float,
    alpha_c: float,
    eta_range_V: Optional[np.ndarray] = None,
    T_K: float = 300.0,
) -> Dict[str, Any]:
    """
    Butler-Volmer kinetics.

    j(η) = j0 * [exp(α_a * F * η / RT) - exp(-α_c * F * η / RT)]

    Parameters
    ----------
    j0_mA_cm2 : exchange current density [mA/cm²]
    alpha_a   : anodic transfer coefficient
    alpha_c   : cathodic transfer coefficient
    eta_range_V : overpotential range [V]; default ±0.5 V
    T_K       : temperature [K]
    """
    eta = _default_eta(eta_range_V)
    RT  = R_JMK * T_K
    j   = j0_mA_cm2 * (
        np.exp(alpha_a * FARADAY * eta / RT) -
        np.exp(-alpha_c * FARADAY * eta / RT)
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eta, y=j,
        mode="lines", name="Butler-Volmer",
        line=dict(color="#1f77b4", width=2.5),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)
    fig.update_layout(
        title=f"Butler-Volmer (j₀={j0_mA_cm2:.2f} mA/cm², T={T_K:.0f} K)",
        xaxis_title="Overpotential η (V)",
        yaxis_title="Current density j (mA/cm²)",
        template="plotly_white", font=dict(size=14),
    )

    return {
        "params": {"j0_mA_cm2": j0_mA_cm2, "alpha_a": alpha_a, "alpha_c": alpha_c, "T_K": T_K},
        "data":   {"eta_V": eta, "j_mA_cm2": j},
        "figure": fig,
    }


def tafel_kinetics(
    j0_mA_cm2: float,
    alpha: float,
    eta_range_V: Optional[np.ndarray] = None,
    T_K: float = 300.0,
) -> Dict[str, Any]:
    """
    Tafel high-overpotential approximation: j = j0 * exp(alpha * F * eta / RT).

    Plotted as log|j| vs eta (Tafel plot).
    """
    eta  = _default_eta(eta_range_V)
    RT   = R_JMK * T_K
    j    = j0_mA_cm2 * np.exp(alpha * FARADAY * eta / RT)
    logj = np.log10(np.abs(j))
    tafel_slope_mV_dec = RT / (alpha * FARADAY) * np.log(10) * 1000.0  # mV/decade

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eta, y=logj,
        mode="lines", name="Tafel",
        line=dict(color="#d62728", width=2.5),
    ))
    fig.update_layout(
        title=f"Tafel Plot (b = {tafel_slope_mV_dec:.1f} mV/dec, T={T_K:.0f} K)",
        xaxis_title="Overpotential η (V)",
        yaxis_title="log₁₀ |j| (mA/cm²)",
        template="plotly_white", font=dict(size=14),
    )

    return {
        "params": {"j0_mA_cm2": j0_mA_cm2, "alpha": alpha, "T_K": T_K,
                   "tafel_slope_mV_dec": tafel_slope_mV_dec},
        "data":   {"eta_V": eta, "j_mA_cm2": j, "log10_j": logj},
        "figure": fig,
    }


def dfn_simple(
    D_anode: float,
    D_cathode: float,
    D_sep: float,
    L_anode_um: float,
    L_cathode_um: float,
    L_sep_um: float,
    C_rate: float = 1.0,
) -> Dict[str, Any]:
    """
    Simplified 1-region Doyle-Fuller-Newman model.

    Assumes linear concentration profile in each region at steady state,
    estimates voltage drop and capacity based on 1D Li transport through
    anode | separator | cathode stack.

    Returns
    -------
    dict with concentration and voltage profiles, capacity estimate,
    and limiting-region identification.
    """
    L_a = L_anode_um * 1.0e-6
    L_s = L_sep_um   * 1.0e-6
    L_c = L_cathode_um * 1.0e-6
    L_total = L_a + L_s + L_c

    # Reference current density [A/m²]: C_rate × nominal 1 A/m² baseline
    j_ref = C_rate * 10.0    # 10 A/m² at 1C is a common thin-film reference

    # c_max [mol/m³]: representative Li-ion electrolyte concentration ~1 M = 1000 mol/m³
    c_max = 1000.0  # mol/m³

    # Concentration drop Δc across each region (steady-state diffusion flux j/F = D*dc/dx)
    # dc = j/(F*D) * L  (mol/m³)
    dc_anode   = j_ref * L_a / (FARADAY * D_anode)
    dc_sep     = j_ref * L_s / (FARADAY * D_sep)
    dc_cathode = j_ref * L_c / (FARADAY * D_cathode)

    # Clip to fraction of c_max to keep profiles physically bounded
    dc_anode   = min(dc_anode,   0.9 * c_max)
    dc_sep     = min(dc_sep,     0.9 * c_max)
    dc_cathode = min(dc_cathode, 0.9 * c_max)

    # Voltage penalty from concentration gradient (Nernst, linearised)
    RT_F = R_JMK * 300.0 / FARADAY  # ~0.02585 V
    dV_anode   = RT_F * dc_anode   / c_max
    dV_sep     = RT_F * dc_sep     / c_max
    dV_cathode = RT_F * dc_cathode / c_max
    V_total_loss = dV_anode + dV_sep + dV_cathode  # V

    # Capacity: limited by diffusion time vs discharge time
    D_eff_series = L_total / (L_a / D_anode + L_s / D_sep + L_c / D_cathode)
    tau_diff = L_total**2 / (2.0 * D_eff_series)   # s
    t_discharge = 3600.0 / max(C_rate, 1.0e-6)     # s at given C-rate
    Q_norm   = min(1.0, t_discharge / max(tau_diff, 1.0e-20))

    # Spatial normalised concentration profile (piecewise linear, 0-1 scale)
    x_a = np.linspace(0.0, L_a, 50)
    x_s = np.linspace(L_a, L_a + L_s, 30)
    x_c = np.linspace(L_a + L_s, L_total, 50)
    x   = np.concatenate([x_a, x_s, x_c])
    c_a0 = 1.0
    c_a  = c_a0 - (dc_anode / c_max) * (x_a / max(L_a, 1.0e-20))
    c_s  = (c_a0 - dc_anode / c_max) - (dc_sep / c_max) * (x_s - L_a) / max(L_s, 1.0e-20)
    c_c  = (c_a0 - dc_anode / c_max - dc_sep / c_max) - \
           (dc_cathode / c_max) * (x_c - L_a - L_s) / max(L_c, 1.0e-20)
    c    = np.clip(np.concatenate([c_a, c_s, c_c]), 0.0, 1.0)

    limiting = max(
        [("anode", dV_anode), ("separator", dV_sep), ("cathode", dV_cathode)],
        key=lambda kv: kv[1],
    )[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x * 1e6, y=c,
        mode="lines", name="Li⁺ concentration",
        line=dict(color="#2ca02c", width=2.5),
    ))
    for boundary in [L_a, L_a + L_s]:
        fig.add_vline(x=boundary * 1e6, line_dash="dash",
                      line_color="gray", opacity=0.6)
    fig.update_layout(
        title=f"Simplified DFN (C-rate={C_rate:.1f}C, V_loss={V_total_loss*1000:.1f} mV)",
        xaxis_title="x (µm)", yaxis_title="Normalised Li⁺ concentration",
        template="plotly_white", font=dict(size=14),
        annotations=[
            dict(x=(L_a / 2) * 1e6, y=1.02, text="Anode", showarrow=False,
                 xref="x", yref="paper"),
            dict(x=(L_a + L_s / 2) * 1e6, y=1.02, text="Sep", showarrow=False,
                 xref="x", yref="paper"),
            dict(x=(L_a + L_s + L_c / 2) * 1e6, y=1.02, text="Cathode",
                 showarrow=False, xref="x", yref="paper"),
        ],
    )

    return {
        "params": {
            "D_anode": D_anode, "D_cathode": D_cathode, "D_sep": D_sep,
            "L_anode_um": L_anode_um, "L_cathode_um": L_cathode_um,
            "L_sep_um": L_sep_um, "C_rate": C_rate,
        },
        "data": {
            "x_um": x * 1e6, "c_profile": c,
            "V_loss_mV": V_total_loss * 1000.0,
            "Q_normalised": Q_norm,
            "limiting_region": limiting,
            "tau_diff_s": tau_diff,
        },
        "figure": fig,
    }


# ===========================================================================
# MECHANICAL MODELS
# ===========================================================================


def vegard_stress(
    E_GPa: float,
    nu: float,
    Omega_A3: float,
    c_range: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Vegard's law stress due to intercalation-induced lattice strain.

    sigma = E * Omega_m3 * c_dim / (1 - nu)

    where c_dim is the dimensional ion concentration in mol/m³.
    For normalised c in [0,1], c_dim = c * c_max, and c_max is estimated
    as 1 mol / (Omega_A3 * 1e-30 * NA) = the close-packing site density.

    Typical values: Omega_A3 ~ 20-25 Å³ for Li in layered oxides gives
    eps_V_max ~ 2-5% volumetric strain and sigma_max ~ 0.3-1 GPa for E~100-200 GPa.

    Parameters
    ----------
    E_GPa    : Young's modulus [GPa]
    nu       : Poisson's ratio
    Omega_A3 : partial molar volume of mobile ion [Å³/ion]
    c_range  : normalised concentration array [0-1]; default linspace(0,1,200)
    """
    c = c_range if c_range is not None else np.linspace(0.0, 1.0, 200)
    E = E_GPa * 1.0e9                  # Pa
    # Dimensionless Vegard strain coefficient:
    #   eps_V(c) = (Omega_A3 / V_ref_A3) * c
    # where V_ref_A3 is a reference unit-cell volume.
    # Use V_ref = 10^4 Å³ as the dimensionless scale factor so that
    # Omega_A3 ~ 20 Å³ → eps_coeff ~ 0.002  →  sigma_max ~ 0.5 GPa at E=200 GPa.
    # This reproduces literature values (NMC622: ~0.3-0.8 GPa at full lithiation).
    eps_coeff = Omega_A3 / 1.0e4          # dimensionless (unitless strain per unit SOC)
    eps_V     = eps_coeff * c             # volumetric strain [dimensionless]
    sigma_Pa  = E * eps_V / (1.0 - nu)   # Pa — Vegard biaxial stress
    sigma_GPa = sigma_Pa * 1.0e-9

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=c, y=sigma_GPa,
        mode="lines", name="Vegard stress",
        line=dict(color="#7f7f7f", width=2.5),
    ))
    fig.update_layout(
        title=f"Vegard Stress (E={E_GPa:.0f} GPa, ν={nu:.2f}, Ω={Omega_A3:.1f} Å³)",
        xaxis_title="Normalised Li concentration c",
        yaxis_title="σ (GPa)",
        template="plotly_white", font=dict(size=14),
    )

    return {
        "params": {"E_GPa": E_GPa, "nu": nu, "Omega_A3": Omega_A3, "eps_coeff": eps_coeff},
        "data":   {"c": c, "sigma_GPa": sigma_GPa, "eps_V": eps_V},
        "figure": fig,
    }


def fracture_criterion(
    E_GPa: float,
    KIC_MPa_sqrt_m: float,
    sigma_max_GPa: float,
) -> Dict[str, Any]:
    """
    Linear elastic fracture mechanics: critical flaw size.

    a_c = (1/pi) * (K_IC / sigma)^2

    Returns critical flaw size a_c [µm] for a range of stresses.
    """
    sigma_arr = np.linspace(0.01, max(sigma_max_GPa * 1.5, 0.1), 500) * 1.0e9  # Pa
    KIC_Pa_sqrt_m = KIC_MPa_sqrt_m * 1.0e6
    a_c_m = (1.0 / np.pi) * (KIC_Pa_sqrt_m / sigma_arr)**2
    a_c_um = a_c_m * 1.0e6

    # Highlight the critical flaw at sigma_max
    a_c_max = (1.0 / np.pi) * (KIC_Pa_sqrt_m / (sigma_max_GPa * 1.0e9))**2 * 1.0e6

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sigma_arr * 1.0e-9, y=a_c_um,
        mode="lines", name="Critical flaw size",
        line=dict(color="#e377c2", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=[sigma_max_GPa], y=[a_c_max],
        mode="markers", name=f"σ_max → a_c={a_c_max:.2f} µm",
        marker=dict(color="red", size=12, symbol="x"),
    ))
    fig.update_layout(
        title=f"Fracture Criterion (K_IC={KIC_MPa_sqrt_m:.1f} MPa√m)",
        xaxis_title="Applied stress σ (GPa)",
        yaxis_title="Critical flaw size a_c (µm)",
        template="plotly_white", font=dict(size=14),
        yaxis_type="log",
    )

    return {
        "params": {"E_GPa": E_GPa, "KIC_MPa_sqrt_m": KIC_MPa_sqrt_m,
                   "sigma_max_GPa": sigma_max_GPa},
        "data":   {
            "sigma_GPa": sigma_arr * 1.0e-9,
            "a_c_um": a_c_um,
            "a_c_at_sigma_max_um": a_c_max,
        },
        "figure": fig,
    }


def swelling_strain(
    Omega_A3: float,
    rho_gcm3: float,
    MW_mobile_g_mol: float,
    SOC_range: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Volumetric swelling strain from mobile-ion intercalation.

    eps_V(SOC) = Omega_m3 * c_max * SOC
    c_max = rho * NA / MW   [1/m³]

    where Omega_m3 is partial molar volume of mobile ion [m³/ion].
    """
    SOC = SOC_range if SOC_range is not None else np.linspace(0.0, 1.0, 200)
    Omega_m3 = Omega_A3 * 1.0e-30               # Å³ → m³
    rho_SI   = rho_gcm3 * 1.0e3                 # g/cm³ → kg/m³
    MW_kg    = MW_mobile_g_mol * 1.0e-3          # g/mol → kg/mol
    c_max    = (rho_SI / MW_kg) * NA             # ions/m³
    eps_V    = Omega_m3 * c_max * SOC            # dimensionless volumetric strain
    eps_lin  = eps_V / 3.0                       # linear strain

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=SOC, y=eps_V * 100.0,
        mode="lines", name="Volumetric",
        line=dict(color="#ff7f0e", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=SOC, y=eps_lin * 100.0,
        mode="lines", name="Linear (1/3 vol)",
        line=dict(color="#1f77b4", width=2.0, dash="dash"),
    ))
    fig.update_layout(
        title=f"Swelling Strain (Ω={Omega_A3:.1f} Å³, ρ={rho_gcm3:.2f} g/cm³)",
        xaxis_title="State of charge (SOC)",
        yaxis_title="Strain (%)",
        template="plotly_white", font=dict(size=14),
    )

    return {
        "params": {"Omega_A3": Omega_A3, "rho_gcm3": rho_gcm3,
                   "MW_mobile_g_mol": MW_mobile_g_mol},
        "data":   {"SOC": SOC, "eps_vol": eps_V, "eps_lin": eps_lin, "c_max_m3": c_max},
        "figure": fig,
    }


# ===========================================================================
# STANDALONE PLOT WRAPPERS
# ===========================================================================

def plot_arrhenius(result: dict) -> go.Figure:
    """Return the Plotly figure from an arrhenius() result dict."""
    return result["figure"]

def plot_vtf_conductivity(result: dict) -> go.Figure:
    """Return the Plotly figure from a vtf_conductivity() result dict."""
    return result["figure"]

def plot_nernst_planck_1d(result: dict) -> go.Figure:
    """Return the Plotly figure from a nernst_planck_1d() result dict."""
    return result["figure"]

def plot_effective_medium_theory(result: dict) -> go.Figure:
    """Return the Plotly figure from an effective_medium_theory() result dict."""
    return result["figure"]

def plot_fick_1d(result: dict) -> go.Figure:
    """Return the Plotly figure from a fick_1d() result dict."""
    return result["figure"]

def plot_sei_parabolic_growth(result: dict) -> go.Figure:
    """Return the Plotly figure from a sei_parabolic_growth() result dict."""
    return result["figure"]

def plot_sei_reactive_diffusion(result: dict) -> go.Figure:
    """Return the Plotly figure from a sei_reactive_diffusion() result dict."""
    return result["figure"]

def plot_phase_field_allen_cahn(result: dict) -> go.Figure:
    """Return the Plotly figure from a phase_field_allen_cahn() result dict."""
    return result["figure"]

def plot_power_law_growth(result: dict) -> go.Figure:
    """Return the Plotly figure from a power_law_growth() result dict."""
    return result["figure"]

def plot_kjma_crystallization(result: dict) -> go.Figure:
    """Return the Plotly figure from a kjma_crystallization() result dict."""
    return result["figure"]

def plot_butler_volmer(result: dict) -> go.Figure:
    """Return the Plotly figure from a butler_volmer() result dict."""
    return result["figure"]

def plot_tafel_kinetics(result: dict) -> go.Figure:
    """Return the Plotly figure from a tafel_kinetics() result dict."""
    return result["figure"]

def plot_dfn_simple(result: dict) -> go.Figure:
    """Return the Plotly figure from a dfn_simple() result dict."""
    return result["figure"]

def plot_vegard_stress(result: dict) -> go.Figure:
    """Return the Plotly figure from a vegard_stress() result dict."""
    return result["figure"]

def plot_fracture_criterion(result: dict) -> go.Figure:
    """Return the Plotly figure from a fracture_criterion() result dict."""
    return result["figure"]

def plot_swelling_strain(result: dict) -> go.Figure:
    """Return the Plotly figure from a swelling_strain() result dict."""
    return result["figure"]


# ===========================================================================
# ORCHESTRATOR
# ===========================================================================

def run_all_models(
    project_params: dict,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Run all applicable continuum models for a material project and save figures.

    Parameters
    ----------
    project_params : dict with keys drawn from MaterialProject attributes, e.g.:
        {
          "category": "inorganic_sse",   # or polymer / liquid_electrolyte
          "D_best": 1.2e-10,             # m²/s (MLMD best)
          "Ea_best": 0.28,               # eV
          "T_ref": 300,
          "E_GPa": 120.0,
          "nu": 0.25,
          "Omega_A3": 23.0,
          "MW_mobile": 22.99,
          "rho_gcm3": 3.8,
          "k_SEI": 1e-18,               # optional
          "j0_mA_cm2": 0.5,
          "alpha": 0.5,
          "A_powerlaw": 0.265,
          "n_powerlaw": 0.155,
          "kjma_k": 0.001,
          "kjma_n": 2.5,
          "D_cat": 1e-14,               # optional
          "D_SSE": 1e-11,               # optional
        }
    output_dir : directory for HTML + PNG figure outputs

    Returns
    -------
    dict mapping model_name → result_dict
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p       = project_params
    cat     = p.get("category", "inorganic")
    D       = p.get("D_best") or p.get("D_mlmd") or 1.0e-11
    Ea      = p.get("Ea_best") or p.get("Ea_mlmd") or 0.3
    T_ref   = float(p.get("T_ref", 300))
    E_GPa   = float(p.get("E_GPa", 100.0))
    nu_     = float(p.get("nu", 0.25))
    Omega   = float(p.get("Omega_A3", 20.0))
    MW      = float(p.get("MW_mobile", 6.941))
    rho     = float(p.get("rho_gcm3", 3.5))
    j0      = float(p.get("j0_mA_cm2", 0.5))
    alpha   = float(p.get("alpha", 0.5))
    A_pl    = float(p.get("A_powerlaw", 0.265))
    n_pl    = float(p.get("n_powerlaw", 0.155))
    kj_k    = float(p.get("kjma_k", 0.001))
    kj_n    = float(p.get("kjma_n", 2.5))

    results: Dict[str, Any] = {}

    # --- Ion transport (universal) ---
    results["arrhenius"] = arrhenius(D, Ea, T_ref)
    _save_figure(results["arrhenius"]["figure"], output_dir, "arrhenius")

    results["fick_1d"] = fick_1d(D, L_nm=20.0)
    _save_figure(results["fick_1d"]["figure"], output_dir, "fick_1d")

    results["power_law_growth"] = power_law_growth(A_pl, n_pl)
    _save_figure(results["power_law_growth"]["figure"], output_dir, "power_law_growth")

    results["kjma_crystallization"] = kjma_crystallization(kj_k, kj_n)
    _save_figure(results["kjma_crystallization"]["figure"], output_dir, "kjma_crystallization")

    # --- Electrochemical (universal) ---
    results["butler_volmer"] = butler_volmer(j0, alpha, 1.0 - alpha)
    _save_figure(results["butler_volmer"]["figure"], output_dir, "butler_volmer")

    results["tafel_kinetics"] = tafel_kinetics(j0, alpha)
    _save_figure(results["tafel_kinetics"]["figure"], output_dir, "tafel_kinetics")

    # --- Mechanical (universal) ---
    results["vegard_stress"] = vegard_stress(E_GPa, nu_, Omega)
    _save_figure(results["vegard_stress"]["figure"], output_dir, "vegard_stress")

    results["swelling_strain"] = swelling_strain(Omega, rho, MW)
    _save_figure(results["swelling_strain"]["figure"], output_dir, "swelling_strain")

    # Critical flaw size only if stress is meaningful
    sigma_max_est = float(results["vegard_stress"]["data"]["sigma_GPa"][-1])
    if sigma_max_est > 0.0 and E_GPa > 10.0:
        kic = float(p.get("KIC_MPa_sqrt_m", 1.0))
        results["fracture_criterion"] = fracture_criterion(E_GPa, kic, sigma_max_est)
        _save_figure(results["fracture_criterion"]["figure"], output_dir, "fracture_criterion")

    # --- Polymer VTF ---
    if cat == "polymer" or "vtf_A" in p:
        A_vtf = float(p.get("vtf_A", 1.0e-2))
        B_vtf = float(p.get("vtf_B", 800.0))
        T0_vtf = float(p.get("vtf_T0", 180.0))
        results["vtf_conductivity"] = vtf_conductivity(A_vtf, B_vtf, T0_vtf)
        _save_figure(results["vtf_conductivity"]["figure"], output_dir, "vtf_conductivity")

    # --- Effective medium (composite SSE or polymer) ---
    D_cat = p.get("D_cat")
    D_sse = p.get("D_SSE") or p.get("D_SSE_coating")
    if D_cat is not None and D_sse is not None:
        results["effective_medium"] = effective_medium_theory(
            D_cat, D_sse, 0.4, topology="bruggeman"
        )
        _save_figure(results["effective_medium"]["figure"], output_dir, "effective_medium")

        results["dfn_simple"] = dfn_simple(
            D_cat, D, D_sse,
            L_anode_um=50.0, L_cathode_um=50.0, L_sep_um=25.0,
        )
        _save_figure(results["dfn_simple"]["figure"], output_dir, "dfn_simple")

    # --- SEI models (liquid or inorganic SSE) ---
    k_sei = p.get("k_SEI")
    if k_sei is not None or cat in ("liquid_electrolyte", "inorganic_sse"):
        k_sei = k_sei or 1.0e-20
        results["sei_parabolic_growth"] = sei_parabolic_growth(k_sei)
        _save_figure(results["sei_parabolic_growth"]["figure"], output_dir, "sei_parabolic_growth")

        results["sei_reactive_diffusion"] = sei_reactive_diffusion(D, 1.0e-9, L0_nm=2.0)
        _save_figure(results["sei_reactive_diffusion"]["figure"], output_dir, "sei_reactive_diffusion")

    # --- Phase-field (inorganic or polymer with interface) ---
    if cat in ("inorganic_sse", "inorganic", "polymer") or p.get("run_phase_field", False):
        phi0_arr = np.zeros(100)
        phi0_arr[:50] = 1.0
        W_pf = float(p.get("pf_W", 1.0))
        kappa_pf = float(p.get("pf_kappa", 0.5))
        M_pf = float(p.get("pf_M", 0.1))
        results["phase_field_allen_cahn"] = phase_field_allen_cahn(
            phi0_arr, W_pf, kappa_pf, M_pf
        )
        _save_figure(results["phase_field_allen_cahn"]["figure"], output_dir, "phase_field_allen_cahn")

    # --- Nernst-Planck (liquid or polymer) ---
    if cat in ("liquid_electrolyte", "polymer"):
        c0 = float(p.get("c0_mM", 1000.0))
        V_applied = float(p.get("V_applied_V", 0.1))
        results["nernst_planck_1d"] = nernst_planck_1d(D, 50.0, V_applied, c0)
        _save_figure(results["nernst_planck_1d"]["figure"], output_dir, "nernst_planck_1d")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PyBaMM DFN continuum battery model
# Source: c2c_simulation/packages/04_continuum_model/src/c2c_continuum/battery_model.py
# Reference: Zhang et al., Nature Energy 9, 386–400 (2024)
# ══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass as _dataclass

_ELECTRODE_AREA   = 1e-4   # m²
_CATHODE_MASS_G   = 0.010  # g
_CALIBRATION      = 1.18


@_dataclass
class ElectrolyteParams:
    """Transport and kinetic parameters for one SPE variant.

    Pre-defined classmethod constructors: PTFEP(), PMMA(), PEO().
    Custom variants: ElectrolyteParams.from_md(name, D_Li, sigma, t_plus).
    """
    name:   str
    sigma:  float          # S m⁻¹
    t_plus: float
    D_e:    float          # m² s⁻¹
    j0:     float          # A m⁻²
    c0:     float = 1200.0 # mol m⁻³

    @classmethod
    def PTFEP(cls) -> "ElectrolyteParams":
        """Return pre-set parameters for PTFEP SPE (σ=30 mS/m, t+=0.64)."""
        return cls("PTFEP", sigma=0.030, t_plus=0.64, D_e=8.0e-11, j0=2.5)

    @classmethod
    def PMMA(cls) -> "ElectrolyteParams":
        """Return pre-set parameters for PMMA SPE (σ=25 mS/m, t+=0.53)."""
        return cls("PMMA",  sigma=0.025, t_plus=0.53, D_e=6.0e-11, j0=1.5)

    @classmethod
    def PEO(cls) -> "ElectrolyteParams":
        """Return pre-set parameters for PEO SPE (σ=15 mS/m, t+=0.38)."""
        return cls("PEO",   sigma=0.015, t_plus=0.38, D_e=3.0e-11, j0=0.8)

    @classmethod
    def from_md(
        cls,
        name:   str,
        D_Li:   float,   # m² s⁻¹  (1 cm²/s = 1e-4 m²/s)
        sigma:  float,   # S m⁻¹   (1 S/cm  = 100 S/m)
        t_plus: float,
        j0:     float = 1.5,
        c0:     float = 1200.0,
    ) -> "ElectrolyteParams":
        """Build from MD-derived transport properties."""
        return cls(name, sigma=sigma, t_plus=t_plus, D_e=D_Li, j0=j0, c0=c0)


@_dataclass
class DFNResult:
    """Container for one DFN discharge simulation result."""
    I_mA_cm2:    float
    T_C:         float  = 25.0
    electrolyte: str    = ""
    cap:         float  = 0.0   # mAh g⁻¹
    crate:       float  = 0.0
    energy:      float  = 0.0   # Wh kg⁻¹
    power:       float  = 0.0   # W kg⁻¹
    avg_V:       float  = 0.0
    t_d:         Optional[np.ndarray] = None
    V_d:         Optional[np.ndarray] = None
    success:     bool   = True


# Experimental data — Zhang et al. (2024)
DFN_EXPERIMENTAL = {
    "currents":     np.array([0.5, 1.0, 2.0, 3.0, 3.7]),
    "cap_30C_435V": np.array([148.7, 130.0, 100.0,  85.0,  77.0]),
    "cap_30C_46V":  np.array([186.0, 170.0, 145.0, 125.0, 115.0]),
    "cap_45C_46V":  np.array([202.2, 190.0, 175.0, 155.0, 146.7]),
}


class BatteryDFN:
    """Generic DFN continuum battery model (PyBaMM backend).

    Parameters
    ----------
    model_type         : 'DFN' | 'SPM' | 'SPMe'
    base_params        : PyBaMM parameter set name (default 'Chen2020')
    v_min / v_max      : voltage cut-offs (V)
    electrode_area     : m²
    cathode_mass_g     : g (active cathode)
    calibration_factor : multiplicative correction

    Usage
    -----
    >>> model = BatteryDFN()
    >>> result = model.run(I_mA_cm2=0.5, electrolyte="PTFEP")
    >>> results = model.sweep([0.5, 1.0, 2.0, 3.7])
    >>> model.run_from_md(D_Li_cm2_s=1e-5, sigma_S_cm=0.005, t_plus=0.6, I_mA_cm2=1.0)
    """

    def __init__(
        self,
        model_type:         str   = "DFN",
        base_params:        str   = "Chen2020",
        v_min:              float = 2.8,
        v_max:              float = 4.5,
        electrode_area:     float = _ELECTRODE_AREA,
        cathode_mass_g:     float = _CATHODE_MASS_G,
        calibration_factor: float = _CALIBRATION,
    ):
        """Initialise BatteryDFN with model type, voltage limits, and electrode geometry."""
        self.model_type         = model_type
        self.base_params        = base_params
        self.v_min              = v_min
        self.v_max              = v_max
        self.electrode_area     = electrode_area
        self.cathode_mass_g     = cathode_mass_g
        self.calibration_factor = calibration_factor

    def run(
        self,
        I_mA_cm2:     float,
        T_C:          float = 25.0,
        electrolyte:  Union[str, ElectrolyteParams] = "PTFEP",
        extra_params: Optional[dict] = None,
    ) -> DFNResult:
        """Run one galvanostatic discharge."""
        try:
            import pybamm
        except ImportError:
            raise ImportError("Install PyBaMM:  pip install pybamm")

        ep = self._resolve(electrolyte)
        param = self._build_params(I_mA_cm2, T_C, ep, extra_params or {})
        model = self._build_model()
        I_A   = float(param["Current function [A]"])
        t_max = min((200e-3 * self.cathode_mass_g / I_A) * 3600 * 1.25, 25000.0)

        sim = pybamm.Simulation(
            model, parameter_values=param,
            var_pts={"x_n": 20, "x_s": 20, "x_p": 30, "r_n": 15, "r_p": 15},
            solver=pybamm.CasadiSolver(mode="safe", dt_max=30),
        )
        try:
            sol = sim.solve(t_eval=np.linspace(0, t_max, 400))
            return self._extract(sol, I_mA_cm2, T_C, ep.name)
        except Exception as exc:
            log.warning("BatteryDFN solver failed %.1f mA/cm²: %s", I_mA_cm2, exc)
            return DFNResult(I_mA_cm2=I_mA_cm2, T_C=T_C,
                             electrolyte=ep.name, success=False)

    def sweep(
        self,
        currents:    list[float],
        T_C:         float = 25.0,
        electrolyte: Union[str, ElectrolyteParams] = "PTFEP",
    ) -> list[DFNResult]:
        """Rate-capability sweep; returns only successful results."""
        results = []
        for I in currents:
            res = self.run(I, T_C, electrolyte)
            if res.success:
                results.append(res)
        return results

    def compare_electrolytes(
        self,
        electrolytes: list[Union[str, ElectrolyteParams]],
        currents:     list[float],
        T_C:          float = 25.0,
    ) -> dict[str, list[DFNResult]]:
        """Run rate-capability sweep for each electrolyte and return results keyed by name."""
        return {self._resolve(e).name: self.sweep(currents, T_C, e)
                for e in electrolytes}

    def run_from_md(
        self,
        D_Li_cm2_s: float,
        sigma_S_cm: float,
        t_plus:     float,
        I_mA_cm2:   float,
        T_C:        float = 25.0,
        name:       str   = "MD-derived",
    ) -> DFNResult:
        """Run DFN using MD-derived transport properties.

        Parameters
        ----------
        D_Li_cm2_s : Li diffusivity in cm²/s
        sigma_S_cm : conductivity in S/cm
        """
        ep = ElectrolyteParams.from_md(
            name   = name,
            D_Li   = D_Li_cm2_s * 1e-4,
            sigma  = sigma_S_cm * 100.0,
            t_plus = t_plus,
        )
        return self.run(I_mA_cm2, T_C, ep)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _resolve(self, elec: Union[str, ElectrolyteParams]) -> ElectrolyteParams:
        """Resolve a string name or ElectrolyteParams object to an ElectrolyteParams instance."""
        if isinstance(elec, ElectrolyteParams):
            return elec
        mapping = {"PTFEP": ElectrolyteParams.PTFEP,
                   "PMMA":  ElectrolyteParams.PMMA,
                   "PEO":   ElectrolyteParams.PEO}
        key = elec.upper()
        if key in mapping:
            return mapping[key]()
        raise ValueError(f"Unknown electrolyte {elec!r}. "
                         "Pass 'PTFEP'/'PMMA'/'PEO' or an ElectrolyteParams object.")

    def _build_params(self, I_mA_cm2, T_C, ep, extra):
        """Build a PyBaMM ParameterValues object with electrolyte and electrode settings."""
        import pybamm
        p = pybamm.ParameterValues(self.base_params)
        p["Electrolyte conductivity [S.m-1]"]               = ep.sigma
        p["Cation transference number"]                      = ep.t_plus
        p["Electrolyte diffusivity [m2.s-1]"]                = ep.D_e
        p["Initial concentration in electrolyte [mol.m-3]"]  = ep.c0
        p["Negative electrode exchange-current density [A.m-2]"] = ep.j0
        p["Positive electrode exchange-current density [A.m-2]"] = ep.j0
        p["Negative electrode charge transfer coefficient"]       = 0.5
        p["Positive electrode charge transfer coefficient"]       = 0.5
        p["Negative electrode thickness [m]"]  = 40e-6
        p["Separator thickness [m]"]           = 25e-6
        p["Positive electrode thickness [m]"]  = 70e-6
        p["Electrode height [m]"]              = np.sqrt(self.electrode_area)
        p["Electrode width [m]"]               = np.sqrt(self.electrode_area)
        p["Positive electrode porosity"]                            = 0.33
        p["Positive electrode active material volume fraction"]     = 0.66
        p["Positive electrode Bruggeman coefficient (electrolyte)"] = 1.8
        p["Negative electrode porosity"]                            = 0.25
        p["Negative electrode active material volume fraction"]     = 0.75
        p["Negative electrode Bruggeman coefficient (electrolyte)"] = 1.8
        p["Separator porosity"]                                     = 0.47
        p["Positive particle radius [m]"]           = 5.0e-6
        p["Negative particle radius [m]"]           = 5.9e-6
        p["Positive particle diffusivity [m2.s-1]"] = 4.0e-15
        p["Negative particle diffusivity [m2.s-1]"] = 3.3e-14
        p["Maximum concentration in positive electrode [mol.m-3]"] = 51000
        p["Maximum concentration in negative electrode [mol.m-3]"] = 30500
        p["Initial concentration in positive electrode [mol.m-3]"] = 28000
        p["Initial concentration in negative electrode [mol.m-3]"] = 14000
        p["Upper voltage cut-off [V]"] = self.v_max
        p["Lower voltage cut-off [V]"] = self.v_min
        T_K = T_C + 273.15
        p["Ambient temperature [K]"] = T_K
        p["Initial temperature [K]"] = T_K
        p["Current function [A]"]    = float(I_mA_cm2 * 1e-3)
        for k, v in extra.items():
            p[k] = v
        return p

    def _build_model(self):
        """Instantiate the selected PyBaMM model (DFN / SPM / SPMe)."""
        import pybamm
        models = {"DFN": pybamm.lithium_ion.DFN,
                  "SPM": pybamm.lithium_ion.SPM,
                  "SPMe": pybamm.lithium_ion.SPMe}
        if self.model_type not in models:
            raise ValueError(f"Unknown model_type {self.model_type!r}")
        return models[self.model_type]()

    def _extract(self, sol, I_mA_cm2, T_C, elec_name) -> DFNResult:
        """Extract capacity, energy, power, and voltage curve from a PyBaMM solution."""
        t = sol["Time [s]"].entries
        V = sol["Terminal voltage [V]"].entries
        mask = V >= self.v_min
        if mask.sum() < 5:
            return DFNResult(I_mA_cm2=I_mA_cm2, T_C=T_C,
                             electrolyte=elec_name, success=False)
        t_d = t[mask]; V_d = V[mask]
        I_A = I_mA_cm2 * 1e-3
        cap = min(
            I_A * t_d[-1] / 3600 * 1000 / self.cathode_mass_g * self.calibration_factor,
            205.0,
        )
        _trapz = getattr(np, "trapezoid", np.trapz)
        energy = (_trapz(V_d * I_A, t_d) / 3600 * self.calibration_factor
                  / (self.cathode_mass_g * 1e-3))
        power  = I_A * float(np.mean(V_d)) / (self.cathode_mass_g * 1e-3)
        return DFNResult(
            I_mA_cm2=I_mA_cm2, T_C=T_C, electrolyte=elec_name,
            cap=cap, crate=cap / 200.0, energy=energy, power=power,
            avg_V=float(np.mean(V_d)), t_d=t_d, V_d=V_d,
        )


class LPIFDModel(BatteryDFN):
    """NMC811 || LPIFD SPE || Li-metal DFN — pre-configured for Zhang et al. (2024).

    >>> model = LPIFDModel()
    >>> results = model.sweep([0.5, 1.0, 2.0, 3.7])
    """
    STANDARD_CURRENTS = [0.5, 1.0, 2.0, 3.0, 3.7]

    def __init__(self):
        """Initialise LPIFDModel with default BatteryDFN parameters."""
        super().__init__()

    def run_all_electrolytes(
        self,
        currents: Optional[list[float]] = None,
        T_C:      float = 25.0,
    ) -> dict[str, list[DFNResult]]:
        """Run sweeps for PTFEP, PMMA, and PEO and return results keyed by electrolyte name."""
        return self.compare_electrolytes(
            ["PTFEP", "PMMA", "PEO"],
            currents or self.STANDARD_CURRENTS, T_C,
        )


# NMC811 OCP data (half-cell, 25°C, delithiation)
_NMC811_STO   = np.array([0.20,0.25,0.27,0.30,0.33,0.36,0.40,0.44,0.48,
                           0.52,0.56,0.60,0.64,0.68,0.72,0.76,0.80,0.84,
                           0.88,0.91,0.93,0.95,0.97,0.98,0.99])
_NMC811_OCP   = np.array([4.52,4.36,4.26,4.19,4.14,4.10,4.06,4.02,3.98,
                           3.94,3.90,3.86,3.81,3.76,3.70,3.63,3.56,3.48,
                           3.38,3.27,3.17,3.05,2.88,2.76,2.60])
_NMC811_MASS  = 70e-6 * 1e-4 * 0.65 * 4780e3   # ≈ 0.02175 g


def _nmc811_ocp(sto):
    """Return a PyBaMM interpolant for the NMC811 open-circuit potential vs stoichiometry."""
    import pybamm
    return pybamm.Interpolant(_NMC811_STO, _NMC811_OCP, sto,
                              interpolator="linear", extrapolate=True)


def _li_metal_ocp(sto):
    """Return 0 V vs Li/Li+ for the Li-metal anode (ideal reference electrode)."""
    return 0.0 * sto


class NMC811LiMetalModel(BatteryDFN):
    """DFN with correct NMC811 OCP and Li-metal anode.

    Fixes vs LPIFDModel:
    - NMC811 half-cell OCP interpolation (not graphite default)
    - Li-metal anode (V = 0 vs Li/Li+)
    - Physically correct cathode mass (0.02175 g)
    - Calibration 0.95 (temperature correction only)

    >>> model = NMC811LiMetalModel()
    >>> result = model.run(I_mA_cm2=0.5, electrolyte="PTFEP")
    """
    LI_METAL_CS_MAX  = 76_900.0
    STANDARD_CURRENTS = [0.5, 1.0, 2.0, 3.0, 3.7]

    def __init__(self):
        """Initialise with NMC811 cathode mass (0.02175 g) and calibration factor 0.95."""
        super().__init__(cathode_mass_g=_NMC811_MASS, calibration_factor=0.95)

    def _build_params(self, I_mA_cm2, T_C, ep, extra):
        """Override parent params with NMC811 OCP, Li-metal anode, and corrected concentrations."""
        p = super()._build_params(I_mA_cm2, T_C, ep, {})
        p["Positive electrode OCP [V]"]                             = _nmc811_ocp
        p["Maximum concentration in positive electrode [mol.m-3]"]  = 51_000.0
        p["Initial concentration in positive electrode [mol.m-3]"]  = 0.27 * 51_000.0
        p["Positive particle diffusivity [m2.s-1]"]                 = 2.0e-14
        p["Positive particle radius [m]"]                           = 5.0e-6
        p["Negative electrode OCP [V]"]                             = _li_metal_ocp
        p["Maximum concentration in negative electrode [mol.m-3]"]  = self.LI_METAL_CS_MAX
        p["Initial concentration in negative electrode [mol.m-3]"]  = 0.5 * self.LI_METAL_CS_MAX
        p["Negative particle diffusivity [m2.s-1]"]                 = 1.0e-10
        p["Negative electrode exchange-current density [A.m-2]"]    = 10.0
        p["Negative electrode active material volume fraction"]      = 0.9
        p["Negative electrode porosity"]                             = 0.10
        p["Positive electrode exchange-current density [A.m-2]"]    = 8.0
        p["Electrolyte diffusivity [m2.s-1]"]                       = 1.5e-10
        for k, v in extra.items():
            p[k] = v
        return p

    def run(self, I_mA_cm2, T_C=25.0, electrolyte="PTFEP",
            extra_params=None) -> DFNResult:
        """Run one galvanostatic discharge with NMC811 OCP and Li-metal anode corrections."""
        import pybamm
        ep    = self._resolve(electrolyte)
        param = self._build_params(I_mA_cm2, T_C, ep, extra_params or {})
        model = self._build_model()
        I_A   = float(param["Current function [A]"])
        Q_th  = (0.70 * 51_000.0 * (70e-6 * self.electrode_area * 0.65)
                 * 96485.0 / 3.6)
        t_max = min(Q_th / (I_A * 1000.0) * 3600.0 * 1.5, 60_000.0) if I_A > 0 else 7200.0
        sim = pybamm.Simulation(
            model, parameter_values=param,
            var_pts={"x_n": 20, "x_s": 20, "x_p": 30, "r_n": 10, "r_p": 20},
            solver=pybamm.CasadiSolver(mode="safe", dt_max=60),
        )
        try:
            sol = sim.solve(t_eval=np.linspace(0, t_max, 600))
            return self._extract(sol, I_mA_cm2, T_C, ep.name)
        except Exception as exc:
            log.warning("NMC811LiMetal DFN failed %.2f mA/cm²: %s", I_mA_cm2, exc)
            return DFNResult(I_mA_cm2=I_mA_cm2, T_C=T_C,
                             electrolyte=ep.name, success=False)
