"""Unit-explicit, validated formulas used by HPCA scientific services."""
from __future__ import annotations

import math

KB_EV_PER_K = 8.617333262145e-5
KB_J_PER_K = 1.380649e-23
ELEMENTARY_CHARGE_C = 1.602176634e-19
ANGSTROM2_PER_PS_TO_M2_PER_S = 1.0e-8


def einstein_diffusivity(msd_slope_a2_per_ps: float, dimensions: int = 3) -> float:
    """Return diffusion in m²/s from an MSD slope in Å²/ps: D=slope/(2d)."""
    if dimensions not in (1, 2, 3):
        raise ValueError("dimensions must be 1, 2, or 3")
    if not math.isfinite(msd_slope_a2_per_ps) or msd_slope_a2_per_ps < 0:
        raise ValueError("MSD slope must be a finite non-negative number")
    return msd_slope_a2_per_ps * ANGSTROM2_PER_PS_TO_M2_PER_S / (2 * dimensions)


def arrhenius_diffusivity(d_ref_m2_per_s: float, activation_ev: float,
                          temperature_k: float, reference_temperature_k: float = 300.0) -> float:
    """Return D(T) in m²/s using a reference-temperature Arrhenius model."""
    if d_ref_m2_per_s <= 0 or not math.isfinite(d_ref_m2_per_s):
        raise ValueError("reference diffusivity must be finite and positive")
    if activation_ev < 0 or not math.isfinite(activation_ev):
        raise ValueError("activation energy must be finite and non-negative")
    if temperature_k <= 0 or reference_temperature_k <= 0:
        raise ValueError("temperatures must be positive Kelvin")
    exponent = -(activation_ev / KB_EV_PER_K) * (
        1.0 / temperature_k - 1.0 / reference_temperature_k)
    return d_ref_m2_per_s * math.exp(exponent)


def nernst_einstein_conductivity(diffusivity_m2_per_s: float,
                                  number_density_per_m3: float,
                                  temperature_k: float,
                                  charge_number: int = 1) -> float:
    """Return uncorrelated-carrier conductivity in S/m."""
    if diffusivity_m2_per_s < 0 or number_density_per_m3 < 0:
        raise ValueError("diffusivity and number density must be non-negative")
    if temperature_k <= 0:
        raise ValueError("temperature must be positive Kelvin")
    if charge_number == 0:
        raise ValueError("charge_number cannot be zero")
    charge = charge_number * ELEMENTARY_CHARGE_C
    return diffusivity_m2_per_s * charge * charge * number_density_per_m3 / (
        KB_J_PER_K * temperature_k)
