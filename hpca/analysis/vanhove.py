#!/usr/bin/env python3
"""
vanhove.py — Self Van Hove correlation, non-Gaussian parameter, and
displacement distribution for battery materials MD trajectories.

All functions operate on unwrapped positions in Angstrom.

Usage
-----
from hpca.analysis.vanhove import self_van_hove, non_gaussian_parameter
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Self Van Hove correlation function G_s(r, t)
# ---------------------------------------------------------------------------

def self_van_hove(
    positions: np.ndarray,
    r_max: float = 10.0,
    n_bins: int = 100,
    dt_frames: List[int] = None,
    box: Optional[np.ndarray] = None,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Return {dt_frame: (r_centers, G_s)} where G_s(r,t) = (1/N) sum_i <delta(r - |r_i(t)-r_i(0)|)>."""
    if dt_frames is None:
        dt_frames = [1, 10, 100, 1000]

    positions = np.asarray(positions, dtype=np.float64)
    n_frames, n_atoms, _ = positions.shape

    r_edges  = np.linspace(0.0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr       = r_edges[1] - r_edges[0]

    result: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for dt in dt_frames:
        if dt >= n_frames:
            warnings.warn(
                f"dt_frames={dt} >= n_frames={n_frames}; skipping.",
                stacklevel=2,
            )
            continue

        # Vectorised: all (frame, atom) pairs with this lag
        disp   = positions[dt:] - positions[: n_frames - dt]  # (n_origins, n_atoms, 3)
        r_mag  = np.linalg.norm(disp, axis=-1).ravel()         # (n_origins * n_atoms,)

        hist, _ = np.histogram(r_mag, bins=r_edges)
        n_origins = n_frames - dt
        shell_vol = 4.0 * np.pi * r_centers ** 2 * dr
        shell_vol = np.where(shell_vol < 1e-30, 1e-30, shell_vol)
        G_s = hist.astype(np.float64) / (n_atoms * n_origins * shell_vol)

        result[dt] = (r_centers.copy(), G_s)

    return result


# ---------------------------------------------------------------------------
# Non-Gaussian parameter α₂(t)
# ---------------------------------------------------------------------------

def non_gaussian_parameter(
    positions: np.ndarray,
    max_lag_frames: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lags, alpha2) where alpha2 = 3*<r^4>/(5*<r^2>^2) - 1."""
    positions = np.asarray(positions, dtype=np.float64)
    n_frames, n_atoms, _ = positions.shape

    max_lag = min(max_lag_frames, n_frames - 1)
    lags    = np.arange(1, max_lag + 1, dtype=np.int64)
    alpha2  = np.zeros(max_lag, dtype=np.float64)

    for i, lag in enumerate(lags):
        disp  = positions[lag:] - positions[: n_frames - lag]  # (n_orig, n_atoms, 3)
        r2    = np.sum(disp ** 2, axis=-1)                      # (n_orig, n_atoms)
        r4    = r2 ** 2
        mean_r2 = float(r2.mean())
        mean_r4 = float(r4.mean())

        if mean_r2 < 1e-30:
            alpha2[i] = 0.0
        else:
            alpha2[i] = (3.0 * mean_r4) / (5.0 * mean_r2 ** 2) - 1.0

    return lags, alpha2


# ---------------------------------------------------------------------------
# Displacement distribution P(Δr, t)
# ---------------------------------------------------------------------------

def displacement_distribution(
    positions: np.ndarray,
    dt_frames: List[int] = None,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Return {dt_frame: (bin_centers, counts)} histogram of scalar displacements."""
    if dt_frames is None:
        dt_frames = [1, 10, 100, 1000]

    positions = np.asarray(positions, dtype=np.float64)
    n_frames, n_atoms, _ = positions.shape

    result: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for dt in dt_frames:
        if dt >= n_frames:
            warnings.warn(
                f"dt_frames={dt} >= n_frames={n_frames}; skipping.",
                stacklevel=2,
            )
            continue

        disp  = positions[dt:] - positions[: n_frames - dt]  # (n_orig, n_atoms, 3)
        r_mag = np.linalg.norm(disp, axis=-1).ravel()

        n_bins   = max(50, min(200, int(np.sqrt(r_mag.size))))
        counts, edges = np.histogram(r_mag, bins=n_bins)
        centers       = 0.5 * (edges[:-1] + edges[1:])

        result[dt] = (centers, counts.astype(np.float64))

    return result
