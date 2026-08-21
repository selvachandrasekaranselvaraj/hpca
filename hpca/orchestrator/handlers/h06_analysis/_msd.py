"""
_msd.py — MSD algorithm (module-level, matches CLAUDE.md §8).
"""
from __future__ import annotations


def _fit_D_from_msd(times, msd):
    """Fit the diffusive-regime slope of an existing MSD(t) curve.

    times: (n,) ps.  msd: (n,) Å².  Returns D_m2s (m²/s).
    Uses the same 0.4–0.8 fractional window as _compute_msd_numpy so a
    cached msd_{T}K.csv reproduces the same D without re-parsing the
    (often much larger) trajectory file.
    """
    from scipy import stats

    n = len(times)
    lo = int(0.4 * n)
    hi = int(0.8 * n)
    if hi <= lo:
        hi = lo + 1
    slope, intercept, r, p, se = stats.linregress(times[lo:hi], msd[lo:hi])
    return (slope / 6.0) * 1e-8  # Å²/ps → m²/s


def _compute_msd_numpy(positions, dt_ps: float, skip_frac: float = 0.2,
                       max_lag_frac: float = 0.5):
    """
    positions: (n_frames, n_atoms, 3) unwrapped coordinates in Angstroms.
    Returns: times (ps), msd (Å²), D_m2s (m²/s), slope (Å²/ps).
    """
    import numpy as np

    skip = int(len(positions) * skip_frac)
    pos = positions[skip:]
    n = len(pos)
    max_lag = max(1, int(n * max_lag_frac))

    msd = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        diff = pos[lag:] - pos[:n - lag]  # (n-lag, n_atoms, 3)
        msd[lag - 1] = np.mean(diff ** 2) * 3

    times = np.arange(1, max_lag + 1) * dt_ps
    D_m2s = _fit_D_from_msd(times, msd)
    slope = D_m2s / 1e-8 * 6.0
    return times, msd, D_m2s, slope
