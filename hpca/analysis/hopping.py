#!/usr/bin/env python3
"""
hopping.py — Ion hopping site detection, event extraction, hop rates,
and Haven ratio for battery materials MLMD trajectories.

Positions must be in Angstrom, unwrapped (no PBC jumps).

Usage
-----
from hpca.analysis.hopping import detect_equilibrium_sites, extract_hop_events
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import label as scipy_label

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _box_lengths(box_frame: np.ndarray) -> np.ndarray:
    """Return (3,) lengths from a single (3,2) box frame."""
    return box_frame[:, 1] - box_frame[:, 0]


def _pbc_delta(dr: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Apply minimum image convention; dr shape (..., 3), L shape (3,)."""
    return dr - L * np.round(dr / L)


def _pbc_dist(a: np.ndarray, b: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Scalar PBC distance(s) between point(s) a and b."""
    dr = _pbc_delta(b - a, L)
    return np.sqrt(np.sum(dr ** 2, axis=-1))


# ---------------------------------------------------------------------------
# Equilibrium site detection
# ---------------------------------------------------------------------------

def detect_equilibrium_sites(
    positions: np.ndarray,
    box: np.ndarray,
    method: str = "grid",
    grid_spacing: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (site_centers [N_sites,3], site_labels [N_sites]) from 3D density peaks."""
    positions = np.asarray(positions, dtype=np.float64)
    box       = np.asarray(box, dtype=np.float64)
    n_frames, n_atoms, _ = positions.shape

    # Use mean box for grid extent
    box_mean = box.mean(axis=0)                     # (3, 2)
    L        = _box_lengths(box_mean)               # (3,)
    lo       = box_mean[:, 0]                       # (3,)

    if method != "grid":
        warnings.warn(f"method='{method}' not implemented; falling back to 'grid'.", stacklevel=2)

    # Build 3D histogram of all visited positions (wrapped into first cell)
    n_bins_x = max(2, int(np.ceil(L[0] / grid_spacing)))
    n_bins_y = max(2, int(np.ceil(L[1] / grid_spacing)))
    n_bins_z = max(2, int(np.ceil(L[2] / grid_spacing)))

    # Wrap positions into [lo, lo+L)
    flat_pos = positions.reshape(-1, 3)             # (n_frames*n_atoms, 3)
    wrapped  = lo + np.mod(flat_pos - lo, L)

    hist, edges = np.histogramdd(
        wrapped,
        bins=[n_bins_x, n_bins_y, n_bins_z],
        range=[(lo[0], lo[0] + L[0]),
               (lo[1], lo[1] + L[1]),
               (lo[2], lo[2] + L[2])],
    )

    # Find local maxima: voxels above mean + 1*std density that are connected peaks
    threshold = hist.mean() + hist.std()
    peaks_mask = hist > threshold

    labeled, n_features = scipy_label(peaks_mask)

    if n_features == 0:
        # Fallback: all non-zero voxels as single site
        labeled, n_features = scipy_label(hist > 0)

    # Centre of mass of each labelled region
    centers  = np.zeros((n_features, 3), dtype=np.float64)
    labels   = np.arange(1, n_features + 1, dtype=np.int64)

    bin_cx = 0.5 * (edges[0][:-1] + edges[0][1:])
    bin_cy = 0.5 * (edges[1][:-1] + edges[1][1:])
    bin_cz = 0.5 * (edges[2][:-1] + edges[2][1:])

    for k in range(1, n_features + 1):
        mask = labeled == k
        w    = hist[mask]
        idx  = np.argwhere(mask)                    # (n_vox, 3)
        cx   = float(np.average(bin_cx[idx[:, 0]], weights=w))
        cy   = float(np.average(bin_cy[idx[:, 1]], weights=w))
        cz   = float(np.average(bin_cz[idx[:, 2]], weights=w))
        centers[k - 1] = [cx, cy, cz]

    return centers, labels


# ---------------------------------------------------------------------------
# Per-frame site assignment
# ---------------------------------------------------------------------------

def assign_sites(
    positions_frame: np.ndarray,
    site_centers: np.ndarray,
    r_assign: float,
) -> np.ndarray:
    """Return site_id per atom (-1 = in transit); no PBC — assumes unwrapped."""
    positions_frame = np.asarray(positions_frame, dtype=np.float64)
    site_centers    = np.asarray(site_centers,    dtype=np.float64)
    n_atoms  = positions_frame.shape[0]
    n_sites  = site_centers.shape[0]

    # (n_atoms, n_sites, 3) distance matrix
    dr = site_centers[np.newaxis, :, :] - positions_frame[:, np.newaxis, :]
    r2 = np.sum(dr ** 2, axis=-1)                   # (n_atoms, n_sites)

    nearest = np.argmin(r2, axis=1)                  # (n_atoms,)
    min_r2  = r2[np.arange(n_atoms), nearest]

    site_ids = np.where(min_r2 < r_assign ** 2, nearest.astype(np.int64), np.int64(-1))
    return site_ids


# ---------------------------------------------------------------------------
# Hop event extraction
# ---------------------------------------------------------------------------

def extract_hop_events(
    positions: np.ndarray,
    site_centers: np.ndarray,
    r_assign: float,
    box: np.ndarray,
) -> List[Tuple[int, int, int, int]]:
    """Return list of (atom_id, from_site, to_site, frame) hop events."""
    positions    = np.asarray(positions,    dtype=np.float64)
    site_centers = np.asarray(site_centers, dtype=np.float64)
    box          = np.asarray(box,          dtype=np.float64)
    n_frames, n_atoms, _ = positions.shape

    events: List[Tuple[int, int, int, int]] = []

    # Assign sites for all frames
    site_seq = np.full((n_frames, n_atoms), -1, dtype=np.int64)
    for f in range(n_frames):
        site_seq[f] = assign_sites(positions[f], site_centers, r_assign)

    # Track last known occupied site per atom
    last_site = site_seq[0].copy()

    for f in range(1, n_frames):
        curr = site_seq[f]
        for a in range(n_atoms):
            if curr[a] == -1:
                continue                             # in transit
            if last_site[a] == -1:
                last_site[a] = curr[a]
                continue
            if curr[a] != last_site[a]:
                events.append((a, int(last_site[a]), int(curr[a]), f))
                last_site[a] = curr[a]

    return events


# ---------------------------------------------------------------------------
# Hop rate
# ---------------------------------------------------------------------------

def hop_rate_per_atom(
    hop_events: List[Tuple[int, int, int, int]],
    n_atoms: int,
    n_frames: int,
    dt_ps: float,
) -> float:
    """Return mean hop rate in hops/atom/ps."""
    total_time_ps = n_frames * dt_ps
    if total_time_ps <= 0 or n_atoms <= 0:
        return 0.0
    return len(hop_events) / (n_atoms * total_time_ps)


# ---------------------------------------------------------------------------
# Haven ratio
# ---------------------------------------------------------------------------

def haven_ratio(
    hop_events: List[Tuple[int, int, int, int]],
    D_tracer: float,
    D_charge: float,
    n_atoms: int,
) -> float:
    """Return Haven ratio H_R = D_tracer / D_charge (dimensionless)."""
    if D_charge is None or D_charge <= 0:
        warnings.warn("D_charge is zero or invalid; Haven ratio undefined.", stacklevel=2)
        return float("nan")
    if D_tracer is None or D_tracer <= 0:
        warnings.warn("D_tracer is zero or invalid; Haven ratio undefined.", stacklevel=2)
        return float("nan")
    return float(D_tracer / D_charge)
