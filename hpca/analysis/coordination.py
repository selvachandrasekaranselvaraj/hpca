#!/usr/bin/env python3
"""
coordination.py — Coordination numbers, bond angles, and polyhedral analysis
for LAMMPS/VASP trajectories from battery materials simulations.

All distance calculations use the minimum image convention for PBC.

Trajectory dict schema (from trajectory.py):
    positions      : np.ndarray (n_frames, n_atoms, 3)  [Angstrom, unwrapped]
    box            : np.ndarray (n_frames, 3, 2)        [[xlo,xhi],[ylo,yhi],[zlo,zhi]]
    species_indices: {element: np.ndarray bool}

Usage
-----
from hpca.analysis.coordination import compute_coordination_number, polyhedral_analysis
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Internal PBC helpers
# ---------------------------------------------------------------------------

def _box_lengths(box_frame: np.ndarray) -> np.ndarray:
    """Return (3,) box lengths from a single (3,2) box frame."""
    return box_frame[:, 1] - box_frame[:, 0]


def _pbc_delta(dr: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Apply minimum image convention in-place; dr shape (..., 3), L shape (3,)."""
    return dr - L * np.round(dr / L)


def _get_type_indices(
    type_indices: Dict[str, np.ndarray],
    species: str,
    n_atoms: int,
) -> np.ndarray:
    """Return integer indices for a species from a bool-mask or int-array dict."""
    mask = type_indices.get(species)
    if mask is None:
        raise KeyError(
            f"Species '{species}' not found. Available: {list(type_indices.keys())}"
        )
    arr = np.asarray(mask)
    if arr.dtype == bool:
        return np.where(arr)[0]
    return arr.astype(np.intp)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_coordination_number(
    positions: np.ndarray,
    central_type: str,
    neighbor_type: str,
    r_cutoff: float,
    box: np.ndarray,
    type_indices: Dict[str, np.ndarray],
) -> np.ndarray:
    """Return CN array (n_frames, n_central) counting neighbors within r_cutoff using PBC."""
    positions = np.asarray(positions, dtype=np.float64)
    box       = np.asarray(box, dtype=np.float64)
    n_frames, n_atoms, _ = positions.shape

    central_idx  = _get_type_indices(type_indices, central_type,  n_atoms)
    neighbor_idx = _get_type_indices(type_indices, neighbor_type, n_atoms)
    n_central    = len(central_idx)

    # Handle same-type pairs: exclude self
    same_type = central_type == neighbor_type

    cn_array = np.zeros((n_frames, n_central), dtype=np.int32)

    for f in range(n_frames):
        L   = _box_lengths(box[f])          # (3,)
        c   = positions[f, central_idx]     # (n_central, 3)
        nb  = positions[f, neighbor_idx]    # (n_neighbor, 3)

        # Broadcast: dr[i, j] = nb[j] - c[i]
        dr  = nb[np.newaxis, :, :] - c[:, np.newaxis, :]   # (n_central, n_nb, 3)
        dr  = _pbc_delta(dr, L)
        r2  = np.sum(dr ** 2, axis=-1)                      # (n_central, n_nb)

        inside = r2 < r_cutoff ** 2

        if same_type:
            # Zero-out self (central index i maps to neighbor index i in same array)
            for i, ci in enumerate(central_idx):
                j = np.searchsorted(neighbor_idx, ci)
                if j < len(neighbor_idx) and neighbor_idx[j] == ci:
                    inside[i, j] = False

        cn_array[f] = inside.sum(axis=1).astype(np.int32)

    return cn_array


def bond_angle_distribution(
    positions: np.ndarray,
    central_type: str,
    neighbor_type: str,
    r_cutoff: float,
    box: np.ndarray,
    type_indices: Dict[str, np.ndarray],
    n_bins: int = 180,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (angles_deg, distribution) for neighbor-central-neighbor triplets."""
    positions = np.asarray(positions, dtype=np.float64)
    box       = np.asarray(box, dtype=np.float64)
    n_frames, n_atoms, _ = positions.shape

    central_idx  = _get_type_indices(type_indices, central_type,  n_atoms)
    neighbor_idx = _get_type_indices(type_indices, neighbor_type, n_atoms)

    edges   = np.linspace(0.0, 180.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist    = np.zeros(n_bins, dtype=np.float64)

    r2_cut = r_cutoff ** 2

    for f in range(n_frames):
        L  = _box_lengths(box[f])
        c  = positions[f, central_idx]    # (n_central, 3)
        nb = positions[f, neighbor_idx]   # (n_nb, 3)

        dr = nb[np.newaxis, :, :] - c[:, np.newaxis, :]  # (n_central, n_nb, 3)
        dr = _pbc_delta(dr, L)
        r2 = np.sum(dr ** 2, axis=-1)                     # (n_central, n_nb)

        for i in range(len(central_idx)):
            nb_mask = r2[i] < r2_cut
            if central_type == neighbor_type:
                j_self = np.searchsorted(neighbor_idx, central_idx[i])
                if j_self < len(neighbor_idx) and neighbor_idx[j_self] == central_idx[i]:
                    nb_mask[j_self] = False
            nb_dr = dr[i, nb_mask]        # (k, 3)
            k     = nb_dr.shape[0]
            if k < 2:
                continue
            nb_r = np.sqrt(np.sum(nb_dr ** 2, axis=1, keepdims=True)) + 1e-30
            nb_hat = nb_dr / nb_r         # (k, 3)

            # Cosine of all unique pairs
            cosines = nb_hat @ nb_hat.T   # (k, k)
            cosines = np.clip(cosines, -1.0, 1.0)
            idx_u, idx_v = np.triu_indices(k, k=1)
            angles_rad = np.arccos(cosines[idx_u, idx_v])
            angles_deg = np.degrees(angles_rad)
            h, _ = np.histogram(angles_deg, bins=edges)
            hist += h.astype(np.float64)

    norm = hist.sum()
    if norm > 0:
        hist /= norm

    return centers, hist


def polyhedral_analysis(
    positions: np.ndarray,
    central_type: str,
    neighbor_types: List[str],
    r_cutoffs: List[float],
    box: np.ndarray,
    type_indices: Dict[str, np.ndarray],
) -> Dict[str, dict]:
    """Return per-neighbor-type dict of mean_CN, CN_distribution, distortion_index."""
    positions = np.asarray(positions, dtype=np.float64)
    box       = np.asarray(box, dtype=np.float64)
    n_frames, n_atoms, _ = positions.shape

    central_idx = _get_type_indices(type_indices, central_type, n_atoms)
    result: Dict[str, dict] = {}

    for nb_type, r_cut in zip(neighbor_types, r_cutoffs):
        cn_arr = compute_coordination_number(
            positions, central_type, nb_type, r_cut, box, type_indices
        )
        # cn_arr: (n_frames, n_central)
        cn_flat = cn_arr.ravel()
        cn_values, counts = np.unique(cn_flat, return_counts=True)
        cn_dist = dict(zip(cn_values.tolist(), (counts / counts.sum()).tolist()))

        # Distortion index: std of per-frame per-atom CN normalised by mean CN
        mean_cn = float(cn_flat.mean())
        std_cn  = float(cn_flat.std())
        di      = (std_cn / mean_cn) if mean_cn > 0 else 0.0

        result[nb_type] = {
            "mean_CN":         mean_cn,
            "CN_distribution": cn_dist,
            "distortion_index": di,
            "r_cutoff":        r_cut,
        }

    return result


def coordination_vs_time(
    positions: np.ndarray,
    central_type: str,
    neighbor_type: str,
    r_cutoff: float,
    box: np.ndarray,
    type_indices: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (frames, mean_CN_per_frame) arrays."""
    cn_arr = compute_coordination_number(
        positions, central_type, neighbor_type, r_cutoff, box, type_indices
    )
    frames       = np.arange(cn_arr.shape[0], dtype=np.int64)
    mean_cn_time = cn_arr.mean(axis=1)
    return frames, mean_cn_time
