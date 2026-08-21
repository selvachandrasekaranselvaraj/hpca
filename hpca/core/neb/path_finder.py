"""hpca/core/neb/path_finder.py — Nonlinear NEB path generation."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize

try:
    from pymatgen.core import Structure
except ImportError:
    import sys
    print("ERROR: pymatgen is required. Install with: pip install pymatgen")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Void Detection
# -----------------------------------------------------------------------------

def analyze_migration_sites(
    structure: Structure,
    mobile_element: str = "Li",
    grid_density: int = 40,
    void_cutoff: float = 2.5,
) -> Dict[str, Any]:
    """
    Detect void (empty) sites in a structure using a grid‑based density search.

    The algorithm creates a 3D grid over the unit cell, computes the minimum distance
    from each grid point to any framework atom (excluding the mobile species), finds
    local maxima (peaks) of this distance field, and identifies voids as peaks above
    the cutoff distance. Close voids are merged.

    Args:
        structure (Structure): Pymatgen Structure of the system.
        mobile_element (str): Symbol of the mobile ion (e.g., 'Li').
        grid_density (int): Number of grid points per dimension.
        void_cutoff (float): Minimum distance (Å) from framework atoms to call a void.

    Returns:
        Dict[str, Any]: Dictionary with keys:
            - "mobile_sites": List of dicts with 'index', 'fractional', 'cartesian'.
            - "void_sites": List of dicts with 'index', 'fractional', 'cartesian', 'radius'.
            - "framework_atoms": List of atom indices (0‑based) that are not mobile.
    """
    mobile_indices = [i for i, sp in enumerate(structure.species)
                      if sp.symbol == mobile_element]
    framework_indices = [i for i in range(len(structure))
                         if i not in mobile_indices]

    frac_coords = structure.frac_coords
    lattice = structure.lattice.matrix

    grid_frac = np.linspace(0, 1, grid_density, endpoint=False)
    grid_coords = np.array(np.meshgrid(grid_frac, grid_frac, grid_frac)).T.reshape(-1, 3)

    dists = []
    for g in grid_coords:
        min_dist = float('inf')
        for idx in framework_indices:
            diff = g - frac_coords[idx]
            diff -= np.round(diff)  # minimum image convention
            cart_diff = diff @ lattice
            d = np.linalg.norm(cart_diff)
            if d < min_dist:
                min_dist = d
        dists.append(min_dist)

    dists = np.array(dists).reshape(grid_density, grid_density, grid_density)

    footprint = np.ones((3, 3, 3), dtype=bool)
    padded = np.pad(dists, 1, mode='wrap')
    maxima = maximum_filter(padded, footprint=footprint, mode='wrap')
    peaks = (padded == maxima) & (padded > void_cutoff)
    peaks = peaks[1:-1, 1:-1, 1:-1]
    peak_positions = np.argwhere(peaks)

    void_sites = []
    for pos in peak_positions:
        frac = (pos + 0.5) / grid_density
        radius = dists[tuple(pos)]
        void_sites.append({
            "fractional": frac.tolist(),
            "cartesian": (frac @ lattice).tolist(),
            "radius": float(radius)
        })

    # Merge close voids (within 0.3 Å)
    merged = []
    for v in void_sites:
        cart = np.array(v["cartesian"])
        too_close = False
        for m in merged:
            if np.linalg.norm(cart - np.array(m["cartesian"])) < 0.3:
                too_close = True
                break
        if not too_close:
            merged.append(v)

    for i, v in enumerate(merged):
        v["index"] = i

    mobile_sites = []
    for i in mobile_indices:
        mobile_sites.append({
            "index": i,
            "fractional": frac_coords[i].tolist(),
            "cartesian": (frac_coords[i] @ lattice).tolist()
        })

    return {
        "mobile_sites": mobile_sites,
        "void_sites": merged,
        "framework_atoms": framework_indices
    }


# -----------------------------------------------------------------------------
# Crystallographic Directions
# -----------------------------------------------------------------------------

def parse_direction(dir_str: str) -> np.ndarray:
    """
    Parse a crystallographic direction string into a normalized vector.

    Supports formats: "[110]", "[1,1,0]", "110", or "1 1 0".

    Args:
        dir_str (str): Direction string (e.g., "[110]").

    Returns:
        np.ndarray: Normalized direction vector of shape (3,).
    """
    dir_str = dir_str.strip("[] ")
    if "," in dir_str:
        parts = dir_str.split(",")
    else:
        parts = list(dir_str)
    vec = np.array([float(p) for p in parts], dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


# -----------------------------------------------------------------------------
# Anchor Chain Construction
# -----------------------------------------------------------------------------

def build_anchor_chain(
    initial_frac: np.ndarray,
    void_sites: List[Dict],
    directions: List[str],
    hop_distance: float,
    lattice: np.ndarray,
    max_anchors: int = 10,
) -> List[np.ndarray]:
    """
    Build a chain of anchor points by stepping along crystallographic directions.

    Starting from the initial position, the function steps along each direction
    by hop_distance (in Å). If the candidate position is within 0.5 Å of a detected
    void, it snaps to that void; otherwise, it keeps the candidate position.

    Args:
        initial_frac (np.ndarray): Starting fractional coordinates (3,).
        void_sites (List[Dict]): List of void site dicts from analyze_migration_sites().
        directions (List[str]): List of direction strings (e.g., ["[100]", "[010]"]).
        hop_distance (float): Step length along each direction (Å).
        lattice (np.ndarray): 3x3 lattice matrix.
        max_anchors (int): Maximum number of anchors (including the start).

    Returns:
        List[np.ndarray]: List of fractional coordinate arrays for the anchor points.
    """
    anchors = [np.array(initial_frac) % 1.0]
    current = anchors[0]

    for dir_str in directions:
        if len(anchors) >= max_anchors:
            break
        dir_vec = parse_direction(dir_str)
        candidate = (current + hop_distance * dir_vec) % 1.0

        best_void = None
        best_dist = float('inf')
        for v in void_sites:
            v_frac = np.array(v["fractional"])
            diff = candidate - v_frac
            diff -= np.round(diff)
            dist = np.linalg.norm(diff @ lattice)
            if dist < best_dist:
                best_dist = dist
                best_void = v_frac

        if best_dist < 0.5:
            anchors.append(np.array(best_void) % 1.0)
        else:
            anchors.append(candidate)
        current = anchors[-1]

    return anchors


# -----------------------------------------------------------------------------
# Non-linear Path Predictor
# -----------------------------------------------------------------------------

def min_image_dist(frac1: np.ndarray, frac2: np.ndarray, lattice: np.ndarray) -> float:
    """
    Compute the minimum image distance (Cartesian) between two fractional points.

    Applies periodic boundary conditions using the minimum image convention.

    Args:
        frac1 (np.ndarray): First fractional coordinate (3,).
        frac2 (np.ndarray): Second fractional coordinate (3,).
        lattice (np.ndarray): 3x3 lattice matrix.

    Returns:
        float: Cartesian distance in Å.
    """
    diff = frac1 - frac2
    diff -= np.round(diff)  # wrap to [-0.5, 0.5]
    cart_diff = diff @ lattice
    return np.linalg.norm(cart_diff)


def objective_nonlinear_path(
    flat_points: np.ndarray,
    n_images: int,
    start_frac: np.ndarray,
    end_frac: np.ndarray,
    framework_fracs: np.ndarray,
    lattice: np.ndarray,
    cutoff_radius: float,
    spring_constant: float,
    repulsion_scale: float,
) -> float:
    """
    Objective function for the non‑linear path optimisation.

    The penalty is the sum of:
        1. Repulsion: 1/d^2 from framework atoms (if d < cutoff_radius).
        2. Spring: squared distance between consecutive images (smoothness).

    Args:
        flat_points (np.ndarray): Flattened array of intermediate fractional coords.
        n_images (int): Number of intermediate images.
        start_frac (np.ndarray): Start node fractional coordinate.
        end_frac (np.ndarray): End node fractional coordinate.
        framework_fracs (np.ndarray): (N_framework, 3) array of framework atoms.
        lattice (np.ndarray): 3x3 lattice matrix.
        cutoff_radius (float): Distance below which repulsion is active (Å).
        spring_constant (float): Weight for the spring penalty.
        repulsion_scale (float): Weight for the repulsion penalty.

    Returns:
        float: Total penalty (higher = worse).
    """
    pts = flat_points.reshape(n_images, 3)
    all_pts = np.vstack([start_frac, pts, end_frac])
    penalty = 0.0

    # Repulsion from framework atoms
    for pt in all_pts:
        for f_frac in framework_fracs:
            d = min_image_dist(pt, f_frac, lattice)
            if d < cutoff_radius:
                penalty += repulsion_scale / (d**2 + 1e-6)

    # Spring smoothness
    for i in range(len(all_pts) - 1):
        diff = all_pts[i+1] - all_pts[i]
        diff -= np.round(diff)
        cart_diff = diff @ lattice
        penalty += spring_constant * np.dot(cart_diff, cart_diff)

    return penalty


def predict_nonlinear_path_segment(
    start_frac: np.ndarray,
    end_frac: np.ndarray,
    framework_fracs: np.ndarray,
    lattice: np.ndarray,
    n_images: int,
    cutoff_radius: float = 2.0,
    spring_constant: float = 1.0,
    repulsion_scale: float = 100.0,
) -> np.ndarray:
    """
    Predict a non‑linear path between two anchor points.

    Uses L‑BFGS‑B optimisation to minimise repulsion from framework atoms and
    enforce smoothness (spring force) on the intermediate images.

    Args:
        start_frac (np.ndarray): Start node fractional coordinate (3,).
        end_frac (np.ndarray): End node fractional coordinate (3,).
        framework_fracs (np.ndarray): (N_framework, 3) array of framework atoms.
        lattice (np.ndarray): 3x3 lattice matrix.
        n_images (int): Number of intermediate images to generate.
        cutoff_radius (float): Repulsion cutoff radius (Å).
        spring_constant (float): Smoothness strength.
        repulsion_scale (float): Repulsion strength.

    Returns:
        np.ndarray: (n_images, 3) array of fractional coordinates.
    """
    # Initial linear guess
    x0 = np.linspace(start_frac, end_frac, n_images+2)[1:-1].flatten()
    bounds = [(0.0, 1.0)] * (n_images * 3)

    res = minimize(
        objective_nonlinear_path,
        x0,
        args=(n_images, np.array(start_frac), np.array(end_frac),
              framework_fracs, lattice, cutoff_radius, spring_constant, repulsion_scale),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-6}
    )

    if not res.success:
        print(f"Warning: optimisation did not converge for segment: {res.message}")

    return res.x.reshape(n_images, 3)


def build_nonlinear_chained_path(
    anchors: List[np.ndarray],
    framework_fracs: np.ndarray,
    lattice: np.ndarray,
    n_images_total: int,
    spacing: float,
    cutoff_radius: float = 2.0,
) -> np.ndarray:
    """
    Build a complete chained non‑linear path across all anchor segments.

    Distributes the total number of images proportionally to the straight‑line
    length of each segment, then calls predict_nonlinear_path_segment() for each.

    Args:
        anchors (List[np.ndarray]): List of anchor fractional coordinates.
        framework_fracs (np.ndarray): (N_framework, 3) array of framework atoms.
        lattice (np.ndarray): 3x3 lattice matrix.
        n_images_total (int): Total number of intermediate images across all segments.
        spacing (float): Target spacing between images (Å).
        cutoff_radius (float): Repulsion cutoff radius (Å).

    Returns:
        np.ndarray: (n_images_total, 3) array of all intermediate coordinates.
    """
    if len(anchors) < 2:
        return np.array([])

    # Compute segment lengths (straight-line, for distribution)
    seg_lengths = []
    for i in range(len(anchors)-1):
        diff = anchors[i+1] - anchors[i]
        diff -= np.round(diff)
        seg_lengths.append(np.linalg.norm(diff @ lattice))
    total_len = sum(seg_lengths)

    if total_len == 0:
        # Fallback to uniform distribution
        n_per_seg = [int(n_images_total / (len(anchors)-1))] * (len(anchors)-1)
        remainder = n_images_total - sum(n_per_seg)
        for i in range(remainder):
            n_per_seg[i] += 1
    else:
        # Allocate images proportional to segment length
        n_per_seg = [max(1, int(n_images_total * l / total_len)) for l in seg_lengths]
        # Adjust to match total
        while sum(n_per_seg) < n_images_total:
            idx = np.argmax([n_per_seg[i] / seg_lengths[i] if seg_lengths[i] > 0 else 0 for i in range(len(seg_lengths))])
            n_per_seg[idx] += 1
        while sum(n_per_seg) > n_images_total:
            idx = np.argmin([n_per_seg[i] / seg_lengths[i] if seg_lengths[i] > 0 else 1e9 for i in range(len(seg_lengths))])
            if n_per_seg[idx] > 1:
                n_per_seg[idx] -= 1
            else:
                break

    all_coords = []
    for i in range(len(anchors)-1):
        start = anchors[i]
        end = anchors[i+1]
        n_img = n_per_seg[i]
        if n_img > 0:
            pts = predict_nonlinear_path_segment(
                start, end, framework_fracs, lattice, n_img,
                cutoff_radius=cutoff_radius
            )
            all_coords.append(pts)

    if all_coords:
        return np.vstack(all_coords)
    else:
        return np.array([])


# -----------------------------------------------------------------------------
# Linear interpolation fallback
# -----------------------------------------------------------------------------

def build_chained_path_linear(
    anchors: List[np.ndarray],
    n_images: int,
    lattice: np.ndarray,
) -> np.ndarray:
    """
    Simple piecewise linear interpolation (fallback if non‑linear predictor fails).

    This function interpolates linearly along each segment between anchors,
    distributing images proportionally to segment length.

    Args:
        anchors (List[np.ndarray]): List of anchor fractional coordinates.
        n_images (int): Total number of intermediate images.
        lattice (np.ndarray): 3x3 lattice matrix.

    Returns:
        np.ndarray: (n_images, 3) array of fractional coordinates.
    """
    anchors = [np.asarray(a) % 1.0 for a in anchors]
    seg_lengths = []
    for i in range(len(anchors) - 1):
        diff = anchors[i+1] - anchors[i]
        diff -= np.round(diff)
        seg_lengths.append(np.linalg.norm(diff @ lattice))
    total_dist = sum(seg_lengths)
    if total_dist == 0:
        return np.tile(anchors[0], (n_images, 1))

    coords = []
    for img_idx in range(1, n_images + 1):
        t = img_idx / (n_images + 1)
        target_dist = t * total_dist
        cum_dist = 0
        for seg_idx, seg_len in enumerate(seg_lengths):
            if target_dist <= cum_dist + seg_len or seg_idx == len(seg_lengths) - 1:
                local_t = (target_dist - cum_dist) / seg_len if seg_len > 0 else 0
                local_t = np.clip(local_t, 0, 1)
                p_start = anchors[seg_idx]
                p_end = anchors[seg_idx+1]
                diff = p_end - p_start
                diff -= np.round(diff)
                coord = (p_start + local_t * diff) % 1.0
                coords.append(coord)
                break
            cum_dist += seg_len
    return np.array(coords)
