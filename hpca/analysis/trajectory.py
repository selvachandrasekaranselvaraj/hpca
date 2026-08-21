#!/usr/bin/env python3
"""
trajectory.py — Parse and manipulate LAMMPS dump and VASP XDATCAR trajectories.

Supported formats
-----------------
- LAMMPS dump (custom: id type xu yu zu element)   — unwrapped positions
- VASP XDATCAR (Direct fractional coordinates)      — needs unwrapping via PBC

All returned positions are in Angstrom.

Returned trajectory dict schema
--------------------------------
{
  "n_frames":       int,
  "n_atoms":        int,
  "atom_types":     list[str],                       # per-atom element, length n_atoms
  "positions":      np.ndarray (n_frames, n_atoms, 3),
  "timesteps":      np.ndarray (n_frames,) int64,
  "box":            np.ndarray (n_frames, 3, 2),     # [[xlo,xhi],[ylo,yhi],[zlo,zhi]]
  "species_indices":  {element: np.ndarray bool},    # True mask, length n_atoms
  "source_format":  str,                             # "lammps" | "xdatcar"
  "lattice":        np.ndarray (3,3) | None,         # first-frame lattice (XDATCAR)
}

Usage
-----
from trajectory import parse_trajectory, get_mobile_indices, slice_trajectory

traj = parse_trajectory("dump_unwrapped.lmp")
li_idx = get_mobile_indices(traj, "Li")
li_pos = traj["positions"][:, li_idx, :]   # (n_frames, n_Li, 3)
"""

from __future__ import annotations

import copy
import re
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
ANGSTROM = "Angstrom"

# ---------------------------------------------------------------------------
# Format auto-detection
# ---------------------------------------------------------------------------

def _detect_format(fpath: Path) -> str:
    """
    Peek at the first non-empty line to distinguish LAMMPS dump from XDATCAR.
    Returns "lammps" or "xdatcar".
    """
    with open(fpath, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line == "ITEM: TIMESTEP":
                return "lammps"
            # XDATCAR: first line is a system name (free text)
            return "xdatcar"
    raise ValueError(f"Cannot determine format of {fpath} — file appears empty.")


# ---------------------------------------------------------------------------
# LAMMPS dump parser
# ---------------------------------------------------------------------------

def parse_lammps_dump(
    fpath: Union[str, Path],
    species: Optional[list[str]] = None,
) -> dict:
    """
    Parse a LAMMPS dump file written with the custom style:
        dump 1 all custom N dump.lmp id type xu yu zu element

    Parameters
    ----------
    fpath   : path to dump_unwrapped.lmp (or any .lmp dump)
    species : optional filter — only keep atoms whose element is in this list.
              If None, keep all atoms.

    Returns
    -------
    Trajectory dict (see module docstring).
    """
    fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(f"LAMMPS dump not found: {fpath}")

    # ------------------------------------------------------------------ first pass: discover header columns
    # We need to find the column order from  "ITEM: ATOMS id type xu yu zu element"
    col_id = col_xu = col_yu = col_zu = col_elem = col_type = None

    with open(fpath, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("ITEM: ATOMS"):
                tokens = line.split()[2:]          # ['id','type','xu','yu','zu','element',...]
                col_map = {tok: i for i, tok in enumerate(tokens)}
                col_id   = col_map.get("id",      col_map.get("ID"))
                col_xu   = col_map.get("xu",      col_map.get("x"))
                col_yu   = col_map.get("yu",      col_map.get("y"))
                col_zu   = col_map.get("zu",      col_map.get("z"))
                col_elem = col_map.get("element", col_map.get("type"))
                col_type = col_map.get("type")
                break

    if col_xu is None:
        raise ValueError(
            f"Cannot identify position columns in {fpath}. "
            "Expected 'xu yu zu' (or 'x y z') in ITEM: ATOMS header."
        )

    # ------------------------------------------------------------------ main parse
    frames_pos  = []         # list of (n_atoms, 3)
    frames_ts   = []         # list of int timestep
    frames_box  = []         # list of (3,2)
    atom_types  = None       # set once from frame 0

    n_atoms  = 0
    ts_val   = 0
    box_buf  = []            # accumulate 3 box lines

    state = "idle"           # finite-state machine

    with open(fpath, "r") as fh:
        lines_iter = iter(fh)
        for raw in lines_iter:
            line = raw.strip()

            if line == "ITEM: TIMESTEP":
                state = "timestep"
                continue

            if state == "timestep":
                ts_val = int(line)
                state = "idle"
                continue

            if line == "ITEM: NUMBER OF ATOMS":
                state = "natoms"
                continue

            if state == "natoms":
                n_atoms = int(line)
                state = "idle"
                continue

            if line.startswith("ITEM: BOX BOUNDS"):
                state = "box"
                box_buf = []
                continue

            if state == "box":
                box_buf.append(list(map(float, line.split()[:2])))
                if len(box_buf) == 3:
                    state = "idle"
                continue

            if line.startswith("ITEM: ATOMS"):
                # read exactly n_atoms lines
                atom_lines = []
                for _ in range(n_atoms):
                    atom_lines.append(next(lines_iter).strip())

                # parse atom lines — sort by atom id so order is stable
                parsed = []
                for al in atom_lines:
                    parts = al.split()
                    elem = parts[col_elem] if col_elem is not None else str(int(float(parts[col_type])))
                    parsed.append((
                        int(float(parts[col_id])) if col_id is not None else len(parsed),
                        elem,
                        float(parts[col_xu]),
                        float(parts[col_yu]),
                        float(parts[col_zu]),
                    ))
                # stable sort by atom id
                parsed.sort(key=lambda x: x[0])

                if atom_types is None:
                    # build from frame 0 — filter by requested species
                    raw_types = [p[1] for p in parsed]
                    if species is not None:
                        keep_mask = np.array([e in species for e in raw_types], dtype=bool)
                    else:
                        keep_mask = np.ones(len(raw_types), dtype=bool)
                    atom_types = [raw_types[i] for i in range(len(raw_types)) if keep_mask[i]]
                    _keep_mask = keep_mask  # reuse each frame

                pos_frame = np.array(
                    [[p[2], p[3], p[4]] for p in parsed], dtype=np.float64
                )
                pos_frame = pos_frame[_keep_mask]  # apply species filter

                frames_pos.append(pos_frame)
                frames_ts.append(ts_val)
                frames_box.append(np.array(box_buf, dtype=np.float64))  # (3,2)
                continue

    if not frames_pos:
        raise ValueError(f"No frames parsed from {fpath}")

    positions = np.stack(frames_pos, axis=0)    # (n_frames, n_atoms, 3)
    timesteps = np.array(frames_ts, dtype=np.int64)
    box       = np.stack(frames_box, axis=0)    # (n_frames, 3, 2)
    n_frames, n_at, _ = positions.shape

    species_indices = _build_species_indices(atom_types)

    return {
        "n_frames":       n_frames,
        "n_atoms":        n_at,
        "atom_types":     atom_types,
        "positions":      positions,
        "timesteps":      timesteps,
        "box":            box,
        "species_indices": species_indices,
        "source_format":  "lammps",
        "lattice":        None,
    }


# ---------------------------------------------------------------------------
# VASP XDATCAR parser
# ---------------------------------------------------------------------------

def parse_xdatcar(fpath: Union[str, Path]) -> dict:
    """
    Parse a VASP XDATCAR file (Direct fractional coordinates).

    Positions are converted from fractional to Cartesian Angstrom using
    the lattice vectors.  PBC wrapping is then removed with unwrap_positions.

    Returns
    -------
    Trajectory dict (see module docstring).
    """
    fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(f"XDATCAR not found: {fpath}")

    with open(fpath, "r") as fh:
        raw_lines = fh.readlines()

    lines = [l.rstrip("\n") for l in raw_lines]
    idx = 0

    # Line 0: system name (ignore)
    idx += 1

    # Line 1: universal scale factor
    scale = float(lines[idx].strip())
    idx += 1

    # Lines 2-4: lattice vectors (3 lines × 3 floats)
    lat = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        lat[i] = list(map(float, lines[idx].split()))
        idx += 1
    lat *= scale

    # Line 5: element symbols
    elem_symbols = lines[idx].split()
    idx += 1

    # Line 6: counts per element
    elem_counts = list(map(int, lines[idx].split()))
    idx += 1

    n_atoms = sum(elem_counts)
    atom_types = []
    for sym, cnt in zip(elem_symbols, elem_counts):
        atom_types.extend([sym] * cnt)

    # ------------------------------------------------------------------ read frames
    frames_frac = []    # fractional coords
    frames_ts   = []
    frames_lat  = [lat.copy()]

    frame_pattern = re.compile(r"Direct\s+configuration=\s*(\d+)", re.IGNORECASE)

    while idx < len(lines):
        line = lines[idx]
        m = frame_pattern.match(line.strip())
        if m:
            ts_val = int(m.group(1))
            idx += 1
            frac = []
            for _ in range(n_atoms):
                if idx >= len(lines):
                    break
                frac.append(list(map(float, lines[idx].split()[:3])))
                idx += 1
            if len(frac) == n_atoms:
                frames_frac.append(np.array(frac, dtype=np.float64))
                frames_ts.append(ts_val)
        else:
            # Check for updated lattice block (NPT XDATCAR has lat+counts repeated)
            # Heuristic: if line has 3 floats and next 2 lines also have 3 floats → new lattice
            parts = line.split()
            if len(parts) == 3:
                try:
                    row0 = list(map(float, parts))
                    row1 = list(map(float, lines[idx + 1].split()))
                    row2 = list(map(float, lines[idx + 2].split()))
                    new_lat = np.array([row0, row1, row2], dtype=np.float64)
                    lat = new_lat * scale
                    frames_lat.append(lat.copy())
                    idx += 3
                    idx += 2      # skip element symbols + counts
                    continue
                except (ValueError, IndexError):
                    pass
            idx += 1

    if not frames_frac:
        raise ValueError(f"No frames parsed from {fpath}")

    n_frames = len(frames_frac)

    # If NPT: lattice may vary per frame; broadcast to match n_frames
    if len(frames_lat) < n_frames:
        # Pad with last known lattice
        while len(frames_lat) < n_frames:
            frames_lat.append(frames_lat[-1].copy())
    lat_arr = np.stack(frames_lat[:n_frames], axis=0)   # (n_frames, 3, 3)

    # Convert fractional → Cartesian
    frac_arr = np.stack(frames_frac, axis=0)             # (n_frames, n_atoms, 3)
    # Cartesian: pos[f] = frac[f] @ lat[f]  (each row of frac is a fractional coord)
    cart = np.einsum("fij,fkj->fki", lat_arr, frac_arr)  # (n_frames, n_atoms, 3)

    # Build box from lattice diagonals (orthogonal assumption; safe for cubic/tetragonal)
    box = np.zeros((n_frames, 3, 2), dtype=np.float64)
    for f in range(n_frames):
        for ax in range(3):
            box[f, ax, 1] = np.linalg.norm(lat_arr[f, ax])

    # Unwrap wrapped positions (XDATCAR is wrapped)
    positions = unwrap_positions(cart, box)

    timesteps = np.array(frames_ts, dtype=np.int64)
    species_indices = _build_species_indices(atom_types)

    return {
        "n_frames":        n_frames,
        "n_atoms":         n_atoms,
        "atom_types":      atom_types,
        "positions":       positions,
        "timesteps":       timesteps,
        "box":             box,
        "species_indices": species_indices,
        "source_format":   "xdatcar",
        "lattice":         lat_arr[0],
    }


# ---------------------------------------------------------------------------
# Convenience auto-dispatcher
# ---------------------------------------------------------------------------

def parse_trajectory(
    fpath: Union[str, Path],
    species: Optional[list[str]] = None,
) -> dict:
    """
    Auto-detect format (LAMMPS dump or VASP XDATCAR) and parse.

    Parameters
    ----------
    fpath   : path to trajectory file
    species : optional species filter (passed to parse_lammps_dump only)
    """
    fpath = Path(fpath)
    fmt = _detect_format(fpath)
    if fmt == "lammps":
        return parse_lammps_dump(fpath, species=species)
    else:
        traj = parse_xdatcar(fpath)
        # Apply species filter post-hoc for XDATCAR
        if species is not None:
            traj = _filter_species(traj, species)
        return traj


# ---------------------------------------------------------------------------
# PBC unwrapping
# ---------------------------------------------------------------------------

def unwrap_positions(
    pos: np.ndarray,
    box: np.ndarray,
) -> np.ndarray:
    """
    Unwrap PBC-wrapped positions using displacement tracking (minimum-image).

    Uses the displacement between consecutive frames to detect crossings and
    accumulates an integer offset per axis per atom.  Works for any box shape
    as long as box sizes are given as lo/hi pairs.

    Parameters
    ----------
    pos : (n_frames, n_atoms, 3) wrapped positions in Angstrom
    box : (n_frames, 3, 2) box lo/hi per axis

    Returns
    -------
    unwrapped : (n_frames, n_atoms, 3)
    """
    pos = np.asarray(pos, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    n_frames, n_atoms, _ = pos.shape

    box_len = box[:, :, 1] - box[:, :, 0]  # (n_frames, 3)

    unwrapped = pos.copy()
    offsets   = np.zeros((n_atoms, 3), dtype=np.float64)

    for f in range(1, n_frames):
        L_prev = box_len[f - 1]   # (3,)
        disp   = unwrapped[f] - unwrapped[f - 1]   # (n_atoms, 3)
        # Minimum-image: round displacement to nearest integer multiple of L
        offsets -= np.round(disp / L_prev[None, :]) * L_prev[None, :]
        unwrapped[f] = pos[f] + offsets

    return unwrapped


# ---------------------------------------------------------------------------
# Trajectory slicing and sub-sampling
# ---------------------------------------------------------------------------

def slice_trajectory(
    traj: dict,
    start_frac: float = 0.2,
    end_frac: float = 1.0,
) -> dict:
    """
    Return a shallow copy of *traj* sliced to [start_frac, end_frac] of frames.

    Parameters
    ----------
    traj       : trajectory dict from parse_lammps_dump / parse_xdatcar
    start_frac : starting fraction (0.0–1.0).  Default 0.2 skips equilibration.
    end_frac   : ending fraction (0.0–1.0).  Default 1.0 uses all remaining.

    Returns
    -------
    New trajectory dict with sliced arrays. atom_types and species_indices
    are shared (not copied) as they don't vary over frames.
    """
    if not (0.0 <= start_frac < end_frac <= 1.0):
        raise ValueError(
            f"Require 0 <= start_frac < end_frac <= 1; got {start_frac}, {end_frac}"
        )
    n = traj["n_frames"]
    i0 = int(n * start_frac)
    i1 = int(n * end_frac)
    if i1 == i0:
        i1 = i0 + 1

    t = copy.copy(traj)
    t["positions"]  = traj["positions"][i0:i1].copy()
    t["timesteps"]  = traj["timesteps"][i0:i1].copy()
    t["box"]        = traj["box"][i0:i1].copy()
    t["n_frames"]   = i1 - i0
    return t


def subsample(traj: dict, every_n: int) -> dict:
    """
    Thin the trajectory by keeping every *every_n*-th frame.

    Useful for quick interactive previews or reducing memory when loading
    long MLMD trajectories (e.g. 1 M step @ dump every 1000 → 1000 frames;
    subsample(traj, 10) → 100 frames).

    Parameters
    ----------
    traj    : trajectory dict
    every_n : stride — keep frames 0, every_n, 2*every_n, ...

    Returns
    -------
    New trajectory dict with thinned arrays.
    """
    if every_n < 1:
        raise ValueError(f"every_n must be >= 1, got {every_n}")
    t = copy.copy(traj)
    sl = slice(0, None, every_n)
    t["positions"]  = traj["positions"][sl].copy()
    t["timesteps"]  = traj["timesteps"][sl].copy()
    t["box"]        = traj["box"][sl].copy()
    t["n_frames"]   = t["positions"].shape[0]
    return t


# ---------------------------------------------------------------------------
# Species helpers
# ---------------------------------------------------------------------------

def get_mobile_indices(traj: dict, mobile_ion: str) -> np.ndarray:
    """
    Return integer index array of mobile ion atoms (e.g. "Li").

    Parameters
    ----------
    traj       : trajectory dict
    mobile_ion : element symbol, e.g. "Li", "Na", "Mg"

    Returns
    -------
    1-D int64 array of atom indices where element == mobile_ion
    """
    mask = traj["species_indices"].get(mobile_ion)
    if mask is None:
        available = list(traj["species_indices"].keys())
        raise KeyError(
            f"Mobile ion '{mobile_ion}' not found in trajectory. "
            f"Available species: {available}"
        )
    return np.where(mask)[0]


def _build_species_indices(atom_types: list[str]) -> dict:
    """Build boolean-mask dict keyed by element symbol."""
    arr = np.array(atom_types)
    unique = sorted(set(atom_types))
    return {elem: (arr == elem) for elem in unique}


def _filter_species(traj: dict, species: list[str]) -> dict:
    """Return a copy of traj keeping only atoms whose element is in *species*."""
    keep = np.array([e in species for e in traj["atom_types"]], dtype=bool)
    t = copy.copy(traj)
    t["positions"]  = traj["positions"][:, keep, :].copy()
    t["atom_types"] = [e for e, k in zip(traj["atom_types"], keep) if k]
    t["n_atoms"]    = int(keep.sum())
    t["species_indices"] = _build_species_indices(t["atom_types"])
    return t


# ---------------------------------------------------------------------------
# ASE integration (optional)
# ---------------------------------------------------------------------------

def frame_to_ase_atoms(traj: dict, frame_idx: int):
    """
    Convert a single frame of *traj* to an :class:`ase.Atoms` object.

    Requires ASE to be installed in the active Python environment.
    Falls back gracefully with an ImportError message rather than crashing.

    Parameters
    ----------
    traj      : trajectory dict
    frame_idx : index of the frame to extract (negative indices supported)

    Returns
    -------
    ase.Atoms | None
    """
    try:
        from ase import Atoms
    except ImportError:
        warnings.warn(
            "ASE is not installed; cannot create Atoms object.  "
            "Install with: pip install ase",
            stacklevel=2,
        )
        return None

    pos  = traj["positions"][frame_idx]        # (n_atoms, 3)
    box  = traj["box"][frame_idx]              # (3, 2)
    cell = [box[ax, 1] - box[ax, 0] for ax in range(3)]

    atoms = Atoms(
        symbols=traj["atom_types"],
        positions=pos,
        cell=cell,
        pbc=True,
    )
    return atoms


# ---------------------------------------------------------------------------
# Trajectory summary
# ---------------------------------------------------------------------------

def trajectory_info(traj: dict) -> str:
    """Return a human-readable summary string for quick inspection."""
    sp_counts = {
        elem: int(mask.sum())
        for elem, mask in traj["species_indices"].items()
    }
    lines = [
        f"Format     : {traj['source_format']}",
        f"Frames     : {traj['n_frames']}",
        f"Atoms      : {traj['n_atoms']}",
        f"Species    : {sp_counts}",
        f"Timesteps  : {traj['timesteps'][0]} → {traj['timesteps'][-1]}",
    ]
    if traj["positions"] is not None:
        p = traj["positions"]
        lines.append(
            f"Pos range  : x=[{p[:,:,0].min():.2f},{p[:,:,0].max():.2f}] "
            f"y=[{p[:,:,1].min():.2f},{p[:,:,1].max():.2f}] "
            f"z=[{p[:,:,2].min():.2f},{p[:,:,2].max():.2f}] Å"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    fpath = sys.argv[1]
    species_filter = sys.argv[2:] if len(sys.argv) > 2 else None
    traj = parse_trajectory(fpath, species=species_filter)
    print(trajectory_info(traj))
