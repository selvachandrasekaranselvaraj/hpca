#!/usr/bin/env python3
"""
vacf.py — Velocity autocorrelation function (VACF), phonon DOS, and
Green-Kubo diffusivity for battery materials MD trajectories.

LAMMPS dump must include vx vy vz columns (dump_modify or dump with vel).

Usage
-----
from hpca.analysis.vacf import parse_dump_velocities, compute_vacf
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# LAMMPS dump velocity parser
# ---------------------------------------------------------------------------

def parse_dump_velocities(
    fpath: Union[str, Path],
    atom_type: Optional[Union[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (velocities [n_frames, n_atoms, 3], times_ps) from a LAMMPS dump with vx vy vz."""
    fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(f"LAMMPS dump not found: {fpath}")

    # First pass: discover column layout from ITEM: ATOMS header
    col_vx = col_vy = col_vz = col_id = col_type = col_elem = None
    with open(fpath, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("ITEM: ATOMS"):
                tokens = line.split()[2:]
                col_map  = {tok: i for i, tok in enumerate(tokens)}
                col_vx   = col_map.get("vx")
                col_vy   = col_map.get("vy")
                col_vz   = col_map.get("vz")
                col_id   = col_map.get("id",      col_map.get("ID"))
                col_type = col_map.get("type")
                col_elem = col_map.get("element")
                break

    if col_vx is None or col_vy is None or col_vz is None:
        raise ValueError(
            f"Columns vx vy vz not found in {fpath}. "
            "Add 'vx vy vz' to your dump command."
        )

    # Normalise atom_type filter
    if atom_type is not None:
        atom_type_str = str(atom_type)
    else:
        atom_type_str = None

    frames_vel: list  = []
    frames_ts:  list  = []
    n_atoms    = 0
    ts_val     = 0
    state      = "idle"

    with open(fpath, "r") as fh:
        lines_iter = iter(fh)
        for raw in lines_iter:
            line = raw.strip()

            if line == "ITEM: TIMESTEP":
                state = "timestep"
                continue

            if state == "timestep":
                ts_val = int(line)
                state  = "idle"
                continue

            if line == "ITEM: NUMBER OF ATOMS":
                state = "natoms"
                continue

            if state == "natoms":
                n_atoms = int(line)
                state   = "idle"
                continue

            if line.startswith("ITEM: BOX BOUNDS"):
                # Skip 3 box lines
                for _ in range(3):
                    next(lines_iter)
                continue

            if line.startswith("ITEM: ATOMS"):
                atom_lines = [next(lines_iter).strip() for _ in range(n_atoms)]

                parsed = []
                for al in atom_lines:
                    parts = al.split()
                    aid   = int(float(parts[col_id])) if col_id is not None else len(parsed) + 1

                    # Determine type string for filtering
                    if col_elem is not None:
                        type_str = parts[col_elem]
                    elif col_type is not None:
                        type_str = parts[col_type]
                    else:
                        type_str = "1"

                    if atom_type_str is not None and type_str != atom_type_str:
                        continue

                    vx = float(parts[col_vx])
                    vy = float(parts[col_vy])
                    vz = float(parts[col_vz])
                    parsed.append((aid, vx, vy, vz))

                parsed.sort(key=lambda x: x[0])
                if parsed:
                    vel_frame = np.array([[p[1], p[2], p[3]] for p in parsed], dtype=np.float64)
                    frames_vel.append(vel_frame)
                    frames_ts.append(ts_val)

    if not frames_vel:
        raise ValueError(f"No velocity frames parsed from {fpath}.")

    # Ensure all frames have the same atom count (use minimum)
    min_atoms  = min(v.shape[0] for v in frames_vel)
    velocities = np.stack([v[:min_atoms] for v in frames_vel], axis=0)  # (n_frames, n_atoms, 3)

    # Convert timesteps → times in ps (LAMMPS metal units: dt in fs, steps × 0.001 = ps)
    ts_arr  = np.array(frames_ts, dtype=np.float64)
    # Delta between consecutive timesteps (assume constant dump frequency)
    if len(ts_arr) > 1:
        dt_steps = float(ts_arr[1] - ts_arr[0])
    else:
        dt_steps = 1.0
    # LAMMPS timestep in metal units: 0.001 ps/step unless otherwise set
    # Store as frame index × 1 for now; caller multiplies by actual dt_ps
    times_ps = (ts_arr - ts_arr[0]) * 0.001   # default 1 fs = 0.001 ps per step

    return velocities, times_ps


# ---------------------------------------------------------------------------
# VACF
# ---------------------------------------------------------------------------

def compute_vacf(
    velocities: np.ndarray,
    max_lag_frac: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lag_times_ps, vacf) where vacf is normalised to 1 at t=0."""
    velocities = np.asarray(velocities, dtype=np.float64)
    if velocities.ndim != 3 or velocities.shape[2] != 3:
        raise ValueError(
            f"velocities must have shape (n_frames, n_atoms, 3); got {velocities.shape}"
        )

    n_frames, n_atoms, _ = velocities.shape
    max_lag = max(1, int(n_frames * max_lag_frac))

    lags_arr = np.arange(0, max_lag, dtype=np.int64)
    vacf_arr = np.zeros(max_lag, dtype=np.float64)

    # lag=0 normalisation
    v0_dot = np.sum(velocities[0] ** 2)          # sum over atoms and dims
    norm   = float(np.mean(velocities ** 2))     # <v·v> at t=0 averaged over all frames and atoms

    if norm < 1e-60:
        warnings.warn("All velocities appear to be zero; VACF will be trivially zero.", stacklevel=2)
        return lags_arr.astype(np.float64), vacf_arr

    for lag in lags_arr:
        # VACF(lag) = mean over time-origins of mean over atoms of v(t+lag)·v(t)
        n_orig = n_frames - lag
        if n_orig <= 0:
            break
        dot = np.sum(velocities[lag:] * velocities[:n_orig], axis=-1)   # (n_orig, n_atoms)
        vacf_arr[lag] = float(dot.mean())

    # Normalise so vacf[0] = 1
    if abs(vacf_arr[0]) > 1e-60:
        vacf_arr /= vacf_arr[0]

    # lag times: assume 1 frame = 1 unit (caller should multiply by dt_ps)
    lag_times = lags_arr.astype(np.float64)

    return lag_times, vacf_arr


# ---------------------------------------------------------------------------
# Phonon DOS from VACF (FFT)
# ---------------------------------------------------------------------------

def phonon_dos_from_vacf(
    lag_times_ps: np.ndarray,
    vacf: np.ndarray,
    zero_pad: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (freq_THz, dos) via FFT of VACF; zero_pad multiplies length before FFT."""
    lag_times_ps = np.asarray(lag_times_ps, dtype=np.float64)
    vacf         = np.asarray(vacf,         dtype=np.float64)

    if len(lag_times_ps) < 2:
        raise ValueError("Need at least 2 lag times for FFT.")

    dt_ps = float(lag_times_ps[1] - lag_times_ps[0])
    if dt_ps <= 0:
        raise ValueError(f"dt_ps must be > 0; got {dt_ps}")

    # Apply Hanning window to reduce spectral leakage
    n       = len(vacf)
    window  = np.hanning(n)
    vacf_w  = vacf * window

    # Zero-pad
    n_pad   = n * zero_pad
    fft_val = np.fft.rfft(vacf_w, n=n_pad)
    dos     = np.abs(fft_val) ** 2

    # Frequency axis: 1 THz = 1/(ps)
    freqs   = np.fft.rfftfreq(n_pad, d=dt_ps)  # cycles per ps = THz

    # Normalise DOS to unit area
    dfreq = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    area  = np.trapz(dos, freqs) if len(freqs) > 1 else dos.sum() * dfreq
    if area > 0:
        dos = dos / area

    return freqs, dos


# ---------------------------------------------------------------------------
# Green-Kubo diffusivity from VACF
# ---------------------------------------------------------------------------

def diffusivity_from_vacf(
    lag_times_ps: np.ndarray,
    vacf: np.ndarray,
) -> float:
    """Return D in m^2/s via D = (1/3) integral_0^inf VACF dt (Green-Kubo); VACF must NOT be normalised."""
    lag_times_ps = np.asarray(lag_times_ps, dtype=np.float64)
    vacf         = np.asarray(vacf,         dtype=np.float64)

    if len(lag_times_ps) < 2:
        raise ValueError("Need at least 2 lag times to integrate.")

    # Integrate VACF: units [Ang^2/ps^2 * ps] = Ang^2/ps
    integral_ang2_ps = float(np.trapz(vacf, lag_times_ps))

    # D = integral / 3
    D_ang2_ps = integral_ang2_ps / 3.0

    # Convert Ang^2/ps → m^2/s  (1 Ang^2/ps = 1e-8 m^2/s)
    D_m2s = D_ang2_ps * 1e-8

    return float(D_m2s)
