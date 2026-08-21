"""
phase.py — Phase transition and structural change analysis
for LAMMPS MD trajectories from battery materials simulations.

Trajectory dict schema (produced by core trajectory reader):
    traj = {
        "positions"  : np.ndarray  shape (n_frames, n_atoms, 3)   [Angstrom, unwrapped]
        "species"    : list[str]   length n_atoms
        "box"        : np.ndarray  shape (n_frames, 3, 2)  [[xlo,xhi],[ylo,yhi],[zlo,zhi]]
        "timesteps"  : np.ndarray  shape (n_frames,)       [LAMMPS integer steps]
        "dt_ps"      : float       timestep in ps (e.g. 0.001)
        "dump_freq"  : int         frames between dumps
        "energies"   : np.ndarray  shape (n_frames,)  optional (pot energy eV)
    }

Author: NREL battery-materials-ai pipeline
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter1d
from scipy.spatial import KDTree
from typing import Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dt_frame_ps(traj: dict) -> float:
    """Return the physical time interval between saved frames in picoseconds."""
    return float(traj.get("dt_ps", 0.001)) * float(traj.get("dump_freq", 1000))


def _species_mask(traj: dict, species: Optional[str]) -> np.ndarray:
    """Return a boolean atom mask selecting *species*; selects all atoms when None."""
    if species is None:
        return np.ones(len(traj["species"]), dtype=bool)
    return np.asarray(traj["species"]) == species


def _box_lengths(traj: dict) -> Optional[np.ndarray]:
    """Box lengths per frame, shape (n_frames, 3)."""
    box = traj.get("box")
    if box is None:
        return None
    box = np.asarray(box, dtype=np.float64)
    if box.ndim == 3:
        return box[:, :, 1] - box[:, :, 0]
    if box.ndim == 2 and box.shape[1] == 2:
        lengths = box[:, 1] - box[:, 0]
        n_frames = traj["positions"].shape[0]
        return np.tile(lengths, (n_frames, 1))
    return None


def _wrap_positions(pos: np.ndarray, box_lengths: np.ndarray) -> np.ndarray:
    """
    Map unwrapped coordinates into [0, L) using modulo (orthorhombic).
    pos : (n, 3), box_lengths : (3,)
    """
    return pos % box_lengths


def _nearest_neighbour_distance(pos: np.ndarray) -> float:
    """
    Mean nearest-neighbour distance for a set of atomic positions.
    Uses a KDTree for efficiency; queries the 2nd neighbour (k=2) because
    the 1st is the atom itself.
    """
    if pos.shape[0] < 2:
        return 1.0
    tree = KDTree(pos)
    dist, _ = tree.query(pos, k=2)
    return float(dist[:, 1].mean())


# ---------------------------------------------------------------------------
# Q6 bond-orientational order: spherical harmonics Y_6^m
# Implemented numerically without special-function libraries.
# ---------------------------------------------------------------------------

def _cart_to_sph(dx: np.ndarray, dy: np.ndarray, dz: np.ndarray):
    """
    Convert Cartesian displacement vectors to (theta, phi) spherical angles.
    Returns theta [0, pi], phi [0, 2pi].
    """
    r     = np.sqrt(dx**2 + dy**2 + dz**2)
    r     = np.where(r == 0, 1e-12, r)
    theta = np.arccos(np.clip(dz / r, -1.0, 1.0))
    phi   = np.arctan2(dy, dx) % (2 * np.pi)
    return theta, phi


def _Y6m(m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Real-valued spherical harmonics Y_6^m (using analytic expressions).
    Only |m| <= 6 implemented; returns complex array consistent with
    the standard Q6 definition.

    We use the scipy-free recurrence approach: express Y_l^m via
    associated Legendre polynomials P_l^|m| computed via recurrence.
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Compute P_6^|m|(cos θ) via recurrence up to order 6
    abs_m = abs(m)

    # Start from P_m^m = (-1)^m (2m-1)!! sin^m(θ)
    Pmm = np.ones_like(cos_t)
    for k in range(1, abs_m + 1):
        Pmm *= -(2 * k - 1) * sin_t

    if abs_m == 6:
        Plm = Pmm
    else:
        # Pmm1 = P_{m+1}^m = cos(θ)(2m+1) P_m^m
        Pmm1 = cos_t * (2 * abs_m + 1) * Pmm
        if abs_m == 5:
            Plm = Pmm1
        else:
            # Recurse from P_m^m and P_{m+1}^m up to P_6^m
            l_prev2 = Pmm
            l_prev1 = Pmm1
            for l in range(abs_m + 2, 7):
                l_curr = (
                    (2 * l - 1) * cos_t * l_prev1 - (l + abs_m - 1) * l_prev2
                ) / (l - abs_m)
                l_prev2 = l_prev1
                l_prev1 = l_curr
            Plm = l_prev1

    # Normalisation constant
    l = 6
    from math import factorial, sqrt, pi
    if m == 0:
        norm = sqrt((2 * l + 1) / (4 * pi))
    else:
        numerator   = (2 * l + 1) * factorial(l - abs_m)
        denominator = 4 * pi * factorial(l + abs_m)
        norm = sqrt(numerator / denominator)
        if m < 0:
            norm *= (-1) ** abs_m

    Y_real = norm * Plm * np.cos(m * phi)
    Y_imag = norm * Plm * np.sin(m * phi)
    return Y_real + 1j * Y_imag


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lindemann_criterion(
    traj: dict,
    species: Optional[str] = None,
    skip_frac: float = 0.2,
) -> dict:
    """
    Compute the Lindemann melting criterion:

        δ_L = sqrt(<u²>) / d_nn

    where <u²> is the mean-squared displacement from the atom's
    time-averaged position and d_nn is the mean nearest-neighbour distance.

    δ_L > 0.10–0.15 indicates melting or significant disordering.

    Returns dict with keys:
        lindemann_per_frame  : np.ndarray (n_use_frames,)  per-frame δ_L
        mean_lindemann       : float
        d_nn_A               : float   mean nearest-neighbour distance in Å
        status               : str     "ordered" | "disordered" | "melting"
        species              : str
        threshold            : float   0.12 (mid-point of accepted range)
    """
    if not traj or "positions" not in traj:
        return {}

    try:
        positions = np.asarray(traj["positions"], dtype=np.float64)    # (F, N, 3)
        n_frames_total, n_atoms, _ = positions.shape

        mask = _species_mask(traj, species)
        if not mask.any():
            return {}

        skip  = int(n_frames_total * skip_frac)
        pos   = positions[skip:][:, mask, :]                           # (F_use, n_sp, 3)
        n_use = pos.shape[0]

        if n_use < 2:
            return {}

        # Reference positions: time-averaged
        pos_mean = pos.mean(axis=0)                                    # (n_sp, 3)

        # Displacement from mean (unwrapped coords — no PBC wrapping needed)
        disp  = pos - pos_mean[None, :, :]                            # (F_use, n_sp, 3)
        u2    = (disp ** 2).sum(axis=-1)                               # (F_use, n_sp)
        msd_f = u2.mean(axis=1)                                        # (F_use,) mean over atoms
        delta_per_frame = np.sqrt(msd_f)                               # rms displacement, per frame

        # Nearest-neighbour distance from mean positions
        d_nn = _nearest_neighbour_distance(pos_mean)

        lindemann_per_frame = delta_per_frame / d_nn if d_nn > 0 else delta_per_frame
        mean_lind = float(lindemann_per_frame.mean())

        threshold = 0.12
        if mean_lind < 0.08:
            status = "ordered"
        elif mean_lind < threshold:
            status = "pre-melting"
        else:
            status = "melting/disordered"

        return {
            "lindemann_per_frame" : lindemann_per_frame,
            "mean_lindemann"      : mean_lind,
            "d_nn_A"              : float(d_nn),
            "status"              : status,
            "species"             : species or "all",
            "threshold"           : threshold,
        }

    except Exception:
        return {}


# ---------------------------------------------------------------------------

def order_parameter(
    traj: dict,
    frame_idx: int,
    reference_frame: int = 0,
    n_neighbors: int = 12,
    r_cutoff: float = 4.0,
) -> dict:
    """
    Compute the bond-orientational order parameter Q6 for each atom in
    `frame_idx`, using up to `n_neighbors` nearest neighbours within
    `r_cutoff` Å.

    Q6_i = sqrt(4π/13 * Σ_m |q_6m(i)|²)
    where q_6m(i) = (1/N_b) Σ_j Y_6^m(r_ij)

    High Q6 (~0.57 for FCC, ~0.51 for HCP) indicates crystalline order;
    low Q6 (~0.28) indicates amorphous/liquid.

    Returns dict with keys:
        Q6_per_atom     : np.ndarray (n_atoms,)
        mean_Q6         : float
        std_Q6          : float
        frame_idx       : int
        reference_Q6    : float   Q6 of reference frame (for comparison)
    """
    if not traj or "positions" not in traj:
        return {}

    try:
        positions = np.asarray(traj["positions"], dtype=np.float64)
        n_frames  = positions.shape[0]

        if frame_idx >= n_frames or reference_frame >= n_frames:
            return {}

        def _frame_Q6(fidx: int) -> np.ndarray:
            """Compute the per-atom Q6 bond-order parameter for frame *fidx*."""
            pos = positions[fidx]                                       # (N, 3)
            box_l = _box_lengths(traj)
            L = box_l[fidx] if box_l is not None else None

            n_atoms = pos.shape[0]
            if L is not None:
                pos_w = _wrap_positions(pos, L)
            else:
                pos_w = pos

            tree = KDTree(pos_w, boxsize=L)
            _, neighbours = tree.query(pos_w, k=min(n_neighbors + 1, n_atoms))

            Q6_atoms = np.zeros(n_atoms)
            for i in range(n_atoms):
                nb = neighbours[i][1:]                                 # exclude self
                nb = nb[nb < n_atoms]

                if len(nb) == 0:
                    continue

                dx = pos_w[nb, 0] - pos_w[i, 0]
                dy = pos_w[nb, 1] - pos_w[i, 1]
                dz = pos_w[nb, 2] - pos_w[i, 2]

                # Minimum image
                if L is not None:
                    dx -= L[0] * np.round(dx / L[0])
                    dy -= L[1] * np.round(dy / L[1])
                    dz -= L[2] * np.round(dz / L[2])

                theta, phi = _cart_to_sph(dx, dy, dz)

                q6m_sum = np.zeros(13, dtype=complex)
                for idx_m, m in enumerate(range(-6, 7)):
                    Ylm = _Y6m(m, theta, phi)
                    q6m_sum[idx_m] = Ylm.mean()

                Q6_atoms[i] = np.sqrt(
                    (4 * np.pi / 13) * np.sum(np.abs(q6m_sum) ** 2)
                )

            return Q6_atoms

        Q6_frame = _frame_Q6(frame_idx)
        Q6_ref   = _frame_Q6(reference_frame)

        return {
            "Q6_per_atom"   : Q6_frame,
            "mean_Q6"       : float(Q6_frame.mean()),
            "std_Q6"        : float(Q6_frame.std()),
            "frame_idx"     : frame_idx,
            "reference_Q6"  : float(Q6_ref.mean()),
        }

    except Exception:
        return {}


# ---------------------------------------------------------------------------

def phase_fraction(
    traj: dict,
    threshold_lindemann: float = 0.12,
    species: Optional[str] = None,
    skip_frac: float = 0.2,
) -> dict:
    """
    Estimate crystalline vs amorphous phase fraction over time using the
    per-frame Lindemann parameter.

    f_crystal(t) = 1  if δ_L(t) < threshold  (frame-level classification)
    f_amorphous  = 1 - f_crystal

    Returns dict with keys:
        times_ns     : np.ndarray
        f_crystal    : np.ndarray  (fraction of atoms per frame in ordered state)
        f_amorphous  : np.ndarray
        lindemann    : np.ndarray  (raw per-frame δ_L for reference)
        threshold    : float
    """
    if not traj or "positions" not in traj:
        return {}

    try:
        lind_data = lindemann_criterion(traj, species=species, skip_frac=skip_frac)
        if not lind_data:
            return {}

        lind_f  = lind_data["lindemann_per_frame"]
        n_use   = len(lind_f)
        dt_ns   = _dt_frame_ps(traj) * 1e-3

        skip    = int(traj["positions"].shape[0] * skip_frac)
        times   = (np.arange(n_use) + skip) * dt_ns

        f_crystal   = (lind_f < threshold_lindemann).astype(float)
        f_amorphous = 1.0 - f_crystal

        # Smooth with a rolling window (5% of frames) for display
        win = max(3, int(n_use * 0.05))
        f_crystal_smooth   = uniform_filter1d(f_crystal,   size=win)
        f_amorphous_smooth = uniform_filter1d(f_amorphous, size=win)

        return {
            "times_ns"           : times,
            "f_crystal"          : f_crystal_smooth,
            "f_amorphous"        : f_amorphous_smooth,
            "f_crystal_raw"      : f_crystal,
            "lindemann"          : lind_f,
            "threshold"          : float(threshold_lindemann),
        }

    except Exception:
        return {}


# ---------------------------------------------------------------------------

def detect_phase_transitions(
    traj: dict,
    window_size: int = 50,
    skip_frac: float = 0.2,
) -> dict:
    """
    Detect potential phase transition frames by identifying discontinuities
    in potential energy and/or the running MSD slope.

    Method:
      1. Compute rolling-window variance of the potential energy.
         Spikes in variance suggest rapid structural rearrangements.
      2. Compute rolling MSD slope and flag frames where the slope changes
         by more than 2σ (change-point detection).
      3. Combine signals; assign confidence = normalised combined z-score.

    Returns dict with keys:
        transition_frames   : list[int]
        transition_types    : list[str]    "energy" | "msd_slope" | "both"
        confidence_scores   : list[float]  [0, 1]
        times_ns            : np.ndarray
        energy_variance     : np.ndarray   rolling energy variance (if available)
        msd_slope           : np.ndarray   rolling MSD slope
    """
    if not traj or "positions" not in traj:
        return {}

    try:
        positions = np.asarray(traj["positions"], dtype=np.float64)
        n_frames, n_atoms, _ = positions.shape
        dt_ns = _dt_frame_ps(traj) * 1e-3
        times_ns = np.arange(n_frames) * dt_ns

        skip = int(n_frames * skip_frac)

        # --- Energy signal ---
        energies = traj.get("energies")
        e_var_signal = np.zeros(n_frames)
        if energies is not None:
            en = np.asarray(energies, dtype=np.float64)
            for i in range(window_size, n_frames):
                win = en[i - window_size:i]
                e_var_signal[i] = float(win.var())

        # --- MSD slope signal (all atoms) ---
        # Use a centred displacement from the start of each window
        msd_slope_signal = np.zeros(n_frames)
        for i in range(window_size, n_frames):
            f0  = i - window_size
            pos_w = positions[f0:i]                                    # (W, N, 3)
            disp  = pos_w - pos_w[0:1]                                 # (W, N, 3)
            msd   = (disp**2).sum(axis=-1).mean(axis=1)                # (W,)
            t_w   = np.arange(window_size) * dt_ns
            if t_w.std() > 0:
                sl, *_ = stats.linregress(t_w, msd)
                msd_slope_signal[i] = max(0.0, float(sl))

        # --- Detect spikes via z-score ---
        def _zscores(arr: np.ndarray, skip: int) -> np.ndarray:
            """Return absolute z-scores relative to the tail region starting at *skip*."""
            region = arr[skip:]
            mu, sigma = region.mean(), region.std()
            if sigma == 0:
                return np.zeros_like(arr)
            return np.abs((arr - mu) / (sigma + 1e-30))

        z_energy = _zscores(e_var_signal, skip)
        z_msd    = _zscores(msd_slope_signal, skip)

        threshold_z = 2.5

        energy_flag  = (z_energy > threshold_z) & (np.arange(n_frames) >= skip)
        msd_flag     = (z_msd    > threshold_z) & (np.arange(n_frames) >= skip)
        combined     = energy_flag | msd_flag

        transition_frames  = []
        transition_types   = []
        confidence_scores  = []

        # Merge nearby frames (within window_size // 2)
        merge_gap  = window_size // 2
        last_added = -merge_gap * 2

        candidate_frames = np.where(combined)[0].tolist()
        for fi in candidate_frames:
            if fi - last_added < merge_gap:
                continue
            t_type = []
            if energy_flag[fi]:
                t_type.append("energy")
            if msd_flag[fi]:
                t_type.append("msd_slope")
            tt = "+".join(t_type) if t_type else "unknown"

            conf_raw = float(max(z_energy[fi], z_msd[fi]))
            conf = min(1.0, conf_raw / (threshold_z * 4))

            transition_frames.append(int(fi))
            transition_types.append(tt)
            confidence_scores.append(conf)
            last_added = fi

        return {
            "transition_frames"  : transition_frames,
            "transition_types"   : transition_types,
            "confidence_scores"  : confidence_scores,
            "times_ns"           : times_ns,
            "energy_variance"    : e_var_signal,
            "msd_slope"          : msd_slope_signal,
        }

    except Exception:
        return {}


# ---------------------------------------------------------------------------

def ion_hopping_analysis(
    positions: np.ndarray,
    dt_ps: float,
    lattice_sites: Optional[np.ndarray] = None,
    hop_threshold_A: float = 1.5,
    min_dwell_frames: int = 5,
) -> dict:
    """
    Detect discrete hop events for a set of mobile-ion trajectories.

    If `lattice_sites` is provided (shape (n_sites, 3)):
        - Assign each ion to its nearest site per frame.
        - A hop = ion changes site assignment and stays at new site for
          at least `min_dwell_frames`.
    If no lattice_sites:
        - Use instantaneous displacement > `hop_threshold_A` relative to
          the previous frame as a proxy hop event.

    Args:
        positions        : (n_frames, n_ions, 3) unwrapped positions in Å
        dt_ps            : ps per saved frame
        lattice_sites    : (n_sites, 3) optional crystallographic sites
        hop_threshold_A  : displacement threshold for no-lattice mode
        min_dwell_frames : minimum frames at a site to confirm a hop

    Returns dict with keys:
        hop_events          : list of dicts with keys
                              {ion_id, frame, t_ps, from_site, to_site,
                               displacement_A}
        hop_rate_per_ns     : float   total hops / ns / ion
        mean_hop_distance_A : float
        n_hops              : int
    """
    if positions is None or positions.ndim != 3 or positions.shape[0] < 2:
        return {}

    try:
        positions = np.asarray(positions, dtype=np.float64)
        n_frames, n_ions, _ = positions.shape
        total_time_ns = n_frames * dt_ps * 1e-3

        hop_events: list[dict] = []

        if lattice_sites is not None:
            # --- Site-assignment mode ---
            sites = np.asarray(lattice_sites, dtype=np.float64)        # (n_sites, 3)
            n_sites = sites.shape[0]

            # Assign every ion to nearest site per frame using broadcasting
            # (n_frames, n_ions, n_sites) — potentially large; use loop over frames
            # but fully vectorised per frame
            site_assignments = np.zeros((n_frames, n_ions), dtype=np.int32)
            for fi in range(n_frames):
                diff = positions[fi, :, None, :] - sites[None, :, :]   # (n_ions, n_sites, 3)
                d2   = (diff**2).sum(axis=-1)                           # (n_ions, n_sites)
                site_assignments[fi] = d2.argmin(axis=1)

            # Detect hop events with dwell-time filter
            for ion in range(n_ions):
                sa = site_assignments[:, ion]                           # (n_frames,)
                for fi in range(1, n_frames):
                    if sa[fi] != sa[fi - 1]:
                        new_site = sa[fi]
                        # Check dwell: minimum `min_dwell_frames` at new site
                        dwell_end = min(fi + min_dwell_frames, n_frames)
                        if np.all(sa[fi:dwell_end] == new_site):
                            disp = float(np.linalg.norm(
                                positions[fi, ion] - positions[fi - 1, ion]
                            ))
                            hop_events.append({
                                "ion_id"       : int(ion),
                                "frame"        : int(fi),
                                "t_ps"         : float(fi * dt_ps),
                                "from_site"    : int(sa[fi - 1]),
                                "to_site"      : int(new_site),
                                "displacement_A": disp,
                            })

        else:
            # --- Displacement threshold mode ---
            disp_vec  = np.diff(positions, axis=0)                     # (F-1, n_ions, 3)
            disp_mag  = np.linalg.norm(disp_vec, axis=-1)              # (F-1, n_ions)

            hop_mask  = disp_mag > hop_threshold_A                     # (F-1, n_ions)
            f_idx, i_idx = np.where(hop_mask)

            for k in range(len(f_idx)):
                fi  = int(f_idx[k])
                ion = int(i_idx[k])
                hop_events.append({
                    "ion_id"        : ion,
                    "frame"         : fi + 1,
                    "t_ps"          : float((fi + 1) * dt_ps),
                    "from_site"     : -1,
                    "to_site"       : -1,
                    "displacement_A": float(disp_mag[fi, ion]),
                })

        n_hops = len(hop_events)
        hop_rate = (n_hops / max(total_time_ns, 1e-12)) / max(n_ions, 1)

        displacements = np.array([e["displacement_A"] for e in hop_events])
        mean_hop_dist = float(displacements.mean()) if displacements.size > 0 else 0.0

        return {
            "hop_events"          : hop_events,
            "hop_rate_per_ns"     : float(hop_rate),
            "mean_hop_distance_A" : mean_hop_dist,
            "n_hops"              : n_hops,
        }

    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------

def plot_lindemann(lindemann_data: dict, project: str, T_K: int) -> go.Figure:
    """
    Plot Lindemann parameter δ_L vs time with a dashed threshold line at 0.12.
    Colours the trace red beyond the threshold.

    Args:
        lindemann_data : output of lindemann_criterion()
        project        : project label for title
        T_K            : temperature in Kelvin
    Returns:
        go.Figure
    """
    if not lindemann_data:
        return go.Figure()

    lf   = lindemann_data.get("lindemann_per_frame", np.array([]))
    thresh = lindemann_data.get("threshold", 0.12)
    sp   = lindemann_data.get("species", "all")
    status = lindemann_data.get("status", "")

    if lf.size == 0:
        return go.Figure()

    dt_ns   = 0.001   # default 1 ps/frame in ns; caller should override if needed
    times   = np.arange(len(lf)) * dt_ns

    # Split into ordered / disordered segments for colour coding
    ordered_mask    = lf <  thresh
    disordered_mask = lf >= thresh

    fig = go.Figure()

    # Ordered (below threshold) — blue
    fig.add_trace(go.Scatter(
        x=times, y=np.where(ordered_mask, lf, np.nan),
        mode="lines",
        name=f"δ_L (ordered, {sp})",
        line=dict(color="#1f77b4", width=1.5),
        connectgaps=False,
    ))

    # Disordered (above threshold) — red
    fig.add_trace(go.Scatter(
        x=times, y=np.where(disordered_mask, lf, np.nan),
        mode="lines",
        name=f"δ_L (disordered, {sp})",
        line=dict(color="#d62728", width=1.5),
        connectgaps=False,
    ))

    fig.add_hline(
        y=thresh,
        line=dict(color="black", dash="dash", width=1.5),
        annotation_text=f"Threshold δ_L = {thresh}",
        annotation_position="top right",
    )

    mean_l = lindemann_data.get("mean_lindemann", float("nan"))
    fig.add_annotation(
        x=0.02, y=0.95, xref="paper", yref="paper",
        text=f"Mean δ_L = {mean_l:.4f}   Status: {status}",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray", borderwidth=1,
        font=dict(size=12),
    )

    fig.update_layout(
        title=dict(text=f"{project} — Lindemann Criterion at {T_K} K", font_size=16),
        xaxis=dict(title="Time (ns)", showgrid=True, gridcolor="#eee"),
        yaxis=dict(title="Lindemann parameter δ_L", showgrid=True, gridcolor="#eee"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=13),
        legend=dict(x=0.01, y=0.85),
    )
    return fig


# ---------------------------------------------------------------------------

def plot_phase_fraction(phase_data: dict, project: str) -> go.Figure:
    """
    Stacked area chart showing crystalline vs amorphous fraction vs time.
    Also overlays the raw Lindemann parameter on a secondary y-axis.

    Args:
        phase_data : output of phase_fraction()
        project    : project label for title
    Returns:
        go.Figure
    """
    if not phase_data:
        return go.Figure()

    times     = phase_data.get("times_ns",    np.array([]))
    f_crys    = phase_data.get("f_crystal",   np.array([]))
    f_amor    = phase_data.get("f_amorphous", np.array([]))
    lind      = phase_data.get("lindemann",   np.array([]))
    threshold = phase_data.get("threshold",   0.12)

    if times.size == 0:
        return go.Figure()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([times, times[::-1]]),
            y=np.concatenate([f_crys, np.zeros(len(times))]),
            fill="toself",
            mode="none",
            fillcolor="rgba(31,119,180,0.5)",
            name="Crystalline",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([times, times[::-1]]),
            y=np.concatenate([np.ones(len(times)), f_crys[::-1]]),
            fill="toself",
            mode="none",
            fillcolor="rgba(214,39,40,0.4)",
            name="Amorphous",
        ),
        secondary_y=False,
    )

    if lind.size > 0:
        fig.add_trace(
            go.Scatter(
                x=times, y=lind,
                mode="lines",
                name="δ_L (Lindemann)",
                line=dict(color="black", width=1, dash="dot"),
                opacity=0.7,
            ),
            secondary_y=True,
        )
        fig.add_hline(
            y=threshold,
            line=dict(color="gray", dash="dash", width=1),
            secondary_y=True,
            annotation_text=f"δ_L threshold = {threshold}",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title=dict(text=f"{project} — Phase Fraction vs Time", font_size=16),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=13),
        legend=dict(x=0.01, y=0.5),
    )
    fig.update_yaxes(
        title_text="Phase fraction", secondary_y=False,
        range=[0, 1.05], showgrid=True, gridcolor="#eee",
    )
    fig.update_yaxes(
        title_text="Lindemann δ_L", secondary_y=True, showgrid=False,
    )
    fig.update_xaxes(title_text="Time (ns)", showgrid=True, gridcolor="#eee")
    return fig


# ---------------------------------------------------------------------------

def plot_ion_hopping(
    hop_data: dict,
    positions: np.ndarray,
    project: str,
    dt_ps: float = 1.0,
) -> go.Figure:
    """
    3D scatter plot of ion hop displacement vectors, coloured by hop time.
    Each arrow originates at the pre-hop position and points in the direction
    of the hop.

    Args:
        hop_data   : output of ion_hopping_analysis()
        positions  : (n_frames, n_ions, 3) positions array
        project    : project label for title
        dt_ps      : ps per frame (used to convert frame index to time)
    Returns:
        go.Figure
    """
    if not hop_data or positions is None:
        return go.Figure()

    events = hop_data.get("hop_events", [])
    if not events:
        return go.Figure()

    try:
        positions = np.asarray(positions, dtype=np.float64)
        n_frames, n_ions, _ = positions.shape

        # Collect origin and vector for each hop
        ox, oy, oz   = [], [], []
        vx, vy, vz   = [], [], []
        times_ps      = []
        displacements = []

        for ev in events:
            fi  = ev["frame"]
            ion = ev["ion_id"]
            if fi < 1 or fi >= n_frames or ion >= n_ions:
                continue

            orig = positions[fi - 1, ion]
            dest = positions[fi,     ion]
            vec  = dest - orig

            ox.append(float(orig[0]))
            oy.append(float(orig[1]))
            oz.append(float(orig[2]))
            vx.append(float(vec[0]))
            vy.append(float(vec[1]))
            vz.append(float(vec[2]))
            times_ps.append(ev["t_ps"])
            displacements.append(ev["displacement_A"])

        if not ox:
            return go.Figure()

        ox = np.array(ox);  oy = np.array(oy);  oz = np.array(oz)
        vx = np.array(vx);  vy = np.array(vy);  vz = np.array(vz)
        t_ps = np.asarray(times_ps)
        disp = np.asarray(displacements)

        # Build cone / scatter: use scatter3d at origin + destination
        fig = go.Figure()

        # Origins coloured by time
        fig.add_trace(go.Scatter3d(
            x=ox, y=oy, z=oz,
            mode="markers",
            marker=dict(
                size=4,
                color=t_ps,
                colorscale="Viridis",
                colorbar=dict(title="Time (ps)", x=1.02),
                opacity=0.7,
            ),
            name="Hop origin",
            hovertemplate="(%{x:.1f}, %{y:.1f}, %{z:.1f})<br>t=%{marker.color:.1f} ps<extra></extra>",
        ))

        # Lines from origin to destination
        for k in range(len(ox)):
            fig.add_trace(go.Scatter3d(
                x=[ox[k], ox[k] + vx[k]],
                y=[oy[k], oy[k] + vy[k]],
                z=[oz[k], oz[k] + vz[k]],
                mode="lines",
                line=dict(
                    color=f"rgba({int(30+220*(t_ps[k]-t_ps.min())/(t_ps.ptp()+1e-9))},100,200,0.5)",
                    width=1.5,
                ),
                showlegend=False,
                hoverinfo="skip",
            ))

        n_hops   = hop_data.get("n_hops", len(events))
        hop_rate = hop_data.get("hop_rate_per_ns", 0.0)
        mean_d   = hop_data.get("mean_hop_distance_A", 0.0)

        fig.update_layout(
            title=dict(
                text=(
                    f"{project} — Ion Hop Vectors<br>"
                    f"<sup>N={n_hops} hops | "
                    f"rate={hop_rate:.2f}/ns/ion | "
                    f"mean Δr={mean_d:.2f} Å</sup>"
                ),
                font_size=15,
            ),
            scene=dict(
                xaxis=dict(title="x (Å)"),
                yaxis=dict(title="y (Å)"),
                zaxis=dict(title="z (Å)"),
                bgcolor="white",
            ),
            paper_bgcolor="white",
            font=dict(family="Arial", size=12),
            showlegend=False,
        )
        return fig

    except Exception:
        return go.Figure()
