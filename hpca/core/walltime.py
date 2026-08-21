"""
hpca/core/walltime.py — Universal walltime estimation.

Single implementation for all handlers.  Reads empirical rates and caps
from platform.yaml via Config; no hard-coded numbers here.

Usage:
    from hpca.core.walltime import vasp_walltime, aimd_walltime, cmd_walltime
    from hpca.core.walltime import mlmd_walltime, mlip_walltime, fmth
"""
from __future__ import annotations

from hpca.core.config import Config


# ── Canonical time formatter ──────────────────────────────────────────────────

def fmth(hours: float) -> str:
    """Float hours → HH:MM:00 walltime string."""
    h = int(hours)
    m = int((hours - h) * 60)
    return f"{h:02d}:{m:02d}:00"


# ── VASP / AIMD ───────────────────────────────────────────────────────────────

def vasp_walltime(natoms: int, nsw: int, job_type: str = "aimd",
                  gamma_only: bool = True, n_nodes: int = 1) -> str:
    """
    Estimate VASP walltime and return HH:MM:00 string.

    job_type: 'aimd' | 'relax' | 'vc_relax' | 'neb' | 'static' | 'bader' | 'dos'
    Rates and caps read from platform.yaml perf_model section.
    """
    cfg = Config.get()
    kp    = 1.0 if gamma_only else 3.5
    scale = 1.0 / max(n_nodes, 1)

    if job_type == "aimd":
        sps   = cfg.perf("vasp_s_per_atom_per_step_aimd") * natoms * kp * scale
        hours = (nsw * sps) / 3600 * cfg.perf("vasp_safety_factor_aimd", 1.25)
        hours = max(hours, cfg.perf("vasp_min_hours_aimd", 8.0))
        cap   = cfg.perf("vasp_max_hours_standard", 44.0)
    elif job_type in ("relax", "vc_relax", "opt", "aimd_relax"):
        sps   = cfg.perf("vasp_s_per_atom_per_step_relax") * natoms * kp * scale
        hours = (nsw * sps) / 3600 * cfg.perf("vasp_safety_factor_relax", 1.30) + 2
        hours = max(hours, cfg.perf("vasp_min_hours_relax", 4.0))
        cap   = cfg.perf("vasp_max_hours_standard", 44.0)
    elif job_type == "neb":
        sps   = cfg.perf("vasp_s_per_atom_per_step_neb") * natoms * kp * scale
        hours = (nsw * sps) / 3600 * cfg.perf("vasp_safety_factor_neb", 1.50) + 4
        hours = max(hours, cfg.perf("vasp_min_hours_neb", 12.0))
        cap   = cfg.perf("vasp_max_hours_long", 238.0)
    else:  # static, bader, dos
        sps   = cfg.perf("vasp_s_per_atom_per_step_static") * natoms * kp * scale
        hours = (nsw * sps) / 3600 * cfg.perf("vasp_safety_factor_static", 1.20) + 2
        hours = max(hours, cfg.perf("vasp_min_hours_static", 2.0))
        cap   = cfg.perf("vasp_max_hours_standard", 44.0)

    return fmth(min(hours, cap))


def aimd_walltime(natoms: int, nsw: int, gamma_only: bool = True) -> str:
    """Convenience wrapper for AIMD dataset boxes (same as vasp_walltime aimd)."""
    return vasp_walltime(natoms, nsw, job_type="aimd", gamma_only=gamma_only)


# ── LAMMPS MLMD (GPU) ─────────────────────────────────────────────────────────

def mlmd_walltime(natoms: int, n_steps: int, n_gpus: int = 4) -> str:
    """
    Estimate LAMMPS+DeepMD/MACE walltime on H100 GPUs.
    Returns HH:MM:00 string.
    """
    cfg  = Config.get()
    ref  = cfg.perf("mlmd_steps_per_hour_per_gpu", 1e9)
    sph  = max(n_gpus, 1) * ref / max(natoms, 100)
    sf   = cfg.perf("mlmd_safety_factor", 1.5)
    hours = n_steps / sph * sf
    if natoms >= 5_000 or n_steps >= 2_000_000:
        hours = max(hours, cfg.perf("mlmd_min_hours_large", 24.0))
    else:
        hours = max(hours, cfg.perf("mlmd_min_hours_small", 8.0))
    return fmth(min(hours, cfg.perf("mlmd_max_hours", 72.0)))


# ── LAMMPS CMD (CPU) ──────────────────────────────────────────────────────────

def cmd_walltime(natoms: int, n_steps: int, n_cpus: int = 104) -> str:
    """
    Estimate LAMMPS OPLS-AA CMD walltime on CPU.
    Returns HH:MM:00 string.
    """
    cfg        = Config.get()
    ns         = n_steps * 1e-6
    ns_ref     = cfg.perf("cmd_ns_per_day_reference", 4.0)
    nat_ref    = cfg.perf("cmd_natoms_reference", 50_000)
    cpu_ref    = cfg.perf("cmd_ncpus_reference", 104)
    ns_per_day = max(0.5, (n_cpus / cpu_ref) * ns_ref * (nat_ref / max(natoms, 1_000)))
    sf         = cfg.perf("cmd_safety_factor", 1.4)
    hours      = (ns / ns_per_day) * 24 * sf
    if natoms > 10_000 or n_steps >= 2_000_000:
        hours = max(hours, cfg.perf("cmd_min_hours_large", 24.0))
    else:
        hours = max(hours, cfg.perf("cmd_min_hours_small", 6.0))
    return fmth(min(hours, cfg.perf("cmd_max_hours", 72.0)))


# ── MLIP training ─────────────────────────────────────────────────────────────

def mlip_walltime(n_frames: int = 0, backend: str = "deepmd",
                  numb_steps: int = 500_000, max_epochs: int = 200) -> str:
    """
    Estimate MLIP training walltime.
    backend: 'deepmd' (CPU) | 'mace' (GPU)
    Returns HH:MM:00 string.
    """
    cfg   = Config.get()
    if backend == "deepmd":
        sph   = cfg.perf("mlip_deepmd_steps_per_hour", 10_000)
        hours = numb_steps / max(sph, 1)
        hours = max(hours, cfg.perf("mlip_min_hours", 2.0))
        cap   = cfg.perf("mlip_max_hours_cpu", 48.0)
    else:  # mace
        eph   = cfg.perf("mlip_mace_epochs_per_hour", 5)
        hours = max_epochs / max(eph, 1)
        hours = max(hours, cfg.perf("mlip_min_hours", 2.0))
        cap   = cfg.perf("mlip_max_hours_gpu", 24.0)
    return fmth(min(hours, cap))
