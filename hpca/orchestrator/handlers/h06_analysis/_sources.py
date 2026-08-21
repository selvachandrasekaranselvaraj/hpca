"""
_sources.py — Trajectory source collection helpers for AnalysisHandler.
"""
from __future__ import annotations

from pathlib import Path


def collect_cmd_sources(project_dir: Path, min_dump: int = 100_000) -> dict[int, Path]:
    """Collect CMD NVT dump files across all supported directory layouts."""
    out: dict[int, Path] = {}
    cmd_dir = project_dir / "cmd"
    if not cmd_dir.exists():
        return out
    for nvt_dump in sorted(cmd_dir.rglob("dump_unwrapped.lmp")):
        parent_name = nvt_dump.parent.name
        gp_name     = nvt_dump.parent.parent.name
        T = None
        if parent_name.endswith("K") and gp_name == "nvt":
            try: T = int(parent_name[:-1])
            except ValueError: pass
        elif gp_name == "nvt":
            try: T = int(parent_name)
            except ValueError: pass
        else:
            try: T = int(gp_name)
            except ValueError: pass
        if T is not None and nvt_dump.stat().st_size > min_dump:
            existing = out.get(T)
            if existing is None or nvt_dump.stat().st_size > existing.stat().st_size:
                out[T] = nvt_dump
    return out


def collect_sources(project_dir: Path, variant: str,
                    min_dump: int = 100_000) -> dict[int, Path]:
    """Return {T: traj_path} for the given analysis variant.

    Variants:
      cmd       — CMD NVT dumps only
      mlmd_dft  — MLMD dumps preferred over AIMD XDATCARs
      combined  — MLMD > AIMD > CMD per temperature
    """
    from hpca.core.paths import mlmd_nvt, dft_aimd
    cmd_srcs = collect_cmd_sources(project_dir, min_dump)
    mlmd: dict[int, Path] = {}
    aimd: dict[int, Path] = {}
    for T in range(200, 1500, 50):
        d = mlmd_nvt(project_dir, T) / "dump_unwrapped.lmp"
        if d.exists() and d.stat().st_size > min_dump:
            mlmd[T] = d
        x = dft_aimd(project_dir, T) / "XDATCAR"
        if x.exists():
            aimd[T] = x

    if variant == "cmd":
        return cmd_srcs

    if variant == "mlmd_dft":
        out: dict[int, Path] = {}
        for T in range(200, 1500, 50):
            if T in mlmd:
                out[T] = mlmd[T]
            elif T in aimd:
                out[T] = aimd[T]
        return out

    if variant == "combined":
        out = {}
        for T in range(200, 1500, 50):
            if T in mlmd:
                out[T] = mlmd[T]
            elif T in aimd:
                out[T] = aimd[T]
            elif T in cmd_srcs:
                out[T] = cmd_srcs[T]
        return out

    return {}


def dt_frame_ps_for(traj_path: Path, yaml: dict, plat_fn) -> float:
    """Return frame interval (ps) appropriate for the given trajectory file.

    plat_fn: callable(section, key, default) — e.g. handler.plat
    """
    if "cmd" in traj_path.parts:
        ts_fs = plat_fn("lammps_md", "timestep_fs_cmd", 2.0)
        de    = plat_fn("lammps_md", "cmd_nvt_dump_freq", 1000)
        return ts_fs * de * 1e-3
    if traj_path.name == "XDATCAR":
        return yaml.get("aimd_timestep_ps", 0.002)
    ts_fs = plat_fn("lammps_md", "timestep_fs_mlmd", 0.5)
    de    = yaml.get("dump_every", 50)
    return ts_fs * de * 1e-3
