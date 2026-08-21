"""
hpca.core.status
~~~~~~~~~~~~~~~~
Query Slurm + filesystem to build a status report for any project.
Writes logs/status.yaml and logs/status.txt inside the project directory.

Usage:
    from hpca.core.status import check_project, check_all
    check_project("/path/to/workspace/LMZC")
    check_all("/path/to/workspace", pattern="*/project.yaml")

CLI:
    python -m hpca.core.status --project /path/to/workspace/LMZC
    python -m hpca.core.status --all /path/to/workspace
"""
from __future__ import annotations
import subprocess, sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ─────────────────────────────────────────────────────────────────────────────
# Slurm helpers
# ─────────────────────────────────────────────────────────────────────────────

def _slurm_jobs(user: str = None) -> list[dict]:
    """Return list of dicts for all user jobs from squeue."""
    user = user or _current_user()
    try:
        out = subprocess.check_output(
            ["squeue", "-u", user,
             "--format=%i|%j|%T|%M|%R|%Z",
             "--noheader"],
            text=True, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    jobs = []
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 6:
            continue
        jobs.append({
            "job_id":   parts[0].strip(),
            "name":     parts[1].strip(),
            "state":    parts[2].strip(),
            "elapsed":  parts[3].strip(),
            "reason":   parts[4].strip(),
            "workdir":  parts[5].strip(),
        })
    return jobs


def _current_user() -> str:
    """Return the current OS username from the USER environment variable."""
    import os
    return os.environ.get("USER", "unknown")


def _jobs_for_project(proj_root: Path, all_jobs: list[dict]) -> list[dict]:
    """Filter jobs whose workdir is under proj_root."""
    root_str = str(proj_root.resolve())
    return [j for j in all_jobs
            if j["workdir"].startswith(root_str) or
               j["workdir"].replace("/kfs2", "").startswith(root_str.replace("/kfs2", ""))]


# ─────────────────────────────────────────────────────────────────────────────
# Filesystem phase detection
# ─────────────────────────────────────────────────────────────────────────────

_PHASE_CHECKS = [
    # (phase_name, relative_path_that_must_exist, label_if_missing)
    ("structure_prep",  "opt/CONTCAR",              "CONTCAR"),
    ("vc_relax",        "vc/CONTCAR",               "CONTCAR"),
    ("bader",           "bader/ACF.dat",            "ACF.dat"),
    ("dos",             "dos/nonscf/DOSCAR",        "DOSCAR"),
    ("neb",             "neb/path_c1/OUTCAR",       "NEB OUTCAR"),
    ("deepmd_train",    "mlff/pot_com.pb",          "pot_com.pb"),
    ("results_msd",     "results/transport",        "transport/"),
    ("results_continuum","results/continuum",       "continuum/"),
    ("manuscript",      "results/manuscript",       "manuscript/"),
]


def _detect_aimd(proj_root: Path, aimd_dirs: list[str]) -> dict:
    """Return a dict mapping each AIMD directory name to its completion status."""
    phases = {}
    for d in aimd_dirs:
        p = proj_root / d
        outcar = p / "OUTCAR"
        xdat   = p / "XDATCAR"
        if xdat.exists():
            phases[d] = "done"
        elif outcar.exists():
            phases[d] = "running_or_partial"
        else:
            phases[d] = "missing"
    return phases


def _detect_mlmd(proj_root: Path, mlmd_dirs: dict) -> dict:
    """Return a dict mapping each MLMD temperature key to its completion status."""
    phases = {}
    for T, d in mlmd_dirs.items():
        p = proj_root / d
        dump = p / "dump_unwrapped.lmp"
        log  = p / "log.lammps"
        if dump.exists():
            phases[str(T)] = "done"
        elif log.exists():
            phases[str(T)] = "running_or_partial"
        else:
            phases[str(T)] = "missing"
    return phases


def _detect_static(proj_root: Path) -> dict:
    """Return a dict mapping each static phase name to 'done' or 'missing'."""
    phases = {}
    for name, rel_path, _ in _PHASE_CHECKS:
        phases[name] = "done" if (proj_root / rel_path).exists() else "missing"
    return phases


# ─────────────────────────────────────────────────────────────────────────────
# Main status builder
# ─────────────────────────────────────────────────────────────────────────────

def check_project(proj_dir: str | Path,
                  all_slurm_jobs: list[dict] = None,
                  write: bool = True) -> dict:
    """
    Build a status dict for one project and optionally write logs/status.yaml.

    Returns:
        {
          "project": "LMZC",
          "timestamp": "2026-06-09T12:00:00",
          "active_jobs": [...],
          "phases": { "aimd": {...}, "mlmd": {...}, "static": {...} },
          "summary": "2 jobs RUNNING | aimd:5/6 done | mlmd:3/6 done"
        }
    """
    proj_root = Path(proj_dir).resolve()
    yaml_path = proj_root / "project.yaml"

    config: dict = {}
    if HAS_YAML and yaml_path.exists():
        with open(yaml_path) as f:
            config = yaml.safe_load(f) or {}

    name      = config.get("name", proj_root.name)
    aimd_dirs = config.get("aimd_dirs", [])
    mlmd_dirs = config.get("mlmd_dirs", {})
    # Handle integer keys from yaml
    mlmd_dirs = {str(k): v for k, v in mlmd_dirs.items()}

    # Slurm
    if all_slurm_jobs is None:
        all_slurm_jobs = _slurm_jobs()
    active = _jobs_for_project(proj_root, all_slurm_jobs)

    # Filesystem phases
    aimd_phases   = _detect_aimd(proj_root, aimd_dirs)
    mlmd_phases   = _detect_mlmd(proj_root, mlmd_dirs)
    static_phases = _detect_static(proj_root)

    # Summary line
    n_aimd_done  = sum(1 for v in aimd_phases.values()  if v == "done")
    n_mlmd_done  = sum(1 for v in mlmd_phases.values()  if v == "done")
    n_run        = sum(1 for j in active if j["state"] == "RUNNING")
    n_pend       = sum(1 for j in active if j["state"] == "PENDING")
    summary_parts = []
    if n_run:  summary_parts.append(f"{n_run} RUNNING")
    if n_pend: summary_parts.append(f"{n_pend} PENDING")
    if aimd_dirs: summary_parts.append(f"aimd:{n_aimd_done}/{len(aimd_dirs)}")
    if mlmd_dirs: summary_parts.append(f"mlmd:{n_mlmd_done}/{len(mlmd_dirs)}")
    summary = " | ".join(summary_parts) or "idle"

    status = {
        "project":     name,
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "summary":     summary,
        "active_jobs": active,
        "phases": {
            "aimd":   aimd_phases,
            "mlmd":   mlmd_phases,
            "static": static_phases,
        },
        "config": {
            "D_best": _best_D(config),
            "Ea_best": _best_Ea(config),
            "category": config.get("category", "unknown"),
        },
    }

    if write:
        logs_dir = proj_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        if HAS_YAML:
            with open(logs_dir / "status.yaml", "w") as f:
                yaml.dump(status, f, default_flow_style=False)
        # Human-readable txt
        _write_txt(status, logs_dir / "status.txt")

    return status


def check_all(search_root: str | Path,
              pattern: str = "*/project.yaml",
              write: bool = True) -> list[dict]:
    """Scan search_root for project.yaml files and check each project.

    Always scans both one level deep (*/project.yaml) and two levels deep
    (**/project.yaml) so projects nested under group directories like _BHNL/
    are always included.
    """
    root = Path(search_root)
    # Always combine one- and two-level scans; dedup via dict keyed on path
    seen: dict[Path, None] = {}
    if pattern != "*/project.yaml":
        for p in sorted(root.glob(pattern)):
            seen[p] = None
    for p in sorted(root.glob("*/project.yaml")):
        seen[p] = None
    for p in sorted(root.glob("*/*/project.yaml")):
        seen[p] = None
    yamls = list(seen.keys())
    if not yamls:
        print(f"No project.yaml found under {root}")
        return []

    all_jobs = _slurm_jobs()
    results  = []
    for y in yamls:
        try:
            s = check_project(y.parent, all_slurm_jobs=all_jobs, write=write)
            results.append(s)
        except Exception as e:
            print(f"  ERROR {y.parent}: {e}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print
# ─────────────────────────────────────────────────────────────────────────────

def _best_D(config: dict) -> Optional[float]:
    """Return the best available diffusivity from MLMD or AIMD config entries."""
    d = config.get("D_mlmd", {})
    if isinstance(d, dict) and d:
        return min(d.values())
    return config.get("D_aimd")


def _best_Ea(config: dict) -> Optional[float]:
    """Return the best available activation energy from MLMD or AIMD config entries."""
    e = config.get("Ea_mlmd", {})
    if isinstance(e, dict) and e:
        return list(e.values())[0]
    return config.get("Ea_aimd")


def _write_txt(status: dict, path: Path):
    """Write a human-readable plain-text status report to path."""
    lines = [
        f"{'='*60}",
        f"Project : {status['project']}",
        f"Updated : {status['timestamp']}",
        f"Summary : {status['summary']}",
        f"{'='*60}",
    ]
    cfg = status.get("config", {})
    if cfg.get("D_best"):
        lines.append(f"D_best  : {cfg['D_best']:.2e} m²/s")
    if cfg.get("Ea_best"):
        lines.append(f"Ea_best : {cfg['Ea_best']:.3f} eV")
    lines.append(f"Category: {cfg.get('category', '?')}")
    lines.append("")

    jobs = status.get("active_jobs", [])
    if jobs:
        lines.append("Active jobs:")
        for j in jobs:
            lines.append(f"  [{j['state']:8s}] {j['job_id']:>10s}  {j['name']:<30s}  {j['elapsed']}")
        lines.append("")

    phases = status.get("phases", {})

    aimd = phases.get("aimd", {})
    if aimd:
        lines.append("AIMD:")
        for d, s in aimd.items():
            icon = "✓" if s == "done" else ("~" if "partial" in s else "✗")
            lines.append(f"  {icon} {d:<30s} {s}")
        lines.append("")

    mlmd = phases.get("mlmd", {})
    if mlmd:
        lines.append("MLMD:")
        for T, s in mlmd.items():
            icon = "✓" if s == "done" else ("~" if "partial" in s else "✗")
            lines.append(f"  {icon} {T+'K':<30s} {s}")
        lines.append("")

    static = phases.get("static", {})
    if static:
        lines.append("Static/results:")
        for name, s in static.items():
            if s == "done":
                lines.append(f"  ✓ {name}")
        missing = [n for n, s in static.items() if s == "missing"]
        if missing:
            lines.append(f"  ✗ pending: {', '.join(missing)}")

    path.write_text("\n".join(lines) + "\n")


def print_all(results: list[dict]):
    """Print a compact summary table for all projects."""
    header = f"{'Project':<20} {'Category':<16} {'D_best':>12} {'Ea':>8} {'Summary'}"
    print(header)
    print("-" * 80)
    for s in results:
        cfg  = s.get("config", {})
        D    = cfg.get("D_best")
        Ea   = cfg.get("Ea_best")
        D_s  = f"{D:.2e}" if D else "—"
        Ea_s = f"{Ea:.3f}" if Ea else "—"
        print(f"{s['project']:<20} {cfg.get('category','?'):<16} {D_s:>12} {Ea_s:>8}  {s['summary']}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Check project simulation status")
    ap.add_argument("--project", help="Path to a single project directory")
    ap.add_argument("--all",     help="Root dir to scan for all project.yaml files")
    ap.add_argument("--no-write", action="store_true",
                    help="Print only, don't write logs/status.yaml")
    args = ap.parse_args()

    write = not args.no_write

    if args.project:
        s = check_project(args.project, write=write)
        txt_path = Path(args.project) / "logs" / "status.txt"
        if txt_path.exists():
            print(txt_path.read_text())
        else:
            print(s)

    elif args.all:
        results = check_all(args.all, write=write)
        print_all(results)

    else:
        ap.print_help()
