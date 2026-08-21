"""
cli.py — Control interface for the HPCA orchestrator daemon.

Commands:
  status  [PROJECT]          — show handler states for all (or one) project
  advance PROJECT            — run one orchestrator poll for PROJECT
  reset   PROJECT HANDLER    — reset handler to PENDING
  start                      — submit orchestrator daemon via sbatch
  stop                       — scancel the running orchestrator job
  logs    [--tail N]         — tail the latest orchestrator log
  chaai   status|trigger     — CHAAI training status and manual trigger
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

#NMCLPS    = Path.cwd() #Path("/path/to/workspace")
#ORCH_DIR  = Path(__file__).parent
#LOG_DIR   = NMCLPS / "logs"
#SUB_SCRIPT = NMCLPS / "sub_orchestrator.sh"
#ORCH_PY   = ORCH_DIR / "hpca_orchestrator.py"

ORCH_DIR   = Path(__file__).resolve().parent
_PLAT_ROOT = ORCH_DIR.parent
NMCLPS     = _PLAT_ROOT.parent.parent

LOG_DIR    = _PLAT_ROOT / "monitor" / "logs"
SUB_SCRIPT = ORCH_DIR / "sub_orchestrator.sh"


SKIP_DIRS: frozenset[str] = frozenset({
    "hpca", "manuscripts", "battery-materials-ai", "alchemi",
    "GNN", "GM", "phase_diagram", "neb_orchestrator", "pipeline",
    "apps", "_archive", "Analysis", "chemical_potentials",
    "CV", "hf_cache", "proc_opt", "proc_opt1", "vasp_data",
    "test", "__pycache__",
})

STAGE_COLORS = {
    "PENDING":  "\033[90m",   # grey
    "RUNNING":  "\033[34m",   # blue
    "COMPLETE": "\033[32m",   # green
    "FAILED":   "\033[31m",   # red
    "SKIPPED":  "\033[33m",   # yellow
}
RESET = "\033[0m"

HANDLER_ORDER = [
    "h00_design",
    "h01_dft.vc_relax", "h01_dft.opt", "h01_dft.bader",
    "h01_dft.dos_scf", "h01_dft.dos_nonscf", "h01_dft.static",
    "h02_aimd", "h03_neb",
    "h04_mlip",
    "h05_lammps",
    "h06_analysis", "h07_electronic",
    "h08_echem", "h09_continuum",
    "h10_plotting", "h11_manuscript",
    "h12_chaai",
]


def _load_state(project_dir: Path) -> dict:
    """Load orchestrator_state.json for project_dir; returns {} on missing or parse error."""
    state_file = project_dir / "logs" / "orchestrator_state.json"
    if not state_file.exists():
        return {}
    try:
        return json.loads(state_file.read_text())
    except Exception:
        return {}


def _discover_projects(root: Path, name_filter: str | None = None) -> list[Path]:
    """Return project directories under root that contain recognisable project content.

    Parameters
    ----------
    name_filter
        Optional case-insensitive substring; only matching project names are returned.
    """
    projects = []
    for p in sorted(root.iterdir()):
        if p.name.startswith(".") or p.name in SKIP_DIRS or not p.is_dir():
            continue
        if name_filter and name_filter.lower() not in p.name.lower():
            continue
        has_content = any([
            (p / "project.yaml").exists(),
            (p / "designed_structures").exists(),
            (p / "dft").exists(),
            (p / "mlmd").exists(),
            (p / "cmd").exists(),
            (p / "POSCAR").exists(),
            (p / "CONTCAR").exists(),
        ])
        if has_content:
            projects.append(p)
    return projects


def _stage_str(stage: str) -> str:
    """Return a 4-character ANSI-coloured stage label for terminal display."""
    color = STAGE_COLORS.get(stage, "")
    label = stage[:4] if stage else "PEND"
    return f"{color}{label}{RESET}"


# ── status ────────────────────────────────────────────────────────────────────
def cmd_status(args: argparse.Namespace) -> None:
    """Print a coloured handler-stage grid for all discovered projects."""
    projects = _discover_projects(NMCLPS, args.project)
    if not projects:
        print("No projects found" + (f" matching '{args.project}'" if args.project else ""))
        return

    # Header
    short_names = [h.split(".")[-1][:8] for h in HANDLER_ORDER]
    print(f"\n{'Project':<18}", end="")
    for n in short_names:
        print(f" {n:>8}", end="")
    print()
    print("-" * (18 + 9 * len(HANDLER_ORDER)))

    for proj_dir in projects:
        state = _load_state(proj_dir)
        handlers = state.get("handlers", {})
        print(f"{proj_dir.name:<18}", end="")
        for h in HANDLER_ORDER:
            stage = handlers.get(h, {}).get("stage", "PEND") if handlers else "PEND"
            label = stage[:4]
            color = STAGE_COLORS.get(stage, "")
            print(f" {color}{label:>8}{RESET}", end="")
        updated = state.get("updated", "")
        if updated:
            updated = updated[:16]
        print(f"  [{updated}]")
        # Show schema validation warnings inline
        try:
            import yaml as _yaml
            from hpca.core.project_schema import validate as _val, migrate as _mig
            _p = proj_dir / "project.yaml"
            if _p.exists():
                _data = _yaml.safe_load(_p.read_text()) or {}
                _errs = _val(_mig(_data))
                for _e in _errs:
                    print(f"  {'':18}  {STAGE_COLORS['FAILED']}SCHEMA{RESET}: {_e}")
        except Exception:
            pass

    print()
    print("Legend: PEND=Pending  RUNN=Running  COMP=Complete  FAIL=Failed  SKIP=Skipped")
    print()

    # Check if daemon is running
    _print_daemon_status()


def _print_daemon_status() -> None:
    """Query squeue and print whether the hpca_orch SLURM job is running."""
    try:
        result = subprocess.run(
            ["squeue", "-u", _get_user(), "--name=hpca_orch", "--noheader",
             "--format=%i %T %M"],
            capture_output=True, text=True, timeout=15
        )
        lines = result.stdout.strip().splitlines()
        if lines:
            print(f"Orchestrator daemon: RUNNING ({lines[0].strip()})")
        else:
            print("Orchestrator daemon: NOT RUNNING  (use 'cli.py start' to launch)")
    except Exception as exc:
        print(f"Orchestrator daemon: status unknown ({exc})")


def _get_user() -> str:
    """Return the current Unix username from $USER (defaults to 'user')."""
    import os
    return os.environ.get("USER", "user")


# ── advance ───────────────────────────────────────────────────────────────────
def cmd_advance(args: argparse.Namespace) -> None:
    """Run one orchestrator poll cycle for the named project."""
    proj_dir = Path(args.project)
    if not proj_dir.is_absolute():
        proj_dir = NMCLPS / args.project
    if not proj_dir.exists():
        print(f"Project directory not found: {proj_dir}")
        sys.exit(1)

    sys.path.insert(0, str(NMCLPS))
    from hpca.core.config import Config as _Cfg
    mat_src = Path(_Cfg.get().hpc("matdesign_src", "") or "/nonexistent")
    if mat_src.exists():
        sys.path.insert(0, str(mat_src))

    from hpca.orchestrator.hpca_orchestrator import HPCAOrchestrator
    orch = HPCAOrchestrator(resume=True)
    print(f"Advancing project: {proj_dir.name}")
    orch.advance_project(proj_dir)
    print("Done.")


# ── reset ─────────────────────────────────────────────────────────────────────
def cmd_reset(args: argparse.Namespace) -> None:
    """Reset a handler's stage to PENDING in the project state file."""
    proj_dir = Path(args.project)
    if not proj_dir.is_absolute():
        proj_dir = NMCLPS / args.project
    state_file = proj_dir / "logs" / "orchestrator_state.json"
    if not state_file.exists():
        print(f"No state file found for {proj_dir.name}")
        sys.exit(1)

    state = json.loads(state_file.read_text())
    handler = args.handler
    if handler not in state.get("handlers", {}):
        print(f"Handler '{handler}' not found in state (never ran?)")
        print(f"Valid handlers: {list(state.get('handlers', {}).keys())}")
        sys.exit(1)

    old_stage = state["handlers"][handler].get("stage", "PENDING")
    state["handlers"][handler] = {"stage": "PENDING"}
    state["updated"] = datetime.now().isoformat()
    state_file.write_text(json.dumps(state, indent=2))
    print(f"Reset {proj_dir.name}/{handler}: {old_stage} → PENDING")


# ── start ─────────────────────────────────────────────────────────────────────
_PROJECT_MARKERS = [
    "project.yaml", "designed_structures", "dft", "mlmd", "cmd", "POSCAR", "CONTCAR",
]

def _resolve_project(raw: str | None) -> Path:
    """Resolve project dir: use CWD if not given, resolve short names against NMCLPS."""
    import os
    if raw is None:
        candidate = Path(os.getcwd()).resolve()
    else:
        p = Path(raw)
        candidate = p.resolve() if p.is_absolute() else (NMCLPS / p).resolve()

    if not candidate.is_dir():
        print(f"ERROR: directory not found: {candidate}")
        sys.exit(1)

    # Validate it looks like a project
    if not any((candidate / m).exists() for m in _PROJECT_MARKERS):
        print(f"ERROR: '{candidate}' does not look like a project directory.")
        print(f"  Expected at least one of: {', '.join(_PROJECT_MARKERS)}")
        print(f"  If this is a new project, create a project.yaml first.")
        sys.exit(1)

    return candidate


def cmd_start(args: argparse.Namespace) -> None:
    """Submit the orchestrator daemon via sbatch for the given project directory."""
    proj = _resolve_project(getattr(args, "project_dir", None))

    proj_log_dir = proj / "logs"
    slurm_dir    = proj / "logs" / "slurm"
    proj_log_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project : {proj.name}")
    print(f"Path    : {proj}")

    # SLURM path: find sub_orchestrator.sh — prefer project's own copy
    sub_script = proj / "sub_orchestrator.sh"
    if not sub_script.exists():
        sub_script = SUB_SCRIPT   # fall back to package-level copy
    if not sub_script.exists():
        print(f"ERROR: sub_orchestrator.sh not found in {proj}/ or {ORCH_DIR}/")
        print(f"  Run 'hpca new' to generate one.")
        sys.exit(1)

    print(f"Orch log: {proj_log_dir}/")
    print(f"SLURM   : {slurm_dir}/")
    print(f"Script  : {sub_script}")

    # Reject if an orchestrator is already running for this exact project path
    try:
        sq = subprocess.run(
            ["squeue", "-u", _get_user(), "--name=hpca-orch", "--noheader",
             "--format=%i %Z"],
            capture_output=True, text=True, timeout=15
        )
        for line in sq.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) >= 2 and str(proj) in parts[1]:
                print(f"\nOrchestrator already running for {proj.name} (job {parts[0]}).")
                print("Use 'hpca stop' first, or check 'hpca status'.")
                sys.exit(1)
    except Exception:
        pass

    cmd = [
        "sbatch", "--parsable",
        f"--chdir={proj}",
        f"--output={slurm_dir}/%J.stdout",
        f"--error={slurm_dir}/%J.stderr",
        str(sub_script),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        job_id = result.stdout.strip()
        print(f"\nOrchestrator submitted: job {job_id}")
        (slurm_dir / "daemon_job_id.txt").write_text(job_id + "\n")
    else:
        print(f"sbatch failed: {result.stderr}")
        sys.exit(1)


# ── stop ──────────────────────────────────────────────────────────────────────
def cmd_stop(args: argparse.Namespace) -> None:
    """Cancel the running hpca_orch SLURM job."""
    stopped = False

    # 1. Try SLURM job
    try:
        result = subprocess.run(
            ["squeue", "-u", _get_user(), "--name=hpca_orch", "--noheader", "--format=%i"],
            capture_output=True, text=True, timeout=15
        )
        job_ids = [j.strip() for j in result.stdout.strip().splitlines() if j.strip()]
        for jid in job_ids:
            subprocess.run(["scancel", jid], timeout=15)
            print(f"Cancelled SLURM job {jid}")
            stopped = True
    except Exception as exc:
        print(f"Warning: squeue/scancel error: {exc}")

    if not stopped:
        print("No running orchestrator found (no SLURM job).")


# ── logs ──────────────────────────────────────────────────────────────────────
def cmd_logs(args: argparse.Namespace) -> None:
    """Print the last N lines of the most recent orchestrator log file."""
    log_files = sorted(LOG_DIR.glob("hpca_orch_*.log"), reverse=True) if LOG_DIR.exists() else []
    if not log_files:
        print(f"No log files found in {LOG_DIR}")
        return
    latest = log_files[0]
    n = args.tail if hasattr(args, "tail") and args.tail else 50
    print(f"=== {latest} (last {n} lines) ===")
    result = subprocess.run(["tail", "-n", str(n), str(latest)],
                            capture_output=True, text=True, timeout=10)
    print(result.stdout)


# ── chaai ─────────────────────────────────────────────────────────────────────
def cmd_chaai(args: argparse.Namespace) -> None:
    """Show CHAAI training data status or manually trigger the fine-tuning pipeline."""
    from hpca.core.paths import load_platform_config as _lpc_cli
    chaai_root = Path(_lpc_cli().get("hpc", {}).get("chaai_root", "Chaai"))
    data_dir   = chaai_root / "training" / "data"
    adapters   = chaai_root / "adapters"

    if args.action == "status":
        print("\n=== CHAAI Training Status ===")
        for jsonl in sorted(data_dir.glob("*.jsonl")):
            lines = sum(1 for _ in jsonl.open())
            print(f"  {jsonl.name}: {lines:,} examples")
        print()
        adapter_dirs = list(adapters.glob("chaai-v*"))
        if adapter_dirs:
            print("Adapters:")
            for d in sorted(adapter_dirs):
                print(f"  {d.name}")
        else:
            print("No adapters trained yet.")
        # Check for running chaai jobs
        result = subprocess.run(
            ["squeue", "-u", _get_user(), "--noheader", "--format=%i %j %T %M"],
            capture_output=True, text=True, timeout=15
        )
        chaai_jobs = [l for l in result.stdout.splitlines() if "chaai" in l.lower()]
        if chaai_jobs:
            print("\nRunning CHAAI jobs:")
            for line in chaai_jobs:
                print(f"  {line}")

    elif args.action == "trigger":
        print("Manually triggering CHAAI training pipeline...")
        sys.path.insert(0, str(NMCLPS))
        from hpca.core.config import Config as _Cfg
        mat_src = Path(_Cfg.get().hpc("matdesign_src", "") or "/nonexistent")
        if mat_src.exists():
            sys.path.insert(0, str(mat_src))
        from hpca.orchestrator.handlers.h12_chaai import CHAAIHandler
        from hpca.orchestrator.state_tracker import load_state

        handler = CHAAIHandler()
        # Use first project with completed analysis as dummy context
        projects = _discover_projects(NMCLPS)
        for proj_dir in projects:
            state = load_state(proj_dir)
            if state.get_stage("h06_analysis") == "COMPLETE":
                job_id = handler.submit(proj_dir, state)
                if job_id:
                    print(f"CHAAI pipeline triggered from {proj_dir.name}: job chain starting at {job_id}")
                else:
                    print("CHAAI submission returned no job_id")
                return
        print("No project with completed analysis found. Cannot trigger CHAAI.")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate command handler."""
    parser = argparse.ArgumentParser(
        description="HPCA orchestrator control CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # status
    p_status = sub.add_parser("status", help="Show handler state for all projects")
    p_status.add_argument("project", nargs="?", default=None,
                          help="Filter by project name substring")

    # advance
    p_adv = sub.add_parser("advance", help="Run one poll for a specific project")
    p_adv.add_argument("project", help="Project name or absolute path")

    # reset
    p_reset = sub.add_parser("reset", help="Reset a handler to PENDING")
    p_reset.add_argument("project", help="Project name or absolute path")
    p_reset.add_argument("handler", help="Handler name, e.g. h02_aimd")

    # start
    p_start = sub.add_parser("start",
                              help="Submit orchestrator daemon for one project "
                                   "(defaults to current directory)")
    p_start.add_argument("project_dir", nargs="?", default=None,
                         help="Project directory (absolute path, short name, or omit to use CWD)")

    # stop
    p_stop = sub.add_parser("stop", help="Stop the running SLURM orchestrator")
    p_stop.add_argument("project_dir", nargs="?", default=None,
                        help="Project directory")

    # logs
    p_logs = sub.add_parser("logs", help="Tail the latest orchestrator log")
    p_logs.add_argument("--tail", type=int, default=50, help="Number of lines (default 50)")

    # chaai
    p_chaai = sub.add_parser("chaai", help="CHAAI training control")
    p_chaai.add_argument("action", choices=["status", "trigger"])

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "status":  cmd_status,
        "advance": cmd_advance,
        "reset":   cmd_reset,
        "start":   cmd_start,
        "stop":    cmd_stop,
        "logs":    cmd_logs,
        "chaai":   cmd_chaai,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
