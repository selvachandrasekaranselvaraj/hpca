#!/usr/bin/env python3
"""
HPCA Battery Materials Pipeline — main CLI entry point.

Usage examples:
    python pipeline.py run        --project NVO  --stages all
    python pipeline.py run        --project NVO  --stages 00,01,04
    python pipeline.py analyze    --project NVO  --type msd,rdf,sei,transport,phase
    python pipeline.py plot       --project NVO  --type arrhenius,msd,sei,comparison
    python pipeline.py continuum  --project NVO  --models all
    python pipeline.py manuscript --project NVO
    python pipeline.py benchmark  --project NVO  --mlip all
    python pipeline.py train-chaai --projects all
    python pipeline.py status
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Make the package importable from wherever the script is invoked.
# We insert the directory that *contains* hpca/ (i.e. /path/to/workspace).
# ---------------------------------------------------------------------------

def _shell_cwd() -> Path:
    """Return the shell's $PWD (preserves symlinks) rather than os.getcwd()."""
    return Path(os.environ.get("PWD", os.getcwd()))


def _resolve_project_dir(arg: Optional[str]) -> Path:
    """Resolve a project directory argument, preserving the user's symlink path."""
    if arg:
        p = Path(arg)
        return (_shell_cwd() / p) if not p.is_absolute() else p
    return _shell_cwd()


_PIPELINE_ROOT = Path(__file__).resolve().parent          # …/hpca/hpca
_PLATFORM_ROOT = _PIPELINE_ROOT.parent                    # …/hpca (package root)
for _p in [str(_PLATFORM_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hpca.core.project import MaterialProject, ProjectRegistry  # noqa: E402
from hpca.core.status  import check_all, check_project, print_all  # noqa: E402
from hpca.core.paths   import load_platform_config  # noqa: E402

# ---------------------------------------------------------------------------
# Stage constants
# ---------------------------------------------------------------------------

ALL_STAGES: List[str] = ["00", "01", "02", "03", "04", "05", "06", "07"]

STAGE_NAMES: Dict[str, str] = {
    "00": "Materials Design",
    "01": "DFT Inputs",
    "02": "MLIP Training",
    "03": "LAMMPS MD",
    "04": "Property Analysis",
    "05": "Characterization",
    "06": "Continuum Models",
    "07": "Manuscript",
}

ANALYSIS_TYPES  = {"msd", "rdf", "sei", "transport", "phase"}
PLOT_TYPES      = {"arrhenius", "msd", "sei", "comparison"}
CONTINUUM_MODELS= {"interdiffusion", "phase_field", "stress", "sei_growth", "kjma", "all"}
MLIP_NAMES      = {"deepmd", "mace-mp0", "uma", "mace-off23", "m3gnet-pbe",
                   "chgnet-pbe", "tensornet-pbe", "esen-omol", "all"}

# ---------------------------------------------------------------------------
# Lazy stage / module imports
# ---------------------------------------------------------------------------

def _import_stages():
    """Lazily import hpca.stages to avoid mandatory heavy deps at startup."""
    try:
        from hpca import stages  # noqa: F401
        return stages
    except ImportError as exc:
        _die(f"Cannot import hpca.stages: {exc}")


def _import_analysis():
    """Lazily import hpca.analysis to avoid mandatory heavy deps at startup."""
    try:
        from hpca import analysis  # noqa: F401
        return analysis
    except ImportError as exc:
        _die(f"Cannot import hpca.analysis: {exc}")


def _import_viz():
    """Lazily import hpca.viz to avoid mandatory heavy deps at startup."""
    try:
        from hpca import viz  # noqa: F401
        return viz
    except ImportError as exc:
        _die(f"Cannot import hpca.viz: {exc}")


def _import_continuum():
    """Lazily import hpca.continuum to avoid mandatory heavy deps at startup."""
    try:
        from hpca import continuum  # noqa: F401
        return continuum
    except ImportError as exc:
        _die(f"Cannot import hpca.continuum: {exc}")


def _import_manuscript():
    """Lazily import hpca.manuscript to avoid mandatory heavy deps at startup."""
    try:
        from hpca import manuscript  # noqa: F401
        return manuscript
    except ImportError as exc:
        _die(f"Cannot import hpca.manuscript: {exc}")


def _import_chaai():
    """Lazily import hpca.chaai to avoid mandatory heavy deps at startup."""
    try:
        from hpca import chaai  # noqa: F401
        return chaai
    except ImportError as exc:
        _die(f"Cannot import hpca.chaai: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _die(msg: str, code: int = 1) -> None:
    """Print an error message to stderr and exit with the given code."""
    print(f"[pipeline] ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def _info(msg: str) -> None:
    """Print an informational message with the [pipeline] prefix."""
    print(f"[pipeline] {msg}")


def _load_registry() -> Dict[str, MaterialProject]:
    """Discover projects by scanning from cwd upward, then from cwd itself."""
    cwd = _shell_cwd()
    # If cwd has a project.yaml, load just that project
    if (cwd / "project.yaml").exists():
        mp = ProjectRegistry.from_project_yaml(cwd / "project.yaml")
        return {mp.name: mp}
    # Otherwise scan one level up (siblings)
    projects = ProjectRegistry.discover(cwd.parent)
    if not projects:
        projects = ProjectRegistry.discover(cwd)
    return projects


def _resolve_project(registry: Dict[str, MaterialProject], name: str) -> MaterialProject:
    """Return the named project from the registry, dying with a helpful message if absent."""
    if name not in registry:
        _die(f"Project '{name}' not found. Known: {sorted(registry)}")
    return registry[name]


def _resolve_stages(raw: str) -> List[str]:
    """Parse a comma-separated stage string (or 'all') into a sorted list of zero-padded IDs."""
    if raw.lower() == "all":
        return ALL_STAGES
    stages = [s.strip().zfill(2) for s in raw.split(",")]
    bad = [s for s in stages if s not in ALL_STAGES]
    if bad:
        _die(f"Unknown stage(s): {bad}. Valid stages: {ALL_STAGES}")
    return stages


def _resolve_csv_set(raw: str, valid: set, label: str) -> List[str]:
    """Parse a comma-separated string (or 'all') against a set of valid choices."""
    if raw.lower() == "all":
        return sorted(valid - {"all"})
    items = [t.strip() for t in raw.split(",")]
    bad = [t for t in items if t not in valid]
    if bad:
        _die(f"Unknown {label}(s): {bad}. Valid: {sorted(valid)}")
    return items


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    """Execute one or more pipeline stages for a named project."""
    registry = _load_registry()
    project  = _resolve_project(registry, args.project)
    stages   = _resolve_stages(args.stages)

    _info(f"Running project={project.name}  stages={stages}")

    stages_mod = _import_stages()

    for stage_id in stages:
        stage_name = STAGE_NAMES[stage_id]
        _info(f"  Stage {stage_id}: {stage_name}")
        runner_fn = getattr(stages_mod, f"run_stage_{stage_id}", None)
        if runner_fn is None:
            _info(f"    [WARN] stages.run_stage_{stage_id} not implemented — skipping")
            continue
        try:
            runner_fn(project, registry=registry)
        except Exception as exc:
            _die(f"Stage {stage_id} ({stage_name}) failed: {exc}")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run post-processing analysis routines (MSD, RDF, SEI, transport, phase) for a project."""
    registry = _load_registry()
    project  = _resolve_project(registry, args.project)
    types    = _resolve_csv_set(args.type, ANALYSIS_TYPES, "analysis type")

    _info(f"Analyzing project={project.name}  types={types}")

    analysis_mod = _import_analysis()

    dispatch = {
        "msd":       "run_msd",
        "rdf":       "run_rdf",
        "sei":       "run_sei",
        "transport": "run_transport",
        "phase":     "run_phase",
    }
    for t in types:
        fn_name = dispatch[t]
        fn = getattr(analysis_mod, fn_name, None)
        if fn is None:
            _info(f"  [WARN] analysis.{fn_name} not implemented — skipping")
            continue
        _info(f"  Running analysis: {t}")
        try:
            fn(project)
        except Exception as exc:
            _die(f"Analysis '{t}' failed: {exc}")


def cmd_plot(args: argparse.Namespace) -> None:
    """Generate publication-quality figures (Arrhenius, MSD, SEI, comparison) for a project."""
    registry = _load_registry()
    project  = _resolve_project(registry, args.project)
    types    = _resolve_csv_set(args.type, PLOT_TYPES, "plot type")

    _info(f"Plotting project={project.name}  types={types}")

    viz_mod = _import_viz()

    dispatch = {
        "arrhenius":  "plot_arrhenius",
        "msd":        "plot_msd",
        "sei":        "plot_sei",
        "comparison": "plot_comparison",
    }
    for t in types:
        fn_name = dispatch[t]
        fn = getattr(viz_mod, fn_name, None)
        if fn is None:
            _info(f"  [WARN] viz.{fn_name} not implemented — skipping")
            continue
        _info(f"  Generating plot: {t}")
        try:
            fn(project)
        except Exception as exc:
            _die(f"Plot '{t}' failed: {exc}")


def cmd_continuum(args: argparse.Namespace) -> None:
    """Run continuum models (PDE/phase-field/SEI/DFN) for a project."""
    registry = _load_registry()
    project  = _resolve_project(registry, args.project)
    models   = _resolve_csv_set(args.models, CONTINUUM_MODELS, "continuum model")
    # "all" expanded by _resolve_csv_set; also strip "all" literal if present
    if "all" in models:
        models = sorted(CONTINUUM_MODELS - {"all"})

    _info(f"Continuum models project={project.name}  models={models}")

    continuum_mod = _import_continuum()

    dispatch = {
        "interdiffusion": "run_interdiffusion",
        "phase_field":    "run_phase_field",
        "stress":         "run_stress",
        "sei_growth":     "run_sei_growth",
        "kjma":           "run_kjma",
    }
    for m in models:
        fn_name = dispatch.get(m)
        if fn_name is None:
            continue
        fn = getattr(continuum_mod, fn_name, None)
        if fn is None:
            _info(f"  [WARN] continuum.{fn_name} not implemented — skipping")
            continue
        _info(f"  Running continuum model: {m}")
        try:
            fn(project)
        except Exception as exc:
            _die(f"Continuum model '{m}' failed: {exc}")


def cmd_manuscript(args: argparse.Namespace) -> None:
    """Auto-generate a .docx manuscript from all pipeline results for a project."""
    registry  = _load_registry()
    project   = _resolve_project(registry, args.project)

    _info(f"Generating manuscript for project={project.name}")

    ms_mod = _import_manuscript()
    fn = getattr(ms_mod, "generate", None)
    if fn is None:
        _die("manuscript.generate not implemented")
    try:
        fn(project)
    except Exception as exc:
        _die(f"Manuscript generation failed: {exc}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Benchmark one or more MLIPs against the DFT reference for a project."""
    registry = _load_registry()
    project  = _resolve_project(registry, args.project)
    _plat_reg = ProjectRegistry(_PIPELINE_ROOT / "config" / "materials.yaml")
    mlip_reg = _plat_reg.mlip_registry() if hasattr(_plat_reg, "mlip_registry") else {}

    all_mlips = set(mlip_reg.keys()) | {"all"}
    mlips     = _resolve_csv_set(args.mlip, all_mlips, "MLIP")
    if "all" in mlips:
        # Filter to MLIPs compatible with this project's category
        mlips = [
            m for m, cfg in mlip_reg.items()
            if project.category in cfg.get("categories", [])
        ]

    _info(f"Benchmarking project={project.name}  mlips={mlips}")

    # Lazy import to avoid mandatory heavy deps at startup
    try:
        from hpca.stages import run_benchmark
    except ImportError:
        run_benchmark = None

    if run_benchmark is None:
        _info("[WARN] stages.run_benchmark not implemented — skipping")
        return

    for mlip in mlips:
        _info(f"  Benchmarking MLIP: {mlip}")
        try:
            run_benchmark(project, mlip=mlip, mlip_config=mlip_reg.get(mlip, {}))
        except Exception as exc:
            _die(f"Benchmark '{mlip}' failed: {exc}")


def cmd_train_chaai(args: argparse.Namespace) -> None:
    """Train the ChaAI surrogate model from one or more project datasets."""
    registry = _load_registry()

    if args.projects.lower() == "all":
        projects = list(registry.values())
    else:
        names    = [n.strip() for n in args.projects.split(",")]
        projects = [_resolve_project(registry, n) for n in names]

    _info(f"ChaAI training for {len(projects)} project(s)")

    chaai_mod = _import_chaai()
    fn = getattr(chaai_mod, "train", None)
    if fn is None:
        _die("chaai.train not implemented")

    try:
        fn(projects, registry=registry)
    except Exception as exc:
        _die(f"ChaAI training failed: {exc}")


def cmd_status(args: argparse.Namespace) -> None:
    """Print stage completion, Slurm jobs, and key metrics for the current or all projects."""
    force_all   = getattr(args, "all", False)
    project_arg = getattr(args, "project_dir", None)

    # Resolve candidate single-project dir: explicit arg > cwd
    if project_arg:
        candidate = Path(project_arg)
    else:
        candidate = _shell_cwd()

    in_project = (candidate / "project.yaml").exists()

    if not force_all and in_project:
        # ── Single-project mode ──────────────────────────────────────────────
        result = check_project(candidate, write=True)
        cfg    = result.get("config", {})
        D      = cfg.get("D_best")
        Ea     = cfg.get("Ea_best")
        D_s    = f"{D:.2e}" if D else "—"
        Ea_s   = f"{Ea:.3f}" if Ea else "—"
        print(f"\nProject:  {result['project']}")
        print(f"Dir:      {candidate}")
        print(f"Category: {cfg.get('category', '?')}")
        print(f"D_best:   {D_s}  |  Ea: {Ea_s} eV")
        print(f"Summary:  {result['summary']}")
        jobs = result.get("active_jobs", [])
        if jobs:
            print(f"\nActive Slurm jobs ({len(jobs)}):")
            for j in jobs:
                print(f"  {j.get('jobid','?'):>10}  {j.get('name','?'):<30}  {j.get('state','?')}")
        else:
            print("\nNo active Slurm jobs.")
        print()
        status_file = candidate / "logs" / "status.txt"
        if status_file.exists():
            _info(f"Status file: {status_file}")
    else:
        # ── All-projects mode — scan cwd and one level up ────────────────────
        scan_root = _shell_cwd().parent if not force_all else _shell_cwd()
        _info(f"Scanning {scan_root} for projects …\n")
        results = check_all(scan_root, write=True)
        if not results:
            _info("No project.yaml files found. Run 'hpca new' in a project directory.")
            return
        print_all(results)
        print()
        _info("Status files written to logs/status.txt in each project directory.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser with all subcommands registered."""
    parser = argparse.ArgumentParser(
        prog="pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            HPCA Battery Materials Pipeline — NREL Kestrel (H100)

            Orchestrates DFT → MLIP → LAMMPS MD → analysis → manuscript
            for all registered battery material projects.
        """),
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # -- run ---------------------------------------------------------------
    p_run = sub.add_parser("run", help="Execute pipeline stages for a project")
    p_run.add_argument("--project", required=True,
                       help="Project name (e.g. NVO, NMC622, LMZC)")
    p_run.add_argument("--stages", default="all",
                       help="Comma-separated stage IDs (00-07) or 'all' (default)")

    # -- analyze -----------------------------------------------------------
    p_ana = sub.add_parser("analyze",
                            help="Run post-processing analyses on simulation data")
    p_ana.add_argument("--project", required=True)
    p_ana.add_argument("--type", default="msd,transport",
                       help=f"Comma-separated types or 'all': {sorted(ANALYSIS_TYPES)}")

    # -- plot --------------------------------------------------------------
    p_plot = sub.add_parser("plot", help="Generate publication-quality figures")
    p_plot.add_argument("--project", required=True)
    p_plot.add_argument("--type", default="arrhenius,msd",
                        help=f"Comma-separated types or 'all': {sorted(PLOT_TYPES)}")

    # -- continuum ---------------------------------------------------------
    p_cont = sub.add_parser("continuum",
                             help="Run continuum (PDE/phase-field/SEI) models")
    p_cont.add_argument("--project", required=True)
    p_cont.add_argument("--models", default="all",
                        help=f"Comma-separated models or 'all': "
                             f"{sorted(CONTINUUM_MODELS - {'all'})}")

    # -- manuscript --------------------------------------------------------
    p_ms = sub.add_parser("manuscript", help="Auto-generate .docx manuscript")
    p_ms.add_argument("--project", required=True)

    # -- benchmark ---------------------------------------------------------
    p_bench = sub.add_parser("benchmark",
                              help="Benchmark one or more MLIPs against DFT reference")
    p_bench.add_argument("--project", required=True)
    p_bench.add_argument("--mlip", default="all",
                         help="Comma-separated MLIP names or 'all'")

    # -- train-chaai -------------------------------------------------------
    p_chaai = sub.add_parser("train-chaai",
                              help="Train the ChaAI surrogate model")
    p_chaai.add_argument("--projects", default="all",
                         help="Comma-separated project names or 'all' (default)")

    # -- status ------------------------------------------------------------
    p_status = sub.add_parser("status",
                   help="Print completion status for current project (or all with --all)")
    p_status.add_argument("project_dir", nargs="?", default=None,
                          help="Project directory (default: cwd if it contains project.yaml)")
    p_status.add_argument("--all", action="store_true",
                          help="Force scan of all projects under the platform root")

    p_validate = sub.add_parser("validate", help="Validate and normalize project.yaml")
    p_validate.add_argument("project_dir", nargs="?", default=None)
    p_validate.add_argument("--json", action="store_true", dest="as_json")

    p_health = sub.add_parser("health", help="Print read-only project health")
    p_health.add_argument("project_dir", nargs="?", default=None)
    p_health.add_argument("--json", action="store_true", dest="as_json")

    # -- init --------------------------------------------------------------
    p_init = sub.add_parser("init",
                             help="Interactive wizard: generate project.yaml for a new project")
    p_init.add_argument("project_dir", nargs="?", default=None,
                        help="Project directory (default: current directory)")

    # -- new ---------------------------------------------------------------
    p_new = sub.add_parser("new",
                            help="Wizard: create project.yaml in a directory (alias for init)")
    p_new.add_argument("project_dir", nargs="?", default=None,
                       help="Project directory (default: current directory)")

    # -- start / resume / restart -----------------------------------------
    for _cmd_name, _cmd_help in [
        ("start",  "Start (or resume) autonomous orchestrator for a project"),
        ("resume", "Resume autonomous orchestrator for a project (alias for start)"),
        ("restart", "Register and restart autonomous processing for a project"),
    ]:
        _p = sub.add_parser(_cmd_name, help=_cmd_help)
        _p.add_argument("project_dir", nargs="?", default=None,
                        help="Project directory (default: current directory)")
        _p.add_argument("--slurm", action="store_true",
                        help="Submit via sbatch (long partition, 10 days) instead of running on this node")
        _p.add_argument("--status-only", action="store_true",
                        help="Just print job/stage status, do not submit")

    # -- log ---------------------------------------------------------------
    p_log = sub.add_parser("log", help="Tail the orchestrator log for a project")
    p_log.add_argument("project_dir", nargs="?", default=None,
                       help="Project directory (default: current directory)")
    p_log.add_argument("-n", "--lines", type=int, default=60,
                       help="Number of tail lines to show (default: 60)")
    p_log.add_argument("-f", "--follow", action="store_true",
                       help="Follow log (like tail -f)")

    # -- stop --------------------------------------------------------------
    p_stop = sub.add_parser("stop", help="Cancel the running orchestrator Slurm job")
    p_stop.add_argument("project_dir", nargs="?", default=None,
                        help="Project directory (default: current directory)")

    return parser


# ---------------------------------------------------------------------------
# init / new commands
# ---------------------------------------------------------------------------

def cmd_init(args) -> None:
    """Run the interactive project wizard to create or update project.yaml."""
    from hpca.tools.project_wizard import run_wizard
    project_dir = _resolve_project_dir(args.project_dir)
    run_wizard(project_dir)


def cmd_new(args) -> None:
    """Run the wizard (like init) and register the new project with the daemon."""
    cmd_init(args)
    _register_with_daemon(args)


def _register_with_daemon(args) -> None:
    """Set project-local RUNNING control and register its canonical project.yaml."""
    project_dir = _resolve_project_dir(args.project_dir)
    yaml_path   = project_dir / "project.yaml"
    if not yaml_path.exists():
        return

    try:
        from hpca.daemon.config import DaemonConfig
        from hpca.daemon.service import start_project
        request = start_project(DaemonConfig(), yaml_path)
        _info(f"Project control: {project_dir / '.hpca' / 'control.json'}")
        _info(f"Daemon request: {request}")
        _info("Daemon will pick it up on its next poll cycle (~60 s).")
        _info(f"Monitor: hpca-daemon project-status {project_dir}")
    except FileNotFoundError as exc:
        _info(f"Cannot register — {exc}")
    except (ValueError, OSError) as exc:
        _info(f"Cannot register project: {exc}")


# ---------------------------------------------------------------------------
# start command
# ---------------------------------------------------------------------------

def cmd_start(args) -> None:
    """Start (or resume) the autonomous orchestrator for a project, on this node or via sbatch."""
    project_dir = _resolve_project_dir(args.project_dir)

    if not (project_dir / "project.yaml").exists():
        _die(f"No project.yaml found in {project_dir} — run 'hpca new' first")

    # On 'hpca resume': reset FAILED handler stages to PENDING so they are retried
    if getattr(args, "command", "") in ("resume", "restart"):
        for child, handlers in _reset_failed_project_tree(project_dir):
            for hname in handlers:
                _info(f"  Resetting {child.name}/{hname}: FAILED → PENDING; attempts → 0")

    if getattr(args, "status_only", False):
        from hpca.daemon.control import control_path, get_desired_state
        _info(f"Project: {project_dir}")
        _info(f"Desired state: {get_desired_state(project_dir)}")
        _info(f"Control: {control_path(project_dir)}")
        return
    if getattr(args, "slurm", False):
        _info("--slurm is no longer needed; the global daemon dispatches compute stages to SLURM.")
    from hpca.daemon.config import DaemonConfig
    from hpca.daemon.service import start_project
    request = start_project(DaemonConfig(), project_dir / "project.yaml")
    _info(f"Project desired state: RUNNING")
    _info(f"Project control: {project_dir / '.hpca' / 'control.json'}")
    _info(f"Daemon request: {request}")


def _reset_failed_project_tree(project_dir: Path) -> list[tuple[Path, list[str]]]:
    """Reset failed state for a parent and its direct production children."""
    from hpca.orchestrator.state_tracker import ProjectState
    roots = [project_dir]
    roots.extend(sorted(path.parent for path in project_dir.glob("*/project.yaml")))
    changed: list[tuple[Path, list[str]]] = []
    for root in roots:
        handlers = ProjectState(root).reset_failed_handlers()
        if handlers:
            changed.append((root, handlers))
    return changed


# ---------------------------------------------------------------------------
# log command
# ---------------------------------------------------------------------------

def cmd_log(args) -> None:
    """Tail the orchestrator log for a project, optionally following it live."""
    import subprocess

    project_dir = _resolve_project_dir(args.project_dir)
    log_file    = project_dir / "logs" / "hpca_orch.log"

    if not log_file.exists():
        # Fall back to central log
        central = _PLATFORM_ROOT / "orchestrator" / "logs"
        candidates = sorted(central.glob("hpca_orch_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            log_file = candidates[0]
            _info(f"No per-project log found; showing central log: {log_file.name}")
        else:
            _die(f"No log file found for {project_dir.name}")

    n = getattr(args, "lines", 60)
    follow = getattr(args, "follow", False)

    if follow:
        subprocess.run(["tail", f"-n{n}", "-f", str(log_file)], check=False)
    else:
        subprocess.run(["tail", f"-n{n}", str(log_file)], check=False)


# ---------------------------------------------------------------------------
# stop command
# ---------------------------------------------------------------------------

def cmd_stop(args) -> None:
    """Request a stop for only the selected project via its local control file."""
    project_dir = _resolve_project_dir(args.project_dir)
    try:
        from hpca.daemon.service import stop_project
        control = stop_project(project_dir)
        _info(f"Project stop control written: {control}")
    except (OSError, ValueError) as exc:
        _die(f"Could not write project stop control: {exc}")


def cmd_validate(args) -> None:
    """Validate canonical project configuration without writing it."""
    import json
    from hpca.core.project_io import read_project_yaml
    from hpca.core.project_schema import migrate, validate
    project_dir = _resolve_project_dir(args.project_dir)
    try:
        data = migrate(read_project_yaml(project_dir))
        errors = validate(data)
    except (OSError, ValueError, TypeError) as exc:
        errors = [str(exc)]
    result = {"project": str(project_dir), "valid": not errors, "errors": errors}
    if args.as_json:
        print(json.dumps(result, sort_keys=True))
    elif errors:
        for error in errors:
            print(f"[pipeline] INVALID: {error}")
    else:
        _info("project.yaml is valid")
    if errors:
        raise SystemExit(2)


def cmd_health(args) -> None:
    """Print a read-only health snapshot."""
    import json
    from hpca.monitor.health import project_health
    result = project_health(_resolve_project_dir(args.project_dir))
    if args.as_json:
        print(json.dumps(result, sort_keys=True))
    else:
        _info(f"{result['project']}: desired={result['desired_state']} "
              f"healthy={result['healthy']} stages={result['stage_counts']}")


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_COMMANDS = {
    "run":          cmd_run,
    "analyze":      cmd_analyze,
    "plot":         cmd_plot,
    "continuum":    cmd_continuum,
    "manuscript":   cmd_manuscript,
    "benchmark":    cmd_benchmark,
    "train-chaai":  cmd_train_chaai,
    "status":       cmd_status,
    "validate":     cmd_validate,
    "health":       cmd_health,
    "init":         cmd_init,
    "new":          cmd_new,
    "start":        cmd_start,
    "resume":       cmd_start,
    "restart":      cmd_start,
    "log":          cmd_log,
    "stop":         cmd_stop,
}


# ---------------------------------------------------------------------------
# Interactive TUI menu  (shown when `hpca` is run with no arguments)
# ---------------------------------------------------------------------------

def _bold(s):
    """Wrap text in bold ANSI escape codes."""
    return f"\033[1m{s}\033[0m"

def _cyan(s):
    """Wrap text in cyan ANSI colour."""
    return f"\033[36m{s}\033[0m"

def _green(s):
    """Wrap text in green ANSI colour."""
    return f"\033[32m{s}\033[0m"

def _yellow(s):
    """Wrap text in yellow ANSI colour."""
    return f"\033[33m{s}\033[0m"

def _red(s):
    """Wrap text in red ANSI colour."""
    return f"\033[31m{s}\033[0m"


def _slurm_status() -> str:
    """Return a one-line summary of running hpca-orch jobs."""
    import subprocess, os
    r = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "--name=hpca-orch",
         "--noheader", "-o", "%.10i %.8T %.12M %j"],
        capture_output=True, text=True,
    )
    lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
    if not lines:
        return _yellow("  No hpca-orch job running")
    return "\n".join(f"  {_green(l)}" for l in lines)


_MENU = [
    ("start",       "Start (or resume) orchestrator for this project"),
    ("new",         "Create / update project.yaml  (setup wizard)"),
    ("status",      "Show all projects + Slurm job status"),
    ("log",         "Tail orchestrator log for this project"),
    ("log -f",      "Follow orchestrator log live  (Ctrl-C to exit)"),
    ("stop",        "Cancel running hpca-orch Slurm job"),
]


def _print_menu(cwd: Path) -> None:
    """Print the numbered TUI menu with current Slurm status."""
    print()
    print(_bold("=" * 54))
    print(_bold("   HPCA — Battery Materials Platform"))
    print(_bold("=" * 54))
    print(f"  Dir:  {_cyan(str(cwd))}")
    print()
    print(_slurm_status())
    print()
    print(_bold("  Choose an action:"))
    for i, (cmd, desc) in enumerate(_MENU, 1):
        print(f"  {_cyan(str(i))}) {_bold(cmd):<12}  {desc}")
    print(f"  {_cyan('h')}){' ':<12}  Help — full command reference")
    print(f"  {_cyan('q')}){' ':<12}  Quit")
    print()


def _print_help() -> None:
    """Print the full command reference table."""
    print(_bold("\n  HPCA Command Reference"))
    print("  ─────────────────────────────────────────────────")
    rows = [
        ("hpca",           "Interactive menu (this screen)"),
        ("hpca start",     "Wizard if needed, then submit orchestrator"),
        ("hpca new",       "Run project setup wizard only"),
        ("hpca status",    "All projects + Slurm job overview"),
        ("hpca log",       "Last 60 lines of project log"),
        ("hpca log -f",    "Live-follow project log"),
        ("hpca stop",      "Cancel hpca-orch job"),
        ("hpca run ...",   "Manual stage runner (see --help)"),
        ("hpca analyze ..",  "Post-process trajectories"),
        ("hpca manuscript ..", "Generate .docx manuscript"),
    ]
    for cmd, desc in rows:
        print(f"  {_bold(cmd):<30}  {desc}")
    print()
    print("  Add to PATH (one-time):  pip install -e /path/to/hpca")
    print()


def _interactive_menu() -> None:
    """Numbered TUI shown when `hpca` is invoked with no arguments."""
    import subprocess, os
    cwd = Path.cwd()

    while True:
        _print_menu(cwd)
        try:
            raw = input("  Choice [1]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if not raw:
            raw = "1"

        if raw in ("q", "quit", "exit"):
            print("  Bye.")
            sys.exit(0)

        if raw in ("h", "help", "--help", "-h"):
            _print_help()
            continue

        # Map number → command string
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(_MENU):
                raw = _MENU[idx][0]
            else:
                print(f"  {_red('Invalid choice')} — enter 1–{len(_MENU)}, h, or q")
                continue

        # Build a fake args namespace and dispatch
        tokens = raw.split()
        cmd    = tokens[0]
        extra  = tokens[1:]   # e.g. ["-f"] for "log -f"

        if cmd not in _COMMANDS:
            print(f"  {_red('Unknown command:')} {cmd}")
            continue

        # Synthesise an argparse Namespace sufficient for each handler
        ns = argparse.Namespace(project_dir=None)

        if cmd in ("start",):
            ns.status_only = False

        if cmd == "log":
            ns.lines  = 60
            ns.follow = "-f" in extra

        # Run it — handlers print their own output
        print()
        try:
            _COMMANDS[cmd](ns)
        except SystemExit:
            pass
        except Exception as exc:
            print(f"  {_red('Error:')} {exc}")

        # After start/stop show a quick Slurm line
        if cmd in ("start", "stop"):
            import time; time.sleep(1)
            print()
            print(_slurm_status())

        print()
        try:
            back = input("  Press Enter for menu  (q + Enter to quit): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if back in ("q", "quit", "exit"):
            print("  Bye.")
            sys.exit(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point: dispatch to the appropriate subcommand handler or show the TUI menu."""
    if len(sys.argv) == 1:
        _interactive_menu()
        return

    parser = build_parser()
    args   = parser.parse_args()
    handler = _COMMANDS.get(args.command)
    if handler is None:
        _die(f"Unknown command: {args.command}")
    handler(args)


if __name__ == "__main__":
    main()
