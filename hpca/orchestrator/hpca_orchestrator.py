"""
hpca_orchestrator.py — Main HPCA simulation-type automation daemon.

Polls all project directories under /path/to/workspace/ every POLL_INTERVAL
seconds, advances each project through the handler DAG, and self-restarts
before SLURM wall-time expires.

Usage:
    python hpca_orchestrator.py [--resume] [--project PROJECT_DIR]
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ORCH_DIR = Path(__file__).parent
_HPCA_ROOT = ORCH_DIR.parent.parent   # …/hpca  (the installed package root)
LOG_DIR = Path.cwd() / "logs"         # default; overridden by --log-dir at startup

# ── Timing — read from platform.yaml orchestrator section ────────────────────
def _orch_cfg() -> dict:
    """Return the orchestrator section of platform.yaml, or {} on failure."""
    try:
        from hpca.core.paths import load_platform_config
        return load_platform_config().get("orchestrator", {})
    except Exception:
        return {}

def _poll_interval() -> int:
    """Return poll interval in seconds from platform.yaml (default 60)."""
    return int(_orch_cfg().get("poll_interval_seconds", 60))

def _wall_buffer_h() -> float:
    """Return wall-time safety buffer in hours from platform.yaml (default 5.0)."""
    return float(_orch_cfg().get("wall_buffer_hours", 5.0))

def _max_parallel_projects() -> int:
    """Return the per-poll ThreadPoolExecutor cap from platform.yaml (default 8).

    Previously this scaled with os.cpu_count() (up to 104 on a full Kestrel
    node), so a project with many combinatorial sub-projects (e.g. LYC's ~67
    doping variants) fanned out to 60-100+ worker threads nearly
    simultaneously every poll cycle. Each thread independently shelled out to
    squeue/scontrol/sacct, which NREL flagged on 2026-08-12 as ~20x the next
    user's RPC load against slurmctld. Combined with the shared TTL-cached
    snapshot in hpca.core.slurm_submit, capping concurrency here also keeps
    the *rest* of check_progress()'s work (file I/O, grep-based parsing) from
    thundering-herding every 60s.
    """
    return int(_orch_cfg().get("max_parallel_projects", 8))

def _log_every_n() -> int:
    """Return heartbeat log frequency (every N polls) from platform.yaml (default 5)."""
    return int(_orch_cfg().get("log_heartbeat_every_n_polls", 5))

# ── Directories to skip when scanning for project subdirs ────────────────────
# These are non-project entries that may appear alongside project directories.
SKIP_DIRS: frozenset[str] = frozenset({
    "__pycache__", ".git", ".claude", "node_modules",
    "apps", "_archive", "hf_cache",
})

# ── Logging ──────────────────────────────────────────────────────────────────
def setup_logging(log_dir: Path, resume: bool) -> logging.Logger:
    """Set up main orchestrator log in orchestrator/logs/ directory.

    Always writes to a single fixed-name hpca_orch.log so wall-time
    self-restarts (which always pass --resume) keep appending to one
    continuous file instead of fragmenting history across a new
    timestamped file per restart.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"
    fh = logging.FileHandler(log_dir / "hpca_orch.log", mode=mode)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger = logging.getLogger("hpca.orch")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def add_project_log_handler(project_dir: Path, logger: logging.Logger) -> logging.FileHandler | None:
    """Add a per-project log file handler. Returns it so caller can remove later.
    Returns None if a handler for this path already exists (avoids duplicate log lines)."""
    proj_log_dir = project_dir / "logs"
    proj_log_dir.mkdir(parents=True, exist_ok=True)
    proj_log = proj_log_dir / "hpca_orch.log"
    proj_log_str = str(proj_log.resolve())
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and str(Path(h.baseFilename).resolve()) == proj_log_str:
            return None
    fh = logging.FileHandler(proj_log_str, mode="a")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return fh


# ── Handler registry ─────────────────────────────────────────────────────────
_HANDLER_REGISTRY = [
    ("hpca.orchestrator.handlers.h00_design",        "MaterialsDesignHandler"),
    ("hpca.orchestrator.handlers.h01_dft",           "DFTHandler"),
    ("hpca.orchestrator.handlers.h02_aimd",          "AIMDHandler"),
    ("hpca.orchestrator.handlers.h03_neb",           "NEBHandler"),
    ("hpca.orchestrator.handlers.h04_mlip",          "MLIPHandler"),
    ("hpca.orchestrator.handlers.h05_cmd",           "ClassicalMDHandler"),
    ("hpca.orchestrator.handlers.h05_lammps",        "LAMMPSHandler"),
    ("hpca.orchestrator.handlers.h06_analysis",      "AnalysisHandler"),
    ("hpca.orchestrator.handlers.h07_electronic",    "ElectronicHandler"),
    ("hpca.orchestrator.handlers.h08_echem",         "EchemHandler"),
    ("hpca.orchestrator.handlers.h09_continuum",     "ContinuumHandler"),
    ("hpca.orchestrator.handlers.h10_plotting",      "PlottingHandler"),
    ("hpca.orchestrator.handlers.h11_manuscript",    "ManuscriptHandler"),
    ("hpca.orchestrator.handlers.h12_chaai",         "CHAAIHandler"),
    ("hpca.orchestrator.handlers.h13_active_learning", "ActiveLearningHandler"),
]

def _load_handlers() -> dict:
    """Import all handler classes and return {name: instance}, skipping failures."""
    import importlib
    # Ensure hpca package root is on sys.path
    hpca_pkg_root = str(_HPCA_ROOT)
    if hpca_pkg_root not in sys.path:
        sys.path.insert(0, hpca_pkg_root)

    log = logging.getLogger("hpca.orch")
    instances = {}
    for mod_name, cls_name in _HANDLER_REGISTRY:
        try:
            mod = importlib.import_module(mod_name)
            handler = getattr(mod, cls_name)()
            instances[handler.name] = handler
        except Exception as exc:
            log.error("Failed to load handler %s.%s: %s", mod_name, cls_name, exc)
    return instances


# ── Project discovery ─────────────────────────────────────────────────────────
_PROJECT_MARKERS = [
    "project.yaml", "designed_structures", "dft", "mlmd", "cmd", "POSCAR", "CONTCAR",
]

def _discover_for_root(root: Path) -> list[Path]:
    """Return list of project directories to manage.

    Single-project mode: if ``root`` itself contains project content
    (e.g. project.yaml, aimd/, ...) treat it as the sole project and return
    ``[root]``.  This is the normal case when the orchestrator is started with
    ``--root /path/to/MyProject``.

    Multi-combination mode: if the project has more than one aimd_combination
    and sub-project directories have been created by h00_design, the orchestrator
    also discovers those sub-projects so each combination's pipeline
    (DFT→AIMD→MLIP→MLMD→CMD) runs in parallel.

    Multi-project mode: scan immediate subdirectories of ``root`` for projects.
    This is the legacy mode when ``root`` is /path/to/workspace.
    """
    if any((root / m).exists() for m in _PROJECT_MARKERS):
        # Check for combinatorial sub-projects to run in parallel
        yaml_path = root / "project.yaml"
        if yaml_path.exists():
            try:
                import yaml as _yaml
                yaml_data = _yaml.safe_load(yaml_path.read_text()) or {}
                from hpca.core.combinations import production_combinations
                production = production_combinations(yaml_data)
                if len(production) > 1:
                    projects = [root]
                    for combo in production:
                        sub = root / combo["name"]
                        if (sub / "project.yaml").exists():
                            projects.append(sub)
                    if len(projects) > 1:
                        return projects
                # Crystal doping variants: each variant is a sub-project
                crystal_variants = yaml_data.get("crystal_doping_variants", [])
                if crystal_variants:
                    projects = [root]
                    for v in crystal_variants:
                        sub = root / v["name"]
                        if (sub / "project.yaml").exists():
                            projects.append(sub)
                    if len(projects) > 1:
                        return projects
            except Exception:
                pass
        return [root]

    projects = []
    for p in sorted(root.iterdir()):
        if p.name.startswith("."):
            continue
        if p.name in SKIP_DIRS:
            continue
        if not p.is_dir():
            continue
        if any((p / m).exists() for m in _PROJECT_MARKERS):
            projects.append(p)
    return projects


# ── Wall-time guard ───────────────────────────────────────────────────────────
class WallTimeGuard:
    """Detect remaining SLURM wall-time and trigger self-restart before expiry."""

    def __init__(self, buffer_h: float | None = None):
        """
        Parameters
        ----------
        buffer_h
            Safety buffer in hours before wall limit triggers self-restart.
            Defaults to wall_buffer_hours from platform.yaml.
        """
        self.buffer = timedelta(hours=buffer_h if buffer_h is not None else _wall_buffer_h())
        self.start_time = datetime.now()
        self.wall_limit: timedelta | None = self._detect_wall_limit()

    def _detect_wall_limit(self) -> timedelta | None:
        """Query scontrol for the current job's TimeLimit; returns None outside SLURM."""
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            return None
        try:
            out = subprocess.check_output(
                ["scontrol", "show", "job", job_id, "--oneliner"],
                text=True, timeout=20
            )
            for token in out.split():
                if token.startswith("TimeLimit="):
                    tl = token.split("=", 1)[1]
                    return self._parse_slurm_time(tl)
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_slurm_time(s: str) -> timedelta | None:
        """Parse a SLURM time string (D-HH:MM:SS or HH:MM:SS) into a timedelta."""
        try:
            if "-" in s:
                days_part, rest = s.split("-", 1)
                days = int(days_part)
            else:
                days, rest = 0, s
            h, m, sec = (int(x) for x in rest.split(":"))
            return timedelta(days=days, hours=h, minutes=m, seconds=sec)
        except Exception:
            return None

    def near_limit(self) -> bool:
        """Return True when remaining wall-time is less than the configured safety buffer."""
        if self.wall_limit is None:
            return False
        elapsed = datetime.now() - self.start_time
        remaining = self.wall_limit - elapsed
        return remaining < self.buffer

    def remaining_str(self) -> str:
        """Return remaining wall-time as 'HH:MM:SS', or 'unknown' if not in SLURM."""
        if self.wall_limit is None:
            return "unknown"
        elapsed = datetime.now() - self.start_time
        remaining = self.wall_limit - elapsed
        total_seconds = max(0, int(remaining.total_seconds()))
        h, rem = divmod(total_seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


# ── Main orchestrator class ───────────────────────────────────────────────────
class HPCAOrchestrator:
    """Main polling daemon that advances all projects through the handler DAG."""

    def __init__(self, root: Path = Path("."), project_filter: str | None = None,
                 resume: bool = False, log_dir: Path = LOG_DIR,
                 inbox_mode: bool = False):
        """
        Parameters
        ----------
        root
            Project directory or parent of multiple projects to manage.
        project_filter
            If set, only process projects whose name contains this substring.
        resume
            Append to the latest log file instead of creating a new one.
        log_dir
            Directory for orchestrator log files.
        inbox_mode
            Discover projects via project_discovery.discover_all() and
            archive/fail-move them when all handlers reach a terminal state.
        """
        self.root = root
        self.project_filter = project_filter
        self.resume = resume
        self.inbox_mode = inbox_mode
        self.log = setup_logging(log_dir, resume)
        self.handlers = _load_handlers()
        self.wall_guard = WallTimeGuard()
        self._shutdown = False
        self._poll_count = 0

        # Become process-group leader so _handle_sigterm can kill daemon-local
        # children (MACE preopt, PACKMOL) without touching the daemon's group.
        try:
            os.setpgrp()
        except OSError:
            pass  # already a session/group leader

        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGINT,  self._handle_sigterm)

    def _handle_sigterm(self, signum, frame):
        """Terminate daemon-local child processes, then exit.

        Without the killpg, in-process subprocesses (MACE preopt, PACKMOL)
        survive as orphans past their timeout — the timeout is enforced by
        this parent — and each hot-reload restart spawns a duplicate racing
        on the same output files.  SLURM jobs are unaffected: they belong to
        the scheduler, not this process tree.
        """
        self.log.info("Caught signal %d — terminating child processes and exiting", signum)
        import os as _os
        import signal as _sig
        # Refund synchronous daemon attempts interrupted by a graceful hot
        # reload.  Genuine failures persist FAILED before signalling and are
        # intentionally not refunded by ProjectState.
        try:
            from hpca.core.project_discovery import discover_projects
            from hpca.orchestrator.state_tracker import ProjectState
            candidates = list(discover_projects(self.root, max_depth=3))
            if (self.root / "project.yaml").exists():
                candidates.insert(0, self.root.resolve())
            for project in dict.fromkeys(candidates):
                rolled_back = ProjectState(project).rollback_interrupted_attempts(_os.getpid())
                for handler_name in rolled_back:
                    self.log.info("Refunded interrupted attempt: %s/%s",
                                  project.name, handler_name)
        except Exception as exc:
            self.log.warning("Could not reconcile interrupted attempts: %s", exc)
        try:
            from hpca.core import child_procs
            n = child_procs.kill_all(_sig.SIGTERM)   # setsid children (preopt, packmol)
            if n:
                self.log.info("Signalled %d registered child process group(s)", n)
        except Exception:
            pass
        try:
            _sig.signal(_sig.SIGTERM, _sig.SIG_IGN)   # don't re-enter from our own killpg
            _os.killpg(_os.getpgid(0), _sig.SIGTERM)  # same-group children
        except Exception:
            pass
        _os._exit(0)

    def run(self) -> None:
        """Main loop: poll all projects every poll_interval seconds until wall-time limit or SIGTERM."""
        self.log.info("=" * 60)
        self.log.info("HPCA Orchestrator started (PID %d)", os.getpid())
        self.log.info("Root: %s  inbox_mode=%s", self.root, self.inbox_mode)
        self.log.info("Project filter: %s", self.project_filter or "(none)")
        self.log.info("Wall limit: %s  Buffer: %.1fh",
                      self.wall_guard.wall_limit or "none", _wall_buffer_h())
        self.log.info("=" * 60)

        while not self._shutdown:
            self._poll_count += 1
            if self._poll_count % _log_every_n() == 1:
                self.log.info("[Heartbeat] poll #%d  wall remaining: %s",
                              self._poll_count, self.wall_guard.remaining_str())

            if self.wall_guard.near_limit():
                self.log.warning("Wall-time limit approaching — self-restarting")
                self._self_restart()
                break

            if self.inbox_mode:
                from hpca.core.project_discovery import discover_all as _pd_discover_all
                projects = _pd_discover_all()
            else:
                projects = _discover_for_root(self.root)
            if self.project_filter:
                projects = [p for p in projects if self.project_filter in p.name]

            # Advance all projects in parallel — each sub-project is fully
            # isolated (own state file, own dirs). Daemon jobs (design, analysis)
            # block the thread but not each other; SLURM submits return instantly.
            n_workers = min(len(projects), _max_parallel_projects())
            if n_workers <= 1:
                for proj_dir in projects:
                    try:
                        self.advance_project(proj_dir)
                    except Exception as exc:
                        self.log.exception("Unhandled error advancing %s: %s", proj_dir.name, exc)
            else:
                from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
                with ThreadPoolExecutor(max_workers=n_workers) as _pool:
                    _futs = {_pool.submit(self.advance_project, p): p for p in projects}
                    for _fut in _as_completed(_futs):
                        try:
                            _fut.result()
                        except Exception as exc:
                            self.log.exception("Unhandled error advancing %s: %s",
                                               _futs[_fut].name, exc)

            self._log_stage_summary(projects)
            projects = self._lifecycle_check(projects)
            _pi = _poll_interval()
            self.log.debug("Poll #%d complete — sleeping %ds", self._poll_count, _pi)
            time.sleep(_pi)

        self.log.info("Orchestrator exiting")

    def _log_stage_summary(self, projects: list[Path]) -> None:
        """Log a compact stage-status line for each project after each poll cycle."""
        from hpca.orchestrator.state_tracker import load_state
        from hpca.registry.stage import get_enabled, HANDLER_ORDER
        for proj_dir in projects:
            try:
                state = load_state(proj_dir)
                yaml_cfg = self._read_project_yaml(proj_dir)
                enabled = set(get_enabled(yaml_cfg))
                counts: dict[str, list[str]] = {"RUNNING": [], "FAILED": [], "PENDING": []}
                n_complete = 0
                for h in HANDLER_ORDER:
                    if h not in enabled:
                        continue
                    s = state.get_stage(h)
                    if s == "COMPLETE":
                        n_complete += 1
                    elif s in counts:
                        counts[s].append(h)
                parts = [f"COMPLETE={n_complete}"]
                if counts["RUNNING"]:
                    parts.append("RUNNING=" + ",".join(counts["RUNNING"]))
                if counts["FAILED"]:
                    parts.append("FAILED=" + ",".join(counts["FAILED"]))
                if counts["PENDING"]:
                    # Only show first few pending to keep line short
                    pend = counts["PENDING"]
                    label = ",".join(pend[:3]) + (f"…+{len(pend)-3}" if len(pend) > 3 else "")
                    parts.append(f"PENDING={label}")
                self.log.debug("[status] %s: %s", proj_dir.name, "  ".join(parts))
            except Exception:
                pass

    def advance_project(self, project_dir: Path) -> None:
        """Advance one project through the handler DAG by one step."""
        # Add per-project log handler so all events for this project also go
        # to {project_dir}/logs/hpca_orch.log
        proj_fh = add_project_log_handler(project_dir, self.log)
        try:
            self._advance_project_inner(project_dir)
        finally:
            if proj_fh is not None:
                self.log.removeHandler(proj_fh)
                proj_fh.close()

    def _advance_project_inner(self, project_dir: Path) -> None:
        """
        Core advance logic: revalidate daemon outputs, check RUNNING handlers, submit new work.

        Called by advance_project() which adds per-project log routing around it.
        """
        from hpca.orchestrator.state_tracker import load_state
        from hpca.core.autonomy import AutonomyPolicy
        from hpca.registry.stage import next_runnable, get_enabled, HANDLER_ORDER

        state = load_state(project_dir)
        yaml_cfg    = self._read_project_yaml(project_dir)
        yaml_cfg    = self._migrate_project_yaml(project_dir, yaml_cfg)
        _val_errors = self._validate_schema(yaml_cfg)
        if _val_errors:
            for _ve in _val_errors:
                self.log.warning("[schema] %s: %s", project_dir.name, _ve)
        enabled = get_enabled(yaml_cfg)
        try:
            autonomy = AutonomyPolicy.from_project(yaml_cfg)
        except (TypeError, ValueError) as exc:
            self.log.warning("[autonomy] %s: %s", project_dir.name, exc)
            return

        recovered = state.migrate_exhausted_local_attempts()
        if recovered:
            self.log.info(
                "Recovered cumulative daemon-attempt state for %s: %s",
                project_dir.name, ",".join(recovered),
            )
        if state.migrate_h05_wait_contract():
            self.log.info("Recovered existing-job wait state for %s/h05_cmd",
                          project_dir.name)
        dft_handler = self.handlers.get("h01_dft")
        if dft_handler is not None:
            recovered_dft = dft_handler.migrate_molecular_sick_job(
                project_dir, state, yaml_cfg)
            if recovered_dft:
                self.log.info("Recovered molecular symmetry failures for %s: %s",
                              project_dir.name, ",".join(recovered_dft))

        # 0. Re-validate COMPLETE handlers whose file outputs may be incomplete or
        #    have become invalid.  This includes SLURM stages: an older completion
        #    predicate may have accepted an actively growing trajectory.
        for handler_name in HANDLER_ORDER:
            if handler_name not in enabled:
                continue
            if state.get_stage(handler_name) != "COMPLETE":
                continue
            handler = self.handlers.get(handler_name)
            if handler is None:
                continue
            try:
                if not handler.is_complete(project_dir, state):
                    self.log.info(
                        "Re-validating COMPLETE %s/%s — resetting to PENDING",
                        project_dir.name, handler_name,
                    )
                    state.set_stage(handler_name, "PENDING")
            except Exception:
                pass

        # 1. Check RUNNING handlers — are they still alive or done?
        for handler_name in HANDLER_ORDER:
            if handler_name not in enabled:
                continue
            if state.get_stage(handler_name) != "RUNNING":
                continue

            # Subtask names like "h01_dft.vc_relax" → look up parent handler "h01_dft"
            handler = self.handlers.get(handler_name)
            if handler is None:
                parent = handler_name.rsplit(".", 1)[0]
                handler = self.handlers.get(parent)
            if handler is None:
                continue

            # Daemon handlers: run check_progress() on every poll, then re-check completion
            if handler.is_daemon:
                self.log.debug("%s/%s: daemon was RUNNING — rechecking", project_dir.name, handler_name)
                try:
                    handler.check_progress(project_dir, state)
                except Exception as exc:
                    self.log.debug("%s/%s daemon check_progress: %s", project_dir.name, handler_name, exc)

            # Check completion
            try:
                if handler.is_complete(project_dir, state):
                    self.log.info("COMPLETE: %s/%s", project_dir.name, handler_name)
                    state.set_stage(handler_name, "COMPLETE",
                                    completed_at=datetime.now().isoformat())
                    if handler.is_daemon:
                        autonomy.clear_successful_local_attempts(handler_name, state)
                    handler.on_complete(project_dir, state)
                    self._chaai_on_complete(handler_name, project_dir, state)
                    continue
            except Exception as exc:
                self.log.warning("%s/%s is_complete error: %s", project_dir.name, handler_name, exc)

            # Daemon handler in RUNNING state but is_complete()=False: reset to PENDING
            # so the next poll calls submit() again (e.g. packing placeholder → retry).
            if handler.is_daemon and state.get_stage(handler_name) == "RUNNING":
                self.log.info("Daemon %s/%s RUNNING but not complete — resetting to PENDING",
                              project_dir.name, handler_name)
                state.set_stage(handler_name, "PENDING")
                continue

            # For SLURM jobs: verify job still alive
            if not handler.is_daemon:
                job_id = state.get_job(handler_name)
                if not handler.job_alive(job_id):
                    # Update progress first — check_progress() may detect per-subtask completion
                    # (e.g. DFT vc_relax done but opt not yet submitted)
                    try:
                        handler.check_progress(project_dir, state)
                    except Exception:
                        pass
                    # Re-check stage after progress update
                    if state.get_stage(handler_name) == "COMPLETE":
                        self.log.info("COMPLETE (via check_progress): %s/%s",
                                      project_dir.name, handler_name)
                        handler.on_complete(project_dir, state)
                        self._chaai_on_complete(handler_name, project_dir, state)
                        continue

                    # Re-check is_complete() after check_progress() may have prepared dataset
                    try:
                        if handler.is_complete(project_dir, state):
                            self.log.info("COMPLETE (post-progress check): %s/%s",
                                          project_dir.name, handler_name)
                            state.set_stage(handler_name, "COMPLETE",
                                            completed_at=datetime.now().isoformat())
                            handler.on_complete(project_dir, state)
                            self._chaai_on_complete(handler_name, project_dir, state)
                            continue
                    except Exception as exc:
                        self.log.debug("post-progress is_complete error: %s", exc)

                    self.log.warning("%s/%s job %s is dead — checking auto-fix",
                                     project_dir.name, handler_name, job_id)
                    try:
                        fixed = handler.auto_fix(project_dir, state)
                    except Exception as exc:
                        self.log.warning("auto_fix error %s/%s: %s", project_dir.name, handler_name, exc)
                        fixed = False

                    if fixed:
                        self.log.info("Auto-fixed %s/%s", project_dir.name, handler_name)
                        # If auto_fix() already resubmitted internally (DFT/AIMD do this),
                        # the state will already be RUNNING — don't double-submit
                        new_stage = state.get_stage(handler_name)
                        if new_stage != "RUNNING":
                            try:
                                new_job = handler.submit(project_dir, state)
                                if new_job:
                                    state.set_stage(handler_name, "RUNNING",
                                                    job=new_job,
                                                    resubmit_at=datetime.now().isoformat())
                                    self.log.info("Resubmitted %s/%s → job %s",
                                                 project_dir.name, handler_name, new_job)
                            except Exception as exc:
                                self.log.error("Resubmit failed %s/%s: %s", project_dir.name, handler_name, exc)
                        else:
                            self.log.info("Auto-fixed and resubmitted %s/%s (internal)",
                                         project_dir.name, handler_name)
                    else:
                        state.set_stage(handler_name, "FAILED",
                                        failed_at=datetime.now().isoformat())
                        self.log.warning("FAILED: %s/%s (no auto-fix)", project_dir.name, handler_name)
                    continue

            # Alive SLURM job: update progress
            try:
                handler.check_progress(project_dir, state)
            except Exception as exc:
                self.log.debug("check_progress error %s/%s: %s", project_dir.name, handler_name, exc)

        # Gate: if project.yaml has schema errors, poll running handlers (above)
        # but do not start any new work until the user fixes the config.
        if _val_errors:
            self.log.debug("[schema] %s: %d error(s) — skipping new submissions",
                           project_dir.name, len(_val_errors))
            return

        # 2. Find handlers that can run now
        runnable = next_runnable(state, enabled, project_yaml=yaml_cfg)
        # Multi-combination / crystal-doping parent: only h00_design and h11_manuscript run
        # at the root level. All pipeline handlers (DFT, NEB, AIMD, …) run on sub-projects.
        # h11_manuscript at the root generates a consolidated cross-composition manuscript.
        from hpca.core.combinations import is_combinatorial_parent
        _is_parent = (is_combinatorial_parent(yaml_cfg)
                      or bool(yaml_cfg.get("crystal_doping_variants")))
        if _is_parent:
            runnable = [h for h in runnable if h in ("h00_design", "h11_manuscript")]
        for handler_name in runnable:
            # Subtask names like "h01_dft.vc_relax" → look up parent handler "h01_dft"
            handler = self.handlers.get(handler_name)
            if handler is None:
                parent = handler_name.rsplit(".", 1)[0]
                handler = self.handlers.get(parent)
            if handler is None:
                self.log.warning("No handler registered for %s", handler_name)
                continue

            may_attempt, denial = autonomy.may_attempt(
                handler_name, state, local=handler.is_daemon)
            if not may_attempt:
                self.log.error("AUTONOMY STOP: %s/%s — %s",
                               project_dir.name, handler_name, denial)
                state.set_stage(handler_name, "FAILED", error=denial,
                                failed_at=datetime.now().isoformat())
                continue

            # Extra filesystem check
            try:
                if not handler.can_run(project_dir, state):
                    continue
            except Exception as exc:
                self.log.debug("%s/%s can_run check failed: %s", project_dir.name, handler_name, exc)
                continue

            # Design approval gate: SLURM handlers wait until user approves.
            # (daemon handlers like h00_design are exempt — they run on login node)
            # h00_design writes DESIGN_COMPLETE.md into designed_structures/.
            if not handler.is_daemon and not handler.simulation_approved(project_dir, yaml_cfg):
                design_md = project_dir / "designed_structures" / "DESIGN_COMPLETE.md"
                if design_md.exists():
                    self.log.info(
                        "AWAITING APPROVAL: %s/%s — review %s then:\n"
                        "  touch %s/design/simulation_approved.flag",
                        project_dir.name, handler_name, design_md,
                        project_dir,
                    )
                else:
                    self.log.debug(
                        "AWAITING DESIGN: %s/%s — design not yet complete",
                        project_dir.name, handler_name,
                    )
                continue

            self.log.info("STARTING: %s/%s", project_dir.name, handler_name)
            autonomy.record_attempt(handler_name, state, local=handler.is_daemon)
            try:
                job_id = handler.submit(project_dir, state)
            except Exception as exc:
                self.log.error("submit error %s/%s: %s", project_dir.name, handler_name, exc)
                state.set_stage(handler_name, "FAILED",
                                error=str(exc),
                                failed_at=datetime.now().isoformat())
                continue
            finally:
                autonomy.finish_attempt(handler_name, state, local=handler.is_daemon)

            if handler.is_daemon:
                # Daemon: submit() ran synchronously; re-check completion.
                # Honor FAILED explicitly set by submit() — don't override with RUNNING.
                if state.get_stage(handler_name) == "FAILED":
                    self.log.warning("Daemon %s/%s submit() set FAILED — honoring",
                                     project_dir.name, handler_name)
                    continue
                try:
                    if handler.is_complete(project_dir, state):
                        self.log.info("COMPLETE (daemon): %s/%s", project_dir.name, handler_name)
                        state.set_stage(handler_name, "COMPLETE",
                                        completed_at=datetime.now().isoformat())
                        autonomy.clear_successful_local_attempts(handler_name, state)
                        handler.on_complete(project_dir, state)
                        self._chaai_on_complete(handler_name, project_dir, state)
                    else:
                        # Daemon ran synchronously but is_complete()=False (e.g. packing
                        # produced a placeholder).  Reset to PENDING so the next poll
                        # retries submit() rather than getting stuck in RUNNING forever.
                        self.log.info("Daemon %s/%s not yet complete — retrying next poll",
                                      project_dir.name, handler_name)
                        state.set_stage(handler_name, "PENDING")
                except Exception as exc:
                    self.log.warning("Daemon %s/%s post-submit check: %s",
                                     project_dir.name, handler_name, exc)
                    state.set_stage(handler_name, "PENDING")
            else:
                if job_id:
                    state.set_stage(handler_name, "RUNNING",
                                    job=job_id,
                                    submitted_at=datetime.now().isoformat())
                    self.log.info("SUBMITTED: %s/%s → job %s",
                                 project_dir.name, handler_name, job_id)
                else:
                    state.set_stage(handler_name, "FAILED",
                                    error="sbatch returned None",
                                    failed_at=datetime.now().isoformat())
                    self.log.error("Submit returned no job_id for %s/%s",
                                  project_dir.name, handler_name)

    def _lifecycle_check(self, projects: list[Path]) -> list[Path]:
        """Archive or fail-move inbox projects whose handlers are all terminal.

        Only acts on projects whose parent is the inbox active dir; returns the
        updated project list with moved projects removed so subsequent
        _all_projects_done() and poll logic see the correct set.
        """
        if not self.inbox_mode or not projects:
            return projects
        try:
            from hpca.core.project_discovery import (
                move_to_archived  as _archive,
                move_to_failed    as _fail_move,
                inbox_active_dir  as _active_dir,
            )
            active_resolved = _active_dir().resolve()
        except Exception as exc:
            self.log.debug("lifecycle_check import error: %s", exc)
            return projects

        from hpca.orchestrator.state_tracker import load_state
        from hpca.registry.stage import get_enabled, HANDLER_ORDER

        terminal = {"COMPLETE", "FAILED", "SKIPPED"}
        to_remove: set[Path] = set()

        for proj_dir in projects:
            try:
                if proj_dir.resolve().parent != active_resolved:
                    continue
                state = load_state(proj_dir)
                yaml_cfg = self._read_project_yaml(proj_dir)
                enabled = set(get_enabled(yaml_cfg))
                stages = [state.get_stage(h) for h in HANDLER_ORDER if h in enabled]
                if not stages or not all(s in terminal for s in stages):
                    continue
                if any(s == "FAILED" for s in stages):
                    self.log.info("Project %s: all-terminal with FAILED → moving to failed inbox",
                                  proj_dir.name)
                    _fail_move(proj_dir)
                else:
                    self.log.info("Project %s: all COMPLETE → moving to archived inbox",
                                  proj_dir.name)
                    _archive(proj_dir)
                to_remove.add(proj_dir)
            except Exception as exc:
                self.log.debug("lifecycle_check %s: %s", proj_dir.name, exc)

        return [p for p in projects if p not in to_remove]

    def _chaai_on_complete(self, handler_name: str, project_dir: Path,
                           state) -> None:
        """Notify CHAAI handler that a stage completed (for data generation)."""
        chaai = self.handlers.get("h12_chaai")
        if chaai is None:
            return
        try:
            chaai.on_stage_complete(handler_name, project_dir, state)
        except Exception as exc:
            self.log.debug("CHAAI notify error: %s", exc)

    def _migrate_project_yaml(self, project_dir: Path, yaml_cfg: dict) -> dict:
        """Apply schema migration; rewrite project.yaml on-disk when fields changed."""
        try:
            from hpca.core.project_schema import migrate
            migrated = migrate(yaml_cfg)
        except Exception:
            return yaml_cfg
        if migrated == yaml_cfg:
            return migrated
        try:
            import yaml
            p = project_dir / "project.yaml"
            p.write_text(yaml.dump(migrated, default_flow_style=False,
                                   sort_keys=False, allow_unicode=True))
            self.log.info("[schema] Migrated project.yaml for %s", project_dir.name)
        except Exception as exc:
            self.log.warning("[schema] Could not write migrated yaml for %s: %s",
                             project_dir.name, exc)
        return migrated

    def _validate_schema(self, yaml_cfg: dict) -> list[str]:
        """Return list of schema validation error strings (empty = valid)."""
        try:
            from hpca.core.project_schema import validate
            return validate(yaml_cfg)
        except Exception as exc:
            self.log.debug("[schema] validate error: %s", exc)
            return []

    @staticmethod
    def _read_project_yaml(project_dir: Path) -> dict:
        """Read project.yaml and return parsed dict, or {} on missing/parse error."""
        p = project_dir / "project.yaml"
        if not p.exists():
            return {}
        try:
            import yaml
            return yaml.safe_load(p.read_text()) or {}
        except Exception:
            return {}

    def _self_restart(self) -> None:
        """Submit sub_orchestrator_daemon.sh via sbatch so the daemon continues after wall-time expiry."""
        sub_script = ORCH_DIR / "sub_orchestrator_daemon.sh"
        if not sub_script.exists():
            self.log.error("sub_orchestrator_daemon.sh not found — cannot self-restart")
            return
        try:
            result = subprocess.run(
                ["sbatch", "--parsable", str(sub_script), "--resume"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                new_job = result.stdout.strip()
                self.log.info("Self-restart submitted as job %s", new_job)
            else:
                self.log.error("Self-restart sbatch failed: %s", result.stderr)
        except Exception as exc:
            self.log.error("Self-restart error: %s", exc)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    """Parse CLI arguments and start the HPCAOrchestrator polling loop."""
    parser = argparse.ArgumentParser(
        description="HPCA simulation-type orchestration daemon"
    )
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing state (append to latest log)")
    parser.add_argument("--project", default=None,
                        help="Filter to only this project name (substring match)")
    parser.add_argument("--root", default=str(Path.cwd()),
                        help="Project directory (or parent of projects) to manage (default: cwd)")
    parser.add_argument("--inbox", action="store_true",
                        help="Discover projects from daemon inbox active dir; "
                             "archive/fail-move them when all handlers are terminal")
    parser.add_argument("--log-dir", default=None,
                        help="Directory for orchestrator log file "
                             "(default: {root}/logs/ for single-project mode)")
    args = parser.parse_args()

    root = Path(args.root)
    if args.log_dir:
        log_dir = Path(args.log_dir)
    elif any((root / m).exists() for m in _PROJECT_MARKERS):
        # Single-project mode: logs go inside the project, never in the hpca package
        log_dir = root / "logs"
    else:
        log_dir = LOG_DIR

    orch = HPCAOrchestrator(
        root=root,
        project_filter=args.project,
        resume=args.resume,
        log_dir=log_dir,
        inbox_mode=args.inbox,
    )
    try:
        orch.run()
    finally:
        # Clean up PID files written by `hpca start` (pipeline.py uses orch_daemon.pid)
        for pid_name in ("orch_daemon.pid", "orchestrator.pid"):
            pid_file = root / "logs" / pid_name
            if not pid_file.exists():
                pid_file = log_dir / pid_name
            if pid_file.exists():
                try:
                    pid_file.unlink()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
