"""
hpca/core/slurm_submit.py — Consolidated SLURM job submission, monitoring,
and management utilities.

This module is the single authoritative source for all SLURM interactions in
the HPCA codebase. Handler files should import from here rather than each
duplicating their own squeue/sacct/sbatch wrappers.

Note: base.py's SimulationHandler.sbatch() and .job_alive() remain intact for
backward compatibility with existing handler code. New code should use the
module-level functions here.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

log = logging.getLogger("hpca.slurm")

# States considered "alive" (job is still occupying a slot or running)
_ALIVE_STATES = frozenset({"RUNNING", "PENDING", "COMPLETING", "CONFIGURING"})

# Terminal states that will never change once observed — safe to cache for a
# long time so repeated lookups of an already-finished job never hit sacct
# again within the TTL below.
_TERMINAL_STATES = frozenset({
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
    "OUT_OF_MEMORY", "DEADLINE", "PREEMPTED", "BOOT_FAIL",
})

# Transient sbatch failure messages that warrant a retry
# ── 0. Shared, TTL-cached whole-user squeue snapshot ─────────────────────────
#
# Background (2026-08-12): NREL flagged the hpca-daemon job for issuing ~20x
# the RPC load of the next-highest cluster user against slurmctld (mostly
# REQUEST_JOB_INFO_SINGLE / REQUEST_FED_INFO / REQUEST_PARTITION_INFO, arriving
# every 10-15ms). Root cause: hpca_orchestrator.py's ThreadPoolExecutor fans
# out to one worker thread per project every poll cycle (uncapped, up to
# os.cpu_count()) and each RUNNING handler independently shelled out to its
# own uncached `squeue -j <single job>` (via job_state() below) or its own
# uncached `squeue -u $USER` (h05_cmd._get_alive_jobs()). With ~67 concurrent
# sub-projects under LYC alone, that is dozens-to-hundreds of fresh squeue
# subprocesses fired within the same few seconds, every 60s, continuously.
#
# Fix: every caller now reads from one process-wide, thread-safe, short-TTL
# snapshot (one real `squeue` call refreshes it; everyone else in that window
# gets the cached dict). Terminal (finished) job states are cached far longer
# since they cannot change. This collapses each poll cycle's O(projects)
# scheduler RPCs down to O(1) per orchestrator process.

_SNAPSHOT_TTL = 20.0     # seconds a squeue snapshot is considered fresh
_TERMINAL_TTL = 600.0    # seconds a terminal (finished) job state is cached

_snapshot_lock = threading.Lock()
_snapshot: dict[str, str] = {}
_snapshot_ts: float = 0.0
_terminal_cache: dict[str, str] = {}
_terminal_cache_ts: dict[str, float] = {}


def _refresh_snapshot_locked(user: str) -> None:
    """Refresh the shared squeue snapshot. Caller must hold _snapshot_lock."""
    global _snapshot, _snapshot_ts
    import subprocess
    try:
        out = subprocess.check_output(
            ["squeue", "-u", user, "--noheader", "--format=%i|%T"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
        )
        snap: dict[str, str] = {}
        for line in out.strip().splitlines():
            parts = line.split("|")
            if len(parts) >= 2:
                snap[parts[0].strip()] = parts[1].strip()
        _snapshot = snap
    except Exception as exc:
        log.debug("squeue snapshot refresh failed: %s", exc)
    _snapshot_ts = time.time()


def user_job_snapshot(user: str | None = None, ttl: float = _SNAPSHOT_TTL) -> dict[str, str]:
    """Return {job_id: state} for all of *user*'s currently queued/running jobs.

    Backed by a single shared TTL-cached `squeue` call — see module docstring
    above. Safe to call from many threads/handlers every poll; only the first
    caller after the TTL expires actually shells out.
    """
    user = user or os.environ.get("USER", "")
    with _snapshot_lock:
        if time.time() - _snapshot_ts > ttl:
            _refresh_snapshot_locked(user)
        return dict(_snapshot)


def alive_job_ids(user: str | None = None) -> set[str]:
    """Return the set of this user's job IDs currently in an alive state."""
    return {jid for jid, state in user_job_snapshot(user).items() if state in _ALIVE_STATES}


# ── 1. Job state querying ─────────────────────────────────────────────────────

def job_state(job_id: str | int) -> str:
    """Return SLURM job state string.

    Returns one of: RUNNING, PENDING, COMPLETING, CONFIGURING, COMPLETED,
    FAILED, CANCELLED, TIMEOUT, NODE_FAIL, or UNKNOWN.

    Strategy: consult the shared squeue snapshot first (fast, covers active
    jobs, and costs no new RPC beyond the shared TTL refresh); fall back to
    sacct for completed/historical jobs, caching terminal results so a job
    that finished once never triggers a repeat sacct call within the TTL.
    """
    import subprocess

    jid = str(job_id)

    # Fast path — shared snapshot covers RUNNING/PENDING/COMPLETING/CONFIGURING
    snap = user_job_snapshot()
    if jid in snap:
        return snap[jid]

    # Already-known terminal state — skip sacct entirely
    with _snapshot_lock:
        cached = _terminal_cache.get(jid)
        cached_ts = _terminal_cache_ts.get(jid, 0.0)
    if cached is not None and time.time() - cached_ts < _TERMINAL_TTL:
        return cached

    # Slow path — sacct covers completed/failed/cancelled jobs
    state = "UNKNOWN"
    try:
        out = subprocess.check_output(
            ["sacct", "-j", jid, "--noheader", "--format=State", "--parsable2"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
        ).strip()
        if out:
            # sacct may return multiple lines (job steps); use the first (job itself)
            first = out.splitlines()[0].strip().split("|")[0].strip()
            # Strip trailing " by ..." annotation that CANCELLED jobs sometimes have
            state = (first.split()[0] if first else "UNKNOWN")
    except Exception as exc:
        log.debug("sacct error for job %s: %s", jid, exc)

    if state in _TERMINAL_STATES:
        with _snapshot_lock:
            _terminal_cache[jid] = state
            _terminal_cache_ts[jid] = time.time()
    return state


def job_alive(job_id: str | int | None) -> bool:
    """Return True if job is RUNNING, PENDING, COMPLETING, or CONFIGURING."""
    if not job_id:
        return False
    return job_state(job_id) in _ALIVE_STATES


def jobs_alive(job_ids: list[str]) -> dict[str, bool]:
    """Batch-check multiple job IDs in a single squeue call.

    Returns a dict mapping each job_id → bool. Jobs not found in squeue are
    then checked via sacct (batch) to distinguish completed from unknown.
    Falls back gracefully to per-job queries if batch calls fail.
    """
    import subprocess

    if not job_ids:
        return {}

    result: dict[str, bool] = {jid: False for jid in job_ids}
    jid_set = set(str(j) for j in job_ids)

    # Single squeue call for all job IDs
    try:
        out = subprocess.check_output(
            ["squeue", "-j", ",".join(jid_set), "--noheader", "--format=%i %T"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
        ).strip()
        seen: set[str] = set()
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                jid, state = parts[0], parts[1]
                if jid in jid_set:
                    result[jid] = state in _ALIVE_STATES
                    seen.add(jid)
        not_seen = jid_set - seen
    except subprocess.CalledProcessError:
        # All jobs may be done; check via sacct below
        not_seen = jid_set
    except Exception as exc:
        log.debug("jobs_alive squeue batch error: %s", exc)
        # Fall back to individual queries
        for jid in job_ids:
            result[str(jid)] = job_alive(jid)
        return result

    # For jobs not in squeue, check sacct to see completed vs unknown
    if not_seen:
        try:
            out = subprocess.check_output(
                ["sacct", "-j", ",".join(not_seen), "--noheader",
                 "--format=JobID,State", "--parsable2"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=15,
            ).strip()
            for line in out.splitlines():
                cols = line.split("|")
                if len(cols) >= 2:
                    jid = cols[0].strip().split(".")[0]  # strip step suffix
                    state = cols[1].strip().split()[0]
                    if jid in not_seen:
                        result[jid] = state in _ALIVE_STATES
        except Exception as exc:
            log.debug("jobs_alive sacct batch error: %s", exc)

    return result


def job_elapsed_seconds(job_id: str | int) -> int | None:
    """Return elapsed seconds for a running or completed job via sacct.

    Returns None if the job is not found or elapsed time is unavailable.
    """
    import subprocess

    jid = str(job_id)
    try:
        out = subprocess.check_output(
            ["sacct", "-j", jid, "--noheader", "--format=Elapsed", "--parsable2"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
        ).strip()
        if not out:
            return None
        elapsed_str = out.splitlines()[0].strip().split("|")[0].strip()
        if not elapsed_str or elapsed_str == "Unknown":
            return None
        return parse_wall_seconds(elapsed_str)
    except Exception as exc:
        log.debug("job_elapsed_seconds error for %s: %s", jid, exc)
        return None


def job_remaining_seconds(job_id: str | int, wall_seconds: int) -> int | None:
    """Estimate remaining walltime as wall_seconds - elapsed.

    Returns None if elapsed time cannot be determined.
    """
    elapsed = job_elapsed_seconds(job_id)
    if elapsed is None:
        return None
    return max(0, wall_seconds - elapsed)


# ── 2. Job submission ─────────────────────────────────────────────────────────

def sbatch(
    script: Path,
    *,
    cwd: Path | None = None,
    extra_args: list[str] | None = None,
    max_retries: int = 3,
    retry_delay_s: float = 30.0,
    dependency: str | None = None,
) -> str | None:
    """Submit a SLURM batch script with retry on transient scheduler failures.

    Args:
        script: path to the .sh submit script
        cwd: working directory for sbatch (defaults to script.parent)
        extra_args: additional sbatch flags, e.g. ["--hold"]
        max_retries: number of attempts before giving up
        retry_delay_s: seconds to wait between attempts
        dependency: SLURM dependency string, e.g. "afterok:12345"

    Returns:
        Job ID string on success, or None after all retries are exhausted.
    """
    import subprocess

    run_cwd = cwd or script.parent
    cmd: list[str] = ["sbatch"]
    if dependency:
        cmd += ["--dependency", dependency]
    if extra_args:
        cmd += extra_args
    cmd.append(str(script))

    for attempt in range(1, max_retries + 1):
        try:
            result = subprocess.run(
                cmd,
                cwd=str(run_cwd),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split()
                if parts:
                    job_id = parts[-1]
                    log.info("sbatch submitted %s → job %s", script.name, job_id)
                    return job_id
                log.warning("sbatch succeeded but no job ID in output: %s",
                            result.stdout.strip())
            else:
                stderr = result.stderr.strip()
                from hpca.scheduler.errors import FailureClass, classify_scheduler_failure
                is_transient = classify_scheduler_failure(stderr) is FailureClass.TRANSIENT
                if is_transient and attempt < max_retries:
                    log.warning("sbatch transient failure (attempt %d/%d): %s — retrying in %.0fs",
                                attempt, max_retries, stderr, retry_delay_s)
                    time.sleep(retry_delay_s)
                    continue
                log.error("sbatch failed for %s: %s", script, stderr)
                return None
        except subprocess.TimeoutExpired:
            log.warning("sbatch timeout (attempt %d/%d) for %s", attempt, max_retries, script)
            if attempt < max_retries:
                time.sleep(retry_delay_s)
        except Exception as exc:
            log.error("sbatch error for %s: %s", script, exc)
            return None

    log.error("sbatch: all %d retries exhausted for %s", max_retries, script)
    return None


# ── 3. SBATCH header builder ──────────────────────────────────────────────────

def sbatch_header(
    job_name: str,
    wall: str,
    n_nodes: int = 1,
    n_tasks: int = 96,
    mem_gb: int | None = None,
    *,
    account: str | None = None,
    gres: str | None = None,
    output: str = "slurm-%j.out",
    cfg=None,
) -> str:
    """Return a complete #SBATCH header block as a string (including #!/bin/bash).

    Reads account from platform.yaml via cfg (or Config.get()) if not provided
    explicitly.  No --partition is written; SLURM auto-selects the partition
    based on wall time.

    Example output::

        #!/bin/bash
        #SBATCH --job-name=LYC_aimd
        #SBATCH --account=your_account
        #SBATCH --nodes=2
        #SBATCH --ntasks-per-node=96
        #SBATCH --time=72:00:00
        #SBATCH --output=slurm-%j.out
    """
    if cfg is None:
        from hpca.core.config import Config
        cfg = Config.get()

    acct = account or cfg.account("standard")

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={acct}",
        f"#SBATCH --nodes={n_nodes}",
        f"#SBATCH --ntasks-per-node={n_tasks}",
        f"#SBATCH --time={wall}",
        f"#SBATCH --output={output}",
    ]

    if mem_gb is not None:
        lines.append(f"#SBATCH --mem={mem_gb}G")

    if gres is not None:
        lines.append(f"#SBATCH --gres={gres}")

    return "\n".join(lines) + "\n"


# ── 4. GPU SBATCH header ──────────────────────────────────────────────────────

def sbatch_gpu_header(
    job_name: str,
    wall: str,
    n_gpus: int = 4,
    n_tasks: int = 4,
    *,
    account: str | None = None,
    gpu_type: str = "h100",
    output: str = "slurm-%j.out",
    cfg=None,
) -> str:
    """Return a GPU #SBATCH header block as a string (including #!/bin/bash).

    Sets --gres=gpu:{gpu_type}:{n_gpus}.  No --partition is written; SLURM
    auto-selects the partition based on wall time and GPU resource request.

    Example output::

        #!/bin/bash
        #SBATCH --job-name=MLMD_run
        #SBATCH --account=your_account
        #SBATCH --nodes=1
        #SBATCH --ntasks-per-node=4
        #SBATCH --gres=gpu:h100:4
        #SBATCH --time=48:00:00
        #SBATCH --output=slurm-%j.out
    """
    if cfg is None:
        from hpca.core.config import Config
        cfg = Config.get()

    acct = account or cfg.account("gpu_h100")
    gres = f"gpu:{gpu_type}:{n_gpus}"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={acct}",
        "#SBATCH --nodes=1",
        f"#SBATCH --ntasks-per-node={n_tasks}",
        f"#SBATCH --gres={gres}",
        f"#SBATCH --time={wall}",
        f"#SBATCH --output={output}",
    ]

    return "\n".join(lines) + "\n"


# ── 5. Environment activation helpers ────────────────────────────────────────

def source_activate_block(env_path: str | Path) -> str:
    """Return shell lines to activate a conda or venv environment.

    Detects conda by checking for a conda-meta/ subdirectory; otherwise
    treats the path as a virtualenv and sources bin/activate.

    Returns a multi-line string (no trailing newline) suitable for embedding
    in a bash submission script.
    """
    p = Path(env_path)
    if (p / "conda-meta").is_dir():
        # Conda environment — use 'conda activate'
        return (
            "# Activate conda environment\n"
            f'eval "$(conda shell.bash hook)"\n'
            f"conda activate {p}"
        )
    else:
        # Standard virtualenv / venv
        return (
            "# Activate virtual environment\n"
            f"source {p}/bin/activate"
        )


def module_load_block(modules: list[str]) -> str:
    """Return 'module load ...' shell lines for a list of module names.

    Prepends 'module purge' to ensure a clean environment.
    Returns a multi-line string (no trailing newline).
    """
    if not modules:
        return ""
    lines = ["module purge"] + [f"module load {m}" for m in modules]
    return "\n".join(lines)


def module_bundle_lines(bundle: str, tolerant: bool = False) -> str:
    """Render 'module load …' for a named ``hpc.modules`` bundle from platform.yaml.

    tolerant=True appends ``2>/dev/null || true`` so a missing module is non-fatal
    (used in generated sbatch scripts that may land on heterogeneous partitions).
    Returns '' when the bundle is empty or undefined (sites without env modules),
    so callers can splice the result into script templates unconditionally.
    """
    from hpca.core.config import Config
    mods = Config.get().modules(bundle)
    if not mods:
        return ""
    suffix = " 2>/dev/null || true" if tolerant else ""
    return f"module load {' '.join(mods)}{suffix}\n"


def ld_library_path_block(paths: list[str]) -> str:
    """Return export LD_LIBRARY_PATH=... shell lines.

    Prepends the given paths to any existing LD_LIBRARY_PATH.
    Returns a single export line (no trailing newline), or empty string
    if paths is empty.
    """
    if not paths:
        return ""
    colon_paths = ":".join(str(p) for p in paths)
    return f"export LD_LIBRARY_PATH={colon_paths}:$LD_LIBRARY_PATH"


# ── 6. Walltime parser / formatter ───────────────────────────────────────────

def parse_wall_seconds(wall: str) -> int:
    """Parse a SLURM walltime string into total seconds.

    Accepts the following formats:
        HH:MM:SS   (e.g. "72:00:00")
        D-HH:MM:SS (e.g. "1-12:00:00")
        MM:SS      (e.g. "05:30")

    Raises ValueError for unrecognized formats.
    """
    wall = wall.strip()

    days = 0
    if "-" in wall:
        day_part, wall = wall.split("-", 1)
        days = int(day_part)

    parts = wall.split(":")
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        h, m, s = 0, int(parts[0]), int(parts[1])
    else:
        raise ValueError(f"Unrecognized walltime format: {wall!r}")

    return days * 86400 + h * 3600 + m * 60 + s


def format_wall(seconds: int) -> str:
    """Format total seconds into HH:MM:SS walltime string.

    Hours may exceed 24 (SLURM accepts this). Days are not used — the output
    is always HH:MM:SS so it is compatible with #SBATCH --time.
    """
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def estimate_vasp_wall(
    n_atoms: int,
    nsw: int,
    kpoints: int = 1,
    mode: str = "relax",
    cfg=None,
) -> str:
    """Estimate VASP walltime using the perf_model in platform.yaml.

    Args:
        n_atoms: number of atoms in the cell
        nsw: number of ionic steps (NSW)
        kpoints: k-point multiplier (1 = Gamma-only, >1 for k-mesh jobs)
        mode: 'relax' | 'aimd' — selects the appropriate rate and safety factor
        cfg: Config instance (defaults to Config.get())

    Returns:
        Walltime string in HH:MM:SS format.
    """
    if cfg is None:
        from hpca.core.config import Config
        cfg = Config.get()

    if mode == "aimd":
        rate = cfg.perf("vasp_s_per_atom_per_step_aimd")
        safety = cfg.perf("vasp_safety_factor_aimd", 1.25)
    else:  # relax or anything else
        rate = cfg.perf("vasp_s_per_atom_per_step_relax")
        safety = cfg.perf("vasp_safety_factor_relax", 1.30)

    # kpoints multiplier: Gamma-only is the baseline; k-mesh scales up
    kpt_factor = max(1.0, float(kpoints))
    total_seconds = n_atoms * nsw * rate * kpt_factor * safety
    return format_wall(int(total_seconds))
