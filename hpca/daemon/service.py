"""Long-running HPCA daemon supervisor."""
from __future__ import annotations

import getpass
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from hpca.daemon.config import DaemonConfig
from hpca.daemon.control import get_desired_state, set_desired_state
from hpca.daemon.inbox import Inbox
from hpca.daemon.leases import Lease
from hpca.daemon.reconciliation import process_matches
from hpca.daemon.schemas import ProjectRequest, utc_now
from hpca.core.atomic import atomic_write_text

log = logging.getLogger("hpca.daemon")


@dataclass
class ManagedProject:
    request: ProjectRequest
    project_root: Path
    process: subprocess.Popen | None
    pid: int
    lease: Lease
    restarts: int = 0


class DaemonService:
    """Validate requests and supervise exactly one orchestrator per project."""

    def __init__(self, config: DaemonConfig):
        self.config = config
        self.inbox = Inbox(config.inbox)
        self.projects: dict[str, ManagedProject] = {}
        self.started = time.monotonic()
        self.stopping = False
        self.successor_job: str | None = None
        self.leader = Lease(config.inbox / "locks" / "daemon.lock")
        self.pkg_mtime = self._pkg_mtime()

    def run(self, once: bool = False) -> None:
        self.inbox.initialize()
        if not self.leader.acquire():
            self._request_handoff()
            deadline = time.monotonic() + min(self.config.successor_before_hours * 3600, 3600)
            while time.monotonic() < deadline and not self.leader.acquire():
                time.sleep(min(self.config.poll_seconds, 10))
            if not self.leader.acquired:
                raise RuntimeError("Existing HPCA daemon did not complete handoff")
        self._install_signals()
        self._write_identity("SERVING")
        try:
            while not self.stopping:
                self.cycle()
                if once:
                    break
                time.sleep(self.config.poll_seconds)
        finally:
            self._drain()
            self._write_identity("STOPPED")
            self.leader.release()

    def cycle(self) -> None:
        if (self.config.inbox / "handoff.request.json").exists():
            self.stopping = True
            return
        self._reconcile_legacy_active()
        self._accept_incoming()
        self._apply_project_controls()
        self._dispatch_queued()
        self._reap()
        self._hot_reload_if_changed()
        self._maybe_submit_successor()
        self._write_identity("SERVING")

    def _reconcile_legacy_active(self) -> None:
        """Convert compatibility pointers left for the former Bash daemon.

        A packaged daemon never executes a plain project YAML directly.  It
        ensures the canonical versioned request exists, then archives the
        pointer.  Project-local desired state remains authoritative.
        """
        import yaml
        for pointer in sorted((self.config.inbox / "active").glob("*.yaml")):
            try:
                data = yaml.safe_load(pointer.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                continue
            if {"request_id", "project_id", "project_yaml"} <= set(data):
                continue
            root_value = data.get("project_path") or data.get("project_root") or data.get("root")
            if not root_value:
                continue
            root = Path(root_value).resolve()
            project_yaml = root / "project.yaml"
            if not project_yaml.is_file() or not any(
                    root.is_relative_to(allowed) for allowed in self.config.allowed_roots):
                continue
            project_id = str(data.get("name") or pointer.stem).strip().lower().replace(" ", "-")
            register_project(self.config, project_yaml, project_id)
            archive = self.config.inbox / "archive" / f"{pointer.stem}.legacy-active.yaml"
            if archive.exists():
                archive = archive.with_name(f"{archive.stem}.{int(time.time())}{archive.suffix}")
            os.replace(pointer, archive)
            self.inbox.event("legacy_pointer.archived", project_id, pointer.stem,
                             f"desired_state={get_desired_state(root)}")

    def _accept_incoming(self) -> None:
        for path, reason in list(self.inbox.malformed("incoming")):
            self.inbox.transition(path, "rejected", reason)
        for path, request in list(self.inbox.requests("incoming")):
            try:
                request.validate(self.config.allowed_roots)
                self.inbox.transition(path, "queued")
            except Exception as exc:
                self.inbox.transition(path, "rejected", str(exc))

    def _dispatch_queued(self) -> None:
        capacity = self.config.max_projects - len(self.projects)
        for path, request in list(self.inbox.requests("queued"))[:max(0, capacity)]:
            project_root = request.validate(self.config.allowed_roots)
            if get_desired_state(project_root) != "RUNNING":
                self.inbox.transition(path, "paused", "project-local desired_state=STOPPED")
                continue
            if self._acquire_and_launch(request, project_root):
                self.inbox.transition(path, "active")

    def _acquire_and_launch(self, request: ProjectRequest, project_root: Path) -> bool:
        """Acquire this project's lease, then adopt its live orchestrator or spawn one.

        Shared by queued dispatch and by resuming requests already sitting in
        ``active/`` whose managed-process entry was lost (e.g. daemon restart).
        Returns whether the project is now tracked in ``self.projects``.
        """
        lease = Lease(self.config.inbox / "locks" / f"project-{request.project_id}.lock")
        if not lease.acquire():
            return False
        try:
            if self._adopt_runtime(request, project_root, lease):
                return True
            process = self._spawn_orchestrator(project_root)
            managed = ManagedProject(request, project_root, process, process.pid, lease)
            self.projects[request.project_id] = managed
            self.inbox.write_runtime(request.project_id, self._runtime(managed, "RUNNING"))
            return True
        except Exception:
            lease.release()
            raise

    def _adopt_runtime(self, request: ProjectRequest, root: Path, lease: Lease) -> bool:
        path = self.inbox.runtime_path(request.project_id)
        if not path.exists():
            return False
        try:
            runtime = json.loads(path.read_text(encoding="utf-8"))
            pid = int(runtime["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False
        if not process_matches(pid, root):
            return False
        self.projects[request.project_id] = ManagedProject(
            request, root, None, pid, lease, int(runtime.get("restarts", 0)))
        return True

    def _reap(self) -> None:
        for project_id, managed in list(self.projects.items()):
            if managed.process is None:
                alive = process_matches(managed.pid, managed.project_root)
                returncode = None if alive else 1
            else:
                returncode = managed.process.poll()
                alive = returncode is None
            if alive:
                continue
            if (returncode not in (None, 0)
                    and managed.restarts < self.config.max_orchestrator_restarts):
                managed.restarts += 1
                managed.process = self._spawn_orchestrator(managed.project_root)
                managed.pid = managed.process.pid
                self.inbox.write_runtime(project_id, self._runtime(managed, "RESTARTED"))
                self.inbox.event("orchestrator.restarted", project_id,
                                 managed.request.request_id,
                                 f"attempt={managed.restarts} previous_exit={returncode}")
                continue
            state = "completed" if returncode in (None, 0) else "failed"
            active_request = self.config.inbox / "active" / f"{managed.request.request_id}.yaml"
            if active_request.exists():
                self.inbox.transition(active_request, state, f"orchestrator exit={returncode}")
            self.inbox.write_runtime(project_id, self._runtime(managed, state.upper()))
            managed.lease.release()
            del self.projects[project_id]

    def _pkg_mtime(self) -> float:
        """Return the newest mtime among all hpca/**/*.py files, plus platform.yaml.

        platform.yaml is read once per orchestrator process and cached in a
        process-wide singleton (hpca.core.config.Config) -- editing it on disk
        does nothing for an orchestrator that was already running, silently.
        (2026-08-19: found HC_Na_electrolyte_interface running an AIMD NPT job
        for 15h+ with a POTIM fix from platform.yaml that was never picked up,
        because its orchestrator process predated the edit and .yaml changes
        weren't watched here -- only .py files were. The job thermally ran
        away to >15,000 K by step 30 and nobody was told. Watching the config
        file's mtime alongside the package's closes that gap: any config edit
        now triggers the same hot-reload/respawn path a code change does.)
        """
        src = Path(__file__).resolve().parents[1]  # hpca/ package root
        newest_py = max((p.stat().st_mtime for p in src.rglob("*.py")), default=0.0)
        platform_yaml = src / "config" / "platform.yaml"
        yaml_mtime = platform_yaml.stat().st_mtime if platform_yaml.is_file() else 0.0
        return max(newest_py, yaml_mtime)

    def _hot_reload_if_changed(self) -> None:
        """Restart every managed orchestrator when hpca package code has changed on disk.

        A code change alone does not affect an already-running orchestrator
        subprocess — Python does not re-import modules a live process already
        loaded. Restarting the subprocess (not this daemon) is what picks up
        new code, mirroring the former Bash daemon's hot-reload behavior.
        """
        current = self._pkg_mtime()
        if current <= self.pkg_mtime:
            return
        self.pkg_mtime = current
        if not self.projects:
            return
        log.info("hpca package changed — hot-reloading %d orchestrator(s)", len(self.projects))
        for project_id, managed in list(self.projects.items()):
            self._terminate_and_wait(managed)
            process = self._spawn_orchestrator(managed.project_root)
            managed.process = process
            managed.pid = process.pid
            managed.restarts = 0
            self.inbox.write_runtime(project_id, self._runtime(managed, "RUNNING"))
        self.inbox.event("daemon.hot_reload", "", "",
                         f"{len(self.projects)} orchestrator(s) restarted")

    def _terminate_and_wait(self, managed: ManagedProject) -> None:
        """Terminate one orchestrator and block until it exits (SIGKILL after the grace period)."""
        grace = self.config.hot_reload_grace_seconds
        if managed.process is not None:
            if managed.process.poll() is None:
                managed.process.terminate()
                try:
                    managed.process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    managed.process.kill()
                    managed.process.wait(timeout=grace)
            return
        if not process_matches(managed.pid, managed.project_root):
            return
        os.kill(managed.pid, signal.SIGTERM)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and process_matches(managed.pid, managed.project_root):
            time.sleep(1)
        if process_matches(managed.pid, managed.project_root):
            os.kill(managed.pid, signal.SIGKILL)

    def _apply_project_controls(self) -> None:
        """Reconcile inbox lifecycle with desired state stored in each project."""
        for path, request in list(self.inbox.requests("queued")):
            root = request.validate(self.config.allowed_roots)
            if get_desired_state(root) == "STOPPED":
                self.inbox.transition(path, "paused", "project-local stop")
        for path, request in list(self.inbox.requests("paused")):
            root = request.validate(self.config.allowed_roots)
            if get_desired_state(root) == "RUNNING":
                self.inbox.transition(path, "queued", "project-local start")
        # Reconcile active requests even when this daemon did not spawn/adopt
        # them.  This is essential after daemon restart or an interrupted
        # handoff: an active inbox pointer must never override a local STOPPED
        # control file, and a RUNNING one must not be left orphaned forever —
        # nothing else re-dispatches a request that is already in active/.
        managed_request_ids = {
            managed.request.request_id for managed in self.projects.values()
        }
        for path, request in list(self.inbox.requests("active")):
            if request.request_id in managed_request_ids:
                continue
            root = request.validate(self.config.allowed_roots)
            if get_desired_state(root) == "STOPPED":
                self.inbox.transition(path, "paused", "orphan active request; project-local stop")
            else:
                self._acquire_and_launch(request, root)
        for project_id, managed in list(self.projects.items()):
            if get_desired_state(managed.project_root) != "STOPPED":
                continue
            if managed.process is not None and managed.process.poll() is None:
                managed.process.terminate()
            elif managed.process is None and process_matches(managed.pid, managed.project_root):
                os.kill(managed.pid, signal.SIGTERM)
            active = self.config.inbox / "active" / f"{managed.request.request_id}.yaml"
            if active.exists():
                self.inbox.transition(active, "paused", "project-local stop")
            self.inbox.write_runtime(project_id, self._runtime(managed, "PAUSED"))
            managed.lease.release()
            del self.projects[project_id]

    def _maybe_submit_successor(self) -> None:
        if self.successor_job or time.monotonic() - self.started < self.config.successor_after_seconds:
            return
        script = self.config.successor_script
        if script is None:
            log.warning("Successor threshold reached but no successor script is configured")
            return
        from hpca.scheduler import get_scheduler
        self.successor_job = get_scheduler().submit(script, cwd=script.parent)
        if not self.successor_job:
            raise RuntimeError(f"Successor submission failed for {script}")
        self.inbox.event("daemon.successor_submitted", "", self.successor_job)

    def _runtime(self, managed: ManagedProject, state: str) -> dict:
        return {"project_id": managed.request.project_id, "request_id": managed.request.request_id,
                "project_root": str(managed.project_root), "pid": managed.pid,
                "state": state, "restarts": managed.restarts,
                "updated_at": utc_now(), "daemon_pid": os.getpid()}

    @staticmethod
    def _spawn_orchestrator(project_root: Path) -> subprocess.Popen:
        """Start one isolated project orchestrator and route output to its durable log."""
        log_path = project_root / "logs" / "hpca_orch.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "hpca.orchestrator.hpca_orchestrator",
                   "--resume", "--root", str(project_root)]
        with log_path.open("a", encoding="utf-8") as stream:
            return subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True)

    def _write_identity(self, state: str) -> None:
        from hpca.core.atomic import atomic_write_json
        atomic_write_json(self.config.inbox / "daemon.json", {
            "pid": os.getpid(), "state": state, "updated_at": utc_now(),
            "successor_job": self.successor_job, "managed_projects": sorted(self.projects)})

    def _install_signals(self) -> None:
        def stop(_signum, _frame):
            self.stopping = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, stop)

    def _request_handoff(self) -> None:
        """Ask the current leader to drain; used by a scheduled successor."""
        from hpca.core.atomic import atomic_write_json
        atomic_write_json(self.config.inbox / "handoff.request.json", {
            "requesting_pid": os.getpid(), "requested_at": utc_now()})

    def _drain(self) -> None:
        for managed in self.projects.values():
            if managed.process is not None and managed.process.poll() is None:
                managed.process.terminate()
            managed.lease.release()
        (self.config.inbox / "handoff.request.json").unlink(missing_ok=True)


def register_project(config: DaemonConfig, project_yaml: Path, project_id: str | None = None) -> Path:
    """Validate and atomically register a canonical project.yaml."""
    path = Path(project_yaml).resolve(strict=True)
    data = __import__("yaml").safe_load(path.read_text(encoding="utf-8")) or {}
    raw_id = project_id or data.get("project", {}).get("id") or data.get("name") or path.parent.name
    normalized = str(raw_id).strip().lower().replace(" ", "-")
    request = ProjectRequest.create(path, normalized, getpass.getuser())
    request.validate(config.allowed_roots)
    return Inbox(config.inbox).submit(request)


def start_project(config: DaemonConfig, project_yaml: Path,
                  project_id: str | None = None) -> Path:
    """Set project-local RUNNING state and ensure the project is registered."""
    path = Path(project_yaml).resolve(strict=True)
    set_desired_state(path.parent, "RUNNING", getpass.getuser())
    request = register_project(config, path, project_id)
    write_legacy_active_pointer(config, path)
    return request


def update_project(config: DaemonConfig, project_yaml: Path,
                   project_id: str | None = None) -> Path:
    """Replace a stopped project's inbox request after validating new content."""
    path = Path(project_yaml).resolve(strict=True)
    if get_desired_state(path.parent) != "STOPPED":
        raise ValueError("Project update requires desired_state=STOPPED")
    data = __import__("yaml").safe_load(path.read_text(encoding="utf-8")) or {}
    raw_id = project_id or data.get("project", {}).get("id") or data.get("name") or path.parent.name
    normalized = str(raw_id).strip().lower().replace(" ", "-")
    request = ProjectRequest.create(path, normalized, getpass.getuser())
    request.validate(config.allowed_roots)
    return Inbox(config.inbox).replace(request)


def stop_project(project_root: Path, config: DaemonConfig | None = None) -> Path:
    """Set project-local STOPPED state for reconciliation on the next daemon cycle."""
    root = Path(project_root).resolve(strict=True)
    control = set_desired_state(root, "STOPPED", getpass.getuser())
    remove_legacy_active_pointer(config or DaemonConfig(), root)
    return control


def _legacy_daemon_detected(config: DaemonConfig) -> bool:
    """Return True only during the transition from the former Bash supervisor."""
    if not (config.inbox / ".daemon_job_id").exists():
        return False
    identity = config.inbox / "daemon.json"
    try:
        return json.loads(identity.read_text(encoding="utf-8")).get("state") != "SERVING"
    except (OSError, json.JSONDecodeError):
        return True


def write_legacy_active_pointer(config: DaemonConfig, project_yaml: Path) -> Path | None:
    """Write an atomic compatibility pointer when the old daemon is running."""
    if not _legacy_daemon_detected(config):
        return None
    import yaml
    path = Path(project_yaml).resolve(strict=True)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    name = str(data.get("name") or path.parent.name).strip()
    if not name or Path(name).name != name:
        raise ValueError(f"Invalid project name for daemon pointer: {name!r}")
    data["root"] = str(path.parent)
    data["project_root"] = str(path.parent)
    destination = config.inbox / "active" / f"{name}.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(destination, yaml.safe_dump(data, sort_keys=False))
    return destination


def remove_legacy_active_pointer(config: DaemonConfig, project_root: Path) -> None:
    """Remove only plain legacy pointers that resolve to *project_root*."""
    import yaml
    root = Path(project_root).resolve()
    for pointer in (config.inbox / "active").glob("*.yaml"):
        try:
            data = yaml.safe_load(pointer.read_text(encoding="utf-8")) or {}
            if {"request_id", "project_id", "project_yaml"} <= set(data):
                continue
            value = data.get("project_path") or data.get("project_root") or data.get("root")
            if value and Path(value).resolve() == root:
                pointer.unlink()
        except (OSError, yaml.YAMLError):
            continue


def legacy_projects(source: Path) -> list[tuple[str, Path]]:
    """Resolve canonical project YAMLs from the legacy flat ``active/*.yaml`` inbox."""
    import yaml
    resolved: list[tuple[str, Path]] = []
    for pointer in sorted((Path(source) / "active").glob("*.yaml")):
        data = yaml.safe_load(pointer.read_text(encoding="utf-8")) or {}
        root = data.get("project_path") or data.get("project_root") or data.get("root")
        if not root:
            raise ValueError(f"Legacy request has no explicit project root: {pointer}")
        project_yaml = Path(root).resolve() / "project.yaml"
        if not project_yaml.is_file():
            raise FileNotFoundError(f"Canonical project.yaml not found: {project_yaml}")
        resolved.append((pointer.stem.lower(), project_yaml))
    return resolved
