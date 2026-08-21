"""Unit tests for the packaged HPCA daemon control plane."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hpca.daemon.config import DEFAULT_INBOX, DaemonConfig
from hpca.daemon.control import get_desired_state
from hpca.daemon.inbox import Inbox
from hpca.daemon.leases import Lease
from hpca.daemon.schemas import ProjectRequest
from hpca.daemon.service import (DaemonService, legacy_projects, register_project,
                                 start_project, stop_project, update_project)
from hpca.daemon.slurm import write_wrapper
from hpca.registry.stage import HANDLER_ORDER, get_stage


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "sample"
    root.mkdir()
    path = root / "project.yaml"
    path.write_text(yaml.safe_dump({
        "name": "sample", "category": "inorganic_sse", "mobile_ion": "Li", "T_ref": 300,
    }))
    return path


def test_successor_timing_is_9_days_4_hours(tmp_path):
    cfg = DaemonConfig(inbox=tmp_path, allowed_roots=(tmp_path,))
    assert cfg.successor_after_seconds == 220 * 3600


def test_default_inbox_is_inside_hpca_repository():
    assert DEFAULT_INBOX.name == "daemon_inbox"
    assert DEFAULT_INBOX.parent.name == "hpca"


def test_register_project_uses_immutable_pointer(tmp_path):
    project = _project(tmp_path)
    cfg = DaemonConfig(inbox=tmp_path / "inbox", allowed_roots=(tmp_path,))
    request_path = register_project(cfg, project)
    value = yaml.safe_load(request_path.read_text())
    assert value["project_id"] == "sample"
    assert value["project_yaml"] == str(project.resolve())
    assert len(value["project_yaml_sha256"]) == 64
    assert register_project(cfg, project) == request_path


def test_changed_project_is_rejected(tmp_path):
    project = _project(tmp_path)
    request = ProjectRequest.create(project, "sample", "tester")
    project.write_text(yaml.safe_dump({"name": "changed", "category": "solid"}))
    with pytest.raises(ValueError, match="changed after registration"):
        request.validate((tmp_path.resolve(),))


def test_project_outside_allowed_root_is_rejected(tmp_path):
    project = _project(tmp_path)
    request = ProjectRequest.create(project, "sample", "tester")
    with pytest.raises(ValueError, match="outside allowed roots"):
        request.validate(((tmp_path / "different").resolve(),))


def test_lease_is_exclusive(tmp_path):
    first = Lease(tmp_path / "project.lock")
    second = Lease(tmp_path / "project.lock")
    assert first.acquire()
    assert not second.acquire()
    first.release()
    assert second.acquire()
    second.release()


def test_inbox_transition_preserves_request(tmp_path):
    inbox = Inbox(tmp_path / "inbox")
    project = _project(tmp_path)
    request = ProjectRequest.create(project, "sample", "tester")
    incoming = inbox.submit(request)
    queued = inbox.transition(incoming, "queued")
    assert queued.exists()
    assert not incoming.exists()
    events = list((inbox.root / "events").glob("*.jsonl"))
    assert events and len(events[0].read_text().splitlines()) == 2


def test_malformed_request_is_discoverable_for_quarantine(tmp_path):
    inbox = Inbox(tmp_path / "inbox")
    inbox.initialize()
    bad = inbox.root / "incoming" / "bad.yaml"
    bad.write_text("not: [valid")
    malformed = list(inbox.malformed("incoming"))
    assert malformed and malformed[0][0] == bad
    assert "Malformed request" in malformed[0][1]


def test_wrapper_has_ten_day_walltime_and_packaged_entry(tmp_path):
    path = write_wrapper(tmp_path / "daemon.sbatch", inbox=tmp_path, account="test")
    text = path.read_text()
    assert "#SBATCH --time=10-00:00:00" in text
    assert f"-m hpca.daemon.cli --inbox {tmp_path} run" in text
    assert path.stat().st_mode & 0o100


def test_stage_registry_places_active_learning_before_analysis():
    assert HANDLER_ORDER.index("h05_lammps") < HANDLER_ORDER.index("h13_active_learning")
    assert HANDLER_ORDER.index("h13_active_learning") < HANDLER_ORDER.index("h06_analysis")
    assert get_stage("h01_dft.opt").handler == "h01_dft"


def test_project_local_start_stop_controls_inbox(tmp_path):
    project = _project(tmp_path)
    cfg = DaemonConfig(inbox=tmp_path / "inbox", allowed_roots=(tmp_path,))
    request_path = start_project(cfg, project)
    assert get_desired_state(project.parent) == "RUNNING"
    inbox = Inbox(cfg.inbox)
    queued = inbox.transition(request_path, "queued")
    stop_project(project.parent)
    assert get_desired_state(project.parent) == "STOPPED"
    service = DaemonService(cfg)
    service._apply_project_controls()
    paused = cfg.inbox / "paused" / queued.name
    assert paused.exists()
    start_project(cfg, project)
    service._apply_project_controls()
    assert (cfg.inbox / "queued" / queued.name).exists()


def test_start_and_stop_bridge_current_legacy_daemon(tmp_path):
    project = _project(tmp_path)
    cfg = DaemonConfig(inbox=tmp_path / "inbox", allowed_roots=(tmp_path,))
    cfg.inbox.mkdir(parents=True)
    (cfg.inbox / ".daemon_job_id").write_text("123\n")

    start_project(cfg, project)
    pointer = cfg.inbox / "active" / "sample.yaml"
    assert pointer.exists()
    value = yaml.safe_load(pointer.read_text())
    assert Path(value["root"]) == project.parent.resolve()

    stop_project(project.parent, cfg)
    assert not pointer.exists()
    assert get_desired_state(project.parent) == "STOPPED"


def test_stopped_orphan_active_request_is_paused_on_daemon_restart(tmp_path):
    """An active pointer cannot override STOPPED after supervisor restart."""
    project = _project(tmp_path)
    cfg = DaemonConfig(inbox=tmp_path / "inbox", allowed_roots=(tmp_path,))
    request = start_project(cfg, project)
    active = Inbox(cfg.inbox).transition(request, "active")
    stop_project(project.parent)

    service = DaemonService(cfg)  # empty managed-process table, as after restart
    service._apply_project_controls()

    assert not active.exists()
    assert (cfg.inbox / "paused" / active.name).exists()
    assert service.projects == {}


def test_running_orphan_active_request_is_resumed_on_daemon_restart(tmp_path, monkeypatch):
    """An active RUNNING pointer with no managed process must be resumed, not left orphaned.

    Regression test: after a daemon restart, self.projects starts empty, so a
    request already sitting in active/ (from before the restart) previously
    had no code path back to a running orchestrator — it just sat there
    forever since only queued/ requests were dispatched.
    """
    project = _project(tmp_path)
    cfg = DaemonConfig(inbox=tmp_path / "inbox", allowed_roots=(tmp_path,))
    request = start_project(cfg, project)
    active = Inbox(cfg.inbox).transition(request, "active")

    class FakeProcess:
        pid = 999999

        def poll(self):
            return None

    monkeypatch.setattr(DaemonService, "_spawn_orchestrator",
                        staticmethod(lambda root: FakeProcess()))

    service = DaemonService(cfg)  # empty managed-process table, as after restart
    service._apply_project_controls()

    assert active.exists()  # already in active/ — no transition needed
    managed = service.projects["sample"]
    assert managed.pid == 999999
    runtime = json.loads((cfg.inbox / "active" / "sample.runtime.json").read_text())
    assert runtime["state"] == "RUNNING"
    assert runtime["pid"] == 999999


def test_hot_reload_restarts_orchestrators_on_code_change(tmp_path, monkeypatch):
    """A code change on disk must restart already-running orchestrators.

    Regression guard: Python does not re-import modules a live process
    already loaded, so an orchestrator subprocess keeps running old code
    forever unless something restarts it — this is that something.
    """
    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid
            self.terminated = False
            self._alive = True

        def poll(self):
            return None if self._alive else 0

        def terminate(self):
            self.terminated = True
            self._alive = False

        def wait(self, timeout=None):
            return 0

    project = _project(tmp_path)
    cfg = DaemonConfig(inbox=tmp_path / "inbox", allowed_roots=(tmp_path,))
    request = start_project(cfg, project)
    Inbox(cfg.inbox).transition(request, "queued")

    spawned = [FakeProcess(111), FakeProcess(222)]
    monkeypatch.setattr(DaemonService, "_spawn_orchestrator",
                        staticmethod(lambda root: spawned.pop(0)))
    mtimes = iter([1000.0, 2000.0, 2000.0])  # init baseline, changed, unchanged
    monkeypatch.setattr(DaemonService, "_pkg_mtime", lambda self: next(mtimes))

    service = DaemonService(cfg)
    service._dispatch_queued()
    old_process = service.projects["sample"].process
    assert old_process.pid == 111

    service._hot_reload_if_changed()
    assert old_process.terminated
    new_process = service.projects["sample"].process
    assert new_process.pid == 222
    runtime = json.loads((cfg.inbox / "active" / "sample.runtime.json").read_text())
    assert runtime["pid"] == 222 and runtime["state"] == "RUNNING"

    # No further code change (mtime unchanged) — must not restart again.
    monkeypatch.setattr(DaemonService, "_spawn_orchestrator", staticmethod(
        lambda root: (_ for _ in ()).throw(AssertionError("must not respawn without a code change"))))
    service._hot_reload_if_changed()
    assert service.projects["sample"].process is new_process


def test_stopped_project_request_can_be_updated(tmp_path):
    project = _project(tmp_path)
    cfg = DaemonConfig(inbox=tmp_path / "inbox", allowed_roots=(tmp_path,))
    old = start_project(cfg, project)
    queued = Inbox(cfg.inbox).transition(old, "queued")
    stop_project(project.parent)
    service = DaemonService(cfg)
    service._apply_project_controls()
    project.write_text(yaml.safe_dump({
        "name": "sample", "category": "inorganic_sse",
        "mobile_ion": "Li", "T_ref": 300, "encut": 520,
    }))
    new = update_project(cfg, project)
    assert new.parent.name == "incoming"
    assert new.name != queued.name
    assert (cfg.inbox / "archive" / f"{queued.stem}.paused.yaml").exists()


def test_active_or_running_project_update_is_rejected(tmp_path):
    project = _project(tmp_path)
    cfg = DaemonConfig(inbox=tmp_path / "inbox", allowed_roots=(tmp_path,))
    request = start_project(cfg, project)
    with pytest.raises(ValueError, match="desired_state=STOPPED"):
        update_project(cfg, project)
    stop_project(project.parent)
    active = Inbox(cfg.inbox).transition(request, "active")
    project.write_text(yaml.safe_dump({
        "name": "sample", "category": "inorganic_sse",
        "mobile_ion": "Li", "T_ref": 300, "encut": 520,
    }))
    with pytest.raises(ValueError, match="active"):
        update_project(cfg, project)
    assert active.exists()


def test_legacy_migration_requires_explicit_project_root(tmp_path):
    source = tmp_path / "legacy"
    (source / "active").mkdir(parents=True)
    (source / "active" / "bad.yaml").write_text("name: bad\n")
    with pytest.raises(ValueError, match="no explicit project root"):
        legacy_projects(source)


def test_legacy_migration_resolves_canonical_yaml(tmp_path):
    project = _project(tmp_path)
    source = tmp_path / "legacy"
    (source / "active").mkdir(parents=True)
    (source / "active" / "Sample.yaml").write_text(
        yaml.safe_dump({"project_path": str(project.parent)}))
    assert legacy_projects(source) == [("sample", project.resolve())]
