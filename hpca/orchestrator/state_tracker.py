"""
state_tracker.py — Per-project JSON state management for HPCA orchestrator.

State file: {PROJECT}/logs/orchestrator_state.json
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("hpca.orch")

STAGES = ["PENDING", "RUNNING", "COMPLETE", "FAILED", "SKIPPED"]
ALLOWED_TRANSITIONS = {
    "PENDING": frozenset(STAGES),
    "RUNNING": frozenset({"RUNNING", "PENDING", "COMPLETE", "FAILED"}),
    "COMPLETE": frozenset({"COMPLETE", "PENDING"}),
    "FAILED": frozenset({"FAILED", "PENDING", "RUNNING"}),
    "SKIPPED": frozenset({"SKIPPED", "PENDING"}),
}
STATE_FILENAME = "orchestrator_state.json"


class StateCorruptionError(RuntimeError):
    """Raised when durable orchestrator state cannot be trusted or migrated."""


class ProjectState:
    """Manages the JSON state file for one project directory."""

    def __init__(self, project_dir: Path):
        """Load or initialise the state file for project_dir."""
        self.project_dir = Path(project_dir)
        self.path = self.project_dir / "logs" / STATE_FILENAME
        self.state: dict = self.load()

    def _default_state(self) -> dict:
        """Return a fresh state dict with schema_version and zeroed counters."""
        return {
            "project": self.project_dir.name,
            "schema_version": 1,
            "revision": 0,
            "updated": datetime.now().isoformat(),
            "chaai_new_examples": 0,
            "handlers": {},
        }

    def load(self) -> dict:
        """Read durable state, creating defaults only when no state file exists."""
        if self.path.exists():
            try:
                value = json.loads(self.path.read_text())
                if not isinstance(value, dict) or value.get("schema_version") != 1:
                    raise ValueError("unsupported or malformed state schema")
                if not isinstance(value.get("handlers", {}), dict):
                    raise ValueError("handlers must be a mapping")
                value.setdefault("revision", 0)
                value.setdefault("handlers", {})
                return value
            except Exception as exc:
                raise StateCorruptionError(
                    f"Refusing to resume from corrupt state {self.path}: {exc}"
                ) from exc
        return self._default_state()

    def save(self) -> None:
        """Persist current state dict to the JSON file, updating the 'updated' timestamp."""
        from hpca.core.atomic import atomic_write_json
        self.state["updated"] = datetime.now().isoformat()
        self.state["revision"] = int(self.state.get("revision", 0)) + 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, self.state)

    def get_handler(self, handler_name: str) -> dict:
        """Return the full state dict for handler_name; defaults to {'stage': 'PENDING'}."""
        return self.state.get("handlers", {}).get(handler_name, {"stage": "PENDING"})

    def set_handler(self, handler_name: str, data: dict) -> None:
        """Merge data into the handler's state dict and persist to disk."""
        if "handlers" not in self.state:
            self.state["handlers"] = {}
        existing = self.state["handlers"].get(handler_name, {})
        existing.update(data)
        self.state["handlers"][handler_name] = existing
        self.save()

    def set_stage(self, handler_name: str, stage: str, **kwargs) -> None:
        """Set handler stage (must be in STAGES) with optional extra key-value metadata."""
        if stage not in STAGES:
            raise ValueError(f"Invalid stage {stage!r}, must be one of {STAGES}")
        previous = self.get_stage(handler_name)
        if stage not in ALLOWED_TRANSITIONS[previous]:
            raise ValueError(f"Invalid state transition for {handler_name}: {previous} -> {stage}")
        data = {"stage": stage}
        data.update(kwargs)
        if previous != stage:
            history = list(self.get_handler(handler_name).get("history", []))
            history.append({"from": previous, "to": stage,
                            "at": datetime.now().isoformat()})
            data["history"] = history[-100:]
        self.set_handler(handler_name, data)

    def get_stage(self, handler_name: str) -> str:
        """Return the current stage string for handler_name (default 'PENDING')."""
        return self.get_handler(handler_name).get("stage", "PENDING")

    def get_job(self, handler_name: str) -> str | None:
        """Return the SLURM job ID stored for handler_name, or None."""
        return self.get_handler(handler_name).get("job")

    def set_job(self, handler_name: str, job_id: str) -> None:
        """Persist a SLURM job ID for handler_name."""
        self.set_handler(handler_name, {"job": job_id})

    def all_complete(self, handler_names: list[str]) -> bool:
        """Return True if every handler in handler_names is in COMPLETE stage."""
        return all(self.get_stage(h) == "COMPLETE" for h in handler_names)

    def any_running(self, handler_names: list[str]) -> bool:
        """Return True if at least one handler in handler_names is in RUNNING stage."""
        return any(self.get_stage(h) == "RUNNING" for h in handler_names)

    def increment_chaai_examples(self, n: int = 1) -> None:
        """Add n to the running CHAAI example counter and persist to disk."""
        self.state["chaai_new_examples"] = self.state.get("chaai_new_examples", 0) + n
        self.save()

    def reset_chaai_examples(self) -> None:
        """Reset the CHAAI example counter to zero and persist to disk."""
        self.state["chaai_new_examples"] = 0
        self.save()

    def get_chaai_count(self) -> int:
        """Return the current accumulated CHAAI training example count."""
        return self.state.get("chaai_new_examples", 0)

    def reset_failed_handlers(self) -> list[str]:
        """Reset failed handlers and their authorized-attempt counters.

        This is called only by an explicit user ``restart``/``resume`` command.
        Historical transitions are preserved; transient failure and job fields
        are cleared so stale metadata cannot poison the next attempt.
        """
        reset: list[str] = []
        handlers = self.state.setdefault("handlers", {})
        autonomy = self.state.setdefault("autonomy", {})
        local = autonomy.setdefault("local_attempts", {})
        submissions = autonomy.setdefault("submissions", {})
        in_progress = autonomy.setdefault("in_progress", {})
        for name, value in handlers.items():
            if not isinstance(value, dict) or value.get("stage") != "FAILED":
                continue
            history = list(value.get("history", []))
            history.append({"from": "FAILED", "to": "PENDING",
                            "at": datetime.now().isoformat()})
            value["stage"] = "PENDING"
            value["history"] = history[-100:]
            value["resumed"] = True
            for key in ("error", "failed_at", "job", "jobs", "submitted_at",
                        "resubmit_at"):
                value.pop(key, None)
            local.pop(name, None)
            submitted = int(submissions.pop(name, 0))
            if submitted:
                autonomy["total_submissions"] = max(
                    0, int(autonomy.get("total_submissions", 0)) - submitted)
            in_progress.pop(name, None)
            reset.append(name)
        if reset:
            self.save()
        return reset

    def migrate_exhausted_local_attempts(self) -> list[str]:
        """Recover states affected by the pre-v2 cumulative local-attempt bug.

        Before successful daemon attempts were cleared, incremental handlers
        such as analysis could consume their lifetime budget while processing
        growing upstream files.  Reset only the distinctive synthetic budget
        failure, once per state file; genuine handler failures remain intact.
        """
        autonomy = self.state.setdefault("autonomy", {})
        migrations = autonomy.setdefault("migrations", {})
        migration = "successful_local_attempts_v2"
        if migrations.get(migration):
            return []
        recovered: list[str] = []
        local = autonomy.setdefault("local_attempts", {})
        for name, value in self.state.setdefault("handlers", {}).items():
            if not isinstance(value, dict) or value.get("stage") != "FAILED":
                continue
            error = str(value.get("error", ""))
            if not (error.startswith(f"{name} local_attempts budget exhausted")
                    and int(local.get(name, 0)) > 0):
                continue
            history = list(value.get("history", []))
            history.append({"from": "FAILED", "to": "PENDING",
                            "at": datetime.now().isoformat(),
                            "reason": migration})
            value["stage"] = "PENDING"
            value["history"] = history[-100:]
            value.pop("error", None)
            value.pop("failed_at", None)
            local.pop(name, None)
            recovered.append(name)
        migrations[migration] = True
        if recovered or migration not in migrations:
            self.save()
        else:
            # Persist the marker even when this project needed no recovery.
            self.save()
        return recovered

    def migrate_h05_wait_contract(self) -> bool:
        """Recover h05 incorrectly failed when it returned an existing live job.

        The old caller interpreted a deliberate ``None`` (no *new* sbatch) as
        submission failure even though h05 had preserved its existing job map.
        This exact state is safe to return to PENDING for normal reconciliation.
        """
        autonomy = self.state.setdefault("autonomy", {})
        migrations = autonomy.setdefault("migrations", {})
        migration = "h05_existing_job_wait_v2"
        if migrations.get(migration):
            return False
        value = self.state.setdefault("handlers", {}).get("h05_cmd", {})
        recovered = bool(
            isinstance(value, dict)
            and value.get("stage") == "FAILED"
            and value.get("error") == "sbatch returned None"
            and value.get("jobs")
        )
        if recovered:
            history = list(value.get("history", []))
            history.append({"from": "FAILED", "to": "PENDING",
                            "at": datetime.now().isoformat(),
                            "reason": migration})
            value["stage"] = "PENDING"
            value["history"] = history[-100:]
            value.pop("error", None)
            value.pop("failed_at", None)
            submissions = autonomy.setdefault("submissions", {})
            count = int(submissions.get("h05_cmd", 0))
            if count > 0:
                submissions["h05_cmd"] = count - 1
                autonomy["total_submissions"] = max(
                    0, int(autonomy.get("total_submissions", 0)) - 1)
        migrations[migration] = True
        self.save()
        return recovered

    def rollback_interrupted_attempts(self, pid: int) -> list[str]:
        """Refund local attempts interrupted by a graceful process shutdown."""
        autonomy = self.state.get("autonomy", {})
        active = autonomy.get("in_progress", {})
        if not isinstance(active, dict):
            return []
        local = autonomy.get("local_attempts", {})
        rolled_back: list[str] = []
        changed = False
        for name, marker in list(active.items()):
            if not isinstance(marker, dict) or int(marker.get("pid", -1)) != int(pid):
                continue
            # A handler that explicitly persisted FAILED before SIGTERM is a
            # genuine failed attempt, not a hot-reload interruption.
            if self.get_stage(name) != "FAILED":
                count = int(local.get(name, 0))
                if count <= 1:
                    local.pop(name, None)
                else:
                    local[name] = count - 1
                rolled_back.append(name)
            active.pop(name, None)
            changed = True
        if changed:
            autonomy["local_attempts"] = local
            autonomy["in_progress"] = active
            self.state["autonomy"] = autonomy
            self.save()
        return rolled_back


def load_state(project_dir: Path) -> ProjectState:
    """Convenience: load and return ProjectState for a project directory."""
    return ProjectState(project_dir)
