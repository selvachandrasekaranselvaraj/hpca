"""Fail-closed policy for unattended HPCA execution."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AutonomyPolicy:
    """Project-scoped authority and resource limits for automatic progression."""

    mode: str = "attended"
    auto_approve_validated_design: bool = False
    max_stage_submissions: int = 5
    max_local_attempts: int = 5
    max_total_submissions: int = 100
    allowed_stages: tuple[str, ...] = ()

    @classmethod
    def from_project(cls, project_yaml: dict[str, Any]) -> "AutonomyPolicy":
        raw = project_yaml.get("autonomy", {}) or {}
        # Wizard workflow v2 asked whether to enable remaining stages "for
        # autonomous execution" but older files failed to persist that answer
        # as an autonomy policy.  Preserve fail-closed behavior for legacy and
        # hand-written files; migrate v2 wizard projects with enabled downstream
        # stages to the unattended semantics the user explicitly selected.
        stages = project_yaml.get("stages", {}) or {}
        has_downstream = any(
            bool(value) if not isinstance(value, dict) else any(value.values())
            for key, value in stages.items() if key != "design"
        )
        migrated_unattended = (
            "autonomy" not in project_yaml
            and int(project_yaml.get("workflow_version", 0) or 0) >= 2
            and has_downstream
        )
        mode = str(raw.get("mode", "unattended" if migrated_unattended else "attended"))
        if mode not in ("attended", "unattended"):
            raise ValueError("autonomy.mode must be attended or unattended")
        default_auto = mode == "unattended"
        policy = cls(
            mode=mode,
            auto_approve_validated_design=bool(
                raw.get("auto_approve_validated_design", default_auto)),
            max_stage_submissions=int(raw.get("max_stage_submissions", 5)),
            max_local_attempts=int(raw.get("max_local_attempts", 5)),
            max_total_submissions=int(raw.get("max_total_submissions", 100)),
            allowed_stages=tuple(str(x) for x in raw.get("allowed_stages", ())),
        )
        if min(policy.max_stage_submissions, policy.max_local_attempts,
               policy.max_total_submissions) < 1:
            raise ValueError("autonomy limits must be positive")
        return policy

    @property
    def unattended(self) -> bool:
        return self.mode == "unattended"

    def stage_allowed(self, stage_name: str) -> bool:
        """Return whether policy grants automatic authority for this stage."""
        parent = stage_name.split(".", 1)[0]
        return not self.allowed_stages or stage_name in self.allowed_stages or parent in self.allowed_stages

    def design_approved(self, project_dir: Path) -> bool:
        """Require an explicit flag, or validated auto-approval in unattended mode."""
        flags = (
            project_dir / "design" / "simulation_approved.flag",
            project_dir / "designed_structures" / "simulation_approved.flag",
        )
        if any(path.exists() for path in flags):
            return True
        if not (self.unattended and self.auto_approve_validated_design):
            return False
        completion = project_dir / "designed_structures" / "DESIGN_COMPLETE.md"
        validation = project_dir / "designed_structures" / "validation.json"
        # Existing h00 projects may predate validation.json.  In unattended mode,
        # DESIGN_COMPLETE plus the handler's internal checks is the migration floor.
        validated = completion.exists() and (validation.exists() or completion.stat().st_size > 0)
        if not validated:
            return False
        # Persist the decision so it is auditable and remains stable across
        # daemon restarts and future policy changes.
        flag = project_dir / "design" / "simulation_approved.flag"
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.touch(exist_ok=True)
        return True

    def may_attempt(self, stage_name: str, state: "Any", *, local: bool) -> tuple[bool, str]:
        """Check stage allowlist and durable attempt budgets."""
        if not self.stage_allowed(stage_name):
            return False, f"stage {stage_name} is outside autonomy.allowed_stages"
        record = state.state.get("autonomy", {})
        total = int(record.get("total_submissions", 0))
        if not local and total >= self.max_total_submissions:
            return False, "project submission budget exhausted"
        key = "local_attempts" if local else "submissions"
        count = int(record.get(key, {}).get(stage_name, 0))
        limit = self.max_local_attempts if local else self.max_stage_submissions
        if count >= limit:
            return False, f"{stage_name} {key} budget exhausted ({count}/{limit})"
        return True, ""

    def record_attempt(self, stage_name: str, state: "Any", *, local: bool) -> None:
        """Persist an attempt before execution so crashes still consume budget."""
        record = dict(state.state.get("autonomy", {}))
        key = "local_attempts" if local else "submissions"
        counts = dict(record.get(key, {}))
        counts[stage_name] = int(counts.get(stage_name, 0)) + 1
        record[key] = counts
        if not local:
            record["total_submissions"] = int(record.get("total_submissions", 0)) + 1
        else:
            active = dict(record.get("in_progress", {}))
            active[stage_name] = {"pid": os.getpid(), "started_at": datetime.now().isoformat()}
            record["in_progress"] = active
        state.state["autonomy"] = record
        state.save()

    def finish_attempt(self, stage_name: str, state: "Any", *, local: bool) -> None:
        """Clear the in-progress marker without refunding a completed attempt."""
        if not local:
            return
        record = dict(state.state.get("autonomy", {}))
        active = dict(record.get("in_progress", {}))
        if stage_name not in active:
            return
        active.pop(stage_name, None)
        record["in_progress"] = active
        state.state["autonomy"] = record
        state.save()

    def clear_successful_local_attempts(self, stage_name: str, state: "Any") -> None:
        """Clear a daemon stage's failure budget after verified completion.

        Daemon stages may legitimately run again when upstream artifacts gain
        new data.  Their safety budget therefore measures consecutive
        unsuccessful executions, not the lifetime number of successful runs.
        """
        record = dict(state.state.get("autonomy", {}))
        counts = dict(record.get("local_attempts", {}))
        active = dict(record.get("in_progress", {}))
        changed = counts.pop(stage_name, None) is not None
        changed = active.pop(stage_name, None) is not None or changed
        if not changed:
            return
        record["local_attempts"] = counts
        record["in_progress"] = active
        state.state["autonomy"] = record
        state.save()
