"""Typed scheduler operations, independent of submission-script rendering."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from hpca.core.slurm_submit import job_alive, job_state, sbatch


class Scheduler(Protocol):
    def submit(self, script: Path, *, cwd: Path | None = None,
               dependency: str | None = None,
               extra_args: list[str] | None = None) -> str | None: ...
    def state(self, job_id: str) -> str: ...
    def alive(self, job_id: str | None) -> bool: ...
    def cancel(self, job_id: str) -> bool: ...


class SlurmScheduler:
    """Production adapter for scheduler commands; contains no script templates."""

    def submit(self, script: Path, *, cwd: Path | None = None,
               dependency: str | None = None,
               extra_args: list[str] | None = None) -> str | None:
        return sbatch(Path(script), cwd=cwd, dependency=dependency, extra_args=extra_args)

    def state(self, job_id: str) -> str:
        return job_state(job_id)

    def alive(self, job_id: str | None) -> bool:
        return job_alive(job_id)

    def cancel(self, job_id: str) -> bool:
        result = subprocess.run(["scancel", str(job_id)], capture_output=True,
                                text=True, timeout=15)
        return result.returncode == 0


@dataclass
class FakeScheduler:
    """In-memory scheduler used by unit and orchestration tests."""

    next_job_id: int = 1
    states: dict[str, str] = field(default_factory=dict)
    submissions: list[dict] = field(default_factory=list)

    def submit(self, script: Path, *, cwd: Path | None = None,
               dependency: str | None = None,
               extra_args: list[str] | None = None) -> str:
        job_id = str(self.next_job_id)
        self.next_job_id += 1
        self.states[job_id] = "PENDING"
        self.submissions.append({"job_id": job_id, "script": Path(script),
                                 "cwd": cwd, "dependency": dependency,
                                 "extra_args": list(extra_args or [])})
        return job_id

    def state(self, job_id: str) -> str:
        return self.states.get(str(job_id), "UNKNOWN")

    def alive(self, job_id: str | None) -> bool:
        return bool(job_id) and self.state(str(job_id)) in {
            "PENDING", "RUNNING", "COMPLETING", "CONFIGURING"
        }

    def cancel(self, job_id: str) -> bool:
        if str(job_id) not in self.states:
            return False
        self.states[str(job_id)] = "CANCELLED"
        return True


_DEFAULT = SlurmScheduler()


def get_scheduler() -> Scheduler:
    return _DEFAULT
