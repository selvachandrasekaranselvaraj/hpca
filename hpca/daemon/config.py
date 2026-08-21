"""Validated daemon configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_INBOX = Path(__file__).resolve().parents[2] / "daemon_inbox"


def default_allowed_roots() -> tuple[Path, ...]:
    """Return hpc.allowed_project_roots from platform.yaml (package parent as last resort)."""
    from hpca.core.config import Config
    roots = Config.get().hpc("allowed_project_roots", [])
    if roots:
        return tuple(Path(p) for p in roots)
    return (Path(__file__).resolve().parents[3],)


@dataclass(frozen=True)
class DaemonConfig:
    """Runtime paths and scheduling policy for one daemon deployment."""

    inbox: Path = DEFAULT_INBOX
    allowed_roots: tuple[Path, ...] = field(default_factory=default_allowed_roots)
    poll_seconds: int = 60
    walltime_hours: int = 240
    successor_before_hours: int = 20
    max_projects: int = 16
    max_orchestrator_restarts: int = 3
    successor_script: Path | None = None
    hot_reload_grace_seconds: int = 90

    def __post_init__(self) -> None:
        object.__setattr__(self, "inbox", Path(self.inbox).resolve())
        object.__setattr__(self, "allowed_roots", tuple(Path(p).resolve() for p in self.allowed_roots))
        if self.poll_seconds < 1:
            raise ValueError("poll_seconds must be positive")
        if self.walltime_hours <= self.successor_before_hours:
            raise ValueError("walltime must exceed successor overlap")
        if self.max_projects < 1:
            raise ValueError("max_projects must be positive")
        if self.max_orchestrator_restarts < 0:
            raise ValueError("max_orchestrator_restarts cannot be negative")
        if self.hot_reload_grace_seconds < 1:
            raise ValueError("hot_reload_grace_seconds must be positive")

    @property
    def successor_after_seconds(self) -> int:
        """Submit replacement after 9d4h for a 10-day/20-hour-overlap policy."""
        return (self.walltime_hours - self.successor_before_hours) * 3600
