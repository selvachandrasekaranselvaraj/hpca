"""Scheduler failure classification used by bounded recovery policy."""
from __future__ import annotations

from enum import Enum


class FailureClass(str, Enum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"


TRANSIENT_MARKERS = (
    "slurmctld is busy", "communication error", "unable to connect",
    "socket timed out", "try again later", "temporarily unavailable",
)


def classify_scheduler_failure(stderr: str) -> FailureClass:
    normalized = str(stderr).casefold()
    return (FailureClass.TRANSIENT if any(marker in normalized for marker in TRANSIENT_MARKERS)
            else FailureClass.PERMANENT)
