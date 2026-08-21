"""Bounded daemon-local task queues for resource-heavy pipeline work.

The orchestrator advances projects concurrently.  These queues add a second,
resource-aware admission layer so CPU/memory-heavy material construction and
MACE preoptimization do not share one unbounded project executor.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Callable, Generic, TypeVar

from hpca.core.paths import load_platform_config


log = logging.getLogger("hpca.orch")
T = TypeVar("T")


@dataclass(frozen=True)
class DaemonTask(Generic[T]):
    """A named unit of daemon-local work submitted to a bounded queue."""

    project_dir: Path
    operation: Callable[[], T]

    @property
    def label(self) -> str:
        return f"{self.__class__.__name__}:{self.project_dir.name}"

    def run(self) -> T:
        log.info("[daemon-queue] START %s", self.label)
        try:
            return self.operation()
        finally:
            log.info("[daemon-queue] END %s", self.label)


@dataclass(frozen=True)
class MaterialDesignTask(DaemonTask[T]):
    """Structure construction/packing work; never runs MACE."""


@dataclass(frozen=True)
class PreoptimizationTask(DaemonTask[T]):
    """Validation, deterministic repair, and policy-selected MACE work."""


def _positive_workers(value: object, default: int) -> int:
    try:
        workers = int(value)
    except (TypeError, ValueError):
        workers = default
    if workers < 1:
        raise ValueError("daemon queue worker counts must be positive integers")
    return workers


class DaemonTaskScheduler:
    """Independent bounded executors for daemon material-design and preopt work."""

    def __init__(self, design_workers: int, preoptimization_workers: int):
        self.design_workers = _positive_workers(design_workers, 4)
        self.preoptimization_workers = _positive_workers(preoptimization_workers, 1)
        self._design = ThreadPoolExecutor(
            max_workers=self.design_workers, thread_name_prefix="hpca-design"
        )
        self._preoptimization = ThreadPoolExecutor(
            max_workers=self.preoptimization_workers, thread_name_prefix="hpca-preopt"
        )
        log.info(
            "[daemon-queue] configured material_design=%d preoptimization=%d",
            self.design_workers,
            self.preoptimization_workers,
        )

    def submit(self, task: DaemonTask[T]) -> Future[T]:
        if isinstance(task, MaterialDesignTask):
            return self._design.submit(task.run)
        if isinstance(task, PreoptimizationTask):
            return self._preoptimization.submit(task.run)
        raise TypeError(f"Unsupported daemon task type: {type(task).__name__}")

    def shutdown(self, wait: bool = True) -> None:
        self._design.shutdown(wait=wait, cancel_futures=False)
        self._preoptimization.shutdown(wait=wait, cancel_futures=False)


_scheduler: DaemonTaskScheduler | None = None
_scheduler_lock = Lock()


def get_daemon_task_scheduler() -> DaemonTaskScheduler:
    """Return the process-wide scheduler configured by ``platform.yaml``.

    Environment overrides are useful for a particular daemon allocation:
    ``HPCA_DESIGN_WORKERS`` and ``HPCA_PREOPT_WORKERS``.
    """
    global _scheduler
    if _scheduler is None:
        with _scheduler_lock:
            if _scheduler is None:
                cfg = load_platform_config().get("daemon_tasks", {}) or {}
                design = os.environ.get("HPCA_DESIGN_WORKERS", cfg.get("material_design_workers", 4))
                preopt = os.environ.get(
                    "HPCA_PREOPT_WORKERS", cfg.get("preoptimization_workers", 1)
                )
                _scheduler = DaemonTaskScheduler(design, preopt)
    return _scheduler

