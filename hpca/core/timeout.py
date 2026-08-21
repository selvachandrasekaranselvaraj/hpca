"""
timeout.py — Thread-based timeout wrapper for long-running functions.

Used by: hpca/orchestrator/handlers/base.py (sbatch timeout),
         hpca/core/mlip_preopt.py (MACE pre-relax timeout).
Configured via platform.yaml handler_timeouts section.
"""
from __future__ import annotations

import concurrent.futures
import logging
from typing import Any, Callable

log = logging.getLogger("hpca.core")


def run_with_timeout(
    fn: Callable,
    args: tuple = (),
    timeout_s: float = 300.0,
    name: str = "",
) -> Any:
    """Run fn(*args) in a thread; raise TimeoutError if it exceeds timeout_s seconds.

    Args:
        fn: Callable to execute.
        args: Positional arguments passed to fn.
        timeout_s: Maximum allowed wall time in seconds.
        name: Human-readable label used in log/error messages.

    Returns:
        The return value of fn(*args).

    Raises:
        TimeoutError: If fn does not complete within timeout_s seconds.
    """
    label = name or getattr(fn, "__name__", "task")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"{label} exceeded {timeout_s:.0f}s timeout")
