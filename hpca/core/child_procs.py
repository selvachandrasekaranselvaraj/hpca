"""Registry of daemon-local child process groups for cleanup on shutdown.

Handlers that spawn long-running subprocesses in their own session
(``preexec_fn=os.setsid`` — e.g. MACE preopt, PACKMOL) register the PID here.
The orchestrator's SIGTERM handler drains the registry so a hot-reload restart
cannot leave orphans racing on the same output files past their timeout
(the timeout is enforced by the parent, which is being killed).
"""
from __future__ import annotations

import logging
import os
import signal
import threading

log = logging.getLogger("hpca.orch")

_lock = threading.Lock()
_pids: set[int] = set()


def register(pid: int) -> None:
    """Track *pid* (a session/group leader) for cleanup on orchestrator shutdown."""
    with _lock:
        _pids.add(pid)


def unregister(pid: int) -> None:
    """Stop tracking *pid* after its parent has reaped it."""
    with _lock:
        _pids.discard(pid)


def kill_all(sig: int = signal.SIGTERM) -> int:
    """Send *sig* to every registered child's process group; return count signalled.

    Safe to call from a signal handler: never raises.
    """
    with _lock:
        pids = list(_pids)
        _pids.clear()
    n = 0
    for pid in pids:
        try:
            os.killpg(os.getpgid(pid), sig)
            n += 1
        except (OSError, ProcessLookupError):
            pass
    return n
