"""HPCA daemon control plane.

The daemon registers projects and supervises one orchestrator per project.  It
does not implement scientific stages; handlers remain responsible for execution.
"""

from hpca.daemon.config import DaemonConfig
from hpca.daemon.service import DaemonService

__all__ = ["DaemonConfig", "DaemonService"]
