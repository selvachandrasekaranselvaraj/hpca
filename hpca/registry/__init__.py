"""Authoritative HPCA registries.

Registries contain stable definitions and lookup logic.  They do not execute
workflow stages, submit jobs, or own orchestration state.
"""

from hpca.registry.incar import build_incar, get_incar, write_incar
from hpca.registry.poscar import get_poscar_source
from hpca.registry.submission import SUBMISSIONS, write_submission
from hpca.registry.validation import require_valid_registries, validate_registries

__all__ = [
    "build_incar",
    "get_incar",
    "get_poscar_source",
    "require_valid_registries",
    "SUBMISSIONS",
    "validate_registries",
    "write_incar",
    "write_submission",
]
