"""Compatibility import for :mod:`hpca.registry.incar`.

New code must import from ``hpca.registry``.  Remove this module in HPCA 2.0.
"""

from hpca.registry.incar import build_incar, get_incar, write_incar

__all__ = ["build_incar", "get_incar", "write_incar"]
