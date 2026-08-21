"""Compatibility import for :mod:`hpca.registry.poscar`.

New code must import from ``hpca.registry``.  Remove this module in HPCA 2.0.
"""

from hpca.registry.poscar import get_poscar_source

__all__ = ["get_poscar_source"]
