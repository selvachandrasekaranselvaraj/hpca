"""Compatibility import for :mod:`hpca.registry.submission`.

New code must import from ``hpca.registry``.  Remove this module in HPCA 2.0.
"""

from hpca.registry.submission import write_submission

__all__ = ["write_submission"]
