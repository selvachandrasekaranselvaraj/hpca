"""Compatibility imports for :mod:`hpca.core.neb.linear`.

New code must import from :mod:`hpca.core.neb`. Remove this module in HPCA 2.0.
"""

from hpca.core.neb.linear import apply_selective_dynamics, find_migrating_atom, make_neb_images

__all__ = ["apply_selective_dynamics", "find_migrating_atom", "make_neb_images"]
