"""Compatibility alias for :mod:`hpca.registry.folder`.

New code must import ``hpca.registry.folder``.  Remove this module in HPCA 2.0.
"""

import sys
from hpca.registry import folder as _canonical

sys.modules[__name__] = _canonical
