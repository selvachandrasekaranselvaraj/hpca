"""
_submit_writers.py — Module-level utilities for h02_aimd submission setup.
"""
from __future__ import annotations

from hpca.core.categories import is_molecular as _cat_is_molecular, is_sse as _cat_is_sse


def incar_key_for_npt(category: str) -> str:
    """Map material category to the matching npt_step0_* template key."""
    if _cat_is_molecular(category):
        return "npt_step0_mol"
    if _cat_is_sse(category):
        return "npt_step0_sse"
    return "npt_step0_int"
