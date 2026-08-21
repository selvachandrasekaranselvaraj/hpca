"""hpca/core/neb — Constrained NEB chain generator subpackage.

Public API for generating NEB scientific input files using a
non-linear path predictor. Does NOT run VASP; only generates inputs.
"""
from .image_tools import generate_neb_chain, apply_constrained_path_dynamics, write_xyz_trajectory
from .path_finder import analyze_migration_sites, build_nonlinear_chained_path, predict_nonlinear_path_segment
from .poscar_io import read_structure, write_poscar
from .linear import apply_selective_dynamics, find_migrating_atom, make_neb_images

__all__ = [
    "generate_neb_chain",
    "apply_constrained_path_dynamics",
    "write_xyz_trajectory",
    "analyze_migration_sites",
    "build_nonlinear_chained_path",
    "predict_nonlinear_path_segment",
    "read_structure",
    "write_poscar",
    "apply_selective_dynamics",
    "find_migrating_atom",
    "make_neb_images",
]
