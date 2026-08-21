"""Policy for deciding whether a structure should receive MACE preoptimization.

The policy is deliberately separate from the MACE execution backend.  It performs
cheap, deterministic checks first and returns an auditable decision; callers may
then run MACE or copy the validated structure directly to the canonical output.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from hpca.core.categories import is_crystalline, is_molecular
from hpca.core.structure_check import check_and_fix_poscar, min_distance_poscar


MACE_OFF_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})
# MACE-MPA-0 is defined for atomic numbers 1--89.  Keeping this explicit makes an
# unsupported chemistry a safe skip instead of a model-load failure.
MACE_MPA_ELEMENTS = frozenset("""
H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn
Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La
Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po
At Rn Fr Ra Ac
""".split())


@dataclass(frozen=True)
class PreoptimizationDecision:
    run_mace: bool
    reason: str
    model: str | None
    elements: tuple[str, ...]
    atom_count: int
    minimum_distance_A: float | None
    overlap_repaired: bool
    estimated_runtime_s: float
    runtime_limit_s: float

    def as_dict(self) -> dict:
        return asdict(self)


def _poscar_identity(path: Path) -> tuple[set[str], int]:
    try:
        lines = path.read_text().splitlines()
        elements = lines[5].split()
        counts = [int(value) for value in lines[6].split()]
        return set(elements), sum(counts)
    except (OSError, IndexError, ValueError):
        return set(), 0


def decide_preoptimization(
    path: Path,
    category: str,
    project_config: dict | None = None,
    platform_config: dict | None = None,
    *,
    generated_structure: bool = False,
) -> PreoptimizationDecision:
    """Validate/repair *path* and decide whether optional MACE should run.

    ``project_config.preoptimization`` overrides platform ``preoptimization``.
    Supported modes are ``auto`` (this decision tree), ``mace`` and ``none``.
    Even forced MACE never bypasses element coverage or the runtime guard.
    """
    project_config = project_config or {}
    defaults = dict((platform_config or {}).get("preoptimization", {}))
    overrides = project_config.get("preoptimization", {}) or {}
    cfg = {**defaults, **overrides}
    mode = str(cfg.get("mode", "auto")).lower()
    if mode not in {"auto", "mace", "none"}:
        raise ValueError("preoptimization.mode must be auto, mace, or none")

    # Parse identity and apply constant-memory guards before any geometry work.
    # min_distance_poscar builds dense N×N and N×N×3 arrays, so calling it for
    # a 60k-atom CMD structure can exhaust the daemon node before the runtime
    # policy gets a chance to reject MACE.
    elements, atom_count = _poscar_identity(path)
    molecular = is_molecular(category)
    model = "mace_off" if molecular and elements.issubset(MACE_OFF_ELEMENTS) else "mace_mp"
    supported = MACE_OFF_ELEMENTS if model == "mace_off" else MACE_MPA_ELEMENTS

    steps = int(cfg.get("steps", 1000))
    seconds_per_atom_step = float(cfg.get("seconds_per_atom_step", 0.002))
    estimate = atom_count * steps * seconds_per_atom_step
    limit = float(cfg.get("max_runtime_s", 1800))
    cheap_common = dict(
        model=model, elements=tuple(sorted(elements)), atom_count=atom_count,
        minimum_distance_A=None, overlap_repaired=False,
        estimated_runtime_s=estimate, runtime_limit_s=limit,
    )

    if mode == "none":
        return PreoptimizationDecision(False, "disabled_by_project_policy", None, **{k: v for k, v in cheap_common.items() if k != "model"})
    if not elements or atom_count < 1:
        return PreoptimizationDecision(False, "invalid_or_unreadable_structure", None, **{k: v for k, v in cheap_common.items() if k != "model"})
    unsupported = elements - supported
    if unsupported:
        return PreoptimizationDecision(False, "unsupported_elements:" + ",".join(sorted(unsupported)), model, **{k: v for k, v in cheap_common.items() if k != "model"})
    if estimate > limit:
        return PreoptimizationDecision(False, "estimated_runtime_exceeds_limit", model, **{k: v for k, v in cheap_common.items() if k != "model"})

    # Detailed validation is reserved for structures small enough to be viable
    # MACE candidates.  Severe overlaps are repaired deterministically first.
    severe_A = float(cfg.get("severe_overlap_A", 0.8))
    repair_A = float(cfg.get("repair_min_distance_A", 1.0))
    original_min = min_distance_poscar(path)
    repaired = original_min < severe_A and check_and_fix_poscar(path, min_dist=repair_A)
    minimum = min_distance_poscar(path) if repaired else original_min
    common = {
        **cheap_common,
        "minimum_distance_A": minimum,
        "overlap_repaired": repaired,
    }
    if mode == "auto" and is_crystalline(category) and not repaired:
        healthy_A = float(cfg.get("crystal_min_distance_A", 1.0))
        if minimum >= healthy_A:
            return PreoptimizationDecision(False, "crystal_already_physically_reasonable", model, **{k: v for k, v in common.items() if k != "model"})

    high_force = bool(overrides.get("high_force", False))
    if mode == "auto" and not (repaired or high_force or (molecular and generated_structure)):
        return PreoptimizationDecision(False, "no_high_force_or_generated_structure_signal", model, **{k: v for k, v in common.items() if k != "model"})
    reason = "forced_by_project_policy" if mode == "mace" else (
        "severe_overlap_repaired" if repaired else
        "high_force_structure" if high_force else "generated_molecular_structure"
    )
    return PreoptimizationDecision(True, reason, model, **{k: v for k, v in common.items() if k != "model"})
