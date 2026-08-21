"""
categories.py — Single source of truth for HPCA material category classification.

Built-in categories cover the six standard types. New categories can be registered
at runtime — no core code edits needed:

    from hpca.core.categories import registry, CategorySpec
    registry.register(CategorySpec(
        name           = "gel_electrolyte",
        material_class = "molecular",
        is_polymer     = True,
        needs_cmd      = True,
        continuum_model= "molecular",
    ))

Category string → capability flags is the ONLY place category-specific logic lives.
Every handler imports predicates from here instead of doing raw string checks.

Cross-ref:
  hpca/config/platform.yaml  — category_defaults (physics constants per category)
  hpca/registry/stage.py — CATEGORY_DEFAULTS (pipeline per category)
  hpca/core/project.py       — MaterialProject.is_sse, .is_molecular etc.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal

# ---------------------------------------------------------------------------
# CategorySpec — one record per material class
# ---------------------------------------------------------------------------

MaterialClass = Literal["molecular", "crystalline", "custom"]

@dataclass
class CategorySpec:
    """Immutable description of one material category.

    Add a new field here when a future stage needs a new capability flag
    (e.g. needs_interface_builder, needs_qm_mm ...). Existing categories
    inherit the field's default value so nothing breaks.
    """
    name: str

    # ── What kind of structure is this? ──────────────────────────────────
    material_class: MaterialClass = "custom"

    # ── Structure generation ──────────────────────────────────────────────
    # True  → use PACKMOL (solvents, salts, polymers, copolymers)
    # False → direct POSCAR/CIF (crystals, solids)
    needs_packmol: bool = False

    # ── Pipeline capabilities ─────────────────────────────────────────────
    # Classical MD (OPLS-AA / force-field)
    needs_cmd: bool = False

    # NEB ion-migration barriers
    needs_neb: bool = False

    # Electrochemical window (ECW) — shared echem/ folder for SSE
    needs_echem: bool = False

    # Bader charges + DOS electronic structure
    needs_electronic: bool = False

    # ── AIMD dataset perturbation style ──────────────────────────────────
    # "random"  → fully randomised fractional coords (liquid/polymer)
    # "rattle"  → Gaussian noise on equilibrium positions (crystal/SSE)
    aimd_rand_kind: str = "random"

    # ── Continuum physics model family ───────────────────────────────────
    # "molecular"  → VTF/WLF diffusion, SEI growth, polymer models
    # "inorganic"  → Marcus theory, KJMA crystallisation, Butler-Volmer
    continuum_model: str = "molecular"

    # ── Sub-category flags ────────────────────────────────────────────────
    is_polymer: bool = False   # chain-builder path in h00_design
    is_sse: bool = False       # solid-state electrolyte (NEB + ECW + OCV)
    is_solid: bool = False     # generic crystalline solid (no NEB/ECW)

    def __post_init__(self) -> None:
        """Enforce derived consistency: SSE and solid categories are always crystalline."""
        if self.is_sse or self.is_solid:
            object.__setattr__(self, "material_class", "crystalline")
            object.__setattr__(self, "aimd_rand_kind", "rattle")
            object.__setattr__(self, "continuum_model", "inorganic")

    # ── Read-only class-level registry reference (set by CategoryRegistry) ─
    _registry: ClassVar["CategoryRegistry | None"] = None


# ---------------------------------------------------------------------------
# CategoryRegistry — global singleton, holds all known categories
# ---------------------------------------------------------------------------

class CategoryRegistry:
    """Runtime registry for material categories.

    Handlers call module-level convenience functions (is_molecular, is_sse …)
    which delegate here.  Users or plugins can extend at import time:

        from hpca.core.categories import registry, CategorySpec
        registry.register(CategorySpec(name="my_new_cat", material_class="molecular", ...))
    """

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self._specs: dict[str, CategorySpec] = {}

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, spec: CategorySpec) -> None:
        """Register or replace a CategorySpec. Call at module import time."""
        if not isinstance(spec, CategorySpec):
            raise TypeError(f"Expected CategorySpec, got {type(spec)}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> CategorySpec:
        """Return spec for name, or the 'custom' fallback if unknown."""
        if name in self._specs:
            return self._specs[name]
        # Unknown category: warn once, return custom fallback
        import logging
        logging.getLogger("hpca.core").warning(
            "[categories] Unknown category %r — treating as 'custom'. "
            "Register it with registry.register(CategorySpec(name=%r, ...))",
            name, name,
        )
        return self._specs.get("custom", CategorySpec(name="custom"))

    def is_known(self, name: str) -> bool:
        """Return True if *name* is a registered category."""
        return name in self._specs

    def all_names(self) -> list[str]:
        """Return a sorted list of all registered category names."""
        return sorted(self._specs.keys())

    # ── Predicate convenience methods ─────────────────────────────────────

    def is_molecular(self, cat: str) -> bool:
        """True for solvents, salts, liquids, polymers, copolymers."""
        return self.get(cat).material_class == "molecular"

    def is_crystalline(self, cat: str) -> bool:
        """True for solids, inorganic SSEs, and any crystal."""
        return self.get(cat).material_class == "crystalline"

    def is_sse(self, cat: str) -> bool:
        """True for solid-state electrolytes with NEB + ECW pipeline."""
        return self.get(cat).is_sse

    def is_solid(self, cat: str) -> bool:
        """True for generic crystalline solids (may or may not be SSE)."""
        return self.get(cat).is_solid or self.get(cat).is_sse

    def is_polymer(self, cat: str) -> bool:
        """True for homopolymers and copolymers (chain-builder path)."""
        return self.get(cat).is_polymer

    def needs_packmol(self, cat: str) -> bool:
        """True when PACKMOL is used to build the initial structure."""
        return self.get(cat).needs_packmol

    def needs_cmd(self, cat: str) -> bool:
        """True when OPLS-AA classical MD is in the pipeline."""
        return self.get(cat).needs_cmd

    def needs_neb(self, cat: str) -> bool:
        """True when NEB barrier calculation is in the pipeline."""
        return self.get(cat).needs_neb

    def needs_echem(self, cat: str) -> bool:
        """True when electrochemical window calculation is in the pipeline."""
        return self.get(cat).needs_echem

    def needs_electronic(self, cat: str) -> bool:
        """True when Bader/DOS electronic analysis is in the pipeline."""
        return self.get(cat).needs_electronic

    def aimd_rand_kind(self, cat: str) -> str:
        """'rattle' for crystals, 'random' for molecular."""
        return self.get(cat).aimd_rand_kind

    def continuum_model(self, cat: str) -> str:
        """'molecular' or 'inorganic'."""
        return self.get(cat).continuum_model

    def material_class(self, cat: str) -> str:
        """Return 'molecular', 'crystalline', or 'custom' for *cat*."""
        return self.get(cat).material_class


# ---------------------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------------------

registry = CategoryRegistry()

# ---------------------------------------------------------------------------
# Built-in category registration
# ---------------------------------------------------------------------------

_BUILTINS: list[CategorySpec] = [
    # ── Molecular categories ─────────────────────────────────────────────────
    # Organic solvents (DME, EC, DMC, DMB, …)
    CategorySpec(
        name="solvent", material_class="molecular",
        needs_packmol=True, needs_cmd=True,
        continuum_model="molecular",
    ),
    # Ionic salts (LiFSI, LiPF6, LiTFSI, …)
    CategorySpec(
        name="salt", material_class="molecular",
        needs_packmol=True, needs_cmd=True,
        continuum_model="molecular",
    ),
    # Mixed solvent+salt liquid electrolyte (legacy default)
    CategorySpec(
        name="liquid_electrolyte", material_class="molecular",
        needs_packmol=True, needs_cmd=True,
        continuum_model="molecular",
    ),
    # Homopolymers (PEO, PVDF, PTFEP, PMMA, …)
    CategorySpec(
        name="polymer", material_class="molecular",
        needs_packmol=True, needs_cmd=True,
        is_polymer=True, continuum_model="molecular",
    ),
    # Mixed-chain copolymers (PVDF-HFP, PVDF-TrFE, …)
    CategorySpec(
        name="copolymer", material_class="molecular",
        needs_packmol=True, needs_cmd=True,
        is_polymer=True, continuum_model="molecular",
    ),

    # ── Crystalline categories ────────────────────────────────────────────────
    # Generic inorganic solid (electrodes, hard carbon, MOFs, interfaces)
    CategorySpec(
        name="solid", material_class="crystalline",
        needs_packmol=False, needs_cmd=False,
        is_solid=True, needs_electronic=True,
        needs_neb=True, needs_echem=True, is_sse=True,
        aimd_rand_kind="rattle", continuum_model="inorganic",
    ),
    # Inorganic solid without SSE extras (legacy)
    CategorySpec(
        name="inorganic", material_class="crystalline",
        needs_packmol=False, needs_cmd=False,
        is_solid=True, needs_electronic=True,
        aimd_rand_kind="rattle", continuum_model="inorganic",
    ),
    # Solid-state electrolyte — full pipeline: NEB + ECW + OCV (legacy default)
    CategorySpec(
        name="inorganic_sse", material_class="crystalline",
        needs_packmol=False, needs_cmd=False,
        is_sse=True, is_solid=True, needs_electronic=True,
        needs_neb=True, needs_echem=True,
        aimd_rand_kind="rattle", continuum_model="inorganic",
    ),

    # ── Hybrid solid/liquid ──────────────────────────────────────────────────
    # Electrode slab + PACKMOL electrolyte sandwich (hpca.core.interface_builder).
    # is_solid=True forces aimd_rand_kind="rattle" — appropriate here since the
    # electrode's covalent network must not be blown apart by full coordinate
    # randomization the way a pure liquid box safely can be. No NEB/ECW/Bader:
    # this category's focus is SEI formation / Na-plating vs. intercalation,
    # not vacancy migration or electronic structure.
    #
    # needs_cmd=True here means a PURE bulk-electrolyte CMD lane (no electrode)
    # — OPLS-AA is a non-reactive molecular force field with no parameters for
    # an extended carbon network, so it cannot represent the electrode or any
    # interfacial (SEI/plating) chemistry. CMD's role is cheap, large-scale,
    # long-timescale bulk transport properties (conductivity, diffusion) as a
    # reference point, computed away from the interface entirely.
    CategorySpec(
        name="electrode_electrolyte_interface", material_class="crystalline",
        needs_packmol=True, needs_cmd=True,
        needs_neb=False, needs_echem=False, needs_electronic=False,
        is_solid=True,
    ),

    # ── Fallback ──────────────────────────────────────────────────────────────
    # Unknown / user-defined; inherits safe defaults (no destructive ops)
    CategorySpec(
        name="custom", material_class="custom",
        needs_packmol=False, needs_cmd=False,
        continuum_model="molecular",
    ),
]

for _spec in _BUILTINS:
    registry.register(_spec)


# ---------------------------------------------------------------------------
# Module-level convenience functions — import these in handlers
# ---------------------------------------------------------------------------

def is_molecular(cat: str) -> bool:
    """True for solvents, salts, liquid_electrolyte, polymer, copolymer."""
    return registry.is_molecular(cat)

def is_crystalline(cat: str) -> bool:
    """True for solid, inorganic, inorganic_sse."""
    return registry.is_crystalline(cat)

def is_sse(cat: str) -> bool:
    """True when NEB + ECW + OCV pipeline is enabled."""
    return registry.is_sse(cat)

def is_solid(cat: str) -> bool:
    """True for any crystalline solid (superset of is_sse)."""
    return registry.is_solid(cat)

def is_polymer(cat: str) -> bool:
    """True for polymer / copolymer (chain-builder in h00_design)."""
    return registry.is_polymer(cat)

def needs_packmol(cat: str) -> bool:
    """True when PACKMOL is used for structure generation."""
    return registry.needs_packmol(cat)

def needs_cmd(cat: str) -> bool:
    """True when OPLS-AA CMD stage is in the pipeline."""
    return registry.needs_cmd(cat)

def needs_neb(cat: str) -> bool:
    """True when NEB barrier calculation is in the pipeline."""
    return registry.needs_neb(cat)

def needs_echem(cat: str) -> bool:
    """True when electrochemical window calculation is in the pipeline."""
    return registry.needs_echem(cat)

def needs_electronic(cat: str) -> bool:
    """True when Bader/DOS electronic analysis is in the pipeline."""
    return registry.needs_electronic(cat)

def aimd_rand_kind(cat: str) -> str:
    """Return 'rattle' for crystals, 'random' for molecular."""
    return registry.aimd_rand_kind(cat)

def continuum_model(cat: str) -> str:
    """Return 'inorganic' or 'molecular' physics model family."""
    return registry.continuum_model(cat)

def material_class(cat: str) -> str:
    """Return 'molecular', 'crystalline', or 'custom'."""
    return registry.material_class(cat)
