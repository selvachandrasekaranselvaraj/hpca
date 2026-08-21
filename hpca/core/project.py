"""
hpca/core/project.py — MaterialProject dataclass and ProjectRegistry.

MaterialProject dataclass and ProjectRegistry for the HPCA platform.

Loading modes:
  1. Per-project yaml (standard — one project.yaml per project directory):
       ProjectRegistry.from_project_yaml("<project_dir>/project.yaml")
       ProjectRegistry.discover("<search_root>")   # scans for project.yaml files

  2. Central config (fallback — no per-project yaml present):
       ProjectRegistry.load_all("hpca/config/materials.yaml")
       Note: materials.yaml no longer stores project entries.
             Use "hpca new" to generate a project.yaml for each project.

Cross-references (update all when adding fields to MaterialProject):
  hpca/core/paths.py           — canonical directory layout (path helpers)
  hpca/config/platform.yaml    — simulation limits and HPC paths
  hpca/orchestrator/handlers/  — all handlers consume MaterialProject properties
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union

import yaml

from hpca.core.paths import (
    dft_base as _dft_base,
    mlmd_base as _mlmd_base,
    cmd_base as _cmd_base,
    continuum_base as _continuum_base,
)


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------

_YAML_PATH: Path = Path(__file__).parent.parent / "config" / "materials.yaml"


def _resolve(root: Path, rel: str) -> Path:
    """Join *root* with relative path *rel* and return the result."""
    return root / rel


# ---------------------------------------------------------------------------
# MaterialProject
# ---------------------------------------------------------------------------

@dataclass
class MaterialProject:
    """Dataclass representing a single simulation project with all metadata and path helpers."""

    name: str
    full_name: str
    mobile_ion: str
    category: str
    T_ref: int
    root: Path

    # Transport from MLMD/AIMD
    D_mlmd: Dict[str, float] = field(default_factory=dict)
    Ea_mlmd: Dict[str, float] = field(default_factory=dict)
    D_aimd: Optional[float] = None
    Ea_aimd: Optional[float] = None

    # Mechanical
    E_GPa: float = 0.0
    nu: float = 0.0
    Omega_A3: float = 0.0
    MW_mobile: float = 0.0
    rho_gcm3: float = 0.0

    # Simulation directory mappings (relative to root)
    aimd_dirs: List[str] = field(default_factory=list)
    mlmd_dirs: Dict[int, str] = field(default_factory=dict)

    # Combinatorial study support
    project_mode: str = "single"                       # "single" | "combinatorial"
    combinations: List[Dict[str, Any]] = field(default_factory=list)
    components: Dict[str, Any] = field(default_factory=dict)

    # Optional project-specific overrides / extras (D_cat, D_SSE, k_SEI, …)
    extras: Dict[str, Any] = field(default_factory=dict)

    # Merged category defaults live here after __post_init__
    _category_defaults: Dict[str, Any] = field(default_factory=dict, repr=False)

    # Layout: see hpca/core/paths.py
    # Category predicates delegate to hpca.core.categories (single source of truth).
    # To add a new category, register it there — no changes needed here.

    # ------------------------------------------------------------------
    # Category predicates (delegated to categories.py)
    # ------------------------------------------------------------------

    @property
    def is_sse(self) -> bool:
        """True when this project is a solid-state electrolyte (NEB + ECW pipeline)."""
        from hpca.core.categories import is_sse as _is_sse
        return _is_sse(self.category)

    @property
    def is_polymer(self) -> bool:
        """True for polymer or copolymer projects (chain-builder path)."""
        from hpca.core.categories import is_polymer as _is_polymer
        return _is_polymer(self.category)

    @property
    def is_liquid(self) -> bool:
        """True for any molecular/liquid category (alias for is_molecular)."""
        from hpca.core.categories import is_molecular as _is_molecular
        return _is_molecular(self.category)

    @property
    def is_inorganic(self) -> bool:
        """True for any crystalline/inorganic category (alias for is_crystalline)."""
        from hpca.core.categories import is_crystalline as _is_crystalline
        return _is_crystalline(self.category)

    @property
    def is_molecular(self) -> bool:
        """True for molecular categories (solvent, salt, polymer, …)."""
        from hpca.core.categories import is_molecular as _is_molecular
        return _is_molecular(self.category)

    @property
    def is_crystalline(self) -> bool:
        """True for crystalline categories (solid, inorganic, inorganic_sse)."""
        from hpca.core.categories import is_crystalline as _is_crystalline
        return _is_crystalline(self.category)

    @property
    def material_class(self) -> str:
        """Return 'molecular', 'crystalline', or 'custom' for this project's category."""
        from hpca.core.categories import material_class as _mc
        return _mc(self.category)

    @property
    def dft_base(self) -> Path:
        """Base directory for DFT calculations (dft/)."""
        return _dft_base(Path(self.root))

    @property
    def mlmd_base(self) -> Path:
        """Base directory for MLMD (mlmd/)."""
        return _mlmd_base(Path(self.root))

    @property
    def cmd_base(self) -> Path:
        """Base directory for CMD (cmd/)."""
        return _cmd_base(Path(self.root))

    @property
    def continuum_base(self) -> Path:
        """Base directory for continuum models (continuum/)."""
        return _continuum_base(Path(self.root))

    # ------------------------------------------------------------------
    # Best-available transport parameters
    # ------------------------------------------------------------------

    @property
    def D_best(self) -> Optional[float]:
        """Return the best available diffusivity: MLMD (deepmd first) > AIMD."""
        if self.D_mlmd:
            # Prefer deepmd; otherwise take the first available model
            return self.D_mlmd.get("deepmd") or next(iter(self.D_mlmd.values()))
        return self.D_aimd

    @property
    def Ea_best(self) -> Optional[float]:
        """Return the best available activation energy: MLMD > AIMD."""
        if self.Ea_mlmd:
            return self.Ea_mlmd.get("deepmd") or next(iter(self.Ea_mlmd.values()))
        return self.Ea_aimd

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def get_aimd_dir(self, T: Union[int, str]) -> Path:
        """Return absolute path for AIMD run at temperature T (K).

        Matches on the trailing numeric component of the stored relative
        path (e.g. 'aimd/300' → T=300).  Falls back to the first entry
        when no exact match is found.
        """
        key = str(int(T))
        for rel in self.aimd_dirs:
            if Path(rel).name == key:
                return _resolve(self.root, rel)
        if self.aimd_dirs:
            return _resolve(self.root, self.aimd_dirs[0])
        raise FileNotFoundError(
            f"[{self.name}] No AIMD directory registered for T={T} K"
        )

    def get_mlmd_dump(self, T: Union[int, str]) -> Path:
        """Return absolute path to dump_unwrapped.lmp for MLMD run at T (K)."""
        T_int = int(T)
        if T_int not in self.mlmd_dirs:
            available = sorted(self.mlmd_dirs.keys())
            raise KeyError(
                f"[{self.name}] MLMD temperature {T_int} K not registered. "
                f"Available: {available}"
            )
        rel = self.mlmd_dirs[T_int]
        return _resolve(self.root, rel) / "dump_unwrapped.lmp"

    @property
    def is_combinatorial(self) -> bool:
        """True when this project defines a combinatorial study with multiple compositions."""
        return bool(self.combinations)

    def get_combination_projects(self) -> List["MaterialProject"]:
        """Return a MaterialProject for each combination, rooted at {root}/{combo_name}."""
        if not self.is_combinatorial:
            return [self]
        sim = self.extras.get("simulation", {})
        concs = sim.get("salt_concs_M", [1.0])
        mlmd_temps = sim.get("mlmd_temps", [])
        aimd_temps = sim.get("aimd_temps", [])
        result = []
        for combo in self.combinations:
            cname = combo["name"]
            croot = self.root / cname
            # Build mlmd_dirs for this combination
            mlmd: Dict[int, str] = {}
            for c in concs:
                cfmt = str(c).replace(".", "p")
                for T in mlmd_temps:
                    mlmd[T] = f"{cname}/dlmd/{cfmt}M/{T}K"
            # aimd_dirs for this combination
            aimd_list = []
            for c in concs:
                cfmt = str(c).replace(".", "p")
                for T in aimd_temps:
                    aimd_list.append(f"{cname}/aimd/{cfmt}M/{T}K")
            cp = MaterialProject(
                name=cname,
                full_name=combo.get("label", cname),
                mobile_ion=self.mobile_ion,
                category=self.category,
                T_ref=self.T_ref,
                root=self.root,
                aimd_dirs=aimd_list,
                mlmd_dirs=mlmd,
                extras={**self.extras, "combination": combo},
            )
            cp._category_defaults = copy.deepcopy(self._category_defaults)
            result.append(cp)
        return result

    def get_deepmd_pot(self) -> Optional[Path]:
        """Return the absolute path to the DeepMD potential file, or None if not set."""
        pot_rel = self.extras.get("deepmd_pot")
        if pot_rel is None:
            return None
        return _resolve(self.root, pot_rel)

    # ------------------------------------------------------------------
    # Merged parameter access (category defaults + extras)
    # ------------------------------------------------------------------

    def param(self, key: str, default: Any = None) -> Any:
        """Look up a parameter: extras → category_defaults → default."""
        if key in self.extras:
            return self.extras[key]
        if key in self._category_defaults:
            return self._category_defaults[key]
        return default

    # ------------------------------------------------------------------
    # Convenience: MLMD temperature list
    # ------------------------------------------------------------------

    @property
    def mlmd_temperatures(self) -> List[int]:
        """Return sorted list of registered MLMD temperatures (K)."""
        return sorted(self.mlmd_dirs.keys())

    @property
    def aimd_temperatures(self) -> List[int]:
        """Return sorted list of AIMD temperatures (K) inferred from aimd_dirs path names."""
        temps = []
        for rel in self.aimd_dirs:
            try:
                temps.append(int(Path(rel).name))
            except ValueError:
                pass
        return sorted(temps)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        name: str,
        raw: Dict[str, Any],
        category_defaults: Optional[Dict[str, Any]] = None,
    ) -> "MaterialProject":
        """Build a MaterialProject from a raw YAML project block.

        All keys not mapped to explicit dataclass fields are collected into
        `extras` so that project-specific parameters (D_cat, D_SSE, k_SEI,
        VTF coefficients, …) are always reachable via project.extras or
        project.param().
        """
        known_fields = {
            "full_name", "mobile_ion", "category", "T_ref", "root",
            "D_mlmd", "Ea_mlmd", "D_aimd", "Ea_aimd",
            "E_GPa", "nu", "Omega_A3", "MW_mobile", "rho_gcm3",
            "aimd_dirs", "mlmd_dirs",
            "project_mode", "combinations", "components",
        }

        d = copy.deepcopy(raw)

        # Coerce mlmd_dirs keys to int where possible (single-conc projects use numeric
        # temperature keys; multi-conc/combinatorial use string keys like "0p25M_300K")
        raw_mlmd = d.pop("mlmd_dirs", {}) or {}
        mlmd_dirs: Dict[int, str] = {}
        for k, v in raw_mlmd.items():
            try:
                mlmd_dirs[int(k)] = v
            except (ValueError, TypeError):
                mlmd_dirs[k] = v  # keep as string key for multi-conc projects

        extras = {k: v for k, v in d.items() if k not in known_fields}
        for k in list(extras):
            d.pop(k, None)

        obj = cls(
            name=name,
            full_name=d.get("full_name", name),
            mobile_ion=d.get("mobile_ion", "Li"),
            category=d.get("category", "inorganic"),
            T_ref=int(d.get("T_ref", 300)),
            root=Path(d.get("root", ".")),
            D_mlmd=d.get("D_mlmd") or {},
            Ea_mlmd=d.get("Ea_mlmd") or {},
            D_aimd=d.get("D_aimd"),
            Ea_aimd=d.get("Ea_aimd"),
            E_GPa=float(d.get("E_GPa", 0.0)),
            nu=float(d.get("nu", 0.0)),
            Omega_A3=float(d.get("Omega_A3", 0.0)),
            MW_mobile=float(d.get("MW_mobile", 0.0)),
            rho_gcm3=float(d.get("rho_gcm3", 0.0)),
            aimd_dirs=d.get("aimd_dirs") or [],
            mlmd_dirs=mlmd_dirs,
            project_mode=d.get("project_mode", "single"),
            combinations=d.get("combinations") or [],
            components=d.get("components") or {},
            extras=extras,
        )

        cat = category_defaults or {}
        obj._category_defaults = copy.deepcopy(cat.get(obj.category, {}))
        return obj

    def __repr__(self) -> str:
        """Return a concise string representation for debugging."""
        return (
            f"MaterialProject(name={self.name!r}, "
            f"ion={self.mobile_ion}, cat={self.category}, "
            f"D_best={self.D_best})"
        )


# ---------------------------------------------------------------------------
# ProjectRegistry
# ---------------------------------------------------------------------------

class ProjectRegistry:
    """Loads and caches the full project catalogue from materials.yaml."""

    def __init__(self, yaml_path: Union[str, Path] = _YAML_PATH) -> None:
        """Initialise registry pointing at *yaml_path* (lazy-loaded on first access)."""
        self._yaml_path = Path(yaml_path)
        self._cache: Optional[Dict[str, MaterialProject]] = None

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    @classmethod
    def load_all(
        cls,
        yaml_path: Union[str, Path] = _YAML_PATH,
    ) -> Dict[str, MaterialProject]:
        """Return a name→MaterialProject dict from the central materials.yaml."""
        registry = cls(yaml_path)
        return registry._load()

    @classmethod
    def from_project_yaml(
        cls,
        project_yaml: Union[str, Path],
        platform_yaml: Union[str, Path] = None,
    ) -> "MaterialProject":
        """
        Load a single MaterialProject from its own project.yaml.
        Merges category defaults from platform.yaml if provided.
        """
        p = Path(project_yaml)
        with open(p) as f:
            data = yaml.safe_load(f) or {}

        cat_defaults: Dict[str, Any] = {}
        platform_path = platform_yaml or (
            Path(__file__).parent.parent / "config" / "platform.yaml"
        )
        if Path(platform_path).exists():
            with open(platform_path) as f:
                plat = yaml.safe_load(f) or {}
            cat_defaults = plat.get("category_defaults", {})

        name = data.get("name", p.parent.name)
        return MaterialProject.from_dict(name, data, cat_defaults)

    @classmethod
    def discover(
        cls,
        search_root: Union[str, Path],
        platform_yaml: Union[str, Path] = None,
    ) -> Dict[str, "MaterialProject"]:
        """
        Scan search_root for project.yaml files (one level deep) and load each.
        Returns name→MaterialProject dict.
        """
        root = Path(search_root)
        yamls = sorted(root.glob("*/project.yaml")) + \
                sorted(root.glob("*/*/project.yaml"))
        projects: Dict[str, MaterialProject] = {}
        for y in yamls:
            try:
                mp = cls.from_project_yaml(y, platform_yaml)
                projects[mp.name] = mp
            except Exception as e:
                print(f"  [registry] skip {y}: {e}")
        return projects

    def get(self, name: str) -> MaterialProject:
        """Return the MaterialProject for *name*, raising KeyError if not found."""
        projects = self._load()
        if name not in projects:
            raise KeyError(
                f"Project '{name}' not found. "
                f"Known projects: {sorted(projects)}"
            )
        return projects[name]

    def names(self) -> List[str]:
        """Return a sorted list of all project names."""
        return sorted(self._load().keys())

    def by_category(self, category: str) -> Dict[str, MaterialProject]:
        """Return a name→project dict filtered to projects with *category*."""
        return {
            n: p for n, p in self._load().items()
            if p.category == category
        }

    def hpc_config(self) -> Dict[str, Any]:
        """Return the raw 'hpc' block from the YAML for cluster settings."""
        with open(self._yaml_path) as fh:
            raw = yaml.safe_load(fh)
        return raw.get("hpc", {})

    def mlip_registry(self) -> Dict[str, Any]:
        """Return the raw 'mlip_registry' block from the YAML."""
        with open(self._yaml_path) as fh:
            raw = yaml.safe_load(fh)
        return raw.get("mlip_registry", {})

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, MaterialProject]:
        """Load and cache all projects from the YAML file; return the cache."""
        if self._cache is not None:
            return self._cache

        if not self._yaml_path.exists():
            raise FileNotFoundError(
                f"Config not found: {self._yaml_path}\n"
                "Ensure hpca/config/materials.yaml is present."
            )

        with open(self._yaml_path) as fh:
            raw = yaml.safe_load(fh)

        cat_defaults: Dict[str, Any] = raw.get("category_defaults", {})
        projects_raw: Dict[str, Any] = raw.get("projects", {})

        self._cache = {
            name: MaterialProject.from_dict(name, data, cat_defaults)
            for name, data in projects_raw.items()
        }
        return self._cache

    def __len__(self) -> int:
        """Return the number of registered projects."""
        return len(self._load())

    def __iter__(self):
        """Iterate over project names."""
        return iter(self._load())

    def __getitem__(self, name: str) -> MaterialProject:
        """Allow dict-style access: registry['project_name']."""
        return self.get(name)

    def __repr__(self) -> str:
        """Return a concise string representation for debugging."""
        return f"ProjectRegistry({self.names()})"
