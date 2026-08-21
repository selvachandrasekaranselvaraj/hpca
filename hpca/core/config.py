"""
hpca/core/config.py — Global platform configuration singleton.

Single cached load of platform.yaml; typed accessor methods used by all
modules so no file is read more than once per process.

Backward compatible: existing code that calls
    load_platform_config()  (from hpca.core.paths)
continues to work unchanged.  New code uses Config.get().

Usage:
    from hpca.core.config import Config
    cfg = Config.get()
    cfg.hpc("vasp_module")               # → "vasp/6.4.2_openMP"
    cfg.ncore(natoms=80)                 # → {"ncore":8,"ntasks":96,"njobs":1}
    cfg.kpoints_mesh(a_angstrom=5.5)     # → [6,6,6]
    cfg.npt_step0("mol")                 # → {isif:3, nsw:20000, ...}
    cfg.gpu_sizing(pot_mb=120)           # → {gpus:4, ntasks:4, ...}
    cfg.nvt_temperatures("sse")          # → [300,320,...,700]
    cfg.slurm_time("dft_vc")            # → "48:00:00"
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

_PLATFORM_YAML = Path(__file__).parent.parent / "config" / "platform.yaml"
_lock = threading.Lock()
_instance: "Config | None" = None


class Config:
    """Singleton wrapper around platform.yaml."""

    def __init__(self, data: dict) -> None:
        """Store the raw platform.yaml dict."""
        self._data = data

    # ── Singleton access ──────────────────────────────────────────────────────

    @classmethod
    def get(cls) -> "Config":
        """Return the process-wide Config singleton, loading platform.yaml on first call."""
        global _instance
        if _instance is None:
            with _lock:
                if _instance is None:
                    _instance = cls(_load_yaml(_PLATFORM_YAML))
        return _instance

    @classmethod
    def reload(cls) -> "Config":
        """Force re-read of platform.yaml (e.g. after hot-edit)."""
        global _instance
        with _lock:
            _instance = cls(_load_yaml(_PLATFORM_YAML))
        return _instance

    # ── Raw access ────────────────────────────────────────────────────────────

    @property
    def raw(self) -> dict:
        """Return the raw platform.yaml dict."""
        return self._data

    def section(self, key: str, default: Any = None) -> Any:
        """Return a top-level section from platform.yaml, or *default* if absent."""
        return self._data.get(key, default or {})

    # ── HPC paths ─────────────────────────────────────────────────────────────

    def hpc(self, key: str, default: str = "") -> str:
        """Return a value from the ``hpc:`` section of platform.yaml."""
        return self._data.get("hpc", {}).get(key, default)

    def account(self, key: str = "standard") -> str:
        """Return a SLURM account name from ``hpc.accounts`` (default: 'standard').

        Missing keys fall back to ``hpc.account_fallback`` — the literal account
        name lives only in platform.yaml, never in code.
        """
        hpc = self._data.get("hpc", {})
        return hpc.get("accounts", {}).get(key) or hpc.get("account_fallback", "")

    def modules(self, bundle: str) -> list:
        """Return a module-name list from ``hpc.modules.{bundle}`` (e.g. 'vasp', 'gpu_md').

        Combine with ``slurm_submit.module_load_block()`` to render the shell lines.
        Empty list when the bundle is undefined (site without environment modules).
        """
        return list(self._data.get("hpc", {}).get("modules", {}).get(bundle, []))

    def partition(self, key: str) -> str:
        """Return a SLURM partition name from ``hpc.partitions``."""
        return self._data.get("hpc", {}).get("partitions", {}).get(key, "")

    # ── Simulation limits ─────────────────────────────────────────────────────

    def limit(self, lane: str, key: str, default: Any = None) -> Any:
        """lane: 'slurm' (only supported lane)."""
        return self._data.get("limits", {}).get(lane, {}).get(key, default)

    def slurm_time(self, stage: str, default: str = "48:00:00") -> str:
        """Return the SLURM walltime string for a named stage from ``slurm_time:``."""
        return self._data.get("slurm_time", {}).get(stage, default)

    # ── NCORE rules ───────────────────────────────────────────────────────────

    def ncore(self, natoms: int) -> dict:
        """Return {ncore, ntasks, njobs} for the given atom count."""
        for rule in self._data.get("ncore_rules", []):
            if natoms <= rule["max_atoms"]:
                return {"ncore": rule["ncore"],
                        "ntasks": rule["ntasks"],
                        "njobs": rule["njobs"]}
        return {"ncore": 8, "ntasks": 96, "njobs": 1}

    # ── K-points ─────────────────────────────────────────────────────────────

    def kpoints_mesh(self, a_angstrom: float) -> list[int]:
        """Return [k1,k2,k3] Monkhorst-Pack mesh for given lattice parameter."""
        for rule in self._data.get("kpoints_rules", []):
            if a_angstrom <= rule["max_a"]:
                return list(rule["mesh"])
        return [1, 1, 1]

    # ── NPT Step 0 ────────────────────────────────────────────────────────────

    def npt_step0(self, category_class: str) -> dict:
        """category_class: 'mol', 'sse', or 'int'."""
        return dict(self._data.get("npt_step0", {}).get(category_class, {}))

    # ── GPU sizing ────────────────────────────────────────────────────────────

    def gpu_sizing(self, pot_mb: float) -> dict:
        """Return {gpus, ntasks, cpus_per_task, mem_gb} from pot_com.pb file size."""
        for rule in self._data.get("gpu_sizing", []):
            if pot_mb <= rule["max_mb"]:
                return {k: rule[k] for k in
                        ("gpus", "ntasks", "cpus_per_task", "mem_gb")}
        last = self._data.get("gpu_sizing", [{}])[-1]
        return {k: last.get(k, 1) for k in
                ("gpus", "ntasks", "cpus_per_task", "mem_gb")}

    # ── NVT temperatures ─────────────────────────────────────────────────────

    def nvt_temperatures(self, category_class: str) -> list[int]:
        """category_class: 'mol', 'sse', or 'int'."""
        by_class = self._data.get("nvt_temperatures", {})
        if isinstance(by_class, dict):
            temps = by_class.get(category_class,
                                 by_class.get("mol", [300, 320, 340, 360, 380, 400, 450, 500]))
        else:
            temps = list(by_class)
        return [int(t) for t in temps]

    # ── NPT validation ────────────────────────────────────────────────────────

    def npt_validation(self) -> dict:
        """Return NPT validation thresholds (density range, temperature tolerance)."""
        return dict(self._data.get("npt_validation", {
            "density_min_g_cm3": 0.5,
            "density_max_g_cm3": 3.0,
            "temperature_tolerance_K": 100,
        }))

    # ── MLIP defaults ─────────────────────────────────────────────────────────

    def mlip(self, key: str, default: Any = None) -> Any:
        """Return a value from the ``mlip_defaults:`` section of platform.yaml."""
        return self._data.get("mlip_defaults", {}).get(key, default)

    # ── AIMD dataset ─────────────────────────────────────────────────────────

    def aimd_dataset(self, key: str, default: Any = None) -> Any:
        """Return a value from the ``aimd_dataset:`` section of platform.yaml."""
        return self._data.get("aimd_dataset", {}).get(key, default)

    # ── Perf model ────────────────────────────────────────────────────────────

    def perf(self, key: str, default: float = 1.0) -> float:
        """Return a float from the ``perf_model:`` section (empirical rates and caps)."""
        return float(self._data.get("perf_model", {}).get(key, default))

    # ── Auto-fix rules ────────────────────────────────────────────────────────

    def auto_fix(self, key: str, default: Any = None) -> Any:
        """Return a value from the ``auto_fix_rules:`` section of platform.yaml."""
        return self._data.get("auto_fix_rules", {}).get(key, default)

    # ── Orchestrator settings ─────────────────────────────────────────────────

    def orch(self, key: str, default: Any = None) -> Any:
        """Return a value from the ``orchestrator:`` section of platform.yaml."""
        return self._data.get("orchestrator", {}).get(key, default)

    # ── Lammps MD params ─────────────────────────────────────────────────────

    def lammps_md(self, key: str, default: Any = None) -> Any:
        """Return a value from the ``lammps_md:`` section of platform.yaml."""
        return self._data.get("lammps_md", {}).get(key, default)

    # ── Category physics defaults ─────────────────────────────────────────────

    def category_defaults(self, category: str) -> dict:
        """Return physics-constant defaults dict for a material category."""
        return dict(self._data.get("category_defaults", {}).get(category, {}))

    # ── Project schema ────────────────────────────────────────────────────────

    def project_schema(self, key: str, default: Any = None) -> Any:
        """Return a value from the ``project_schema:`` section of platform.yaml."""
        return self._data.get("project_schema", {}).get(key, default)


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict; returns {} on any error."""
    try:
        import yaml
        if not path.exists():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        import warnings
        warnings.warn(f"hpca Config: could not load {path}: {exc}")
        return {}


def account_fallback() -> str:
    """Return ``hpc.account_fallback`` from platform.yaml.

    Module-level convenience so dict-style config readers share one source for
    the default SLURM account — the literal name exists only in platform.yaml.
    """
    return Config.get().hpc("account_fallback", "")
