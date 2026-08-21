"""
h00_design.py — Materials Design handler (daemon-local).

Crystal   : CIF/MP → POSCAR, supercell, NEB vacancy
Polymer   : PACKMOL chain building via matdesign
Liquid    : PACKMOL cell packing per (combination, concentration) using .vasp molecules
"""
from __future__ import annotations

import logging
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .base import SimulationHandler
from hpca.core.categories import (
    is_molecular as _cat_is_molecular,
    is_crystalline as _cat_is_crystalline,
    is_polymer as _cat_is_polymer,
)

if TYPE_CHECKING:
    from hpca.orchestrator.state_tracker import ProjectState

# Layout: see hpca/core/paths.py
from hpca.core.paths import (
    designed_structures as _designed_structures, preopt as _preopt,
    dft_preopt as _dft_preopt, contcar_preopt, dft_opt, load_platform_config,
)
from hpca.registry.folder import preoptimization_policy
from hpca.orchestrator.daemon_tasks import (
    MaterialDesignTask,
    PreoptimizationTask,
    get_daemon_task_scheduler,
)

log = logging.getLogger("hpca.orch")

# Paths resolved from platform.yaml at runtime via self.hpc_path() / self.platform_config()
# Cross-ref: hpca/config/platform.yaml hpc section
PRERELAX_SCRIPT        = str(Path(__file__).parents[1] / "prerelax_mace.py")
PRERELAX_LAMMPS_SCRIPT = str(Path(__file__).parents[1] / "prerelax_mace_lammps.py")

# Global shared molecule library (platform.yaml hpc.input_structures_library).
# h00_design checks this after per-project input_structures/ and before SMILES generation.
# For combinatorial projects the per-project dir is already a symlink to the parent, so
# the lookup order is effectively: parent input_structures/ → global library → SMILES/fetch.
_GLOBAL_INPUT_LIB: Path | None = (
    Path(p) if (p := load_platform_config().get("hpc", {}).get("input_structures_library", "")) else None
)

# PACKMOL binary: check env var first, then platform.yaml hpc.packmol_bin
_hpc_packmol = load_platform_config().get("hpc", {}).get("packmol_bin", "")
_PACKMOL_CANDIDATES = [
    os.environ.get("PACKMOL_BIN", ""),
    _hpc_packmol,
]
PACKMOL_BIN = next((p for p in _PACKMOL_CANDIDATES
                    if p and Path(os.path.expanduser(p)).exists()), "")


def _find_mol_vasp(mol_name: str, local_dir: Path) -> Path | None:
    """Return the first existing {mol_name}.vasp from local_dir or the global library.

    For combinatorial sub-projects local_dir is already a symlink to the parent's
    input_structures/, so no explicit parent traversal is needed here.
    Lookup order: local_dir → global library.
    """
    local = local_dir / f"{mol_name}.vasp"
    if local.exists():
        return local
    if _GLOBAL_INPUT_LIB:
        gl = _GLOBAL_INPUT_LIB / f"{mol_name}.vasp"
        if gl.exists():
            return gl
    return None


def _find_oligo_vasp(pname: str, n_units: int, local_dir: Path,
                     fallback: bool = True) -> Path | None:
    """Return first existing {pname}_oligo{n_units}.vasp from local_dir or global library.

    With fallback=True (default) also checks shorter cached oligomers (n_units-5, 10, 5).
    With fallback=False only looks for the exact n_units oligomer.
    """
    candidates = [n_units, n_units - 5, 10, 5] if fallback else [n_units]
    for n in candidates:
        if n < 2:
            continue
        for base in ([local_dir] + ([_GLOBAL_INPUT_LIB] if _GLOBAL_INPUT_LIB else [])):
            p = base / f"{pname}_oligo{n}.vasp"
            if p.exists():
                if n != n_units:
                    log.debug("[h00_design] Using %s %d-mer (no %d-mer cached)", pname, n, n_units)
                return p
    return None


def _save_to_global_lib(src: Path, dest_name: str) -> None:
    """Atomically copy src to the global library if the library is writable."""
    if not (_GLOBAL_INPUT_LIB and _GLOBAL_INPUT_LIB.is_dir()):
        return
    dst = _GLOBAL_INPUT_LIB / dest_name
    if dst.exists():
        return
    try:
        import shutil as _sh
        tmp = dst.with_suffix(".tmp")
        _sh.copy2(src, tmp)
        tmp.replace(dst)   # atomic rename on POSIX
        log.info("[h00_design] Saved %s to global library", dest_name)
    except Exception as exc:
        log.debug("[h00_design] Could not save %s to global library: %s", dest_name, exc)


# Atoms per molecule for common electrolyte species (used for box sizing)
_NATOMS_PER_MOL: dict[str, int] = {
    "DME": 16, "DMB": 34, "DOL": 9,  "EC": 10, "DMC": 10, "EMC": 12, "DEC": 18,
    "PC":  12, "FEC": 10, "VC":  9,  "ACN": 6, "TMS": 11, "SN":  7,
    "GBL": 9,  "DMSO": 6, "THF": 9,  "DEE": 12, "TEGDME": 26,
    "LiFSI": 10, "LiTFSI": 15, "LiPF6": 8, "LiClO4": 6,
    "NaPF6": 8,  "NaFSI": 10,  "LiBF4": 6, "LiDFOB": 9,
    "NaTFSI": 15, "NaClO4": 6,
    "PEO": 8, "PVDF": 5, "PVDF-HFP": 7, "PVDF-TrFE": 7,
    "PMMA": 13, "PTFEP": 18,
}
_SPECIES_FB = load_platform_config().get("species_fallbacks", {}) or {}
_DEFAULT_NATOMS_PER_MOL = int(_SPECIES_FB.get("natoms_per_mol", 12))  # unknown-species fallback (warned)

# Molecular weights (g/mol) for common electrolyte components
_MW: dict[str, float] = {
    "DMB": 118.17, "DME": 90.12,  "DOL": 74.08,  "EC": 88.06,
    "DMC": 90.08,  "EMC": 104.10, "PC": 102.09,  "FEC": 106.05, "DEC": 118.13,
    "ACN": 41.05,  "TMS": 88.15,  "SN": 80.09,   "GBL": 86.09,
    "DMSO": 78.13, "THF": 72.11,  "DEE": 74.12,  "TEGDME": 222.28,
    "LiFSI": 187.07, "LiTFSI": 287.08, "LiPF6": 151.90,
    "LiClO4": 106.39, "NaPF6": 167.95, "LiDFOB": 139.77,
    "NaFSI": 203.07, "NaTFSI": 303.08,
    "PEO": 44.05, "PVDF": 64.03,
}

# Densities (g/cm³) used for molarity → mole-fraction conversion
_DENSITY: dict[str, float] = {
    "DMB": 0.789, "DME": 0.862,  "DOL": 1.060, "EC": 1.321,
    "DMC": 1.069, "EMC": 1.006,  "PC": 1.200,  "FEC": 1.454, "DEC": 0.975,
    "ACN": 0.786, "TMS": 0.848,  "SN": 1.070,  "GBL": 1.129,
    "DMSO": 1.100, "THF": 0.889, "DEE": 0.713, "TEGDME": 1.009,
    "LiFSI": 1.55, "LiTFSI": 1.33, "LiPF6": 1.50, "LiClO4": 2.42,
    "NaPF6": 2.02, "NaFSI": 1.55, "NaTFSI": 1.30, "LiDFOB": 1.65,
}
_DEFAULT_DENSITY = float(_SPECIES_FB.get("density_gcm3", 1.0))


def _merge_molecule_data() -> None:
    """Extend the species tables from hpca/data/molecular_properties.json and
    the platform.yaml ``molecule_data:`` section (yaml has highest priority).

    Any site can add new solvents/salts/polymers via platform.yaml without
    code changes — this is what makes box sizing work for arbitrary species.
    """
    try:
        from hpca.data import load as _ld
        for name, props in (_ld("molecular_properties") or {}).items():
            if not isinstance(props, dict):
                continue
            if "mw" in props:
                _MW.setdefault(name, float(props["mw"]))
            if "density_gcm3" in props:
                _DENSITY.setdefault(name, float(props["density_gcm3"]))
            if "natoms" in props:
                _NATOMS_PER_MOL.setdefault(name, int(props["natoms"]))
    except Exception as exc:
        log.debug("[h00_design] molecular_properties.json merge failed: %s", exc)
    try:
        for name, props in (load_platform_config().get("molecule_data") or {}).items():
            if not isinstance(props, dict):
                continue
            if "natoms" in props:
                _NATOMS_PER_MOL[name] = int(props["natoms"])
            if "mw" in props:
                _MW[name] = float(props["mw"])
            if "density_gcm3" in props:
                _DENSITY[name] = float(props["density_gcm3"])
    except Exception as exc:
        log.debug("[h00_design] platform.yaml molecule_data merge failed: %s", exc)


_merge_molecule_data()


def _refresh_species_tables(names: "set[str]", project_dir: Path) -> None:
    """Derive exact natoms (and missing MW) for species from their molecule
    .vasp files in input_structures/ (project first, global library second).

    Makes box sizing exact for ANY molecule the design flow has fetched —
    the curated tables become a fallback, not a limit.  Species that end up
    with no data at all are logged as WARNING so a silently mis-sized box
    can never happen again.
    """
    from hpca.core.vasp_job import poscar_element_counts
    try:
        from hpca.data import load as _ld
        masses = _ld("atomic_masses") or {}
    except Exception:
        masses = {}
    for name in names:
        if not name:
            continue
        mol_file = None
        for base in (project_dir / "input_structures", _GLOBAL_INPUT_LIB):
            if base and (base / f"{name}.vasp").exists():
                mol_file = base / f"{name}.vasp"
                break
        if mol_file is not None:
            counts = poscar_element_counts(mol_file)
            n = sum(counts.values())
            if n > 0:
                if _NATOMS_PER_MOL.get(name) not in (None, n):
                    log.info("[h00_design] %s: natoms %s (table) → %d (from %s)",
                             name, _NATOMS_PER_MOL.get(name), n, mol_file.name)
                _NATOMS_PER_MOL[name] = n
                mw = sum(masses.get(el, 0.0) * c for el, c in counts.items())
                if mw > 0 and name not in _MW:
                    _MW[name] = mw
        if name not in _NATOMS_PER_MOL:
            log.warning("[h00_design] Unknown species '%s': no molecule file or "
                        "molecule_data entry — using default %d atoms/mol. "
                        "Add it to platform.yaml molecule_data: for exact box sizing.",
                        name, _DEFAULT_NATOMS_PER_MOL)
        if name not in _DENSITY:
            log.warning("[h00_design] Species '%s': no density known — using %.1f g/cm³ "
                        "(platform.yaml molecule_data: {%s: {density_gcm3: ...}} to fix)",
                        name, _DEFAULT_DENSITY, name)


def _species_in_yaml(yaml_data: dict) -> "set[str]":
    """Collect every species name referenced by the project's composition."""
    names: set[str] = set()
    sim = yaml_data.get("simulation", {}) or {}
    for spec in (sim.get("comp_spec", {}) or {},):
        for cat_key in ("solvents", "salts", "polymers", "copolymers"):
            for entry in spec.get(cat_key, []) or []:
                if isinstance(entry, dict) and entry.get("name"):
                    names.add(entry["name"])
    for entry in sim.get("solvents", []) or []:
        if isinstance(entry, dict) and entry.get("name"):
            names.add(entry["name"])
    if sim.get("salt"):
        names.add(sim["salt"])
    for tier_key in ("molecule_counts_aimd", "molecule_counts_mlmd", "molecule_counts_cmd"):
        names.update((sim.get(tier_key) or {}).keys())
    return names


# OPLS atom-type label → element symbol (for dump_modify element)
_OPLS_ELEMENT: dict[str, str] = {
    "OS": "O",    "CT_O": "C",  "CT_M": "C",  "CT_C": "C",
    "C_CO": "C",  "O_CO": "O",  "OS_E": "O",
    "NI": "N",    "SF": "S",    "OY": "O",    "FS": "F",
    "CF": "C",    "FT": "F",    "Li": "Li",
    "CT_H2": "C", "CT_F2": "C", "CT_F1": "C", "CT_F3": "C", "FP": "F",
    "P_N": "P",   "N_P": "N",   "CT_B": "C",  "CQ": "C",    "HC": "H",
}


def _read_atom_count(data_path: Path) -> str:
    """Return atom count string from a LAMMPS data file header."""
    try:
        for ln in data_path.read_text().splitlines():
            parts = ln.split()
            if len(parts) >= 2 and parts[1] == "atoms":
                return parts[0]
    except Exception:
        pass
    return "?"


def _parse_elements_from_data(data_path: Path) -> list[str]:
    """Return element symbols in atom type ID order by parsing Masses section."""
    elements: dict[int, str] = {}
    try:
        in_masses = False
        for line in data_path.read_text().splitlines():
            stripped = line.strip()
            if stripped == "Masses":
                in_masses = True
                continue
            if in_masses:
                if not stripped:
                    continue
                if not stripped[0].isdigit():
                    break
                parts = stripped.split()
                if len(parts) >= 2:
                    type_id = int(parts[0])
                    opls_type = stripped.split("#", 1)[1].strip() if "#" in stripped else ""
                    elements[type_id] = _OPLS_ELEMENT.get(opls_type, "C")
    except Exception:
        pass
    return [elements[k] for k in sorted(elements.keys())]


class MaterialsDesignHandler(SimulationHandler):
    """Daemon-local handler: builds crystal, polymer, or liquid electrolyte cells."""

    name      = "h00_design"
    is_daemon = True

    @staticmethod
    def _migrate_legacy_dft_preopt(project_dir: Path) -> None:
        """Move the former root-level DFT preopt artifact to its canonical DFT directory."""
        legacy = _preopt(project_dir) / "contcar_dft_preopt.vasp"
        target = contcar_preopt(project_dir, "dft")
        if legacy.exists() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            legacy.replace(target)
            log.info("[h00_design] Migrated legacy DFT preopt %s → %s", legacy, target)

    @staticmethod
    def _record_preopt_decision(project_dir: Path, tier: str, decision) -> None:
        """Persist the latest policy result for audit and autonomous debugging."""
        path = preoptimization_policy(project_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            current = json.loads(path.read_text()) if path.exists() else {}
        except (OSError, json.JSONDecodeError):
            current = {}
        current[tier] = decision.as_dict()
        path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")

    @staticmethod
    def _supercell_factors(spec: str) -> list[int]:
        """Parse an NxMxK wizard supercell value."""
        try:
            factors = [int(x) for x in str(spec).lower().split("x")]
        except ValueError:
            factors = []
        if len(factors) != 3 or any(x < 1 for x in factors):
            raise ValueError(f"Invalid AIMD supercell {spec!r}; expected NxMxK")
        return factors

    @staticmethod
    def _target_supercell_factors(n_atoms: int, target: int) -> list[int]:
        """Choose near-isotropic integer replication factors closest to an atom target."""
        if n_atoms < 1 or target <= n_atoms:
            return [1, 1, 1]
        linear = (target / n_atoms) ** (1 / 3)
        limit = max(2, int(linear) + 3)
        candidates = ([a, b, c] for a in range(1, limit + 1)
                      for b in range(a, limit + 1) for c in range(b, limit + 1))
        return min(candidates, key=lambda f: (
            abs(n_atoms * f[0] * f[1] * f[2] - target),
            f[2] - f[0],
            sum(f),
        ))

    def can_run(self, project_dir: Path, state: "ProjectState") -> bool:
        """Always runnable — design is the first pipeline stage."""
        return True

    def is_complete(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True when all structure files and preopt contcars are written."""
        self._migrate_legacy_dft_preopt(project_dir)
        yaml_data   = self.read_project_yaml(project_dir)
        from hpca.core.combinations import production_combinations
        production = production_combinations(yaml_data)

        # Crystal doping sub-project: complete once the policy-selected canonical
        # preoptimization input exists (it may be validated/copy-through).
        # Covers both mono-dopant (doping_n) and bi/multi-dopant (doping_elements).
        # Also completes if downstream stages are already running (old-format sub-projects
        # that predate the designed_structures/ layout).
        if "doping_n" in yaml_data or "doping_elements" in yaml_data:
            preopt_dft = contcar_preopt(project_dir, "dft")
            if preopt_dft.exists():
                return True
            # Bypass: downstream DFT already running (sub-project in old layout)
            dft_vc_poscar = project_dir / "dft" / "vc" / "POSCAR"
            dft_opt_contcar = project_dir / "dft" / "opt" / "CONTCAR"
            if dft_vc_poscar.exists() or dft_opt_contcar.exists():
                return True
            return False

        # Crystal doping: complete once every sub-project dir has project.yaml
        crystal_variants = yaml_data.get("crystal_doping_variants", [])
        if crystal_variants:
            return all(
                (project_dir / v["name"] / "project.yaml").exists()
                for v in crystal_variants
            )

        # Multi-combination: complete once every sub-project directory has a project.yaml.
        # Sub-projects discovered by the orchestrator then run their own pipelines.
        if len(production) > 1:
            return all(
                (project_dir / c["name"] / "project.yaml").exists()
                for c in production
            )

        # Primary: DESIGN_COMPLETE.md written by submit()
        ds_dir     = _designed_structures(project_dir)
        # DFT + MLMD are required: they gate h01_dft and h05_lammps respectively.
        # CMD is handled by h05_cmd.submit() independently — not a gate here.
        core_preopt_done = (
            contcar_preopt(project_dir, "dft").exists()
            and contcar_preopt(project_dir, "mlmd").exists()
        )
        if not (ds_dir / "DESIGN_COMPLETE.md").exists():
            # Fallback: DESIGN_COMPLETE.md not yet written but core preopt is done
            return core_preopt_done
        # DESIGN_COMPLETE.md exists — trust policy-produced canonical inputs.
        if core_preopt_done:
            return True
        # Preopt not yet written: verify raw POSCARs exist as non-placeholder
        for tier in ("dft", "mlmd"):
            p = ds_dir / f"poscar_{tier}.vasp"
            if not p.exists() or not p.stat().st_size:
                log.warning("[h00_design] poscar_%s.vasp missing — holding COMPLETE", tier)
                return False
        return True

    def submit(self, project_dir: Path, state: "ProjectState") -> str | None:
        """Build structures, validate them, and optionally run MACE in-process.

        Crystal sub-projects normally skip MACE when their geometry is reasonable.
        Liquid/polymer/gel projects pack PACKMOL boxes for all three tiers then
        run LAMMPS MACE NPT pre-optimisation sequentially on the daemon node.
        """
        self._migrate_legacy_dft_preopt(project_dir)
        yaml_data = self.read_project_yaml(project_dir)
        category    = yaml_data.get("category", "inorganic_sse")
        system_type = yaml_data.get("system_type", "")

        # Refresh species tables from actual molecule files so box sizing is
        # exact for any species, not just the curated table entries.
        if _cat_is_molecular(category) or "liquid" in system_type or system_type == "gel":
            _refresh_species_tables(_species_in_yaml(yaml_data), project_dir)

        # Crystal doping sub-project: universal design pipeline applies here too.
        # Parent wrote designed_structures/poscar_{dft,mlmd}.vasp; apply policy.
        # Layout: designed_structures/poscar_dft.vasp → dft/preopt/CONTCAR
        #         → dft/aimd_relax → dft/vc_relax → dft/opt → AIMD dataset, NEB ...
        if "doping_n" in yaml_data or "doping_elements" in yaml_data:
            ds_dir     = _designed_structures(project_dir)
            preopt_dir = _preopt(project_dir)
            poscar_dft  = ds_dir / "poscar_dft.vasp"
            poscar_mlmd = ds_dir / "poscar_mlmd.vasp"

            if not poscar_dft.exists():
                log.warning("[h00_design] Crystal sub-project %s: poscar_dft.vasp missing — "
                            "waiting for parent h00_design", project_dir.name)
                return None

            import shutil as _sh
            preopt_dir.mkdir(parents=True, exist_ok=True)

            def _preopt_doped_structures() -> None:
                # Material-agnostic DFT preopt:
                # designed POSCAR → dft/preopt/CONTCAR.
                preopt_dft = contcar_preopt(project_dir, "dft")
                if not preopt_dft.exists():
                    preopt_dft.parent.mkdir(parents=True, exist_ok=True)
                    _sh.copy2(poscar_dft, preopt_dft)
                    self._mlip_prerelax(preopt_dft, category, yaml_data, "dft")
                    log.info("[h00_design] DFT preopt → %s", preopt_dft)

                # Policy-selected preopt: raw MLMD POSCAR → canonical input.
                preopt_mlmd = preopt_dir / "contcar_mlmd_preopt.vasp"
                if not preopt_mlmd.exists() and poscar_mlmd.exists():
                    _sh.copy2(poscar_mlmd, preopt_mlmd)
                    self._mlip_prerelax(preopt_mlmd, category, yaml_data, "mlmd")
                    log.info("[h00_design] MLMD preopt → %s", preopt_mlmd)

            get_daemon_task_scheduler().submit(PreoptimizationTask(
                project_dir, _preopt_doped_structures
            )).result()

            self._write_design_complete(project_dir, yaml_data, cmd_systems=[], test_results=[])
            design_dir = project_dir / "design"
            design_dir.mkdir(exist_ok=True)
            (design_dir / "simulation_approved.flag").touch()
            state.set_stage("h00_design", "COMPLETE", completed=datetime.now().isoformat())
            log.info("[h00_design] COMPLETE for %s", project_dir.name)
            return None

        log.info("[h00_design] Starting for %s (category=%s, system_type=%s)",
                 project_dir.name, category, system_type or "default")

        # ── Standard workflow: designed_structures/ + preopt/ ──────────────────
        # Cross-ref: hpca/core/paths.py for all path functions used here.
        from hpca.core.combinations import production_combinations
        production = production_combinations(yaml_data)
        # Material construction and preoptimization use independent bounded
        # daemon queues.  Waiting here preserves h00's durable stage contract,
        # while other project threads can use the other queue concurrently.
        daemon_tasks = get_daemon_task_scheduler()
        daemon_tasks.submit(MaterialDesignTask(
            project_dir, lambda: self._build_all_designed_structures(project_dir, yaml_data)
        )).result()
        if len(production) <= 1:
            # For multi-combination, preopt runs inside each sub-project's h00_design.
            def _preopt_and_finalize_cmd() -> None:
                self._run_preopt_all(project_dir, yaml_data)
                # Coordinates are only valid after the preoptimization task.
                self._rebuild_system_cmd_data(project_dir, yaml_data)

            daemon_tasks.submit(PreoptimizationTask(
                project_dir, _preopt_and_finalize_cmd
            )).result()
        self._write_design_complete(project_dir, yaml_data, cmd_systems=[], test_results=[])
        state.set_stage("h00_design", "COMPLETE", completed=datetime.now().isoformat())
        log.info("[h00_design] COMPLETE for %s", project_dir.name)
        return None

        # ── Legacy: amorphous designs (kept for reference; not reached) ────────
        if False and yaml_data.get("design_mode", "") == "amorphous":
            self._build_amorphous_system(project_dir, yaml_data)
        elif design_mode == "nanoparticle_substrate":
            self._build_nanoparticle_substrate(project_dir, yaml_data)
        elif design_mode == "lpifd_polymer":
            self._build_lpifd_system(project_dir, yaml_data)
        elif _cat_is_molecular(category) or "liquid" in system_type or system_type == "gel":
            self._build_liquid_electrolyte_system(project_dir, yaml_data)
        elif _cat_is_polymer(category) and system_type not in ("gel",):
            self._build_polymer_system(project_dir, yaml_data)
        else:
            if not _cat_is_crystalline(category):
                log.warning("[h00_design] Category '%s' (system_type='%s') has no dedicated "
                            "design branch — falling through to the crystal path. "
                            "Register the category in hpca/core/categories.py if this "
                            "is not a crystalline material.", category, system_type)
            self._build_crystal_system(project_dir, yaml_data)

        # For CMD gel/liquid projects: pre-build system.data and run a short test
        sim = yaml_data.get("simulation", {})
        cmd_systems: list = []
        test_results: list[tuple[str, bool, str]] = []
        if sim.get("classical_md") and (
            _cat_is_molecular(category) or "liquid" in system_type or system_type == "gel"
        ):
            cmd_systems = self._prebuild_cmd_systems(project_dir, yaml_data)
            test_results = self._run_test_cmd(project_dir, yaml_data)
            if test_results:
                self._write_design_complete(project_dir, yaml_data,
                                            cmd_systems=cmd_systems,
                                            test_results=test_results)
                approved = self._auto_approve_or_kill(project_dir, state, test_results)
                if not approved:
                    # Raise so the orchestrator catches it and records FAILED —
                    # returning None would let is_complete() see DESIGN_COMPLETE.md
                    # and overwrite the FAILED state with COMPLETE.
                    raise RuntimeError("FF test failed — fix errors and run hpca resume")
                state.set_stage("h00_design", "COMPLETE",
                                completed=datetime.now().isoformat())
                log.info("[h00_design] COMPLETE (all FF tests passed) for %s",
                         project_dir.name)
                return None

        self._write_design_complete(project_dir, yaml_data,
                                    cmd_systems=cmd_systems,
                                    test_results=test_results)
        state.set_stage("h00_design", "COMPLETE",
                        completed=datetime.now().isoformat())
        log.info("[h00_design] COMPLETE for %s", project_dir.name)
        log.info("[h00_design] Review designed_structures/DESIGN_COMPLETE.md then:\n"
                 "  touch %s/designed_structures/simulation_approved.flag", project_dir)
        return None

    # ── New workflow: designed_structures + preopt ────────────────────────────

    @staticmethod
    def _scale_mol_counts_to_natoms(mol_counts: dict, natoms_max: int) -> dict:
        """Scale molecule counts proportionally so total atoms ≤ natoms_max."""
        if not mol_counts:
            return {}
        total = sum(n * _NATOMS_PER_MOL.get(name, _DEFAULT_NATOMS_PER_MOL)
                    for name, n in mol_counts.items())
        if total <= natoms_max:
            return dict(mol_counts)
        scale = natoms_max / total
        scaled: dict[str, int] = {}
        for name, n in mol_counts.items():
            scaled[name] = max(1, round(n * scale))
        return scaled

    @staticmethod
    def _dft_autofill(
        sm_species: "list[str]",
        poly_entries: "list[dict]",
        target: int = 250,
        hard_cap: int = 300,
    ) -> "tuple[dict, int]":
        """Auto-calculate DFT box counts to fill ~target atoms with equal species coverage.

        Returns (mc_dft, dft_poly_budget):
          mc_dft          — {sm_name: count} for PACKMOL
          dft_poly_budget — atom budget reserved for polymer monomers (passed to
                            _poly_specs_for_tier so it computes n_units per type)

        Strategy:
          1. List every species (SM molecules + one entry per polymer monomer type).
          2. Divide target equally: per_species_budget = target // n_species.
          3. Initial count: n = max(1, per_species_budget // atoms_per_unit).
          4. Greedy fill: add 1 unit at a time to the species with the smallest
             atom-per-unit cost until target is reached or nothing more fits.
        """
        effective_target = min(target, hard_cap)
        aps: "dict[str, int]" = {}   # atoms per unit for each species
        for s in sm_species:
            aps[s] = _NATOMS_PER_MOL.get(s, _DEFAULT_NATOMS_PER_MOL)
        poly_names: "list[str]" = []
        for p in poly_entries:
            m = p.get("monomer", "")
            if m:
                aps[m] = _NATOMS_PER_MOL.get(m, _DEFAULT_NATOMS_PER_MOL)
                poly_names.append(m)

        all_names = list(sm_species) + poly_names
        if not all_names:
            return {}, 0

        per_budget = effective_target // len(all_names)
        counts: "dict[str, int]" = {n: max(1, per_budget // max(1, aps[n])) for n in all_names}
        total = sum(counts[n] * aps[n] for n in all_names)

        # Greedy fill: add 1 unit at a time to smallest-cost species
        by_cost = sorted(all_names, key=lambda n: aps[n])
        while total < effective_target:
            added = False
            for n in by_cost:
                if total + aps[n] <= effective_target:
                    counts[n] += 1
                    total += aps[n]
                    added = True
                    break
            if not added:
                break

        mc_dft = {n: counts[n] for n in sm_species if n in counts}
        dft_poly_budget = sum(counts[n] * aps[n] for n in poly_names if n in counts)
        return mc_dft, dft_poly_budget

    @staticmethod
    def _auto_mol_counts_from_comp(comp_spec: dict, natoms_target: int) -> dict:
        """Compute {molecule: count} from composition + atom target.

        Uses salt_molarity (mol/L) when specified; falls back to vol_pct.
        """
        c        = float(comp_spec.get("salt_molarity", 0.0))
        solvents = comp_spec.get("solvents", [])
        salts    = comp_spec.get("salts",    [])

        if c > 0 and solvents and salts:
            # Analytic total salt mole fraction. For a mixed salt, use its
            # requested molar ratio for the effective MW/atom count and split
            # the resulting integer formula units by the same ratio.
            salt_raw = [(s["name"], max(0.0, float(s.get("ratio", 1.0)))) for s in salts]
            salt_total = sum(r for _, r in salt_raw) or 1.0
            salt_mf = [(nm, r / salt_total) for nm, r in salt_raw]
            MW_salt = sum(f * _MW.get(nm, 100.0) for nm, f in salt_mf)
            at_salt = sum(f * _NATOMS_PER_MOL.get(nm, _DEFAULT_NATOMS_PER_MOL)
                          for nm, f in salt_mf)

            raw      = [(s["name"], float(s.get("ratio", s.get("vol_pct", 1.0)))) for s in solvents]
            mole_w   = [(nm, r / _MW.get(nm, 100.0)) for nm, r in raw]
            total_mw = sum(w for _, w in mole_w) or 1.0
            solv_mf  = [(nm, w / total_mw) for nm, w in mole_w]

            MW_solv_avg  = sum(f * _MW.get(nm, 100.0) for nm, f in solv_mf)
            at_solv_avg  = sum(f * _NATOMS_PER_MOL.get(nm, _DEFAULT_NATOMS_PER_MOL) for nm, f in solv_mf)
            rho_solv_avg = sum(f * _DENSITY.get(nm, _DEFAULT_DENSITY) for nm, f in solv_mf)

            denom  = rho_solv_avg * 1000.0 + c * (MW_solv_avg - MW_salt)
            x_salt = max(0.01, min(0.95, c * MW_solv_avg / max(denom, 1e-9)))

            avg_atoms = x_salt * at_salt + (1.0 - x_salt) * at_solv_avg
            if avg_atoms > 0:
                N_mol  = natoms_target / avg_atoms
                N_salt = max(1, round(x_salt * N_mol))
                N_solv = max(1, round((1.0 - x_salt) * N_mol))
                counts: dict[str, int] = {}
                remaining = N_salt
                for idx, (nm, fraction) in enumerate(salt_mf):
                    count = remaining if idx == len(salt_mf) - 1 else max(1, round(fraction * N_salt))
                    counts[nm] = count
                    remaining = max(0, remaining - count)
                for nm, f in solv_mf:
                    counts[nm] = max(1, round(f * N_solv))
                return counts

        # vol_pct path
        species: dict[str, float] = {}
        for s in solvents:
            if s.get("vol_pct", 0) > 0:
                species[s["name"]] = s["vol_pct"]
        for s in salts:
            if s.get("vol_pct", 0) > 0:
                species[s["name"]] = s["vol_pct"]
        for s in comp_spec.get("polymers", []):
            if s.get("vol_pct", 0) > 0:
                species[s["name"]] = s["vol_pct"]
        for s in comp_spec.get("copolymers", []):
            if s.get("vol_pct", 0) > 0:
                species[s["name"]] = s["vol_pct"]
        if not species:
            return {}
        total_pct = sum(species.values())
        norm = {name: pct / total_pct for name, pct in species.items()}
        weighted = sum(r * _NATOMS_PER_MOL.get(n, _DEFAULT_NATOMS_PER_MOL)
                       for n, r in norm.items())
        if weighted == 0:
            return {}
        k = natoms_target / weighted
        return {name: max(1, round(r * k)) for name, r in norm.items()}

    def _build_all_designed_structures(self, project_dir: Path, yaml_data: dict) -> None:
        """Dispatch to category-specific designed_structures builder."""
        category    = yaml_data.get("category", "inorganic_sse")
        system_type = yaml_data.get("system_type", "")

        # Electrode|electrolyte interface projects: hpca.core.interface_builder
        # produces designed_structures/poscar_{dft,mlmd,cmd}.vasp directly
        # (electrode slab + PACKMOL electrolyte sandwich) — none of the
        # crystal-system branch below applies (no CIF/structure_files source,
        # no doping variants, no supercell-scaling relationship between
        # tiers). If those three files are already present, this project's
        # design work is done; regenerating them here would run crystal-
        # system logic built for a single-material CIF/POSCAR source against
        # a POSCAR it can't interpret.
        #
        # Keyed on system_type (not category): a project may run under an
        # already-registered category (e.g. "solid") to pass validation in a
        # daemon process that hasn't reloaded a newly-added CategorySpec yet
        # — category additions only take effect for freshly spawned processes
        # like this orchestrator, never the long-lived daemon that validated
        # the request. system_type carries the real intent regardless.
        if system_type == "electrode_liquid":
            ds_dir = _designed_structures(project_dir)
            required = [ds_dir / f"poscar_{tier}.vasp" for tier in ("dft", "mlmd", "cmd")]
            missing = [p for p in required if not p.exists()]
            if not missing:
                log.info("[h00_design] %s: pre-built interface structures found — "
                         "skipping design build", project_dir.name)
                return
            log.warning("[h00_design] %s: system_type=electrode_liquid but "
                        "missing %s — this system_type has no automatic structure "
                        "builder wired in yet; build them with "
                        "hpca.core.interface_builder and place them at "
                        "designed_structures/poscar_{dft,mlmd,cmd}.vasp",
                        project_dir.name, [p.name for p in missing])
            return

        from hpca.core.combinations import production_combinations
        production = production_combinations(yaml_data)

        # Multi-combination: spawn one sub-project per production composition.
        # Each sub-project gets its own project.yaml and designed_structures/.
        # The orchestrator discovers sub-project dirs on the next poll and runs
        # each combination's DFT→AIMD→MLIP→MLMD→CMD pipeline in parallel.
        if len(production) > 1:
            self._build_combinatorial_subprojects(project_dir, yaml_data)
            return

        if _cat_is_molecular(category) or "liquid" in system_type or system_type == "gel":
            self._build_designed_structures_liquid(project_dir, yaml_data)
        elif _cat_is_polymer(category):
            self._build_designed_structures_polymer(project_dir, yaml_data)
        else:
            self._build_designed_structures_crystal(project_dir, yaml_data)

    def _build_designed_structures_liquid(self, project_dir: Path, yaml_data: dict) -> None:
        """Build poscar_dft/mlmd/cmd.vasp + system_cmd.data in designed_structures/."""
        import shutil as _sh
        sim = yaml_data.get("simulation", {})
        category   = yaml_data.get("category", "liquid_electrolyte")
        ds_dir = _designed_structures(project_dir)
        ds_dir.mkdir(parents=True, exist_ok=True)

        # ── Composition — build a combo dict with "solvents" + "salt" keys ─────
        # _pack_liquid_cell expects combo["solvents"] = [{name, ratio}, ...]
        # and combo["salt"] = "<salt_name>".
        # Priority: simulation.solvents/salt (always present) over cmd_combinations.
        solvents = sim.get("solvents", [])
        salt     = sim.get("salt", "")
        if not solvents:
            # Fall back to cmd_combinations components structure
            for raw_combo in yaml_data.get("cmd_combinations", []):
                for c in raw_combo.get("components", {}).get("solvent", {}).get("components", []):
                    solvents.append({"name": c["name"], "ratio": c.get("ratio", 1)})
                if not salt:
                    for c in raw_combo.get("components", {}).get("salt", {}).get("components", []):
                        salt = c.get("name", "")
                break
        combo = {"name": yaml_data.get("name", "system"),
                 "solvents": solvents, "salt": salt}

        rho  = (sim.get("tier_cmd", {}).get("density_gcm3")
                or sim.get("target_density_gcm3", 1.0))

        # ── Molecule counts per tier ─────────────────────────────────────────
        mc_cmd  = sim.get("molecule_counts_cmd",  {})
        mc_mlmd = sim.get("molecule_counts_mlmd", {})
        mc_dft  = sim.get("molecule_counts_aimd", {})   # DFT/AIMD scale

        dft_max  = self.sim_limit("slurm", "dft_atoms")  or 200
        mlmd_max = max(sim.get("tier_mlmd", {}).get("natoms") or 0,
                       self.sim_limit("slurm", "mlmd_atoms") or dft_max)
        cmd_max  = max(sim.get("tier_cmd",  {}).get("natoms") or 0,
                       self.sim_limit("slurm", "cmd_atoms")  or dft_max)

        # Auto-compute mc_cmd from vol_pct composition when not explicitly set
        if not mc_cmd:
            mc_cmd = self._auto_mol_counts_from_comp(sim.get("comp_spec", {}), cmd_max)
            if mc_cmd:
                log.info("[h00_design] Auto-computed molecule counts (cmd): %s", mc_cmd)

        box_cmd  = sim.get("tier_cmd",  {}).get("box_A")
        box_mlmd = sim.get("tier_mlmd", {}).get("box_A")

        # ── Polymer gel: atom-budget-aware tier scaling ───────────────────────
        # For gel systems, each tier's atom budget must be split between polymers
        # and small molecules according to the polymer volume fraction f_poly.
        # n_chains and n_units for each polymer type are derived from the resulting
        # per-tier polymer budget — not from hardcoded heuristics.
        system_type  = yaml_data.get("system_type", "")
        poly_entries = yaml_data.get("polymers", [])
        poly_names: set[str] = {p.get("monomer", "") for p in poly_entries if p.get("monomer")}

        # Compute polymer volume fraction f_poly
        f_poly = 0.0
        if poly_entries:
            comp_spec = sim.get("comp_spec", {})
            f_poly = (comp_spec.get("polymer", 0) + comp_spec.get("copolymer", 0)) / 100.0
            if f_poly <= 0 and mc_cmd:
                # Fallback: infer f_poly from CMD-scale polymer vs total atom count
                sm_atoms_cmd = sum(
                    n * _NATOMS_PER_MOL.get(k, _DEFAULT_NATOMS_PER_MOL)
                    for k, n in mc_cmd.items() if k not in poly_names
                )
                poly_atoms_cmd = sum(
                    p.get("n_chains", 1) * p.get("chain_length", 10)
                    * _NATOMS_PER_MOL.get(p.get("monomer", ""), _DEFAULT_NATOMS_PER_MOL)
                    for p in poly_entries if p.get("monomer")
                )
                total_cmd = sm_atoms_cmd + poly_atoms_cmd
                f_poly = poly_atoms_cmd / total_cmd if total_cmd > 0 else 0.3
            if f_poly > 0:
                log.info("[h00_design] Polymer gel: f_poly=%.2f → splitting tier budgets", f_poly)

        # Per-tier atom budgets: polymer portion and small-molecule portion
        dft_poly_max  = round(dft_max  * f_poly)
        dft_sm_max    = dft_max  - dft_poly_max
        mlmd_poly_max = round(mlmd_max * f_poly)
        mlmd_sm_max   = mlmd_max - mlmd_poly_max

        # Remove polymer names from molecule counts (polymers are packed via
        # polymer_specs, not as discrete molecule files)
        if poly_names:
            mc_dft  = {k: v for k, v in mc_dft.items()  if k not in poly_names}
            mc_mlmd = {k: v for k, v in mc_mlmd.items() if k not in poly_names}
            mc_cmd  = {k: v for k, v in mc_cmd.items()  if k not in poly_names}

        # DFT: auto-fill to ~250 atoms with equal coverage of all species.
        # Concentration ratios are irrelevant — DFT cells generate MLIP training data.
        sm_species = list(mc_cmd.keys()) if mc_cmd else list((mc_dft or {}).keys())
        mc_dft, dft_poly_budget = self._dft_autofill(
            sm_species, poly_entries, target=250, hard_cap=dft_max
        )
        log.info("[h00_design] DFT autofill: %s  polymer_budget=%d atoms",
                 mc_dft, dft_poly_budget)

        # MLMD: scale from CMD counts, leaving room for polymer chains.
        sm_mlmd_max = mlmd_sm_max if poly_entries else mlmd_max
        if not mc_mlmd and mc_cmd:
            mc_mlmd = self._scale_mol_counts_to_natoms(mc_cmd, sm_mlmd_max)

        def _poly_specs_for_tier(tier_key: str, poly_budget: int) -> list[dict] | None:
            """Compute polymer chain specs that fit within poly_budget atoms.

            n_chains and n_units are derived from the atom budget, not hardcoded.
            Budget is distributed across polymer types proportionally to their
            CMD-scale atom contribution (n_chains × chain_length × atoms_per_mono).
            Explicit chain_counts_{tier} in project.yaml override the budget-derived
            n_chains but n_units is still computed from the remaining budget.
            """
            if not poly_entries:
                return None
            cc = sim.get(f"chain_counts_{tier_key}", {})

            # CMD-scale weight per polymer type for proportional budget distribution
            cmd_weights: dict[str, float] = {}
            for p in poly_entries:
                m = p.get("monomer", "")
                if not m:
                    continue
                cmd_weights[m] = (
                    p.get("n_chains", 1) * p.get("chain_length", 10)
                    * _NATOMS_PER_MOL.get(m, _DEFAULT_NATOMS_PER_MOL)
                )
            total_weight = sum(cmd_weights.values()) or 1

            specs = []
            for p in poly_entries:
                monomer   = p.get("monomer", "")
                chain_len = p.get("chain_length", 10)
                cratio    = p.get("copolymer_ratio")
                if not monomer:
                    continue
                atoms_per_mono = _NATOMS_PER_MOL.get(monomer, _DEFAULT_NATOMS_PER_MOL)

                if tier_key == "cmd":
                    # CMD: use full chain specs from project.yaml
                    n_chains = max(1, cc.get(monomer) or p.get("n_chains", 1))
                    n_units  = max(4, chain_len)
                elif tier_key == "aimd":
                    # DFT: 1 chain, n_units derived from dft_poly_budget allocation
                    n_chains = 1
                    type_budget = round(poly_budget * cmd_weights.get(monomer, 1) / total_weight)
                    n_units = max(1, type_budget // max(1, atoms_per_mono))
                    n_units = min(n_units, chain_len)
                else:
                    # MLMD: budget-derived chain length, minimum 3 chains × 2-mer
                    min_chains = 3
                    n_chains   = max(min_chains, cc.get(monomer) or min_chains)
                    type_budget = round(poly_budget * cmd_weights.get(monomer, 1) / total_weight)
                    n_units = max(2, type_budget // max(1, n_chains * atoms_per_mono))
                    n_units = min(n_units, chain_len)   # don't exceed full chain length

                smiles = self._polymer_oligomer_smiles(monomer, n_units, cratio)
                if smiles is None:
                    log.warning("[h00_design] No SMILES for %s (tier=%s) — skipping", monomer, tier_key)
                    continue
                log.info("[h00_design]   polymer %s [%s]: %d chain(s) × %d-mer (~%d atoms)",
                         monomer, tier_key, n_chains, n_units, n_chains * n_units * atoms_per_mono)
                specs.append({"name": monomer, "n_chains": n_chains,
                               "n_units": n_units, "smiles": smiles})
            return specs if specs else None

        # For DFT/MLMD: recalculate polymer budget as (tier_total_max - actual_SM_atoms)
        # so that SM + polymer never exceeds the tier atom limit.
        # tier_aimd.natoms / tier_mlmd.natoms may have been computed as SM-only by the
        # wizard; adding full polymer chains on top would blow past the DFT limit.
        def _remaining_poly_budget(mc: dict, total_max: int) -> int:
            """Return atom budget remaining for polymer after accounting for small-molecule atoms."""
            sm_atoms = sum(n * _NATOMS_PER_MOL.get(k, _DEFAULT_NATOMS_PER_MOL)
                           for k, n in mc.items())
            return max(0, total_max - sm_atoms)

        if poly_entries:
            mlmd_poly_max = _remaining_poly_budget(mc_mlmd, mlmd_max)
        else:
            mlmd_poly_max = 0

        tier_cfgs = [
            ("dft",  mc_dft,  None,     "aimd", dft_poly_budget),  # auto-filled to ~250 atoms
            ("mlmd", mc_mlmd, box_mlmd, "mlmd", mlmd_poly_max),
            ("cmd",  mc_cmd,  box_cmd,  "cmd",  0),
        ]

        # ── Pack POSCARs (DFT / MLMD / CMD in parallel) ─────────────────────
        preopt_dir = _preopt(project_dir)
        preopt_dir.mkdir(parents=True, exist_ok=True)
        # Pre-compute poly_specs in the main thread (uses closures from above).
        # Each tuple: (tier, mc, box_A, poly_specs)
        tier_cfgs_ext = [
            (tier, mc, box_A, _poly_specs_for_tier(cc_key, poly_budget))
            for tier, mc, box_A, cc_key, poly_budget in tier_cfgs
        ]
        from concurrent.futures import ThreadPoolExecutor as _TPool, as_completed as _ac
        with _TPool(max_workers=3) as _pool:
            _futs = {
                _pool.submit(
                    self._pack_one_tier,
                    tier, mc, box_A, poly_specs,
                    ds_dir=ds_dir, preopt_dir=preopt_dir,
                    combo=combo, rho=rho,
                    category=category, system_type=system_type,
                ): tier
                for tier, mc, box_A, poly_specs in tier_cfgs_ext
            }
            for _fut in _ac(_futs):
                try:
                    _fut.result()
                except Exception as _exc:
                    log.error("[h00_design] Packing %s tier raised: %s", _futs[_fut], _exc)

        # ── Build preopted_system_cmd.data (OPLS-AA topology for CMD LAMMPS) ──
        system_data = _preopt(project_dir) / "preopted_system_cmd.data"
        if not system_data.exists() and mc_cmd:
            log.info("[h00_design] Building preopted_system_cmd.data for CMD LAMMPS")
            cc_cmd   = sim.get("chain_counts_cmd", {})
            mol_data = self._build_mol_data(project_dir, mc_cmd, cc_cmd, yaml_data)
            if mol_data:
                try:
                    from hpca.sim.forcefield import build_mixed_system
                    L = build_mixed_system(mol_data, system_data, box_size=box_cmd)
                    log.info("[h00_design] preopted_system_cmd.data written (box=%.1f Å)", L)
                except Exception as exc:
                    log.warning("[h00_design] preopted_system_cmd.data build failed: %s", exc)

    def _rebuild_system_cmd_data(self, project_dir: Path, yaml_data: dict) -> None:
        """Rebuild system_cmd.data from contcar_cmd_preopt.vasp + OPLS-AA topology.

        Called after _run_preopt_all() so contcar_cmd_preopt.vasp exists.
        Coordinates come from the MACE-relaxed geometry; bonds/angles/dihedrals/
        charges come from the OPLS-AA templates for each molecule type.

        Only runs for liquid_electrolyte / gel categories without grid-placed
        polymer chains (i.e. systems where PACKMOL was used for CMD packing).
        Skipped silently when contcar_cmd_preopt.vasp is absent or is a copy of
        the grid-placed original (first line contains "grid-placed").
        """
        category    = yaml_data.get("category", "")
        system_type = yaml_data.get("system_type", "")
        is_liquid   = _cat_is_molecular(category) or "liquid" in system_type or system_type == "gel"
        if not is_liquid:
            return

        preopt_dir  = _preopt(project_dir)
        ds_dir      = _designed_structures(project_dir)
        preopt_cmd  = preopt_dir / "contcar_cmd_preopt.vasp"
        system_data = preopt_dir / "preopted_system_cmd.data"

        if not preopt_cmd.exists():
            log.debug("[h00_design] contcar_cmd_preopt.vasp absent — skipping preopted_system_cmd.data rebuild")
            return

        # Skip grid-placed structures — they weren't packed with PACKMOL so the
        # element-order → molecule mapping assumption does not hold.
        first_line = preopt_cmd.read_text().splitlines()[0].lower() if preopt_cmd.stat().st_size else ""
        if "grid-placed" in first_line:
            log.info("[h00_design] contcar_cmd_preopt.vasp is grid-placed — skipping rebuild")
            return

        sim     = yaml_data.get("simulation", {})
        mc_cmd  = sim.get("molecule_counts_cmd", {})
        cc_cmd  = sim.get("chain_counts_cmd", {})
        if not mc_cmd:
            return

        # Remove polymer names — polymers are grid-placed and handled separately.
        poly_entries = yaml_data.get("polymers", [])
        poly_names   = {p.get("monomer", "") for p in poly_entries if p.get("monomer")}
        mc_cmd_sm    = {k: v for k, v in mc_cmd.items() if k not in poly_names}
        if not mc_cmd_sm:
            return

        # Build mol_data in PACKMOL structure-block order: solvents first (in
        # combo["solvents"] order), then salt/extras.  This must match the order
        # used in _pack_liquid_cell so element-block cursors align correctly.
        solvents    = sim.get("solvents", [])
        salt        = sim.get("salt", "")
        solv_names  = [s["name"] for s in solvents]
        extra_names = [k for k in mc_cmd_sm if k not in solv_names]
        ordered_names = solv_names + extra_names

        from hpca.sim.forcefield import MolData
        mol_data_ordered: list = []
        for mol_name in ordered_names:
            count = mc_cmd_sm.get(mol_name)
            if not count:
                continue
            vasp_path = _find_mol_vasp(mol_name, project_dir / "input_structures")
            if vasp_path and vasp_path.exists():
                try:
                    md = MolData.from_file(vasp_path, name=mol_name, count=count)
                    mol_data_ordered.append(md)
                    continue
                except Exception as exc:
                    log.warning("[h00_design] MolData(%s) failed: %s", mol_name, exc)
            # Fallback: builtin geometry
            from hpca.sim.forcefield import MOLECULES
            if mol_name.upper() in MOLECULES:
                try:
                    md = MolData.from_builtin(mol_name.upper(), count=count)
                    mol_data_ordered.append(md)
                    continue
                except Exception as exc:
                    log.warning("[h00_design] MolData builtin(%s) failed: %s", mol_name, exc)
            log.error("[h00_design] _rebuild_system_cmd_data: cannot find %s — aborting rebuild", mol_name)
            return

        if not mol_data_ordered:
            return

        try:
            from hpca.sim.forcefield import build_system_data_from_poscar
            L = build_system_data_from_poscar(preopt_cmd, mol_data_ordered, system_data)
            log.info("[h00_design] preopted_system_cmd.data rebuilt from validated coordinates (box=%.1f Å)", L)
        except Exception as exc:
            log.warning("[h00_design] preopted_system_cmd.data rebuild failed: %s — keeping template-grid version", exc)

    def _build_designed_structures_polymer(self, project_dir: Path, yaml_data: dict) -> None:
        """Build 3 POSCARs for polymer systems (DFT/MLMD/CMD scale)."""
        sim    = yaml_data.get("simulation", {})
        ds_dir = _designed_structures(project_dir)
        ds_dir.mkdir(parents=True, exist_ok=True)

        # Build the full polymer system first (existing logic)
        self._build_polymer_system(project_dir, yaml_data)

        # For polymer: the POSCAR variants differ only in chain counts
        # DFT: minimal (1–2 chains, short), MLMD: medium, CMD: full
        # If system.lmp exists, derive POSCARs from it via pymatgen conversion
        system_lmp = project_dir / "design" / "system.lmp"
        if system_lmp.exists():
            for tier in ("dft", "mlmd", "cmd"):
                dest = ds_dir / f"poscar_{tier}.vasp"
                if not dest.exists():
                    import shutil as _sh
                    _sh.copy2(system_lmp, dest)
                    log.info("[h00_design] polymer poscar_%s.vasp → from system.lmp", tier)

    def _build_designed_structures_crystal(self, project_dir: Path, yaml_data: dict) -> None:
        """Build 3 POSCARs for crystal systems (supercell at DFT/MLMD/CMD scale)."""
        # Doping variants: create one sub-project per entry instead of single crystal
        if yaml_data.get("crystal_doping_variants"):
            self._build_crystal_doping_subprojects(project_dir, yaml_data)
            return

        ds_dir = _designed_structures(project_dir)
        ds_dir.mkdir(parents=True, exist_ok=True)

        # Build crystal using existing logic
        self._build_crystal_system(project_dir, yaml_data)

        # Apply the requested AIMD supercell physically; the wizard value is not
        # metadata-only. MLMD/CMD are independently expanded toward their target.
        poscar_src = (dft_opt(project_dir) / "POSCAR")
        if not poscar_src.exists():
            poscar_src = project_dir / "design" / "vars" / "d100" / "POSCAR"
        if poscar_src.exists():
            from pymatgen.core import Structure
            base = Structure.from_file(str(poscar_src))
            aimd_struct = base.copy()
            aimd_struct.make_supercell(self._supercell_factors(
                yaml_data.get("simulation", {}).get("aimd_supercell", "1x1x1")
            ))
            target = int(yaml_data.get("simulation", {}).get("mlmd_natoms_target", 5000))
            mlmd_factors = self._target_supercell_factors(len(base), target)
            mlmd_struct = base.copy()
            mlmd_struct.make_supercell(mlmd_factors)
            for tier, structure in (("dft", aimd_struct), ("mlmd", mlmd_struct),
                                    ("cmd", mlmd_struct)):
                dest = ds_dir / f"poscar_{tier}.vasp"
                if not dest.exists():
                    dest.write_text(structure.to(fmt="poscar"))
                    log.info("[h00_design] crystal poscar_%s.vasp: %d atoms", tier, len(structure))

    # ── Crystal doping sub-project creation ──────────────────────────────────────

    @staticmethod
    def _apply_substitution_doping(base_struct, host_el: str, dopant_el: str,
                                    n_sub: int, seed: int = 42,
                                    substitution_indices: list[int] | None = None):
        """Return new Structure with n_sub host_el sites replaced by dopant_el."""
        import random
        rng = random.Random(seed)
        sc = base_struct.copy()
        host_indices = [i for i, site in enumerate(sc) if site.species_string == host_el]
        if n_sub > len(host_indices):
            raise ValueError(
                f"n_sub={n_sub} exceeds available {host_el} sites ({len(host_indices)})"
            )
        chosen = substitution_indices or rng.sample(host_indices, n_sub)
        if len(chosen) != n_sub or any(idx not in host_indices for idx in chosen):
            raise ValueError("Explicit substitution indices do not match host sites")
        for idx in chosen:
            sc[idx] = dopant_el
        return sc

    @staticmethod
    def _apply_codoping(base_struct, host_el: str,
                        dopant_entries: list[dict], seed: int = 42):
        """Return new Structure with sequential co-doping substitutions.

        dopant_entries: [{element: str, n_substitutions: int}, ...]
        Each dopant consumes from the remaining pool of host_el sites.
        """
        import random
        rng = random.Random(seed)
        sc = base_struct.copy()
        remaining = [i for i, site in enumerate(sc) if site.species_string == host_el]
        for entry in dopant_entries:
            el    = entry["element"]
            n_sub = int(entry["n_substitutions"])
            if n_sub > len(remaining):
                raise ValueError(
                    f"Not enough {host_el} sites for {el}: need {n_sub}, "
                    f"only {len(remaining)} left"
                )
            chosen = rng.sample(remaining, n_sub)
            for idx in chosen:
                sc[idx] = el
            remaining = [i for i in remaining if i not in chosen]
        return sc

    def _build_crystal_doping_subprojects(self, project_dir: Path, yaml_data: dict) -> None:
        """Create one sub-project dir per entry in crystal_doping_variants.

        Each sub-project gets: dft/opt/POSCAR (doped), project.yaml, neb/POSCAR_vac.
        The parent project's h00_design is marked COMPLETE once all sub-dirs are written;
        the orchestrator discovers each sub-project on the next poll and runs the full
        inorganic_sse pipeline (h01_dft → h03_neb → h07 → h08 → ...) independently.
        """
        import yaml as _yaml
        variants = yaml_data.get("crystal_doping_variants", [])
        if not variants:
            log.warning("[h00_design] crystal_doping_variants is empty — nothing to build")
            return

        _ensure_cladue_env()

        # Load base structure from mp_id, cif, or vasp key
        base_struct = None
        if "mp_id" in yaml_data:
            base_struct = self._fetch_from_mp(yaml_data["mp_id"])
        elif "cif" in yaml_data:
            base_struct = self._load_from_cif(yaml_data["cif"])

        if base_struct is None:
            log.error("[h00_design] crystal_doping_variants: no structure source "
                      "(add mp_id or cif key to project.yaml)")
            return

        sc_spec = yaml_data.get("simulation", {}).get(
            "aimd_supercell", yaml_data.get("supercell", "1x1x1"))
        factors = self._supercell_factors(sc_spec) if isinstance(sc_spec, str) else sc_spec
        base_struct.make_supercell(factors)
        log.info("[h00_design] Applied AIMD supercell %s → %d atoms", factors, len(base_struct))

        mobile_ion = yaml_data.get("mobile_ion", "Li")
        category   = yaml_data.get("category", "inorganic_sse")

        for v in variants:
            name      = v["name"]
            host_el   = v.get("host_element")
            dopant_el = v.get("dopant_element")
            n_sub     = int(v.get("n_substitutions", 0))
            sub_dir   = project_dir / name
            sub_dir.mkdir(parents=True, exist_ok=True)

            # Apply doping substitution (mono) or co-doping (di/trinary)
            dopant_entries = v.get("dopant_elements")  # di/trinary co-doping
            stable_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
            try:
                if host_el and dopant_entries:
                    struct = self._apply_codoping(
                        base_struct, host_el, dopant_entries,
                        seed=stable_seed,
                    )
                elif host_el and dopant_el and n_sub > 0:
                    struct = self._apply_substitution_doping(
                        base_struct, host_el, dopant_el, n_sub,
                        seed=stable_seed,
                        substitution_indices=v.get("substitution_indices"),
                    )
                else:
                    struct = base_struct.copy()
            except Exception as exc:
                log.error("[h00_design] Doping '%s' failed: %s", name, exc)
                continue

            log.info("[h00_design] Sub-project '%s': %s (%d atoms)",
                     name, struct.formula, len(struct))

            # ── Universal design layout: designed_structures/ → dft/preopt/ → dft/ ──
            ds_dir = _designed_structures(sub_dir)
            ds_dir.mkdir(parents=True, exist_ok=True)

            # DFT-scale POSCAR (raw doped structure; policy runs in child h00_design)
            poscar_dft = ds_dir / "poscar_dft.vasp"
            if not poscar_dft.exists():
                poscar_dft.write_text(struct.to(fmt="poscar"))
                log.info("[h00_design] Wrote %s", poscar_dft)

            # MLMD-scale supercell (scale up to mlmd_natoms_target)
            poscar_mlmd = ds_dir / "poscar_mlmd.vasp"
            if not poscar_mlmd.exists():
                mlmd_target = yaml_data.get("simulation", {}).get("mlmd_natoms_target", 5000)
                n_atoms = len(struct)
                factors = self._target_supercell_factors(n_atoms, mlmd_target)
                mlmd_struct = struct.copy()
                mlmd_struct.make_supercell(factors)
                poscar_mlmd.write_text(mlmd_struct.to(fmt="poscar"))
                log.info("[h00_design] Wrote %s (%dx%dx%d supercell, %d atoms)",
                         poscar_mlmd, *factors, len(mlmd_struct))

            # NEB vacancy POSCAR (Li removed at one site)
            self._make_neb_vacancy_poscar(sub_dir, struct, mobile_ion)

            # Write sub-project project.yaml (strip doping-parent keys)
            sub_yaml_path = sub_dir / "project.yaml"
            if not sub_yaml_path.exists():
                sub_yaml = {k: v for k, v in yaml_data.items()
                            if k != "crystal_doping_variants"}
                sub_yaml["name"]            = name
                sub_yaml["doping_host"]     = host_el or ""
                if dopant_entries:
                    sub_yaml["doping_elements"] = dopant_entries
                else:
                    sub_yaml["doping_element"]  = dopant_el or ""
                    sub_yaml["doping_n"]        = n_sub
                sub_yaml_path.write_text(
                    _yaml.dump(sub_yaml, default_flow_style=False,
                               sort_keys=False, allow_unicode=True)
                )
                log.info("[h00_design] Sub-project yaml: %s", sub_yaml_path)

        log.info("[h00_design] Crystal doping: %d sub-projects created under %s",
                 len(variants), project_dir)

    # ── Combinatorial sub-project creation ───────────────────────────────────────

    def _build_combinatorial_subprojects(self, project_dir: Path, yaml_data: dict) -> None:
        """Create one sub-project directory per molarity-resolved production unit.

        Only writes project.yaml + symlinks — returns immediately.  Packing,
        preopt, and MD submission are handled by the orchestrator polling each
        sub-project independently, so faster sub-projects start DFT/MD without
        waiting for the slowest packing to complete.
        """
        import yaml as _yaml

        from hpca.core.combinations import aimd_dataset_combination, production_combinations
        production = production_combinations(yaml_data)

        if not production:
            log.warning("[h00_design] multi-combination project has no production combinations")
            return

        log.info("[h00_design] Multi-combination: creating %d molarity-resolved sub-project yamls",
                 len(production))

        for combo_meta in production:
            combo_name = combo_meta["name"]
            combo_dir  = project_dir / combo_name
            combo_dir.mkdir(parents=True, exist_ok=True)

            combo_yaml_path = combo_dir / "project.yaml"
            if not combo_yaml_path.exists():
                dataset_meta = aimd_dataset_combination(yaml_data, combo_meta)
                sub_yaml = self._build_sub_project_yaml(
                    project_dir, yaml_data, combo_meta, combo_meta,
                    combo_variants=[combo_meta], aimd_combo_meta=dataset_meta,
                )
                combo_yaml_path.write_text(
                    _yaml.dump(sub_yaml, default_flow_style=False,
                               sort_keys=False, allow_unicode=True)
                )
                log.info("[h00_design] Sub-project yaml written: %s", combo_name)

            # Symlink parent's input_structures/ so molecule .vasp files are found
            # without re-fetching from PubChem for every combination.
            parent_input = project_dir / "input_structures"
            combo_input  = combo_dir / "input_structures"
            if parent_input.is_dir() and not combo_input.exists():
                combo_input.symlink_to(parent_input.resolve())

        # Archive any old combined POSCARs in the parent's designed_structures/ so
        # downstream handlers don't pick them up on the parent project.
        ds_dir = _designed_structures(project_dir)
        for tier in ("dft", "mlmd", "cmd"):
            old = ds_dir / f"poscar_{tier}.vasp"
            archived = ds_dir / f"poscar_{tier}.vasp.combined"
            if old.exists() and not archived.exists():
                old.rename(archived)
                log.info("[h00_design] Archived parent poscar_%s.vasp → .combined", tier)

        log.info("[h00_design] Combinatorial: %d sub-project dirs ready for independent processing",
                 len(production))

    def _build_sub_project_yaml(
        self,
        project_dir: Path,
        parent_yaml: dict,
        combo_meta:  dict,
        combo_detail: dict,
        combo_variants: list[dict] | None = None,
        aimd_combo_meta: dict | None = None,
    ) -> dict:
        """Generate a single-combination project.yaml for a combinatorial sub-project.

        Species counts and box sizes are recomputed from the combination's composition
        using the same atom-count targets as the parent project's tiers.
        """
        sim        = parent_yaml.get("simulation", {})
        combo_name = combo_meta["name"]
        cat_pcts   = sim.get("cat_pcts", {})

        # ── Per-combination species list from cmd_combinations components ────────
        combo_species_by_cat: dict[str, list[dict]] = {}
        for cat, cat_data in combo_detail.get("components", {}).items():
            combo_species_by_cat[cat] = list(cat_data.get("components", []))

        # ── Build per-combination comp_spec ─────────────────────────────────────
        parent_comp     = sim.get("comp_spec", {})
        parent_solvents = {s["name"]: s.get("vol_pct", 0) for s in parent_comp.get("solvents", [])}
        parent_salts    = {s["name"]: s.get("vol_pct", 0) for s in parent_comp.get("salts",    [])}

        combo_solvent_parts = combo_species_by_cat.get("solvent", [])
        combo_salt_parts    = combo_species_by_cat.get("salt",    [])
        combo_solvent_names = [part["name"] for part in combo_solvent_parts]
        combo_salt_names    = [part["name"] for part in combo_salt_parts]

        # combo_meta may carry its own salt_molarity (multi-molarity combinatorial series)
        combo_salt_molarity = combo_meta.get("salt_molarity",
                                             parent_comp.get("salt_molarity", 0.0))

        if combo_salt_molarity > 0:
            # Molarity mode: solvents carry ratio, no vol_pct — skip redistribution
            combo_solvents = [{"name": p["name"], "ratio": p.get("ratio", 1.0)}
                              for p in combo_solvent_parts]
            combo_salts    = [{"name": p["name"], "ratio": p.get("ratio", 1.0)}
                              for p in combo_salt_parts]
        else:
            # Redistribute vol_pct: each combination gets the full solvent/salt fraction
            # from the parent (cat_pcts), split only among the species in this combo.
            total_solvent_pct = cat_pcts.get("solvent",
                                sum(parent_solvents.values()) or 70.0)
            total_salt_pct    = cat_pcts.get("salt",
                                sum(parent_salts.values())    or 30.0)

            solv_raw = {s: parent_solvents.get(s, 1.0) for s in combo_solvent_names}
            solv_sum = sum(solv_raw.values()) or 1.0
            combo_solvents = [
                {"name": s, "vol_pct": v / solv_sum * total_solvent_pct}
                for s, v in solv_raw.items()
            ]
            salt_raw = {s: parent_salts.get(s, 1.0) for s in combo_salt_names}
            salt_sum = sum(salt_raw.values()) or 1.0
            combo_salts = [
                {"name": s, "vol_pct": v / salt_sum * total_salt_pct}
                for s, v in salt_raw.items()
            ]

        # Polymers/copolymers are shared across all combinations and are packed
        # via polymer_specs (oligomer generation from SMILES) — NOT as discrete
        # molecules from molecule_counts.  Exclude them from the comp_spec used
        # for auto-mol-count computation to avoid looking for PEO.vasp etc.
        combo_comp_spec: dict = {
            "solvents":   combo_solvents,
            "salts":      combo_salts,
            "polymers":   [],
            "copolymers": [],
            **({"salt_molarity": combo_salt_molarity} if combo_salt_molarity > 0 else {}),
        }

        # ── Compute molecule counts per tier (small molecules only) ───────────────
        dft_max  = self.sim_limit("slurm", "dft_atoms")  or 200
        mlmd_max = self.sim_limit("slurm", "mlmd_atoms") or dft_max
        cmd_max  = self.sim_limit("slurm", "cmd_atoms")  or dft_max

        # AIMD for molecular/polymer/liquid projects generates a compact
        # reference dataset; it deliberately does not reproduce bulk molarity.
        # Give all selected chemical species representative pseudo-fractions.
        aimd_comp_spec = {
            "solvents": [{"name": p["name"], "vol_pct": p.get("ratio", 1.0)}
                         for p in combo_solvent_parts],
            "salts": [{"name": p["name"], "vol_pct": p.get("ratio", 1.0)}
                      for p in combo_salt_parts],
            "polymers": [],
            "copolymers": [],
        }
        mc_dft  = self._auto_mol_counts_from_comp(aimd_comp_spec, dft_max)
        mc_mlmd = self._auto_mol_counts_from_comp(combo_comp_spec, mlmd_max)
        mc_cmd  = self._auto_mol_counts_from_comp(combo_comp_spec, cmd_max)

        # ── Box geometry from molecule counts + density ──────────────────────────
        rho = (sim.get("tier_cmd", {}).get("density_gcm3")
               or sim.get("target_density_gcm3", 1.0))

        # Estimate polymer atom contribution to box volume for gel sub-project yamls.
        # _auto_mol_counts_from_comp uses SM-only comp_spec, so mc_* excludes polymers.
        # Without polymer volume, box_A is undersized → PACKMOL timeout on MLMD/DFT.
        _poly_entries = parent_yaml.get("polymers", [])
        _poly_names_set: set[str] = {p.get("monomer", "") for p in _poly_entries if p.get("monomer")}
        # Estimate f_poly from parent comp_spec polymer + copolymer vol_pct fractions
        _parent_comp = sim.get("comp_spec", {})
        _f_poly = sum(float(p.get("vol_pct", 0.0))
                      for p in _parent_comp.get("polymers", [])) / 100.0
        _f_poly += sum(float(p.get("vol_pct", 0.0))
                       for p in _parent_comp.get("copolymers", [])) / 100.0
        if _f_poly <= 0 and _poly_entries:
            _f_poly = 0.3  # conservative default when no vol_pct breakdown
        _dft_poly_atoms  = round(dft_max  * _f_poly) if _poly_entries else 0
        _mlmd_poly_atoms = round(mlmd_max * _f_poly) if _poly_entries else 0

        def _tier_info(mc: dict, poly_atoms: int = 0) -> dict:
            """Compute atom count, molecular weight, and estimated cubic box size for a molecule mix."""
            n_atoms = sum(n * _NATOMS_PER_MOL.get(nm, _DEFAULT_NATOMS_PER_MOL)
                          for nm, n in mc.items()) + poly_atoms
            MW      = sum(n * _MW.get(nm, 100.0) for nm, n in mc.items())
            MW     += poly_atoms * 10.0  # ~10 g/mol per organic atom (C/H/O/F average)
            V_A3    = MW / rho / 6.022e23 * 1e24 if MW > 0 else (n_atoms * 20.0)
            box_A   = V_A3 ** (1.0 / 3.0)
            return {"natoms": n_atoms, "box_A": box_A, "species": mc, "density_gcm3": rho}

        # ── Build per-combination simulation section ──────────────────────────────
        combo_sim: dict = {**sim}
        combo_sim["comp_spec"]            = combo_comp_spec
        combo_sim["salt_molarity"]        = combo_salt_molarity
        combo_sim["solvents"]             = [{"name": s["name"], "ratio": 1} for s in combo_solvents]
        combo_sim["salt"]                 = combo_salt_names[0] if combo_salt_names else ""
        combo_sim["molecule_counts_aimd"] = mc_dft
        combo_sim["molecule_counts_mlmd"] = mc_mlmd
        combo_sim["molecule_counts_cmd"]  = mc_cmd
        combo_sim["tier_aimd"]            = _tier_info(mc_dft,  poly_atoms=_dft_poly_atoms)
        combo_sim["tier_mlmd"]            = _tier_info(mc_mlmd, poly_atoms=_mlmd_poly_atoms)
        combo_sim["tier_cmd"]             = _tier_info(mc_cmd)
        # Remove the old parent-wide generic key; keep the per-tier _aimd/_mlmd/_cmd keys
        combo_sim.pop("molecule_counts", None)

        aimd_dirs = [f"aimd/{T}K" for T in sim.get("aimd_temps", [300, 400, 500])]
        mlmd_dirs = {f"{T}K": f"dlmd/{T}K"
                     for T in sim.get("mlmd_temps", [300, 320, 340, 360, 380, 400, 500, 600])}

        from hpca.core.combinations import production_combinations
        n_parallel = len(production_combinations(parent_yaml))

        doc: dict = {
            "name":              combo_name,
            "full_name":         combo_meta.get("label", combo_name),
            "category":          parent_yaml.get("category",      "polymer"),
            "system_type":       parent_yaml.get("system_type",   "gel"),
            "execution_mode":    parent_yaml.get("execution_mode", "slurm"),
            "workflow_version":  2,
            "mobile_ion":        parent_yaml.get("mobile_ion",    "Li"),
            "mobile_ions":       parent_yaml.get("mobile_ions",   ["Li"]),
            "T_ref":             parent_yaml.get("T_ref",         300),
            "root":              str(project_dir / combo_name),
            "grand_combinations_total": 1,
            "n_parallel_subprojects": n_parallel,
            "aimd_combinations": [aimd_combo_meta or combo_meta],
            "mlmd_combinations": [combo_meta],
            "cmd_combinations":  (combo_variants if combo_variants
                                  else [combo_detail] if combo_detail else []),
            "aimd_dirs":         aimd_dirs,
            "mlmd_dirs":         mlmd_dirs,
            "cmd_dirs":          ["cmd"],
            "simulation":        combo_sim,
            "stages":            parent_yaml.get("stages", {}),
            "autonomy":          parent_yaml.get("autonomy", {}),
            "production_combination": combo_name,
            "aimd_dataset_key": (aimd_combo_meta or combo_meta).get("name", combo_name),
        }
        # Molarity series: expose the sweep as composition_variants so MLMD/CMD
        # handlers can build one box per molarity inside this sub-project.
        # Single-molarity projects get no key — current single-box behavior.
        _variant_entries = [v for v in (combo_variants or []) if "salt_molarity" in v]
        if len(_variant_entries) > 1:
            _prefix = f"{combo_name}_"
            doc["composition_variants"] = [
                {
                    "name": (v["name"][len(_prefix):]
                             if v.get("name", "").startswith(_prefix)
                             else f"{v['salt_molarity']:g}".replace(".", "p") + "M"),
                    "salt_molarity": v["salt_molarity"],
                }
                for v in _variant_entries
            ]

        # Carry polymer chain specs (chain_length, n_chains, copolymer_ratio)
        # so _build_designed_structures_liquid() can generate oligomers.
        if parent_yaml.get("polymers"):
            doc["polymers"] = parent_yaml["polymers"]
        return doc

    def _pack_one_tier(
        self,
        tier: str,
        mc: dict,
        box_A: "float | None",
        poly_specs: "list[dict] | None",
        *,
        ds_dir: Path,
        preopt_dir: Path,
        combo: dict,
        rho: float,
        category: str = "",
        system_type: str = "",
    ) -> None:
        """Pack one tier (dft/mlmd/cmd) → poscar_{tier}.vasp.

        Thread-safe: writes only to tier-unique paths.
        Also called by h05_cmd.submit() for self-contained CMD packing.
        """
        out_poscar = ds_dir / f"poscar_{tier}.vasp"
        tier_preopt_dir = _dft_preopt(ds_dir.parent) if tier == "dft" else preopt_dir
        preopt_out = contcar_preopt(ds_dir.parent, tier)
        if preopt_out.exists():
            log.info("[h00_design] poscar_%s.vasp preopt done — skip re-packing", tier)
            return
        # CMD is owned by h05_cmd.submit(); if preopted_system_cmd.data already exists, skip
        # repacking to avoid running PACKMOL on 50k-atom boxes unnecessarily.
        if tier == "cmd" and (preopt_dir / "preopted_system_cmd.data").exists():
            log.info("[h00_design] preopted_system_cmd.data exists — skipping CMD repack (h05_cmd owns CMD)")
            return
        if out_poscar.exists():
            if self._poscar_is_valid(out_poscar):
                log.info("[h00_design] poscar_%s.vasp exists and valid — skip", tier)
                return
            log.info("[h00_design] poscar_%s.vasp invalid — regenerating", tier)
            for _stale in (tier_preopt_dir / "POSCAR" if tier == "dft"
                           else tier_preopt_dir / f"poscar_{tier}_preopt.vasp",
                           preopt_out):
                if _stale.exists():
                    _stale.unlink()
                    log.info("[h00_design] Removed stale preopt file: %s", _stale.name)
        if not mc and not poly_specs:
            log.warning("[h00_design] No molecule counts for %s tier — skip", tier)
            return
        n_total = sum(mc.values()) if mc else 0
        n_atoms = sum(n * _NATOMS_PER_MOL.get(name, _DEFAULT_NATOMS_PER_MOL)
                      for name, n in (mc or {}).items())
        if poly_specs:
            n_atoms += sum(
                ps["n_chains"] * ps["n_units"] * _NATOMS_PER_MOL.get(ps["name"], 50)
                for ps in poly_specs
            )
        log.info("[h00_design] Packing poscar_%s.vasp: small_mol=%s poly=%s → ~%d atoms",
                 tier, mc, [ps["name"] for ps in (poly_specs or [])], n_atoms)

        # Liquid/polymer: 1.5× linear box so PACKMOL packs at ~30% SM density.
        # VASP ISIF=3 (DFT) and LAMMPS NPT (MLMD/CMD) compress to target density.
        _is_liq_or_poly = _cat_is_molecular(category) or system_type == "gel"
        _BOX_SCALE = 1.5   # linear enlargement factor for liquid/polymer packing

        def _auto_box_2x(n_atoms_total: int) -> float:
            """Compute enlarged box from atom count + density for PACKMOL packing."""
            mass_g = n_atoms_total * 10.0 * 1.66054e-24   # rough average MW ~10 g/mol per atom
            target = max(15.0, ((mass_g / rho) * 1e24) ** (1.0 / 3.0))
            return max(20.0, target * _BOX_SCALE)

        # MLMD/CMD: if box_A provided, scale it for PACKMOL
        pack_box_A = box_A
        if box_A and _is_liq_or_poly and tier in ("mlmd", "cmd"):
            pack_box_A = box_A * _BOX_SCALE
            log.info("[h00_design] %s liquid/polymer: enlarged box %.1f Å (target %.1f Å, ~30%% SM density)",
                     tier.upper(), pack_box_A, box_A)

        project_dir = ds_dir.parent
        if poly_specs and tier != "dft":
            # MLMD/CMD: grid placement for large polymer chains (PACKMOL can't fit them)
            use_box = pack_box_A
            if tier == "mlmd" and use_box is None:
                sm_a = sum(n * _NATOMS_PER_MOL.get(k, _DEFAULT_NATOMS_PER_MOL)
                           for k, n in (mc or {}).items())
                poly_a = sum(
                    ps["n_chains"] * ps["n_units"]
                    * _NATOMS_PER_MOL.get(ps["name"], _DEFAULT_NATOMS_PER_MOL)
                    for ps in poly_specs
                )
                use_box = _auto_box_2x(sm_a + poly_a)
                log.info("[h00_design] %s gel box: %.1f Å (%d SM + %d poly atoms, 1.5× linear)",
                         tier.upper(), use_box, sm_a, poly_a)
            poscar_text = self._pack_cmd_with_grid_polymers(
                project_dir, combo, conc_M=1.0, n_total=n_total,
                rho=rho, molecule_counts=mc, box_A=use_box, polymer_specs=poly_specs)
        else:
            # DFT (with or without polymer): always use PACKMOL directly.
            # DFT has ≤300 atoms total — PACKMOL handles 1 chain × few monomers fine.
            # Avoids "grid-placed" marker which would skip MACE preopt.
            if pack_box_A is None and _is_liq_or_poly:
                n_atoms_all = sum(n * _NATOMS_PER_MOL.get(k, _DEFAULT_NATOMS_PER_MOL)
                                  for k, n in (mc or {}).items())
                if poly_specs:
                    n_atoms_all += sum(
                        ps["n_chains"] * ps["n_units"]
                        * _NATOMS_PER_MOL.get(ps["name"], _DEFAULT_NATOMS_PER_MOL)
                        for ps in poly_specs
                    )
                _mass_g = n_atoms_all * 10.0 * 1.66054e-24
                _target = max(15.0, ((_mass_g / rho) * 1e24) ** (1.0 / 3.0))
                pack_box_A = max(20.0, _target * 1.1)   # 1.1× natural density; VASP ISIF=3 compresses
                log.info("[h00_design] DFT box: %.1f Å (1.1× natural density, ISIF=3 compresses)", pack_box_A)
            poscar_text = self._pack_liquid_cell(
                project_dir, combo, conc_M=1.0, n_total=n_total,
                rho=rho, molecule_counts=mc, box_A=pack_box_A, polymer_specs=poly_specs)
        if poscar_text:
            out_poscar.write_text(poscar_text)
            log.info("[h00_design] poscar_%s.vasp written", tier)
        else:
            log.error("[h00_design] Packing failed for poscar_%s.vasp", tier)

    def _preopt_one_tier(
        self,
        tier: str,
        *,
        ds_dir: Path,
        preopt_dir: Path,
        params: dict,
        category: str = "",
    ) -> None:
        """LAMMPS MACE NPT pre-optimize one tier (variable cell, sequential).

        Runs LAMMPS NPT at 300 K, 1 bar to compress oversized PACKMOL boxes.
        Falls back to ASE UnitCellFilter + FIRE (CPU) if LAMMPS is unavailable.
        On any failure the packed POSCAR is used unchanged as the starting structure.
        """
        import shutil as _sh_t
        import signal as _sig, os as _os
        src = ds_dir / f"poscar_{tier}.vasp"
        if not src.exists():
            log.warning("[h00_design] poscar_%s.vasp not found — skipping preopt", tier)
            return
        if tier == "dft":
            preopt_dir = _dft_preopt(ds_dir.parent)
            preopt_in = preopt_dir / "POSCAR"
        else:
            preopt_in = preopt_dir / f"poscar_{tier}_preopt.vasp"
        preopt_out = contcar_preopt(ds_dir.parent, tier)
        preopt_dir.mkdir(parents=True, exist_ok=True)
        if not preopt_in.exists():
            _sh_t.copy2(src, preopt_in)
        if preopt_out.exists():
            log.info("[h00_design] contcar_%s_preopt.vasp exists — skip", tier)
            return

        from hpca.core.preoptimization import decide_preoptimization
        project_dir = ds_dir.parent
        yaml_data = params.get("project_config", {})
        decision = decide_preoptimization(
            src, category, yaml_data, self.platform_config(),
            generated_structure=_cat_is_molecular(category),
        )
        self._record_preopt_decision(project_dir, tier, decision)
        if decision.overlap_repaired:
            # The repair operates on the canonical designed structure.  Keep the
            # backend input synchronized with the validated coordinates.
            _sh_t.copy2(src, preopt_in)
        if not decision.run_mace:
            _sh_t.copy2(src, preopt_out)
            log.info("[h00_design] MACE preopt skipped for %s: %s", tier, decision.reason)
            return
        first_line = src.read_text().splitlines()[0].lower() if src.stat().st_size else ""
        if "grid-placed" in first_line:
            log.info("[h00_design] %s is grid-placed — copying as contcar directly", tier)
            _sh_t.copy2(src, preopt_out)
            return

        steps   = params["steps"]
        timeout = params["timeout"]
        ntasks  = params.get("ntasks", 1)
        # Select MACE model: MP0 if any metal/non-organic element present, else OFF23
        _MACE_OFF_ELEMENTS = {"H","C","N","O","F","P","S","Cl","Br","I"}
        _poscar_elems = set()
        try:
            for ln in src.read_text().splitlines()[5:6]:
                _poscar_elems = set(ln.split())
        except Exception:
            pass
        model_type = "mace_off" if _poscar_elems and _poscar_elems.issubset(_MACE_OFF_ELEMENTS) else "mace_mp"
        log.info("[h00_design] LAMMPS MACE NPT preopt %s: tier=%s steps=%d timeout=%ds model=%s",
                 src.name, tier, steps, timeout, model_type)

        _py = self.hpc_path("python_deepmd") or self.hpc_path("python_cladue") or sys.executable
        cmd_args = [
            _py, PRERELAX_LAMMPS_SCRIPT, str(src),
            f"steps={steps}", f"timeout={timeout}",
            f"ntasks={ntasks}", f"model_type={model_type}",
        ]
        from hpca.core import child_procs as _child_procs
        try:
            proc = subprocess.Popen(
                cmd_args,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                preexec_fn=_os.setsid,
            )
            # setsid children survive the orchestrator's own killpg on SIGTERM —
            # register so the shutdown handler can clean up this session too.
            _child_procs.register(proc.pid)
            try:
                proc.communicate(timeout=timeout + 120)  # outer guard: 2 min beyond inner
                if proc.returncode == 0:
                    log.info("[h00_design] MACE LAMMPS preopt converged for %s (rc=0)", tier)
                else:
                    log.warning("[h00_design] MACE LAMMPS preopt rc=%d for %s — using packed geometry",
                                proc.returncode, tier)
                    # script restores .orig on failure; use src as-is
            except subprocess.TimeoutExpired:
                log.warning("[h00_design] MACE LAMMPS preopt outer timeout (%ds) for %s",
                            timeout + 120, tier)
                try:
                    _os.killpg(_os.getpgid(proc.pid), _sig.SIGTERM)
                except OSError:
                    pass
                proc.wait()
            finally:
                _child_procs.unregister(proc.pid)
        except FileNotFoundError:
            log.warning("[h00_design] python not found — skipping preopt for %s", tier)
        except Exception as exc:
            log.warning("[h00_design] MACE LAMMPS preopt error for %s: %s — using packed geometry",
                        tier, exc)

        # The script writes preopt result in-place to src (poscar_{tier}.vasp).
        # Copy the (possibly updated) src to the contcar output path.
        if src.exists():
            _sh_t.copy2(src, preopt_out)
        else:
            log.error("[h00_design] poscar_%s.vasp missing after preopt — skipping tier", tier)
            return
        # Clean up any .orig backup left by the script on failure
        orig = Path(str(src) + ".orig")
        if orig.exists():
            orig.unlink()
        log.info("[h00_design] contcar_%s_preopt.vasp written", tier)

    def _run_preopt_all(self, project_dir: Path, yaml_data: dict) -> None:
        """Preoptimize DFT into dft/preopt and MLMD/CMD into preopt.

        This method runs inside the bounded preoptimization queue.  Its tier
        order remains deterministic; concurrency is across independent projects.
        LAMMPS NPT compresses oversized PACKMOL boxes (8× volume) to near-target density.
        Falls back to ASE variable-cell if LAMMPS is unavailable (e.g., no GPU).
        Timeout → uses PACKMOL-packed geometry as contcar (still valid starting point).
        """
        ds_dir     = _designed_structures(project_dir)
        preopt_dir = _preopt(project_dir)
        preopt_dir.mkdir(parents=True, exist_ok=True)
        category       = yaml_data.get("category", "liquid_electrolyte")

        policy = {**self.platform_config().get("preoptimization", {}),
                  **(yaml_data.get("preoptimization", {}) or {})}
        _steps   = int(policy.get("steps", 1000))
        _timeout = int(policy.get("max_runtime_s", 1800))
        _ntasks  = int(policy.get("ntasks", 1))

        _PREOPT_PARAMS: dict[str, dict] = {
            "dft":  {"steps": _steps, "timeout": _timeout, "ntasks": _ntasks, "project_config": yaml_data},
            "mlmd": {"steps": _steps, "timeout": _timeout, "ntasks": _ntasks, "project_config": yaml_data},
            "cmd":  {"steps": _steps, "timeout": _timeout, "ntasks": _ntasks, "project_config": yaml_data},
        }

        # Deterministic tier order inside one project.  The daemon scheduler
        # controls concurrency across projects independently of design packing.
        for tier in ("dft", "mlmd", "cmd"):
            try:
                self._preopt_one_tier(
                    tier,
                    ds_dir=ds_dir, preopt_dir=preopt_dir,
                    params=_PREOPT_PARAMS[tier],
                    category=category,
                )
            except Exception as exc:
                log.error("[h00_design] Preopt %s tier raised: %s", tier, exc)

    # ── Liquid electrolyte systems ─────────────────────────────────────────────

    def _build_liquid_electrolyte_system(self, project_dir: Path,
                                          yaml_data: dict) -> None:
        """Legacy: pack POSCAR cells for each AIMD directory from combinations/solvents."""
        sim             = yaml_data.get("simulation", {})
        n_total         = sim.get("n_molecules_aimd", 8)
        # Prefer tier-computed density; fall back to old scalar key
        rho             = (sim.get("tier_aimd", {}).get("density_gcm3")
                           or sim.get("target_density_gcm3", 0.85))
        aimd_dirs       = yaml_data.get("aimd_dirs", [])
        category        = yaml_data.get("category", "liquid_electrolyte")
        # Prefer per-tier species counts (new wizard); fall back to legacy key
        molecule_counts = (sim.get("molecule_counts_aimd")
                           or sim.get("molecule_counts"))

        if not aimd_dirs:
            log.warning("[h00_design] No aimd_dirs — nothing to build")
            return

        # ── Resolve combination list ─────────────────────────────────────────
        # Old format: combinations: [{name, solvents, salt}, ...]
        # New format: simulation.solvents + simulation.salt (single combo = project)
        combos = yaml_data.get("combinations", [])
        if not combos:
            solvents = sim.get("solvents", [])
            salt     = sim.get("salt", "LiFSI")
            if solvents and salt:
                proj_name = yaml_data.get("name", project_dir.name)
                combos = [{"name": proj_name, "solvents": solvents, "salt": salt}]
            else:
                log.warning("[h00_design] No combinations or simulation.solvents+salt — nothing to build")
                return

        combo_map = {c["name"]: c for c in combos}

        # ── Detect dir format ────────────────────────────────────────────────
        # Tier (2-part): aimd/temp             e.g. aimd/300K  (new vol% wizard)
        # New  (3-part): aimd/conc/temp         e.g. aimd/0p5M/250K
        # Old  (4-part): combo/aimd/conc/temp  e.g. DMB_LiFSI/aimd/0p5M/250K
        sample_parts  = Path(aimd_dirs[0]).parts
        is_tier_format = len(sample_parts) == 2  # aimd/temp  — single composition
        is_new_format  = len(sample_parts) == 3  # aimd/conc/temp

        if is_tier_format:
            # New vol%-wizard format: one POSCAR shared across all temperatures
            combo = combos[0]
            missing = [d for d in aimd_dirs
                       if not (project_dir / d / "POSCAR").exists()]
            if not missing:
                log.info("[h00_design] All tier-format POSCARs exist — skip")
                return
            log.info("[h00_design] Building single AIMD cell (tier format) for %d temp dirs",
                     len(aimd_dirs))

            # For gel/polymer systems, include short polymer oligomers so the
            # AIMD training data captures polymer-ion and polymer-solvent interactions.
            system_type = yaml_data.get("system_type", "")
            box_A       = sim.get("tier_aimd", {}).get("box_A")   # wizard pre-computed
            poly_specs  = (self._extract_aimd_polymers(yaml_data)
                           if system_type in ("gel", "polymer") else None)

            poscar_text = self._pack_liquid_cell(
                project_dir, combo, 1.0, n_total, rho, molecule_counts,
                polymer_specs=poly_specs, box_A=box_A)
            if poscar_text is None:
                log.error("[h00_design] Tier-format cell build failed")
                return
            first_dir = project_dir / aimd_dirs[0]
            first_dir.mkdir(parents=True, exist_ok=True)
            first_poscar = first_dir / "POSCAR"
            first_poscar.write_text(poscar_text)
            self._mlip_prerelax(first_poscar, category)
            relaxed_text = first_poscar.read_text()
            for d in aimd_dirs[1:]:
                cell_dir = project_dir / d
                cell_dir.mkdir(parents=True, exist_ok=True)
                (cell_dir / "POSCAR").write_text(relaxed_text)
            log.info("[h00_design] Tier-format: POSCAR written to %d dirs", len(aimd_dirs))
            return

        if is_new_format:
            # Group by concentration (parts[1]); single combo = the project itself
            conc_dirs: dict[str, list] = defaultdict(list)
            for d in aimd_dirs:
                parts = Path(d).parts  # ('aimd', '0p5M', '250K')
                if len(parts) >= 3:
                    conc_dirs[parts[1]].append(d)

            combo = combos[0]

            # Collect work items — skip already-built concentrations
            pending_new: list[tuple] = []
            for conc_str, dirs in sorted(conc_dirs.items()):
                if all((project_dir / d / "POSCAR").exists() for d in dirs):
                    log.info("[h00_design] %s: all POSCARs exist — skip", conc_str)
                    continue
                conc_M = float(conc_str.replace("p", ".").rstrip("M"))
                pending_new.append((conc_str, dirs, conc_M))

            def _build_conc(args: tuple) -> None:
                """Pack and write a single concentration cell (new-style layout) in a thread pool worker."""
                conc_str, dirs, conc_M = args
                log.info("[h00_design] Building cell: conc=%s  %.2f M", conc_str, conc_M)
                poscar_text = self._pack_liquid_cell(
                    project_dir, combo, conc_M, n_total, rho, molecule_counts)
                if poscar_text is None:
                    log.error("[h00_design] Cell build failed for %s", conc_str)
                    return
                first_dir = project_dir / dirs[0]
                first_dir.mkdir(parents=True, exist_ok=True)
                first_poscar = first_dir / "POSCAR"
                first_poscar.write_text(poscar_text)
                self._mlip_prerelax(first_poscar, category)
                relaxed_text = first_poscar.read_text()
                for d in dirs[1:]:
                    cell_dir = project_dir / d
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    (cell_dir / "POSCAR").write_text(relaxed_text)
                log.info("[h00_design] %s: POSCAR written to %d dirs", conc_str, len(dirs))

            if pending_new:
                nw = min(len(pending_new), 96)
                log.info("[h00_design] Building %d cells in parallel (workers=%d)",
                         len(pending_new), nw)
                with concurrent.futures.ThreadPoolExecutor(max_workers=nw) as pool:
                    list(pool.map(_build_conc, pending_new))

        else:
            # Old (4-part) format: group by (combo_name, conc_str)
            combo_conc: dict[tuple, list] = defaultdict(list)
            for d in aimd_dirs:
                parts = Path(d).parts   # ('DMB_LiFSI', 'aimd', '0p5M', '250K')
                if len(parts) >= 4:
                    combo_conc[(parts[0], parts[2])].append(d)

            pending_old: list[tuple] = []
            for (combo_name, conc_str), dirs in sorted(combo_conc.items()):
                if all((project_dir / d / "POSCAR").exists() for d in dirs):
                    log.info("[h00_design] %s/%s: all POSCARs exist — skip",
                             combo_name, conc_str)
                    continue
                combo = combo_map.get(combo_name)
                if combo is None:
                    log.warning("[h00_design] Unknown combination '%s' — skip", combo_name)
                    continue
                conc_M = float(conc_str.replace("p", ".").rstrip("M"))
                pending_old.append((combo_name, conc_str, dirs, combo, conc_M))

            def _build_combo(args: tuple) -> None:
                """Pack and write a single combination/concentration cell (legacy layout) in a thread pool worker."""
                combo_name, conc_str, dirs, combo, conc_M = args
                log.info("[h00_design] Building cell: %s  %.2f M", combo_name, conc_M)
                poscar_text = self._pack_liquid_cell(
                    project_dir, combo, conc_M, n_total, rho, molecule_counts)
                if poscar_text is None:
                    log.error("[h00_design] Cell build failed for %s/%s",
                              combo_name, conc_str)
                    return
                first_dir = project_dir / dirs[0]
                first_dir.mkdir(parents=True, exist_ok=True)
                first_poscar = first_dir / "POSCAR"
                first_poscar.write_text(poscar_text)
                self._mlip_prerelax(first_poscar, category)
                relaxed_text = first_poscar.read_text()
                for d in dirs[1:]:
                    cell_dir = project_dir / d
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    (cell_dir / "POSCAR").write_text(relaxed_text)
                log.info("[h00_design] %s/%s: POSCAR written to %d dirs",
                         combo_name, conc_str, len(dirs))

            if pending_old:
                nw = min(len(pending_old), 96)
                log.info("[h00_design] Building %d combo cells in parallel (workers=%d)",
                         len(pending_old), nw)
                with concurrent.futures.ThreadPoolExecutor(max_workers=nw) as pool:
                    list(pool.map(_build_combo, pending_old))

    def _pack_liquid_cell(self, project_dir: Path, combo: dict,
                           conc_M: float, n_total: int, rho: float,
                           molecule_counts: dict | None = None,
                           polymer_specs: list | None = None,
                           box_A: float | None = None) -> str | None:
        """Pack one liquid cell with PACKMOL; return POSCAR string or None.

        If molecule_counts is provided those exact counts are used.
        If box_A is provided (wizard pre-computed, includes all species) it is used
        directly instead of estimating from molecular weights.
        polymer_specs: [{name, n_chains, n_units, smiles}] for gel/polymer systems.
        """
        solvents  = combo.get("solvents", [])
        salt_name = combo.get("salt", "LiFSI")

        if molecule_counts:
            # Explicit composition from wizard manual mode
            n_solvents: dict[str, int] = {
                s["name"]: molecule_counts.get(s["name"], 1) for s in solvents
            }
            n_salt       = molecule_counts.get(salt_name, 1)
            n_solv_total = sum(n_solvents.values())
            if box_A:
                # Use wizard-precomputed box that already accounts for polymer mass
                box = box_A
            else:
                MW_total = (n_salt * _MW.get(salt_name, 187.07)
                            + sum(n * _MW.get(nm, 100.0) for nm, n in n_solvents.items()))
                V_A3 = MW_total / rho / 6.022e23 * 1e24
                box  = V_A3 ** (1.0 / 3.0)
        else:
            # Auto-compute from concentration and target molecule count
            total_ratio  = sum(s.get("ratio", 1) for s in solvents) or 1
            MW_solv_avg  = sum(s.get("ratio", 1) * _MW.get(s["name"], 100.0)
                               for s in solvents) / total_ratio
            MW_salt      = _MW.get(salt_name, 187.07)

            rho_solv_mol = rho * 1000.0 / MW_solv_avg
            n_salt       = max(1, round(n_total * conc_M / (conc_M + rho_solv_mol)))
            n_solv_total = max(1, n_total - n_salt)

            n_solvents = {}
            remaining = n_solv_total
            for i, s in enumerate(solvents):
                if i == len(solvents) - 1:
                    n_solvents[s["name"]] = max(1, remaining)
                else:
                    frac = s.get("ratio", 1) / total_ratio
                    n    = max(1, round(n_solv_total * frac))
                    n_solvents[s["name"]] = n
                    remaining -= n

            MW_total = (n_salt * MW_salt
                        + sum(n * _MW.get(nm, 100.0) for nm, n in n_solvents.items()))
            V_A3  = MW_total / rho / 6.022e23 * 1e24
            box   = V_A3 ** (1.0 / 3.0)

        log.info("[h00_design]   n_salt=%d  n_solvs=%s  box=%.1f Å", n_salt, n_solvents, box)

        # Collect ALL non-polymer small molecules (may include multiple salts)
        _poly_names = {ps["name"] for ps in (polymer_specs or [])}
        _solv_names = {s["name"] for s in solvents}
        if molecule_counts:
            n_extra: dict[str, int] = {
                k: v for k, v in molecule_counts.items()
                if k not in _solv_names and k not in _poly_names
            }
        else:
            n_extra = {salt_name: n_salt}

        # Ensure .vasp structure files exist in input_structures/
        # Lookup order: local input_structures/ (symlinked to parent for combos)
        #               → global library (platform.yaml hpc.input_structures_library)
        #               → legacy project root copy
        #               → fetch from PubChem/RDKit (saves result back to local + global lib)
        import shutil as _shutil
        input_dir = project_dir / "input_structures"
        input_dir.mkdir(parents=True, exist_ok=True)
        all_mol_names = [s["name"] for s in solvents] + list(n_extra.keys())
        for mol_name in all_mol_names:
            vasp_path = input_dir / f"{mol_name}.vasp"
            # 1. Check local dir (and global lib via _find_mol_vasp)
            if not vasp_path.exists():
                found = _find_mol_vasp(mol_name, input_dir)
                if found and found != vasp_path:
                    _shutil.copy2(found, vasp_path)
                    log.info("[h00_design] %s.vasp ← %s", mol_name, found.parent.name)
            # 2. Legacy: bare file in project root
            if not vasp_path.exists():
                legacy = project_dir / f"{mol_name}.vasp"
                if legacy.exists():
                    _shutil.copy2(legacy, vasp_path)
                    log.info("[h00_design] Copied %s.vasp from project root", mol_name)
            # 3. Fetch from PubChem/RDKit; save to local and global library
            if not vasp_path.exists():
                log.info("[h00_design] Fetching structure for %s", mol_name)
                try:
                    sys.path.insert(0, str(Path(__file__).parents[3]))
                    from hpca.sim.structure_fetch import fetch_structure
                    role = "solvent" if mol_name in _solv_names else "salt"
                    fetch_structure(mol_name, role, input_dir)
                    if vasp_path.exists():
                        _save_to_global_lib(vasp_path, f"{mol_name}.vasp")
                except Exception as exc:
                    log.warning("[h00_design] structure_fetch failed for %s: %s", mol_name, exc)
            if not vasp_path.exists():
                log.error("[h00_design] Missing %s.vasp — cannot pack", mol_name)
                return None

        # Run PACKMOL
        packmol_bin = PACKMOL_BIN  # already resolved at module load
        if not packmol_bin:
            log.warning("[h00_design] PACKMOL not found at any known path — using grid placement")
            log.warning("[h00_design]   Set PACKMOL_BIN env var to the packmol binary path")
            mol_counts = {**n_solvents, **n_extra}
            grid_poscar = _grid_placement_poscar(combo["name"], mol_counts, input_dir, box)
            if grid_poscar:
                log.info("[h00_design] Grid-placed POSCAR written (%d molecules)", n_salt + n_solv_total)
                return grid_poscar
            log.error("[h00_design] Grid placement also failed — cannot pack without PACKMOL")
            return None

        with tempfile.TemporaryDirectory() as _tmpdir:
            tmpdir = Path(_tmpdir)

            # Convert .vasp → .xyz for PACKMOL
            xyz_paths: dict[str, Path] = {}
            for mol_name in all_mol_names:
                if mol_name in xyz_paths:
                    continue
                vasp_path = input_dir / f"{mol_name}.vasp"
                xyz_path  = tmpdir / f"{mol_name}.xyz"
                try:
                    _ensure_cladue_env()
                    from pymatgen.core import Structure
                    struct = Structure.from_file(str(vasp_path))
                    lines  = [str(len(struct)), mol_name]
                    for site in struct:
                        x, y, z = site.coords
                        lines.append(f"{site.species_string}  {x:.6f}  {y:.6f}  {z:.6f}")
                    xyz_path.write_text("\n".join(lines) + "\n")
                    xyz_paths[mol_name] = xyz_path
                except Exception as exc:
                    log.warning("[h00_design] XYZ convert failed for %s: %s", mol_name, exc)
                    return None

            # ── Polymer oligomers (gel/polymer systems) ──────────────────────
            poly_xyz: dict[str, Path] = {}
            poly_n:   dict[str, int]  = {}
            if polymer_specs:
                from hpca.sim.structure_fetch import fetch_from_smiles as _fetch_smiles
                for ps in polymer_specs:
                    pname   = ps["name"]
                    n_units = ps["n_units"]
                    # 1. Exact match in local + global lib
                    vasp_p = _find_oligo_vasp(pname, n_units, input_dir, fallback=False)
                    # 2. Generate from SMILES and cache so next sub-project reuses it
                    if vasp_p is None:
                        target = input_dir / f"{pname}_oligo{n_units}.vasp"
                        log.info("[h00_design] Generating %s %d-mer from SMILES", pname, n_units)
                        _fetch_smiles(pname, ps["smiles"], target)
                        if target.exists():
                            _save_to_global_lib(target, f"{pname}_oligo{n_units}.vasp")
                        vasp_p = target if target.exists() else None
                    # 3. Last resort: shorter cached oligomer
                    if vasp_p is None:
                        vasp_p = _find_oligo_vasp(pname, n_units, input_dir, fallback=True)
                    if vasp_p is None:
                        log.warning("[h00_design] %s oligomer unavailable — skipping", pname)
                        continue
                    xyz_p = tmpdir / f"{pname}_oligo.xyz"
                    try:
                        from pymatgen.core import Structure as _S
                        st = _S.from_file(str(vasp_p))
                        xyz_lines = [str(len(st)), pname]
                        for site in st:
                            x, y, z = site.coords
                            xyz_lines.append(f"{site.species_string}  {x:.6f}  {y:.6f}  {z:.6f}")
                        xyz_p.write_text("\n".join(xyz_lines) + "\n")
                        poly_xyz[pname] = xyz_p
                        poly_n[pname]   = ps["n_chains"]
                    except Exception as exc:
                        log.warning("[h00_design] XYZ convert failed for %s: %s", pname, exc)

            # Write PACKMOL input
            out_xyz = tmpdir / "packed.xyz"
            packmol_tol = "2.5" if polymer_specs else "2.0"
            inp_lines = [
                f"tolerance {packmol_tol}",
                "seed -1",
                "maxit 500",
                "nloop 1000",
                "filetype xyz",
                f"output {out_xyz}",
                "",
            ]
            for s in solvents:
                inp_lines += [
                    f"structure {xyz_paths[s['name']]}",
                    f"  number {n_solvents[s['name']]}",
                    f"  inside box 0.0 0.0 0.0 {box:.3f} {box:.3f} {box:.3f}",
                    "end structure",
                    "",
                ]
            for mol_name, count in n_extra.items():
                if mol_name in xyz_paths:
                    inp_lines += [
                        "",
                        f"structure {xyz_paths[mol_name]}",
                        f"  number {count}",
                        f"  inside box 0.0 0.0 0.0 {box:.3f} {box:.3f} {box:.3f}",
                        "end structure",
                    ]
            for pname, xyz_p in poly_xyz.items():
                inp_lines += [
                    "",
                    f"structure {xyz_p}",
                    f"  number {poly_n[pname]}",
                    f"  inside box 0.0 0.0 0.0 {box:.3f} {box:.3f} {box:.3f}",
                    "end structure",
                ]
            inp_text = "\n".join(inp_lines) + "\n"
            inp_file = tmpdir / "packmol.inp"
            inp_file.write_text(inp_text)

            try:
                # Write input to file — some PACKMOL builds can't seek on stdin pipes
                n_pack_total = n_solv_total + n_salt + sum(poly_n.values())
                # Base timeout on atom count, not molecule count, so large-molecule
                # systems (DMB, TFEC, polymers) get proportionally more time.
                n_atoms_approx = sum(
                    cnt * _NATOMS_PER_MOL.get(nm, _DEFAULT_NATOMS_PER_MOL)
                    for nm, cnt in molecule_counts.items()
                )
                n_atoms_approx += sum(
                    _NATOMS_PER_MOL.get(ps["name"], 50) * ps.get("n_chains", 1) * ps.get("n_units", 1)
                    for ps in (polymer_specs or [])
                )
                pack_timeout = min(7200, max(1800, 300 + n_atoms_approx * 2))
                with open(inp_file) as fin:
                    result = subprocess.run(
                        [packmol_bin],
                        stdin=fin,
                        capture_output=True, text=True, timeout=pack_timeout,
                    )
                if not out_xyz.exists():
                    log.error("[h00_design] PACKMOL failed: %s", result.stderr[-500:])
                    return None
            except Exception as exc:
                log.warning("[h00_design] PACKMOL error: %s — will retry next cycle", exc)
                return None

            # Convert packed.xyz → POSCAR via pymatgen
            try:
                _ensure_cladue_env()
                from pymatgen.core import Molecule, Lattice, Structure
                lines = out_xyz.read_text().splitlines()
                n_at  = int(lines[0])
                elems, coords = [], []
                for line in lines[2:2 + n_at]:
                    parts = line.split()
                    elems.append(parts[0])
                    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                lattice = Lattice.cubic(box)
                struct  = Structure(lattice, elems, coords, coords_are_cartesian=True)
                # Sort by species so all same-element atoms are grouped (VASP requirement)
                return struct.get_sorted_structure().to(fmt="poscar")
            except Exception as exc:
                log.error("[h00_design] POSCAR conversion failed: %s", exc)
                return None

    # ── Polymer oligomer helpers ───────────────────────────────────────────────

    @staticmethod
    def _polymer_oligomer_smiles(monomer: str, n_units: int,
                                  copolymer_ratio: dict | None = None) -> str | None:
        """Return SMILES for a short polymer oligomer (n_units repeat units)."""
        m = monomer.upper().replace("-", "").replace("_", "")

        if "PEO" in m:
            # HO-(CH2CH2O)n-H
            return "O" + "CCO" * n_units

        elif "PVDF" in m:
            # VDF: -CH2-CF2-   HFP: -CF2-CF(CF3)-
            n_hfp = 0
            if copolymer_ratio and "HFP" in copolymer_ratio:
                total  = sum(copolymer_ratio.values()) or 1
                n_hfp  = max(0, round(n_units * copolymer_ratio["HFP"] / total))
            n_vdf = max(1, n_units - n_hfp)
            vdf_block = "CC(F)(F)" * n_vdf
            hfp_block = "C(F)(F)C(F)(C(F)(F)F)" * n_hfp
            return vdf_block + hfp_block

        elif "PMMA" in m:
            return "C(CC(=O)OC)(C)" * n_units + "C"

        elif "PTFEP" in m:
            # poly(bis(2,2,2-trifluoroethoxy)phosphazene) [-N=P(OCH2CF3)2-]n
            # RDKit 3D embedding fails for n>10 with the =N backbone SMILES;
            # cap at 10 units and use the SP3 single-bond form which embeds reliably.
            n = min(n_units, 10)
            side = "P(OCC(F)(F)F)(OCC(F)(F)F)"
            return "N" + side + ("=N" + side) * (n - 1) + "=N"

        return None

    def _extract_aimd_polymers(self, yaml_data: dict) -> list[dict]:
        """Return polymer oligomer specs for inclusion in the AIMD cell.

        For each polymer listed in yaml_data['polymers'] that also appears in
        molecule_counts_aimd, produce a spec with SMILES for a short chain
        (n_monomers_aimd / n_chains_aimd repeat units).
        """
        sim          = yaml_data.get("simulation", {})
        mc_aimd      = sim.get("molecule_counts_aimd", {}) or {}
        ch_aimd      = sim.get("chain_counts_aimd", {})    or {}
        solvent_names = {s["name"] for s in sim.get("solvents", [])}
        salt_name     = sim.get("salt", "")

        specs = []
        for p in yaml_data.get("polymers", []):
            monomer = p.get("monomer", "")
            if not monomer or monomer in solvent_names or monomer == salt_name:
                continue
            n_monomers = mc_aimd.get(monomer, 0)
            n_chains   = max(1, ch_aimd.get(monomer, 1))
            if not n_monomers:
                continue
            n_units = max(2, n_monomers // n_chains)
            cratio  = p.get("copolymer_ratio")
            smiles  = self._polymer_oligomer_smiles(monomer, n_units, cratio)
            if smiles is None:
                log.warning("[h00_design] No SMILES template for %s — skipping in AIMD cell", monomer)
                continue
            specs.append({"name": monomer, "n_chains": n_chains,
                           "n_units": n_units, "smiles": smiles})
            log.info("[h00_design]   polymer %s: %d chain(s) × %d-mer", monomer, n_chains, n_units)
        return specs

    # ── Crystal systems ────────────────────────────────────────────────────────

    def _build_crystal_system(self, project_dir: Path, yaml_data: dict) -> None:
        """Fetch crystal from MP/CIF, make supercell, MACE prerelax, and build AIMD variants."""
        dest = dft_opt(project_dir)
        dest.mkdir(parents=True, exist_ok=True)
        category = yaml_data.get("category", "inorganic_sse")
        augment  = yaml_data.get("augment_aimd", True)

        struct     = None
        poscar_path = dest / "POSCAR"

        if not poscar_path.exists() and not (project_dir / "vc" / "CONTCAR").exists():
            if "mp_id" in yaml_data:
                struct = self._fetch_from_mp(yaml_data["mp_id"])
            elif "cif" in yaml_data:
                struct = self._load_from_cif(yaml_data["cif"])

            if struct is None:
                log.warning("[h00_design] No structure source (mp_id or cif) in project.yaml")
                return

            if "supercell" in yaml_data:
                sc = yaml_data["supercell"]
                log.info("[h00_design] Building supercell %s", sc)
                struct.make_supercell(sc)

            struct.to(fmt="poscar", filename=str(poscar_path))
            log.info("[h00_design] Wrote %s (%d atoms)", poscar_path, len(struct))
            self._mlip_prerelax(poscar_path, category)

            mobile_ion = yaml_data.get("mobile_ion", "Li")
            self._make_neb_vacancy_poscar(project_dir, struct, mobile_ion)

        elif poscar_path.exists() and augment:
            # Load existing POSCAR so we can still build variants
            try:
                _ensure_cladue_env()
                from pymatgen.core import Structure
                struct = Structure.from_file(str(poscar_path))
            except Exception as exc:
                log.warning("[h00_design] Cannot reload opt/POSCAR for variants: %s", exc)

        # Generate comprehensive structural variants for AIMD dataset augmentation
        if augment and struct is not None:
            self._build_crystal_variants(project_dir, struct, yaml_data)

    def _fetch_from_mp(self, mp_id: str):
        """Return pymatgen Structure for mp_id from Materials Project, or None on failure."""
        try:
            _ensure_cladue_env()
            from mp_api.client import MPRester
            log.info("[h00_design] Fetching %s from Materials Project", mp_id)
            with MPRester(MP_API_KEY) as mpr:
                struct = mpr.get_structure_by_material_id(mp_id)
            log.info("[h00_design] Fetched %s: %s", mp_id, struct.formula)
            return struct
        except Exception as exc:
            log.error("[h00_design] MP fetch failed for %s: %s", mp_id, exc)
            return None

    def _load_from_cif(self, cif_path: str):
        """Return pymatgen Structure loaded from a CIF file, or None on failure."""
        try:
            _ensure_cladue_env()
            from pymatgen.core import Structure
            struct = Structure.from_file(cif_path)
            log.info("[h00_design] Loaded CIF %s: %s", cif_path, struct.formula)
            return struct
        except Exception as exc:
            log.error("[h00_design] CIF load failed for %s: %s", cif_path, exc)
            return None

    def _make_neb_vacancy_poscar(self, project_dir: Path, struct,
                                  mobile_ion: str) -> None:
        """Write neb/POSCAR_vac with one mobile_ion site removed from struct."""
        try:
            _ensure_cladue_env()
            neb_dir = project_dir / "neb"
            neb_dir.mkdir(parents=True, exist_ok=True)
            target_idx = next(
                (i for i, site in enumerate(struct)
                 if site.species_string == mobile_ion), None)
            if target_idx is None:
                log.warning("[h00_design] No %s found — skip NEB vacancy", mobile_ion)
                return
            vac_struct = struct.copy()
            vac_struct.remove_sites([target_idx])
            vac_path = neb_dir / "POSCAR_vac"
            vac_struct.to(fmt="poscar", filename=str(vac_path))
            log.info("[h00_design] Wrote %s (vacancy at site %d)", vac_path, target_idx)
        except Exception as exc:
            log.warning("[h00_design] NEB vacancy POSCAR failed: %s", exc)

    # ── Crystal augmentation variants ─────────────────────────────────────────

    def _build_crystal_variants(self, project_dir: Path,
                                  base_struct, yaml_data: dict) -> None:
        """Generate all structural variant POSCARs in design/vars/ for AIMD augmentation.

        Variant types:
          d090–d110  : 5 volume-scaled cells (0.90–1.10 × lattice)
          dis_0–dis_2: 3 randomly disordered cells (positions shuffled among species)
          ast_0–ast_1: 2 antisite-defect cells (pair of atoms swapped between species)
          surf_001/011/111: 3 surface slabs (slab_builder: ASE → pymatgen → pure-Python)
          iface_0    : interface (needs film.vasp + sub.vasp in project root)
          part_0     : spherical nanoparticle cluster
          part_sub_0 : nanoparticle on substrate (needs substrate.vasp)
          bilayer_0  : two-layer bilayer (slab_builder.build_multilayer)
        All variants MACE pre-relaxed before DFT.
        n_atoms_target (default 96): supercell size target for volume/disorder/antisite.
        """
        vars_dir = project_dir / "design" / "vars"
        vars_dir.mkdir(parents=True, exist_ok=True)

        category       = yaml_data.get("category", "inorganic_sse")
        n_atoms_target = yaml_data.get("n_atoms_target", 96)
        base_poscar    = dft_opt(project_dir) / "POSCAR"

        # Build supercell of base struct targeting n_atoms_target
        base_sc = self._make_supercell_target(base_struct, n_atoms_target)

        # ── 1. Volume-scaled cells (parallel) ────────────────────────────
        def _build_vol_variant(scale_pct: int) -> None:
            """Write a volume-scaled POSCAR variant for the given lattice scale percentage."""
            var_key = f"d{scale_pct:03d}"
            var_dir = vars_dir / var_key
            var_dir.mkdir(exist_ok=True)
            poscar  = var_dir / "POSCAR"
            if poscar.exists():
                return
            try:
                _ensure_cladue_env()
                from pymatgen.core import Structure, Lattice
                sc = base_sc.copy()
                if scale_pct != 100:
                    sf      = scale_pct / 100.0
                    new_mat = sc.lattice.matrix * sf
                    sc      = Structure(Lattice(new_mat), sc.species, sc.frac_coords)
                poscar.write_text(sc.to(fmt="poscar"))
                self._mlip_prerelax(poscar, category)
                log.info("[h00_design] variant %s: %d atoms", var_key, len(sc))
            except Exception as exc:
                log.warning("[h00_design] variant %s failed: %s", var_key, exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as _pool:
            list(_pool.map(_build_vol_variant, [90, 95, 100, 105, 110]))

        # ── 2. Random species disorder ────────────────────────────────────
        for i in range(3):
            var_key = f"dis_{i}"
            var_dir = vars_dir / var_key
            var_dir.mkdir(exist_ok=True)
            poscar  = var_dir / "POSCAR"
            if poscar.exists():
                continue
            try:
                sc = self._random_disorder_struct(base_sc)
                if sc is not None:
                    poscar.write_text(sc.to(fmt="poscar"))
                    self._mlip_prerelax(poscar, category)
                    log.info("[h00_design] variant %s", var_key)
            except Exception as exc:
                log.warning("[h00_design] variant %s failed: %s", var_key, exc)

        # ── 3. Antisite defects ──────────────────────────────────────────
        for i in range(2):
            var_key = f"ast_{i}"
            var_dir = vars_dir / var_key
            var_dir.mkdir(exist_ok=True)
            poscar  = var_dir / "POSCAR"
            if poscar.exists():
                continue
            try:
                sc = self._antisite_defect_struct(base_sc)
                if sc is not None:
                    poscar.write_text(sc.to(fmt="poscar"))
                    self._mlip_prerelax(poscar, category)
                    log.info("[h00_design] variant %s", var_key)
            except Exception as exc:
                log.warning("[h00_design] variant %s failed: %s", var_key, exc)

        # ── 4–8. Geometry-derived variants ──────────────────────────────
        self._build_surface_variants(project_dir, vars_dir, base_poscar, category)
        self._build_interface_variants(project_dir, vars_dir, category)
        self._build_particle_variants(project_dir, vars_dir, base_sc, n_atoms_target, category)
        self._build_particle_substrate_variants(project_dir, vars_dir, category)
        self._build_bilayer_variants(project_dir, vars_dir, base_poscar, category)

        n_vars = len(list(vars_dir.glob("*/POSCAR")))
        log.info("[h00_design] Crystal variants done: %d variants in %s", n_vars, vars_dir)

    @staticmethod
    def _make_supercell_target(struct, n_atoms_target: int):
        """Return supercell with atom count closest to n_atoms_target."""
        import math
        n = len(struct)
        if n == 0 or n >= n_atoms_target:
            return struct
        factor = max(1, round((n_atoms_target / n) ** (1.0 / 3.0)))
        sc = struct.copy()
        sc.make_supercell([factor, factor, factor])
        return sc

    def _pack_cmd_with_grid_polymers(
            self, project_dir: Path, combo: dict, conc_M: float, n_total: int,
            rho: float, molecule_counts: dict | None, box_A: float | None,
            polymer_specs: list) -> "str | None":
        """Pack CMD box: PACKMOL for small molecules, then grid-place polymer chains.

        This avoids PACKMOL timeout when packing many long polymer chains in a large box.
        The small-molecule PACKMOL run converges fast; polymer chains are placed on a
        regular grid inside the box so MD equilibration can relax them into the melt.
        """
        import math

        # Phase 1: pack small molecules only (no polymer specs)
        poscar_sm = self._pack_liquid_cell(
            project_dir, combo, conc_M=conc_M, n_total=n_total,
            rho=rho, molecule_counts=molecule_counts, box_A=box_A,
            polymer_specs=None)
        if poscar_sm is None:
            return None
        if "placeholder" in poscar_sm.lower().splitlines()[0].lower():
            # PACKMOL timed out for small molecules — fall back to atom-grid placement
            input_dir_fb = project_dir / "input_structures"
            sm_counts = {k: v for k, v in (molecule_counts or {}).items()}
            log.info("[h00_design] CMD grid fallback: grid-placing %d small molecules in %.1f Å box",
                     sum(sm_counts.values()), box_A or 60.0)
            poscar_sm = _grid_placement_poscar(combo["name"], sm_counts, input_dir_fb, box_A or 60.0)
            if poscar_sm is None:
                log.warning("[h00_design] CMD grid fallback failed — no poscar_cmd")
                return None

        # Phase 2: load oligomer structures and place on grid
        input_dir = project_dir / "input_structures"
        chains_to_place: list[tuple[str, object]] = []  # (name, Structure)
        for ps in polymer_specs:
            pname   = ps["name"]
            n_units = ps["n_units"]
            n_chains = ps["n_chains"]
            # 1. Exact match in local + global lib
            vasp_p = _find_oligo_vasp(pname, n_units, input_dir, fallback=False)
            # 2. Generate from SMILES and cache so next sub-project reuses it
            if vasp_p is None:
                try:
                    from hpca.sim.structure_fetch import fetch_from_smiles as _fetch_smiles
                    log.info("[h00_design] Generating %s %d-mer from SMILES for CMD grid", pname, n_units)
                    target = input_dir / f"{pname}_oligo{n_units}.vasp"
                    _fetch_smiles(pname, ps["smiles"], target)
                    if target.exists():
                        _save_to_global_lib(target, f"{pname}_oligo{n_units}.vasp")
                    vasp_p = target if target.exists() else None
                except Exception as exc:
                    log.warning("[h00_design] %s oligomer SMILES generation failed: %s — skip", pname, exc)
            # 3. Last resort: shorter cached oligomer
            if vasp_p is None:
                vasp_p = _find_oligo_vasp(pname, n_units, input_dir, fallback=True)
            if vasp_p is None:
                log.warning("[h00_design] %s oligomer unavailable for CMD grid — skip", pname)
                continue
            try:
                _ensure_cladue_env()
                from pymatgen.core import Structure as _Struct
                chain_struct = _Struct.from_file(str(vasp_p))
                for _ in range(n_chains):
                    chains_to_place.append((pname, chain_struct))
            except Exception as exc:
                log.warning("[h00_design] Cannot load %s oligomer for grid: %s", pname, exc)

        if not chains_to_place:
            log.info("[h00_design] CMD: no polymer chains available — using small-molecule-only box")
            return poscar_sm

        box = box_A or 60.0
        n_chains_total = len(chains_to_place)

        # Shuffle so different polymer species are interleaved throughout the box
        # (e.g. PEO, PTFEP, PEO, PTFEP instead of all-PEO then all-PTFEP)
        import random as _random_mod
        _random_mod.shuffle(chains_to_place)

        try:
            _ensure_cladue_env()
            from pymatgen.core import Structure as _Struct, Lattice as _Lattice
            import numpy as _np

            def _random_rotation(rng: "_np.random.Generator") -> "_np.ndarray":
                """Random 3×3 proper rotation matrix via QR decomposition of random Gaussian."""
                H = rng.standard_normal((3, 3))
                Q, R = _np.linalg.qr(H)
                Q *= _np.sign(_np.diag(R))
                if _np.linalg.det(Q) < 0:
                    Q[:, 0] *= -1
                return Q

            struct = _Struct.from_str(poscar_sm, fmt="poscar")
            placed_centers: list = []
            min_sep = 8.0        # Å, minimum center-to-center gap between chains
            max_tries = 300
            rng = _np.random.default_rng(seed=42)

            for idx, (pname, chain_struct) in enumerate(chains_to_place):
                coords = _np.array(chain_struct.cart_coords)
                center = coords.mean(axis=0)
                coords_c = coords - center

                # Random rotation so chains are not all aligned the same way
                rot = _random_rotation(rng)
                coords_r = coords_c @ rot.T

                # Rejection-sample a position with minimum separation from prior chains
                lo, hi = min_sep / 2.0, box - min_sep / 2.0
                lo = min(lo, box * 0.05)
                hi = max(hi, box * 0.95)
                pos = None
                for _ in range(max_tries):
                    candidate = rng.uniform(lo, hi, size=3)
                    ok = True
                    for pc in placed_centers:
                        diff = candidate - _np.asarray(pc)
                        diff -= box * _np.round(diff / box)   # minimum image
                        if _np.linalg.norm(diff) < min_sep:
                            ok = False
                            break
                    if ok:
                        pos = candidate
                        break

                if pos is None:
                    # Fall back to a deterministic grid slot so nothing is dropped
                    grid_n = max(1, math.ceil(n_chains_total ** (1.0 / 3.0)) + 1)
                    grid_step = box / grid_n
                    ix = idx % grid_n
                    iy = (idx // grid_n) % grid_n
                    iz = (idx // (grid_n * grid_n)) % grid_n
                    pos = _np.array([(ix + 0.5) * grid_step,
                                     (iy + 0.5) * grid_step,
                                     (iz + 0.5) * grid_step])
                    log.debug("[h00_design] chain %d (%s): scatter rejected, grid fallback", idx, pname)

                placed_centers.append(pos)
                gx, gy, gz = float(pos[0]), float(pos[1]), float(pos[2])
                for i, site in enumerate(chain_struct):
                    cx = (coords_r[i, 0] + gx) % box
                    cy = (coords_r[i, 1] + gy) % box
                    cz = (coords_r[i, 2] + gz) % box
                    struct.append(site.species_string, [cx, cy, cz], coords_are_cartesian=True)

            log.info("[h00_design] Phase-2: placed %d polymer chains "
                     "(mixed order, random scatter+rotation) in %.1f Å box (%d atoms total)",
                     n_chains_total, box, len(struct))
            poscar_str = struct.get_sorted_structure().to(fmt="poscar")
            # Inject "grid-placed" marker so _preopt_one_tier skips MACE on this
            # structure. Pymatgen writes the formula as line 1; replace it with the
            # combo name + marker so the bypass check finds it.
            _lines = poscar_str.splitlines()
            _lines[0] = f"{combo.get('name', 'system')} (grid-placed, {n_chains_total} chains, {len(struct)} atoms)"
            return "\n".join(_lines) + "\n"
        except Exception as exc:
            log.warning("[h00_design] CMD grid polymer placement failed: %s — using small-mol box", exc)
            return poscar_sm

    @staticmethod
    def _replicate_poscar_to_target(poscar_path: Path, n_atoms_target: int, box_A: float) -> "str | None":
        """Replicate a POSCAR by supercell to reach ~n_atoms_target, then scale to box_A."""
        try:
            _ensure_cladue_env()
            from pymatgen.core import Structure
            struct = Structure.from_file(str(poscar_path))
            n_orig = len(struct)
            if n_orig == 0:
                return None
            factor = max(1, round((n_atoms_target / n_orig) ** (1.0 / 3.0)))
            sc = struct.copy()
            sc.make_supercell([factor, factor, factor])
            if box_A and box_A > 0:
                sc = sc.scale_lattice(box_A ** 3)
            return sc.get_sorted_structure().to(fmt="poscar")
        except Exception as exc:
            log.error("[h00_design] Supercell replicate failed: %s", exc)
            return None

    @staticmethod
    def _random_disorder_struct(struct):
        """Shuffle fractional positions across all sites (random mixing)."""
        import random
        species = [str(s) for s in struct.species]
        frac    = [site.frac_coords.copy() for site in struct.sites]
        random.shuffle(frac)
        _ensure_cladue_env()
        from pymatgen.core import Structure
        return Structure(struct.lattice, species, frac).get_sorted_structure()

    @staticmethod
    def _antisite_defect_struct(struct):
        """Swap one pair of atoms between the two most common species."""
        import random
        species = [str(s) for s in struct.species]
        unique  = list(dict.fromkeys(species))
        if len(unique) < 2:
            return None
        sp1, sp2 = unique[0], unique[1]
        idx1 = [i for i, s in enumerate(species) if s == sp1]
        idx2 = [i for i, s in enumerate(species) if s == sp2]
        if not idx1 or not idx2:
            return None
        sc = struct.copy()
        sc.replace(random.choice(idx1), sp2)
        sc.replace(random.choice(idx2), sp1)
        return sc.get_sorted_structure()

    def _build_surface_variants(self, project_dir: Path, vars_dir: Path,
                                  base_poscar: Path, category: str) -> None:
        """Surface slabs via hpca.core.slab_builder (ASE → pymatgen → pure-Python)."""
        miller_map = [("001", (0, 0, 1)), ("011", (0, 1, 1)), ("111", (1, 1, 1))]
        if not base_poscar.exists():
            return

        from hpca.core.slab_builder import build_surface_slab
        for miller_str, miller_idx in miller_map:
            vname  = f"surf_{miller_str}"
            vdir   = vars_dir / vname
            vdir.mkdir(exist_ok=True)
            poscar = vdir / "POSCAR"
            if poscar.exists():
                continue
            try:
                slab_poscar = build_surface_slab(
                    bulk_poscar=base_poscar,
                    miller=miller_idx,
                    n_layers=4,
                    vacuum_A=15.0,
                    min_slab_A=10.0,
                )
                poscar.write_text(slab_poscar)
                self._mlip_prerelax(poscar, category)
                log.info("[h00_design] variant %s", vname)
            except Exception as exc:
                log.warning("[h00_design] Surface %s failed: %s", vname, exc)

    def _build_interface_variants(self, project_dir: Path,
                                    vars_dir: Path, category: str) -> None:
        """Interface slab via hpca.core.slab_builder (needs film.vasp + sub.vasp)."""
        vdir   = vars_dir / "iface_0"
        vdir.mkdir(exist_ok=True)
        poscar = vdir / "POSCAR"
        if poscar.exists():
            return

        film = project_dir / "film.vasp"
        sub  = project_dir / "sub.vasp"
        if not film.exists() or not sub.exists():
            log.debug("[h00_design] iface_0 skipped (film.vasp or sub.vasp missing)")
            return

        try:
            from hpca.core.slab_builder import build_interface
            iface_poscar = build_interface(
                slab_a_poscar=sub,
                slab_b_poscar=film,
                vacuum_A=15.0,
                gap_A=2.5,
                max_strain=0.05,
                fix_bottom=True,
            )
            poscar.write_text(iface_poscar)
            self._mlip_prerelax(poscar, category)
            log.info("[h00_design] variant iface_0")
        except Exception as exc:
            log.warning("[h00_design] build_interface failed: %s", exc)

    def _build_particle_variants(self, project_dir: Path, vars_dir: Path,
                                   base_sc, n_atoms_target: int, category: str) -> None:
        """Spherical nanoparticle cluster carved from bulk supercell."""
        vdir   = vars_dir / "part_0"
        vdir.mkdir(exist_ok=True)
        poscar = vdir / "POSCAR"
        if poscar.exists():
            return
        try:
            import math
            import numpy as np
            _ensure_cladue_env()
            from pymatgen.core import Structure, Lattice

            # Expand bulk to get enough atoms for sphere selection
            rep = max(2, int(math.ceil((n_atoms_target * 6 / len(base_sc)) ** (1.0/3.0))))
            sc  = base_sc.copy()
            sc.make_supercell([rep, rep, rep])

            center       = sc.lattice.matrix.sum(axis=0) / 2.0
            vol_per_atom = sc.volume / len(sc)
            r_target     = (3.0 * n_atoms_target * vol_per_atom / (4.0 * math.pi)) ** (1.0/3.0)

            cart = sc.cart_coords
            dists = np.linalg.norm(cart - center, axis=1)
            mask  = dists <= r_target

            sel_species = [sc.species[i] for i in range(len(sc)) if mask[i]]
            sel_coords  = cart[mask]

            if len(sel_species) < 10:
                log.warning("[h00_design] part_0: only %d atoms in sphere — skip", len(sel_species))
                return

            box_size   = r_target * 2.0 + 15.0
            shift      = np.array([box_size/2, box_size/2, box_size/2]) - center
            new_coords = sel_coords + shift

            cluster = Structure(Lattice.cubic(box_size), sel_species,
                                new_coords, coords_are_cartesian=True)
            cluster.to(fmt="poscar", filename=str(poscar))
            self._mlip_prerelax(poscar, category)
            log.info("[h00_design] variant part_0: %d atoms (r=%.1f Å)", len(cluster), r_target)
        except Exception as exc:
            log.warning("[h00_design] part_0 failed: %s", exc)

    def _build_particle_substrate_variants(self, project_dir: Path,
                                             vars_dir: Path, category: str) -> None:
        """NP-on-substrate via build_substrate_np.py (needs substrate.vasp)."""
        import os, shutil

        vdir   = vars_dir / "part_sub_0"
        vdir.mkdir(exist_ok=True)
        poscar = vdir / "POSCAR"
        if poscar.exists():
            return

        substrate_vasp = project_dir / "substrate.vasp"
        bulk_vasp      = dft_opt(project_dir) / "POSCAR"
        if not substrate_vasp.exists() or not bulk_vasp.exists():
            log.debug("[h00_design] part_sub_0 skipped (substrate.vasp or dft/opt/POSCAR missing)")
            return

        try:
            from hpca.core.np_builder import build_substrate_with_nanoparticles as _build_np
            output_files = _build_np(
                substrate_file=str(substrate_vasp),
                bulk_file=str(bulk_vasp),
                np_radius=5.0, distances=[3],
                n_x=1, n_y=1, z_gap=2.0, boundary_pad=5.0, z_reps=1,
                vacuum=15.0, save_np=True, out_dir=vdir,
            )
            if output_files:
                shutil.copy(vdir / output_files[0], poscar)
                self._mlip_prerelax(poscar, category)
                log.info("[h00_design] variant part_sub_0")
        except Exception as exc:
            log.warning("[h00_design] part_sub_0 failed: %s", exc)

    def _build_bilayer_variants(self, project_dir: Path, vars_dir: Path,
                                  base_poscar: Path, category: str) -> None:
        """Two-layer bilayer via hpca.core.slab_builder.build_multilayer."""
        vdir   = vars_dir / "bilayer_0"
        vdir.mkdir(exist_ok=True)
        poscar = vdir / "POSCAR"
        if poscar.exists():
            return
        if not base_poscar.exists():
            return

        try:
            from hpca.core.slab_builder import build_multilayer
            bilayer_poscar = build_multilayer(
                base_poscar=base_poscar,
                n_repeats=2,
                gap_A=2.5,
                vacuum_A=15.0,
            )
            poscar.write_text(bilayer_poscar)
            self._mlip_prerelax(poscar, category)
            log.info("[h00_design] variant bilayer_0")
        except Exception as exc:
            log.warning("[h00_design] build_multilayers failed: %s", exc)

    # ── LPIFD polymer design (Zhang et al. Nature Energy 2024) ────────────────

    def _build_lpifd_system(self, project_dir: Path, yaml_data: dict) -> None:
        """Build LPIFD polymer electrolyte cells for all combinations in project.yaml.

        For each combination (PEO_LPIFD / PMMA_LPIFD / PTFEP_LPIFD):
          1. Build PVDF-HFP chains via polymer.py (VDF:HFP=4:1, 50 monomers)
          2. Use weight_ratio [1.0, 0.4, 2.4] to set molecule counts
          3. Pack with PACKMOL → system.data under <combo>/cmd/
          4. MACE pre-relax (CPU, fmax=0.2 for polymer, 300 steps)
          5. Copy pre-relaxed structure to AIMD starting POSCARs

        Box size estimated from target density (PVDF-HFP ≈ 1.78 g/cm³ blend).
        """
        design_dir = project_dir / "design"
        design_dir.mkdir(parents=True, exist_ok=True)

        combinations = yaml_data.get("combinations", [])
        if not combinations:
            log.warning("[h00_design] lpifd_polymer: no combinations in project.yaml")
            return

        sim = yaml_data.get("simulation", {})
        chain_monomers = sim.get("chain_monomers", 50)
        hfp_fraction   = sim.get("hfp_fraction",   0.20)
        category       = yaml_data.get("category", "polymer")

        for combo in combinations:
            cname        = combo.get("name", "")
            li_polymer   = combo.get("li_polymer", "PEO")
            salt         = combo.get("salt", "LiFSI")
            wt_ratio     = combo.get("weight_ratio", [1.0, 0.4, 2.4])
            n_chains_lp  = combo.get("n_chains_polymer", 2)
            n_chains_fd  = combo.get("n_chains_diluter", 3)
            hfp_frac     = combo.get("hfp_fraction", hfp_fraction)
            n_mon        = combo.get("chain_monomers", chain_monomers)

            cmd_dir = project_dir / cname / "cmd"
            system_data = cmd_dir / "system.data"
            if system_data.exists():
                log.info("[h00_design] %s: system.data exists — skip", cname)
                continue

            log.info("[h00_design] Building LPIFD cell: %s  (Li-polymer=%s, VDF:HFP=%.0f:%.0f)",
                     cname, li_polymer, (1-hfp_frac)*100, hfp_frac*100)

            ff_dir = cmd_dir / "ff"
            ff_dir.mkdir(parents=True, exist_ok=True)

            mol_data = []

            # ── PVDF-HFP chains (F diluter) ───────────────────────────────────
            try:
                from hpca.sim.polymer import PolymerMolData
                from hpca.sim.forcefield import write_lmp
                pvdf_pm = PolymerMolData.pvdf_hfp(
                    n_units=n_mon, hfp_fraction=hfp_frac, seed=42, count=n_chains_fd)
                pvdf_md = pvdf_pm.to_mol_data()
                write_lmp(pvdf_md.atoms, pvdf_md.bonds, pvdf_md.types,
                          ff_dir / "PVDF_HFP.lmp", "PVDF_HFP")
                mol_data.append(pvdf_md)
                log.info("[h00_design]   PVDF-HFP: %d chains × %d monomers  MW=%.0f g/mol",
                         n_chains_fd, n_mon, pvdf_pm.mw)
            except Exception as exc:
                log.error("[h00_design] PVDF-HFP chain build failed: %s", exc)
                continue

            # ── Li-polymer chains ─────────────────────────────────────────────
            from hpca.sim.forcefield import MOLECULES, MolData, write_lmp as _wlmp
            lp_key = li_polymer.upper()
            if lp_key in MOLECULES:
                try:
                    lp_md = MolData.from_builtin(lp_key, count=n_chains_lp)
                    _wlmp(lp_md.atoms, lp_md.bonds, lp_md.types,
                          ff_dir / f"{li_polymer}.lmp", li_polymer)
                    mol_data.append(lp_md)
                    log.info("[h00_design]   %s: %d chains (builtin)", li_polymer, n_chains_lp)
                except Exception as exc:
                    log.warning("[h00_design]   %s builtin failed: %s — skipping", li_polymer, exc)
            else:
                log.warning("[h00_design]   %s not in MOLECULES builtins — skipping", li_polymer)

            # ── Salt molecules ────────────────────────────────────────────────
            # Count from weight ratio: LiFSI/PVDF-HFP = 2.4/0.4 → 6:1 by weight
            # With MW(LiFSI)=187 and MW(PVDF-HFP repeat)~64/150:
            # Approximate: n_salt from wt_ratio[2] / wt_ratio[1] * n_chains_fd * (MW_pvdf/MW_salt)
            MW_salt  = {"LiFSI": 187.07, "LiTFSI": 287.08, "LiPF6": 151.90}.get(salt, 187.07)
            MW_pvdf_chain = pvdf_pm.mw
            wt_diluter = wt_ratio[1] if len(wt_ratio) > 1 else 0.4
            wt_salt    = wt_ratio[2] if len(wt_ratio) > 2 else 2.4
            # mass_diluter = n_chains_fd * MW_pvdf_chain
            # n_salt = mass_diluter * (wt_salt/wt_diluter) / MW_salt
            n_salt = max(10, round(n_chains_fd * MW_pvdf_chain * (wt_salt / wt_diluter) / MW_salt))
            salt_key = salt.upper().replace("-", "")
            if salt_key in MOLECULES:
                try:
                    salt_md = MolData.from_builtin(salt_key, count=n_salt)
                    _wlmp(salt_md.atoms, salt_md.bonds, salt_md.types,
                          ff_dir / f"{salt}.lmp", salt)
                    mol_data.append(salt_md)
                    log.info("[h00_design]   %s: %d ion pairs", salt, n_salt)
                except Exception as exc:
                    log.warning("[h00_design]   salt %s failed: %s", salt, exc)

            if not mol_data:
                log.error("[h00_design] %s: no molecule data built — skip", cname)
                continue

            # ── Pack into LAMMPS system.data ──────────────────────────────────
            try:
                from hpca.sim.forcefield import build_mixed_system
                L_box = build_mixed_system(mol_data, system_data)
                log.info("[h00_design] %s: system.data  box=%.1f Å  atoms=%d",
                         cname, L_box,
                         sum(len(m.atoms) * m.count for m in mol_data))
            except Exception as exc:
                log.error("[h00_design] build_mixed_system failed for %s: %s", cname, exc)
                continue

            # ── MACE pre-relax (CPU) ──────────────────────────────────────────
            # For polymer blends: write a small POSCAR from the first chain only
            # (full ~50k atom system is too large for MACE pre-relax on login node)
            log.info("[h00_design] %s: MACE pre-relax skipped for large polymer cell "
                     "(~%d atoms — equilibration handles geometry)", cname,
                     sum(len(m.atoms) * m.count for m in mol_data))

            # ── Write AIMD starting POSCARs from a small fragment ─────────────
            self._write_aimd_starting_poscars(project_dir, cname, yaml_data, pvdf_md, lp_key)

        log.info("[h00_design] LPIFD design complete: %d systems built",
                 len([c for c in combinations
                      if (project_dir / c.get("name","") / "cmd" / "system.data").exists()]))

    def _write_aimd_starting_poscars(self, project_dir: Path, cname: str,
                                      yaml_data: dict, pvdf_md, lp_key: str) -> None:
        """Write small AIMD starting POSCARs (fragment, ~150 atoms) for each temperature."""
        aimd_dirs = yaml_data.get("aimd_dirs", [])
        # Filter dirs belonging to this combination
        combo_aimd = [d for d in aimd_dirs if cname.lower() in d.lower()]
        if not combo_aimd:
            return

        # Build a minimal POSCAR from the polymer chain geometry (first 5 monomers)
        try:
            from hpca.sim.polymer import PolymerMolData
            fragment = PolymerMolData.pvdf_hfp(n_units=5, hfp_fraction=0.20, seed=42, count=1)
            frag_md  = fragment.to_mol_data()

            # Simple box estimate: 20 Å cube
            box = 20.0
            lines = ["AIMD fragment for " + cname,
                     "1.0",
                     f"  {box:.6f}  0.000000  0.000000",
                     f"  0.000000  {box:.6f}  0.000000",
                     f"  0.000000  0.000000  {box:.6f}"]

            # Count elements
            from collections import Counter
            elem_counts: Counter = Counter(a["element"] for a in frag_md.atoms)
            lines.append("  ".join(elem_counts.keys()))
            lines.append("  ".join(str(v) for v in elem_counts.values()))
            lines.append("Cartesian")
            for atom in frag_md.atoms:
                lines.append(f"  {atom['x']:.6f}  {atom['y']:.6f}  {atom['z']:.6f}")
            poscar_text = "\n".join(lines) + "\n"

            for d in combo_aimd:
                aimd_dir = project_dir / d
                aimd_dir.mkdir(parents=True, exist_ok=True)
                poscar_path = aimd_dir / "POSCAR"
                if not poscar_path.exists():
                    poscar_path.write_text(poscar_text)
                    log.info("[h00_design]   POSCAR → %s", poscar_path)
        except Exception as exc:
            log.warning("[h00_design] AIMD POSCAR generation failed for %s: %s", cname, exc)

    # ── CMD / MLMD system pre-building ────────────────────────────────────────

    def _prebuild_cmd_systems(self, project_dir: Path, yaml_data: dict) -> list:
        """Build LAMMPS system.data for each cmd_dir using OPLS-AA FF.

        Uses .vasp files for small molecules and oligo .vasp files for polymers.
        Returns list of (rel_dir, status, n_atoms_str).
        """
        from hpca.sim.forcefield import MolData, build_mixed_system

        sim      = yaml_data.get("simulation", {})
        cmd_dirs = yaml_data.get("cmd_dirs", []) or sim.get("cmd_dirs", [])
        mc_cmd   = sim.get("molecule_counts_cmd") or {}
        cc_cmd   = sim.get("chain_counts_cmd")    or {}
        box_A    = sim.get("tier_cmd", {}).get("box_A")

        if not cmd_dirs or not mc_cmd:
            return []

        results = []
        for rel_dir in cmd_dirs:
            cmd_root    = project_dir / rel_dir
            system_data = cmd_root / "system.data"

            if system_data.exists():
                n_atoms = _read_atom_count(system_data)
                results.append((rel_dir, "BUILT", n_atoms))
                log.info("[h00_design] CMD system.data already exists: %s (%s atoms)",
                         rel_dir, n_atoms)
                continue

            log.info("[h00_design] Pre-building CMD system.data for %s", rel_dir)
            mol_data = self._build_mol_data(project_dir, mc_cmd, cc_cmd, yaml_data)
            if not mol_data:
                results.append((rel_dir, "FAILED — could not resolve all molecules", "?"))
                continue

            try:
                cmd_root.mkdir(parents=True, exist_ok=True)
                L = build_mixed_system(mol_data, system_data, box_size=box_A)
                n_atoms = str(sum(len(m.atoms) * m.count for m in mol_data))
                results.append((rel_dir, "BUILT", n_atoms))
                log.info("[h00_design] CMD system.data written: %s  box=%.1f Å  atoms=%s",
                         rel_dir, L, n_atoms)
            except Exception as exc:
                log.warning("[h00_design] CMD system.data build failed for %s: %s", rel_dir, exc)
                results.append((rel_dir, f"FAILED ({exc})", "?"))

        return results

    def _build_mol_data(self, project_dir: Path,
                         mol_counts: dict, chain_counts: dict,
                         yaml_data: dict) -> list:
        """Assemble a list of MolData from .vasp files and polymer oligo files."""
        from hpca.sim.forcefield import MolData, MOLECULES

        mol_data = []
        for mol_name, n_units in mol_counts.items():
            n_chains = chain_counts.get(mol_name, 0)

            if n_chains > 0:
                # Polymer: one entry per chain type; oligo file scales chain count
                count = n_chains
                vasp_p = None
                input_dir = project_dir / "input_structures"
                for search_dir in (input_dir, project_dir):
                    for p in sorted(search_dir.glob(f"{mol_name}_oligo*.vasp")):
                        vasp_p = p
                        break
                    if vasp_p:
                        break

                if vasp_p:
                    try:
                        md = MolData.from_file(vasp_p, name=mol_name, count=count)
                        mol_data.append(md)
                        log.info("[h00_design]   %s: %d chain(s) from %s (%d atoms/chain)",
                                 mol_name, count, vasp_p.name, len(md.atoms))
                        continue
                    except Exception as exc:
                        log.warning("[h00_design]   MolData(%s) from .vasp: %s", mol_name, exc)

                # Fallback: polymer builder (PVDF-HFP)
                m = mol_name.upper().replace("-", "_")
                if "PVDF" in m and "HFP" in m:
                    try:
                        from hpca.sim.polymer import PolymerMolData
                        n_mon = max(5, n_units // max(1, n_chains))
                        pm = PolymerMolData.pvdf_hfp(n_units=n_mon,
                                                      hfp_fraction=0.10, count=count)
                        md = pm.to_mol_data()
                        mol_data.append(md)
                        log.info("[h00_design]   %s: %d chains × %d-mer (polymer builder)",
                                 mol_name, count, n_mon)
                        continue
                    except Exception as exc:
                        log.warning("[h00_design]   PVDF-HFP builder failed: %s", exc)

                # Fallback: PEO from builtin
                if "PEO" in m:
                    try:
                        md = MolData.from_builtin("PEO", count=count)
                        mol_data.append(md)
                        log.info("[h00_design]   %s: %d chains (builtin 3-mer)", mol_name, count)
                        continue
                    except Exception as exc:
                        log.warning("[h00_design]   PEO builtin failed: %s", exc)

                log.error("[h00_design]   No source for polymer %s — aborting", mol_name)
                return []

            else:
                # Small molecule (solvent / salt)
                count = n_units
                input_dir = project_dir / "input_structures"
                vasp_p = input_dir / f"{mol_name}.vasp"
                if not vasp_p.exists():
                    vasp_p = project_dir / f"{mol_name}.vasp"
                if vasp_p.exists():
                    try:
                        md = MolData.from_file(vasp_p, name=mol_name, count=count)
                        mol_data.append(md)
                        log.info("[h00_design]   %s: %d molecules (%d atoms each)",
                                 mol_name, count, len(md.atoms))
                        continue
                    except Exception as exc:
                        log.warning("[h00_design]   MolData(%s) from .vasp: %s", mol_name, exc)

                # Builtin forcefield registry fallback
                key = mol_name.upper().replace("-", "")
                if key in MOLECULES:
                    try:
                        md = MolData.from_builtin(key, count=count)
                        mol_data.append(md)
                        log.info("[h00_design]   %s: %d molecules (builtin)", mol_name, count)
                        continue
                    except Exception as exc:
                        log.warning("[h00_design]   builtin %s failed: %s", key, exc)

                # PubChem fetch fallback — downloads 3D SDF and caches as .vasp
                try:
                    from hpca.sim.structure_fetch import fetch_from_pubchem as _fetch_pub
                    vasp_p = input_dir / f"{mol_name}.vasp"
                    input_dir.mkdir(parents=True, exist_ok=True)
                    _fetch_pub(mol_name, vasp_p)
                    if vasp_p.exists():
                        md = MolData.from_file(vasp_p, name=mol_name, count=count)
                        mol_data.append(md)
                        log.info("[h00_design]   %s: %d molecules (fetched+cached)",
                                 mol_name, count)
                        continue
                except Exception as exc:
                    log.warning("[h00_design]   fetch_molecule(%s) failed: %s", mol_name, exc)

                log.error("[h00_design]   No source for molecule %s — aborting", mol_name)
                return []

        return mol_data

    def _run_test_cmd(self, project_dir: Path,
                       yaml_data: dict) -> list[tuple[str, bool, str]]:
        """Run 200-step NVT force-field test for each cmd_dir composition.

        Uses the CPU LAMMPS binary on the login node (no SLURM, no GPU).
        Returns list of (rel_dir, passed, message).
        """
        if not Path(_CPU_LAMMPS_BIN).exists():
            log.warning("[h00_design] CPU LAMMPS not found: %s — skipping FF test",
                        _CPU_LAMMPS_BIN)
            return []

        sim      = yaml_data.get("simulation", {})
        mc_cmd   = sim.get("molecule_counts_cmd") or {}
        cc_cmd   = sim.get("chain_counts_cmd")    or {}
        cmd_dirs = yaml_data.get("cmd_dirs", []) or sim.get("cmd_dirs", [])

        if not cmd_dirs or not mc_cmd:
            return []

        # All cmd_dirs share the same aimd-scale test system (molecule_counts_aimd).
        # Run the LAMMPS test once and broadcast the result to every dir.
        first_dir = cmd_dirs[0]
        passed, msg = self._test_one_composition(
            project_dir, yaml_data, first_dir, mc_cmd, cc_cmd)
        status = "PASS" if passed else "FAIL"
        log.info("[h00_design] FF test %s (representative run) — %s: %s",
                 status, first_dir, msg)
        if len(cmd_dirs) > 1:
            log.info("[h00_design] Broadcasting FF test result to all %d cmd_dirs",
                     len(cmd_dirs))
        return [(rel_dir, passed, msg) for rel_dir in cmd_dirs]

    def _test_one_composition(self, project_dir: Path, yaml_data: dict,
                               rel_dir: str, mc_cmd: dict,
                               cc_cmd: dict) -> tuple[bool, str]:
        """Build a small test cell for one cmd_dir and run 200-step NVT.

        Uses only small molecules (non-chain) so the box stays small enough
        for Ewald (requires box > 2 × (cutoff + skin) = 26 Å).
        Returns (passed, message).
        """
        sim = yaml_data.get("simulation", {})

        # Always use aimd counts for the FF test — they give a small, fast cell.
        # mc_cmd counts are full-scale (thousands of molecules) and make the
        # test run for minutes. mc_aimd counts give ~100 atoms in a ~30 Å box.
        mc_aimd  = sim.get("molecule_counts_aimd") or {}
        cc_aimd  = sim.get("chain_counts_aimd")    or {}
        mc_small = {n: v for n, v in mc_aimd.items() if not cc_aimd.get(n)}
        cc_small: dict = {}
        if not mc_small:
            # Final fallback: use cmd counts filtered to small molecules
            mc_small = {n: v for n, v in mc_cmd.items() if not cc_cmd.get(n)}
        if not mc_small:
            return False, "No small-molecule species found for FF test"

        # Enforce ≥ 30 Å box (Ewald requires box > 2×(11+2) = 26 Å)
        sim_rho  = sim.get("target_density_gcm3", 1.0)
        mw_small = sum(n * _MW.get(name, 100.0) for name, n in mc_small.items())
        vol_A3   = mw_small / sim_rho / 6.022e23 * 1e24
        box_test = max(30.0, vol_A3 ** (1.0 / 3.0))

        safe        = rel_dir.replace("/", "_").replace("\\", "_")
        test_dir    = project_dir / "design" / f"test_cmd_{safe}"
        test_dir.mkdir(parents=True, exist_ok=True)
        system_data = test_dir / "system.data"
        lammps_in   = test_dir / "in.test.lammps"
        log_file    = test_dir / "test.log"

        if not system_data.exists():
            log.info("[h00_design] Building test system for %s: %s  box=%.1f Å",
                     rel_dir, mc_small, box_test)
            mol_data = self._build_mol_data(project_dir, mc_small, cc_small, yaml_data)
            if not mol_data:
                return False, "Could not assemble test system (MolData failed)"
            try:
                from hpca.sim.forcefield import build_mixed_system
                build_mixed_system(mol_data, system_data, box_size=box_test)
                log.info("[h00_design] Test system.data: %d atoms",
                         sum(len(m.atoms) * m.count for m in mol_data))
            except Exception as exc:
                return False, f"build_mixed_system failed: {exc}"

        # Get element list from data file for dump_modify
        elems    = _parse_elements_from_data(system_data)
        elem_str = " ".join(elems) if elems else "C H O N S F Li"

        lammps_in.write_text(f"""\
# HPCA h00_design — FF verification test (OPLS-AA)
# minimize → 200-step NVT

# Structure
units           real
boundary        p p p
atom_style      full

# Variables
variable read_data_file string "system.data"
variable dump_file1 string "test_dump_unwrapped.lmp"
variable dump_file2 string "test_dump.lmp"
variable T equal 300
variable timestep equal 0.5
variable thermo_freq equal 50
variable dump_freq equal 50

# Force field
pair_style      lj/cut/coul/long 11.0 11.0
kspace_style    ewald 1.0e-4
bond_style      harmonic
angle_style     harmonic
dihedral_style  opls
improper_style  cvff

read_data       ${{read_data_file}}

special_bonds lj/coul 0.0 0.0 0.5 coul 0.0 0.0 1.0 angle yes dihedral yes

neighbor     2.0 bin
neigh_modify every 1 delay 0 check yes

timestep     ${{timestep}}
thermo_style custom step temp pe etotal
thermo       ${{thermo_freq}}

# Stage 1: minimize to remove grid-placement overlaps
min_style    cg
minimize     1.0e-4 1.0e-6 500 5000

reset_timestep 0

dump 1 all custom ${{dump_freq}} ${{dump_file1}} id mol type element xu yu zu
dump_modify 1 element {elem_str}
dump 2 all custom ${{dump_freq}} ${{dump_file2}} id mol type element x y z
dump_modify 2 element {elem_str}

velocity all create ${{T}} 87287 loop geom

# Stage 2: short NVT to verify stable dynamics
fix nvt all nvt temp ${{T}} ${{T}} 100.0
run 200
unfix nvt

write_data   system_test_final.data nocoeff
""")

        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{_CONDA_ENV_LIB}:{env.get('LD_LIBRARY_PATH', '')}"
        env["OMPI_MCA_btl"]    = "^openib"
        env["OMPI_MCA_osc"]    = "pt2pt"
        env["OMP_NUM_THREADS"] = "4"

        log.info("[h00_design] Running 200-step NVT test for %s ...", rel_dir)
        try:
            result = subprocess.run(
                [_CPU_LAMMPS_BIN, "-in", "in.test.lammps",
                 "-log", "test.log", "-screen", "none"],
                cwd=str(test_dir),
                env=env,
                capture_output=True, text=True, timeout=300,
            )
        except subprocess.TimeoutExpired:
            return False, "Test timed out (>300 s)"
        except Exception as exc:
            return False, f"Subprocess error: {exc}"

        log_text = log_file.read_text() if log_file.exists() else result.stdout

        if "ERROR" in log_text or "Lost atoms" in log_text:
            for line in log_text.splitlines():
                if "ERROR" in line or "Lost atoms" in line:
                    return False, f"LAMMPS error: {line.strip()}"
            return False, f"LAMMPS error (see {test_dir}/test.log)"

        final_pe = None
        final_T  = None
        for line in log_text.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                try:
                    step = int(parts[0])
                    if step >= 150:
                        final_T  = float(parts[1])
                        final_pe = float(parts[2])
                except (ValueError, IndexError):
                    pass

        if result.returncode != 0 and final_pe is None:
            return False, f"LAMMPS exited {result.returncode}"

        if final_pe is not None:
            return True, (f"200 steps  T={final_T:.0f} K  PE={final_pe:.1f} kcal/mol")
        return True, "200 steps completed"

    def _auto_approve_or_kill(self, project_dir: Path,
                               state: "ProjectState",
                               test_results: list[tuple[str, bool, str]]) -> bool:
        """Auto-approve on all-pass; kill daemon + cancel SLURM jobs on any failure.

        Returns True if all tests passed (flag written), False otherwise.
        """
        import signal as _signal

        all_pass = all(passed for _, passed, _ in test_results)

        if all_pass:
            flag = _designed_structures(project_dir) / "simulation_approved.flag"
            flag.touch()
            log.info("[h00_design] All FF tests PASSED — simulation_approved.flag written")
            return True

        failures = [(d, m) for d, p, m in test_results if not p]
        log.error("[h00_design] FF test FAILED for %d composition(s):", len(failures))
        for rel_dir, msg in failures:
            log.error("[h00_design]   %s — %s", rel_dir, msg)
        log.error("[h00_design] Fix errors, then run: hpca resume")

        # Cancel all tracked SLURM jobs
        all_jobs: list[str] = []
        for hdata in state.state.get("handlers", {}).values():
            jobs = hdata.get("jobs", {})
            if isinstance(jobs, dict):
                all_jobs.extend(v for v in jobs.values() if v)
            jid = hdata.get("job")
            if isinstance(jid, str) and jid:
                all_jobs.append(jid)
        for jid in set(all_jobs):
            try:
                from hpca.scheduler import get_scheduler
                if get_scheduler().cancel(str(jid)):
                    log.info("[h00_design] cancelled job %s", jid)
            except Exception:
                pass

        # Persist FAILED state before signalling shutdown
        state.set_stage("h00_design", "FAILED",
                        error="FF test failed — fix errors and run hpca resume",
                        failed_at=datetime.now().isoformat())

        # Signal orchestrator to stop cleanly (SIGTERM handler sets _shutdown=True)
        log.warning("[h00_design] Sending SIGTERM to orchestrator (PID %d)",
                    os.getpid())
        os.kill(os.getpid(), _signal.SIGTERM)
        return False

    # ── Design completion gate ─────────────────────────────────────────────────

    def _write_design_complete(self, project_dir: Path, yaml_data: dict,
                                cmd_systems: list | None = None,
                                test_results: list | None = None,
                                test_result: tuple | None = None) -> None:
        """Write designed_structures/DESIGN_COMPLETE.md summarising what was built and how to approve."""
        design_dir = _designed_structures(project_dir)
        design_dir.mkdir(parents=True, exist_ok=True)

        name        = yaml_data.get("name", project_dir.name)
        category    = yaml_data.get("category", "")
        system_type = yaml_data.get("system_type", "")
        sim         = yaml_data.get("simulation", {})
        benchmark   = yaml_data.get("benchmark", {})

        # ── CMD inventory ─────────────────────────────────────────────────────
        cmd_dirs   = yaml_data.get("cmd_dirs", []) or sim.get("cmd_dirs", [])
        cmd_temps  = sim.get("cmd_temps", [])

        # cmd_systems is a pre-built list from _prebuild_cmd_systems(); fall back
        # to scanning disk if it was not passed (e.g. non-CMD project types).
        if cmd_systems:
            built = list(cmd_systems)
        else:
            built = []
            for rel_dir in cmd_dirs:
                system_data = project_dir / rel_dir / "system.data"
                if system_data.exists():
                    n_atoms = _read_atom_count(system_data)
                    built.append((rel_dir, "BUILT", n_atoms))
                else:
                    built.append((rel_dir, "pending (built after approval)", "?"))

        # ── AIMD inventory ────────────────────────────────────────────────────
        aimd_dirs  = yaml_data.get("aimd_dirs", [])
        aimd_built = sum(1 for d in aimd_dirs if (project_dir / d / "POSCAR").exists())

        # ── AIMD cell composition ─────────────────────────────────────────────
        mc_aimd = sim.get("molecule_counts_aimd") or {}
        ch_aimd = sim.get("chain_counts_aimd")    or {}
        aimd_species_lines = []
        if mc_aimd:
            for sp, n in mc_aimd.items():
                n_ch = ch_aimd.get(sp)
                suffix = f" ({n_ch} chain{'s' if n_ch!=1 else ''})" if n_ch else ""
                aimd_species_lines.append(f"  - {sp}: {n} molecules/units{suffix}")

        # ── Job counts and HPC cost ───────────────────────────────────────────
        n_cmd_jobs   = len(cmd_dirs) * len(cmd_temps) * 2   # NPT + NVT per T
        n_aimd_jobs  = len(aimd_dirs)
        est_aus_cmd  = n_cmd_jobs  * 52  * 12    # 52 CPU, ~12 h/job
        est_aus_aimd = n_aimd_jobs * 104 * 72    # 2 nodes × 52 CPU, 72 h

        # ── NPT protocol text ─────────────────────────────────────────────────
        is_solid = _cat_is_crystalline(category)
        equil_steps = sim.get("cmd_equil_steps", 100_000)
        prod_steps  = sim.get("cmd_prod_steps",  2_000_000)
        dt_fs   = 1.0 if is_solid else 2.0
        equil_ns = equil_steps * dt_fs * 1e-6
        prod_ns  = prod_steps  * dt_fs * 1e-6

        if is_solid:
            npt_rows = [
                "| Stage | Ensemble | Temperature | Duration |",
                "|-------|----------|-------------|----------|",
                "| 1 | Minimize | — | energy minimization |",
                f"| 2 | NVT | 2 × T_sim | {equil_steps//5:,} steps (heat) |",
                f"| 3 | NPT | 2 × T_sim | {equil_steps*3//10:,} steps (high-T equil) |",
                f"| 4 | NPT | T_sim | {equil_steps//2:,} steps (cool + equil) |",
                f"| 5 | NVT | T_sim | {prod_steps:,} steps ({prod_ns:.2f} ns, production) |",
            ]
            npt_header = "## CMD Protocol (Solid — 4-stage NPT, 1 fs timestep)"
        else:
            npt_rows = [
                "| Step | Ensemble | Duration |",
                "|------|----------|----------|",
                f"| Equilibration | NPT, 1 bar | {equil_steps:,} steps ({equil_ns:.2f} ns) |",
                f"| Production    | NVT        | {prod_steps:,} steps ({prod_ns:.2f} ns) |",
            ]
            npt_header = f"## CMD Protocol (Liquid/Gel — single-stage NPT, {dt_fs:.0f} fs timestep)"

        # ── Assemble markdown ─────────────────────────────────────────────────
        lines = [
            f"# Design Complete — {name}",
            "",
            f"**Category:** {category}  |  **System type:** {system_type or '—'}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]
        if benchmark:
            lines += [
                "## Benchmark Reference",
                f"- Paper: {benchmark.get('paper', '')}",
                f"- DOI: {benchmark.get('doi', '')}",
                "",
            ]

        lines += ["## Systems (CMD directories)", ""]
        if built:
            lines += ["| Directory | Status | Atoms |", "|-----------|--------|-------|"]
            for rel_dir, status, n_atoms in built:
                lines.append(f"| `{rel_dir}` | {status} | {n_atoms} |")
        else:
            lines.append("*(no cmd_dirs defined — check project.yaml)*")

        lines += [
            "",
            f"**AIMD starting POSCARs:** {aimd_built}/{len(aimd_dirs)} written",
        ]
        if aimd_species_lines:
            lines += ["**AIMD cell composition:**"] + aimd_species_lines

        # ── Test CMD result (per composition) ────────────────────────────────
        if test_results:
            all_pass = all(p for _, p, _ in test_results)
            hdr_icon = "✓ ALL PASS" if all_pass else "✗ FAILED"
            lines += [
                "",
                "## Force-Field Verification (200-step NVT per composition)",
                "",
                f"**Overall:** {hdr_icon}",
                "",
                "| Composition | Result | Details |",
                "|-------------|--------|---------|",
            ]
            for rd, passed, msg in test_results:
                icon = "✓ PASS" if passed else "✗ FAIL"
                lines.append(f"| `{rd}` | {icon} | {msg} |")
            if not all_pass:
                lines += [
                    "",
                    "> **Action required:** Fix LAMMPS errors in "
                    "`design/test_cmd_*/test.log`, then run `hpca resume` to retry.",
                ]
        elif test_result is not None:
            passed, msg = test_result
            icon = "✓ PASS" if passed else "✗ FAIL"
            lines += [
                "",
                "## Force-Field Verification",
                "",
                f"**Test result:** {icon}  —  {msg}",
            ]
            if not passed:
                lines += [
                    "",
                    "> **Action required:** Check `design/test_cmd/test.log`, "
                    "then run `hpca resume`.",
                ]

        lines += [
            "",
            "## Simulation Schedule",
            "",
            "| Track | Method | When | Jobs | Est. cost |",
            "|-------|--------|------|------|-----------|",
            f"| **A — CMD**  | OPLS-AA LAMMPS | **immediately after approval** "
            f"| {n_cmd_jobs} | ~{est_aus_cmd:,} AUs |",
            f"| **B — AIMD** | VASP NVT       | in parallel with A "
            f"| {n_aimd_jobs} | ~{est_aus_aimd:,} AUs |",
            "| **C — MLIP** | DeepMD training | after B completes | 1 | ~576 AUs (GPU) |",
            "| **D — MLMD** | DeepMD LAMMPS  | after C completes | TBD | TBD |",
            "",
            npt_header,
            "",
        ] + npt_rows + [
            "",
            "OPLS-AA force field, LJ 14 Å cutoff, Ewald long-range electrostatics.",
            "",
            "## Next Step",
            "",
            "When all FF tests pass, `simulation_approved.flag` is created automatically.",
            "The orchestrator immediately starts all tracks.",
            "",
            "To approve manually (e.g., after fixing errors and re-running):",
            "```bash",
            f"touch {project_dir}/designed_structures/simulation_approved.flag",
            "```",
            "",
            "To resume after a failure:",
            "```bash",
            "hpca resume          # daemon (login-node) mode",
            "hpca resume --slurm  # SLURM submission mode",
            "```",
            "",
            "---",
            "*Generated by hpca h00_design MaterialsDesignHandler*",
        ]

        md_path = design_dir / "DESIGN_COMPLETE.md"
        md_path.write_text("\n".join(lines) + "\n")
        log.info("[h00_design] Design summary written to %s", md_path)

    # ── Polymer systems ────────────────────────────────────────────────────────

    def _build_polymer_system(self, project_dir: Path, yaml_data: dict) -> None:
        """Pack a polymer-salt PACKMOL box and write POSCAR/LAMMPS data to design/."""
        design_dir = project_dir / "design"
        design_dir.mkdir(parents=True, exist_ok=True)
        if MATDESIGN_SRC not in sys.path:
            sys.path.insert(0, MATDESIGN_SRC)

        polymer_name = yaml_data.get("polymer_name", "PEO")
        n_units  = yaml_data.get("n_units", 30)
        n_chains = yaml_data.get("n_chains", 4)
        salt     = yaml_data.get("salt", "LiFSI")
        salt_count = yaml_data.get("salt_count", 8)
        solvents = yaml_data.get("solvents", {})
        box_size = yaml_data.get("box_size", 40.0)

        composition = {polymer_name: n_chains, salt: salt_count}
        composition.update(solvents)
        log.info("[h00_design] Building polymer system: %s", composition)

        try:
            from matdesign.electrolyte.packing import pack_electrolyte_from_names
            packed = pack_electrolyte_from_names(
                composition, box_size=box_size, polymer_n_units=n_units)
            system_lmp = design_dir / "system.lmp"
            from ase.io import write
            write(str(system_lmp), packed, format="lammps-data")
            log.info("[h00_design] Wrote %s", system_lmp)
        except ImportError as exc:
            log.warning("[h00_design] matdesign not available (%s) — placeholder", exc)
            _write_placeholder_lammps(design_dir / "system.lmp", polymer_name, n_chains)

        if yaml_data.get("design_matrix"):
            self._build_design_matrix(project_dir, yaml_data)

        log.info("[h00_design] Polymer system complete for %s", project_dir.name)

    # ── MLIP pre-relaxation ────────────────────────────────────────────────────

    def _mlip_prerelax(self, poscar_path: Path, category: str,
                       yaml_data: dict | None = None, tier: str = "dft") -> None:
        """Relax poscar_path in-place with MACE-MPA-0 via a subprocess.

        Uses the deepmd-lammps-gpu conda env which has MACE installed.
        fmax=0.1 for liquids/polymers, fmax=0.05 for solids.
        Timeout=300 s for liquids, 600 s for solids.
        On failure: logs a warning and leaves POSCAR untouched (restored by script).
        """
        from hpca.core.preoptimization import decide_preoptimization
        decision = decide_preoptimization(
            poscar_path, category, yaml_data or {}, self.platform_config(),
            generated_structure=_cat_is_molecular(category),
        )
        self._record_preopt_decision(poscar_path.parents[2] if tier == "dft" else poscar_path.parents[1],
                                    tier, decision)
        if not decision.run_mace:
            log.info("[h00_design] MACE pre-relax skipped for %s: %s", tier, decision.reason)
            return

        is_liquid_or_polymer = _cat_is_molecular(category)
        fmax    = "0.1"  if is_liquid_or_polymer else "0.05"
        policy = {**self.platform_config().get("preoptimization", {}),
                  **((yaml_data or {}).get("preoptimization", {}) or {})}
        timeout = min(300 if is_liquid_or_polymer else 600,
                      int(decision.runtime_limit_s))
        steps = int(policy.get("steps", 1000))

        _py = self.hpc_path("python_deepmd") or self.hpc_path("python_cladue") or sys.executable
        cmd = [_py, PRERELAX_SCRIPT, str(poscar_path),
               f"fmax={fmax}", f"steps={steps}", "device=cpu"]
        log.info("[h00_design] MACE pre-relax: %s (fmax=%s, timeout=%ds)",
                 poscar_path, fmax, timeout)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode == 0:
                summary = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "OK"
                log.info("[h00_design] %s → %s", poscar_path.name, summary)
            else:
                log.warning("[h00_design] MACE pre-relax failed for %s (rc=%d): %s",
                            poscar_path, result.returncode,
                            result.stderr.strip()[-300:] if result.stderr else "")
        except subprocess.TimeoutExpired:
            log.warning("[h00_design] MACE pre-relax timed out (%ds) for %s — continuing",
                        timeout, poscar_path)
        except FileNotFoundError:
            log.warning("[h00_design] python not found at %s — skipping MACE pre-relax", _py)
        except Exception as exc:
            log.warning("[h00_design] MACE pre-relax error for %s: %s — continuing", poscar_path, exc)
        # Remove the .orig backup created by prerelax_mace.py regardless of success/failure
        Path(str(poscar_path) + ".orig").unlink(missing_ok=True)

    # ── Amorphous structure design ─────────────────────────────────────────────

    def _build_amorphous_system(self, project_dir: Path, yaml_data: dict) -> None:
        """Generate an amorphous structure from a crystal via MACE melt-quench.

        project.yaml keys:
          design_mode: amorphous
          input_structure: <mp_id or /path/to/file.cif>   # crystal to amorphize
          amorphous_T_melt: 3000     # K, melting temperature (default 3000)
          amorphous_T_quench: 300    # K, quench temperature (default 300)
          amorphous_steps_melt: 500  # MD steps at T_melt (default 500)
          amorphous_steps_quench: 300 # MD steps during quench (default 300)
          supercell: [2, 2, 2]       # optional supercell before amorphizing

        Output: design/amorphous/POSCAR  (amorphous structure ready for AIMD)
        """
        AMORPHIZE_SCRIPT = str(Path(__file__).parents[1] / "amorphize_mace.py")
        amorphous_dir = project_dir / "design" / "amorphous"
        amorphous_dir.mkdir(parents=True, exist_ok=True)
        out_poscar = amorphous_dir / "POSCAR"

        if out_poscar.exists():
            log.info("[h00_design] Amorphous POSCAR already exists: %s", out_poscar)
            return

        # Build crystal structure first
        input_struct = yaml_data.get("input_structure", "")
        crystal_poscar = amorphous_dir / "crystal_POSCAR"

        if input_struct.startswith("mp-"):
            struct = self._fetch_from_mp(input_struct)
        elif input_struct and Path(input_struct).exists():
            struct = self._load_from_cif(input_struct)
        else:
            log.error("[h00_design] amorphous design needs input_structure (mp_id or CIF path)")
            return

        supercell = yaml_data.get("supercell", [2, 2, 2])
        if struct is not None:
            struct.make_supercell(supercell)
            crystal_poscar.write_text(struct.to(fmt="poscar"))
        else:
            log.error("[h00_design] Could not load input_structure for amorphous design")
            return

        T_melt   = yaml_data.get("amorphous_T_melt", 3000)
        T_quench = yaml_data.get("amorphous_T_quench", 300)
        steps_melt   = yaml_data.get("amorphous_steps_melt", 500)
        steps_quench = yaml_data.get("amorphous_steps_quench", 300)

        _py = self.hpc_path("python_deepmd") or self.hpc_path("python_cladue") or sys.executable
        cmd = [
            _py, AMORPHIZE_SCRIPT,
            str(crystal_poscar),
            f"T_melt={T_melt}", f"T_quench={T_quench}",
            f"steps_melt={steps_melt}", f"steps_quench={steps_quench}",
            f"out={out_poscar}",
            "device=cpu",
        ]
        log.info("[h00_design] MACE melt-quench: %s → amorphous (T_melt=%dK, T_quench=%dK)",
                 crystal_poscar.name, T_melt, T_quench)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode == 0:
                log.info("[h00_design] Amorphous structure generated: %s", out_poscar)
                # Copy result to main design dir for downstream handlers
                main_poscar = project_dir / "design" / "POSCAR"
                if out_poscar.exists():
                    import shutil
                    shutil.copy(out_poscar, main_poscar)
            else:
                log.error("[h00_design] amorphize_mace.py failed (rc=%d): %s",
                          result.returncode, result.stderr[-500:] if result.stderr else "")
        except subprocess.TimeoutExpired:
            log.error("[h00_design] amorphize_mace.py timed out (1800s) — structure too large?")
        except Exception as exc:
            log.error("[h00_design] amorphize_mace.py error: %s", exc)

    # ── Nano-particle deposited substrate designer ─────────────────────────────

    def _build_nanoparticle_substrate(self, project_dir: Path, yaml_data: dict) -> None:
        """Build substrate + nanoparticle VASP structures via build_substrate_np.py API.

        Two scale modes controlled by np_scale:
          "aimd"  (default when np_radius < 8): 2 tiny NPs on thin substrate.
                  Target < 200 atoms for feasible DFT/AIMD dataset generation.
                  Defaults: np_radius=5, nx=1, ny=2, z_reps=1, boundary_pad=5, vacuum=15
          "mlmd"  (default when np_radius ≥ 8): 2×2 NPs on full slab for MLMD.
                  Defaults: np_radius=20, nx=2, ny=2, z_reps=4, boundary_pad=25, vacuum=30

        project.yaml keys:
          design_mode: nanoparticle_substrate
          substrate_file: /path/to/substrate.vasp   (required)
          bulk_file: /path/to/bulk.vasp             (required — NP material)
          np_scale: aimd          # "aimd" or "mlmd"
          np_radius: 5.0          # Å — 5 Å ≈ 10-30 atoms (aimd); 20 Å for mlmd
          np_distances: [3, 5, 8] # surface-to-surface NP gaps (Å)
          np_nx: 1                # 1 column of NPs
          np_ny: 2                # 2 rows → 2 particles total (aimd)
          np_z_gap: 2.0           # substrate-to-NP gap (Å)
          np_boundary_pad: 5.0    # cell edge margin (Å)
          np_z_reps: 1            # 1 = use pre-built slab; 4 = tile unit cell
          np_vacuum: 15.0         # vacuum on both z sides (Å)

        Outputs:
          design/nanoparticle_substrate/<prefix>_d{X}A.vasp — one per distance
          Atom counts logged; AIMD-feasible structures (≤200 atoms) flagged.
        """
        out_dir = project_dir / "design" / "nanoparticle_substrate"
        out_dir.mkdir(parents=True, exist_ok=True)

        substrate_file = yaml_data.get("substrate_file", "")
        bulk_file      = yaml_data.get("bulk_file", "")
        if not substrate_file or not bulk_file:
            log.error("[h00_design] nanoparticle_substrate requires substrate_file and bulk_file")
            return
        if not Path(substrate_file).exists():
            log.error("[h00_design] substrate_file not found: %s", substrate_file)
            return
        if not Path(bulk_file).exists():
            log.error("[h00_design] bulk_file not found: %s", bulk_file)
            return

        # Scale-mode defaults
        np_radius = yaml_data.get("np_radius", None)
        np_scale  = yaml_data.get("np_scale", "aimd" if (np_radius or 5) < 8 else "mlmd")
        if np_radius is None:
            np_radius = 5.0 if np_scale == "aimd" else 20.0

        if np_scale == "aimd":
            # Small: 2 particles (1×2 grid), thin slab, small vacuum
            defaults = dict(nx=1, ny=2, z_reps=1, boundary_pad=5.0, vacuum=15.0,
                            z_gap=2.0, distances=[3, 5, 8])
        else:
            # Full MLMD scale: 4 particles (2×2), full slab
            defaults = dict(nx=2, ny=2, z_reps=4, boundary_pad=25.0, vacuum=30.0,
                            z_gap=4.0, distances=[5, 10, 15, 20, 25])

        distances    = yaml_data.get("np_distances", defaults["distances"])
        nx           = yaml_data.get("np_nx",          defaults["nx"])
        ny           = yaml_data.get("np_ny",          defaults["ny"])
        z_gap        = yaml_data.get("np_z_gap",       defaults["z_gap"])
        boundary_pad = yaml_data.get("np_boundary_pad", defaults["boundary_pad"])
        z_reps       = yaml_data.get("np_z_reps",      defaults["z_reps"])
        vacuum       = yaml_data.get("np_vacuum",       defaults["vacuum"])

        log.info("[h00_design] Building substrate+NP structures: scale=%s, radius=%.1f Å, "
                 "distances=%s, grid=%dx%d", np_scale, np_radius, distances, nx, ny)
        if np_scale == "aimd":
            log.info("[h00_design] AIMD mode: targeting ≤200 atoms for DFT dataset generation")

        try:
            from hpca.core.np_builder import build_substrate_with_nanoparticles as _build_np
            output_files = _build_np(
                substrate_file=substrate_file,
                bulk_file=bulk_file,
                np_radius=np_radius,
                distances=distances,
                n_x=nx,
                n_y=ny,
                z_gap=z_gap,
                boundary_pad=boundary_pad,
                z_reps=z_reps,
                vacuum=vacuum,
                save_np=True,
                out_dir=out_dir,
            )

            # Count atoms and advise on suitability
            try:
                from ase.io import read as _ase_read
                for fpath in output_files:
                    atoms = _ase_read(str(out_dir / fpath), format="vasp")
                    n = len(atoms)
                    suitability = "AIMD-ok" if n <= 200 else ("MLMD-ok" if n <= 2000 else "large")
                    log.info("[h00_design]   %s — %d atoms [%s]", fpath, n, suitability)
                    if np_scale == "aimd" and n > 200:
                        log.warning("[h00_design]   %s has %d atoms — too large for AIMD. "
                                    "Reduce np_radius or np_nx/ny.", fpath, n)
            except Exception:
                log.info("[h00_design] NP substrate structures written: %s", output_files)

        except Exception as exc:
            log.error("[h00_design] build_substrate_np failed: %s", exc)

    def _build_design_matrix(self, project_dir: Path, yaml_data: dict) -> None:
        """Write combinatorial chain-length × salt-count parameter grid to design/lpifd_matrix/."""
        matrix_dir = project_dir / "design" / "lpifd_matrix"
        matrix_dir.mkdir(parents=True, exist_ok=True)
        chain_lengths = yaml_data.get("chain_lengths", [10, 20, 30, 40, 50])
        salt_counts   = yaml_data.get("salt_counts",   [2, 4, 6, 8, 10, 12, 14])
        polymer_name  = yaml_data.get("polymer_name", "PEO")
        salt          = yaml_data.get("salt", "LiFSI")
        log.info("[h00_design] Building design matrix %dx%d",
                 len(chain_lengths), len(salt_counts))
        for n_units in chain_lengths:
            for n_salt in salt_counts:
                entry_dir = matrix_dir / f"n{n_units}_s{n_salt}"
                entry_dir.mkdir(exist_ok=True)
                (entry_dir / "design_params.yaml").write_text(
                    f"polymer_name: {polymer_name}\nn_units: {n_units}\n"
                    f"salt: {salt}\nsalt_count: {n_salt}\nn_chains: 4\nbox_size: 40.0\n")
        log.info("[h00_design] Design matrix written to %s", matrix_dir)


# ── Module-level helpers ───────────────────────────────────────────────────────

def _ensure_cladue_env() -> None:
    """Prepend the cladue conda site-packages directory to sys.path if not already present."""
    cladue_site = load_platform_config().get("hpc", {}).get("cladue_site_packages", "")
    if cladue_site and cladue_site not in sys.path:
        sys.path.insert(0, cladue_site)


def _placeholder_poscar(name: str, n_molecules: int, box: float) -> str:
    """Return a minimal single-H POSCAR string flagged as placeholder when PACKMOL is unavailable."""
    return (
        f"{name} (PLACEHOLDER — PACKMOL unavailable, ~{n_molecules} molecules)\n"
        "1.0\n"
        f"  {box:.6f}  0.000000  0.000000\n"
        f"  0.000000  {box:.6f}  0.000000\n"
        f"  0.000000  0.000000  {box:.6f}\n"
        "H\n1\nCartesian\n  0.000000  0.000000  0.000000\n"
    )


def _grid_placement_poscar(name: str, mol_counts: dict[str, int],
                            input_dir: Path, box: float) -> str | None:
    """Place atoms on a uniform grid without PACKMOL.

    Reads each molecule's VASP file to get the correct element composition,
    then places every atom on an individual grid point. Molecular bonding
    is not preserved, but all inter-atomic distances equal box/n^(1/3) ≈ 2 Å,
    which MACE and VASP can handle. VASP AIMD/relaxation will restructure the
    atoms into molecules during the first ionic steps.

    Returns POSCAR string, or None if any molecule file is missing.
    """
    import math

    # Collect full atom list (element → count) from all molecule files
    all_species: list[str] = []
    for mol_name, count in mol_counts.items():
        if count <= 0:
            continue
        vasp_path = input_dir / f"{mol_name}.vasp"
        if not vasp_path.exists():
            return None
        try:
            from pymatgen.core import Structure
            st = Structure.from_file(str(vasp_path))
            for _ in range(count):
                all_species.extend(str(s.specie) for s in st)
        except Exception:
            return None

    n_atoms = len(all_species)
    if n_atoms == 0:
        return None

    # Place atoms on a uniform simple-cubic grid; min_dist = box / n_side ≈ 2 Å
    n_side = max(1, math.ceil(n_atoms ** (1.0 / 3.0)))
    stride = box / n_side

    grid_pts: list[tuple[float, float, float]] = []
    for ix in range(n_side):
        for iy in range(n_side):
            for iz in range(n_side):
                grid_pts.append((ix * stride + stride * 0.5,
                                  iy * stride + stride * 0.5,
                                  iz * stride + stride * 0.5))

    # Sort by element for VASP (species must be contiguous)
    order = sorted(range(n_atoms), key=lambda i: all_species[i])
    sorted_sp = [all_species[i] for i in order]
    sorted_pts = grid_pts[:n_atoms]
    sorted_pts = [sorted_pts[i] for i in order]

    seen: dict[str, int] = {}
    for sp in sorted_sp:
        seen[sp] = seen.get(sp, 0) + 1

    n_mols = sum(mol_counts.values())
    lines = [
        f"{name} (grid-placed, no PACKMOL, {n_mols} molecules)",
        "1.0",
        f"  {box:.6f}  0.000000  0.000000",
        f"  0.000000  {box:.6f}  0.000000",
        f"  0.000000  0.000000  {box:.6f}",
        "  ".join(seen.keys()),
        "  ".join(str(v) for v in seen.values()),
        "Cartesian",
    ]
    for x, y, z in sorted_pts:
        lines.append(f"  {x:.6f}  {y:.6f}  {z:.6f}")
    return "\n".join(lines) + "\n"


def _write_placeholder_lammps(path: Path, polymer_name: str, n_chains: int) -> None:
    """Write an empty LAMMPS data file placeholder when polymer packing is unavailable."""
    path.write_text(
        f"# Placeholder LAMMPS data: {polymer_name} x {n_chains} chains\n"
        "# Re-generate with: matdesign.electrolyte.packing.pack_electrolyte_from_names\n"
        "\n0 atoms\n0 bonds\n0 angles\n0 dihedrals\n0 impropers\n"
        "\n0 atom types\n\n0.0 40.0 xlo xhi\n0.0 40.0 ylo yhi\n0.0 40.0 zlo zhi\n"
    )
