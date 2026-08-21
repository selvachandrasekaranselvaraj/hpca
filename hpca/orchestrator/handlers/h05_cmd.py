"""
h05_cmd.py — Classical force-field MD handler (OPLS-AA, LAMMPS).

Workflow:
  1. cmd/npt/     — NPT equilibration at 300 K, up to 200 ps (SLURM)
                    Source: preopt/contcar_cmd_preopt.vasp  OR  designed_structures/system_cmd.data
  2. cmd/nvt/{T}/ — NVT production at each temperature
                    Source: cmd/npt/ final frame

Temperatures (canonical list from platform.yaml limits.nvt_temperatures):
  300, 320, 340, 360, 380, 400, 500, 600 K

Atom limits (from platform.yaml limits):
  slurm: cmd_atoms ≤ 50000

Cross-ref:
  hpca/core/paths.py              — cmd_npt(), cmd_nvt(), contcar_preopt(), designed_structures()
  hpca/config/platform.yaml       — limits and hpc paths
  hpca/orchestrator/handlers/h00_design.py — writes preopt/contcar_cmd_preopt.vasp
"""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from .base import SimulationHandler
from hpca.core.categories import is_crystalline as _cat_is_crystalline
from hpca.core.paths import cmd_npt, cmd_nvt, designed_structures, contcar_preopt, preopt, load_platform_config
from hpca.core.config import account_fallback as _account_fallback

if TYPE_CHECKING:
    from hpca.orchestrator.state_tracker import ProjectState

log = logging.getLogger("hpca.orch")
# Layout: see hpca/core/paths.py

_CPU_TIME_LIMIT     = "48:00:00"
# Paths resolved from platform.yaml at runtime via self.hpc_path()
# Cross-ref: hpca/config/platform.yaml hpc section

_DUMP_MIN_SIZE = 1_000_000   # 1 MB -- minimum valid production dump


def _get_opls_element() -> dict[str, str]:
    """Return OPLS atom-type label → element symbol mapping (lazy-loaded from hpca.data)."""
    from hpca.data import load
    return load("opls_elements")  # type: ignore[return-value]


def _parse_elements_from_data(data_path: Path, n_types: int | None = None) -> list[str]:
    """Return element symbols in atom type ID order by parsing Masses section.

    n_types: if given, truncate result to this many types.  Use this when
    reading system.data (which has OPLS comments) to get labels for a
    write_data output (which collapses types and drops comments).
    """
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
                    elements[type_id] = _get_opls_element().get(opls_type, "C")
    except Exception:
        pass
    result = [elements[k] for k in sorted(elements.keys())]
    if n_types is not None:
        result = result[:n_types]
    return result


def _count_atom_types(data_path: Path) -> int | None:
    """Return the atom types count from a LAMMPS data file header."""
    try:
        for line in data_path.read_text().splitlines()[:30]:
            if "atom types" in line:
                return int(line.split()[0])
    except Exception:
        pass
    return None


class ClassicalMDHandler(SimulationHandler):
    """SLURM handler: classical OPLS-AA LAMMPS MD. Starts as soon as project.yaml is ready."""

    name      = "h05_cmd"
    is_daemon = False

    # Dielectric constants are loaded lazily from hpca.data dielectric_constants.json.
    # For mixtures with >1 component, a mole-fraction-weighted average is used.
    _DEFAULT_DIELECTRIC: float = 5.0   # fallback for unrecognised species

    @staticmethod
    def _get_dielectric() -> dict[str, float]:
        """Return solvent/salt/polymer → dielectric constant mapping (lazy-loaded from hpca.data)."""
        from hpca.data import load
        return load("dielectric_constants")  # type: ignore[return-value]

    # Gate temperature: gate all other temperatures behind this NPT completing first.
    _T_GATE: int = 300

    @classmethod
    def _compute_dielectric(cls, components: list[tuple[str, int]]) -> float:
        """Mole-fraction-weighted average dielectric constant for the mixture.

        Single-component: use that species' value directly.
        Multi-component: weighted average over all species.
        """
        if not components:
            return cls._DEFAULT_DIELECTRIC
        total = sum(count for _, count in components)
        if total == 0:
            return cls._DEFAULT_DIELECTRIC
        dielectric = cls._get_dielectric()
        eps = sum(
            dielectric.get(name, cls._DEFAULT_DIELECTRIC) * count / total
            for name, count in components
        )
        return round(eps, 2)

    @staticmethod
    def _parse_npt_thermo(log_path: Path) -> tuple[float | None, float | None]:
        """Parse the final thermo row from a LAMMPS log.

        thermo_style: step time temp press pe vol density cella cellb cellc
        cols (0-idx):  0    1    2     3    4   5   6        7     8     9

        Returns (density_g_cm3, temp_K) or (None, None) on parse failure.
        """
        try:
            with open(log_path, "rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                fh.seek(max(0, size - 16384))
                tail = fh.read().decode("utf-8", errors="replace")
        except Exception:
            return None, None

        for line in reversed(tail.splitlines()):
            parts = line.split()
            if len(parts) >= 7:
                try:
                    int(parts[0])           # step must be an integer
                    d = float(parts[6])     # density column
                    t = float(parts[2])     # temp column
                    if d > 0 and t > 0:
                        return d, t
                except (ValueError, IndexError):
                    continue
        return None, None

    # Density bounds for liquid/polymer electrolytes (g/cm³).
    _RHO_MIN: float = 0.5   # below this the box expanded into gas/foam
    _RHO_MAX: float = 3.0   # above this is unphysically compressed
    _T_TOL:   int   = 100   # K -- acceptable temperature drift from target
    # Approximate average mass per atom (g/mol) for organic electrolyte mixtures.
    # Used for density estimation from dump box + atom count when system.data unavailable.
    _AVG_MASS_PER_ATOM: float = 10.0

    @staticmethod
    def _dump_density(dump_path: Path, system_data: Path | None = None) -> float | None:
        """Estimate density (g/cm³) from first frame of a LAMMPS dump file.

        Uses exact masses from system.data Masses section when available;
        falls back to a 10 g/mol-per-atom approximation otherwise.
        """
        import re as _re, math as _math
        try:
            with open(dump_path, "rb") as fh:
                header = fh.read(8192).decode("utf-8", errors="replace")
        except Exception:
            return None

        m_n = _re.search(r"NUMBER OF ATOMS\n(\d+)", header)
        m_b = _re.search(
            r"BOX BOUNDS.*?\n"
            r"([\-\d.eE+]+)\s+([\-\d.eE+]+)\n"
            r"([\-\d.eE+]+)\s+([\-\d.eE+]+)\n"
            r"([\-\d.eE+]+)\s+([\-\d.eE+]+)",
            header,
        )
        if not (m_n and m_b):
            return None

        n_atoms = int(m_n.group(1))
        lx = float(m_b.group(2)) - float(m_b.group(1))
        ly = float(m_b.group(4)) - float(m_b.group(3))
        lz = float(m_b.group(6)) - float(m_b.group(5))
        vol_cm3 = lx * ly * lz * 1e-24   # Å³ → cm³

        # Try exact masses from system.data
        total_mass_g = None
        if system_data and system_data.exists():
            try:
                txt = system_data.read_text(errors="replace")
                masses_m = _re.search(r"Masses\s*\n([\s\S]+?)(?:\n[A-Z])", txt)
                atom_types_m = _re.search(r"Atoms\s*\n([\s\S]+?)(?:\n[A-Z]|\Z)", txt)
                if masses_m:
                    type_mass: dict[int, float] = {}
                    for line in masses_m.group(1).strip().splitlines():
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                type_mass[int(parts[0])] = float(parts[1])
                            except ValueError:
                                pass
                    if type_mass and atom_types_m:
                        # Count atoms per type from Atoms section
                        type_count: dict[int, int] = {}
                        for line in atom_types_m.group(1).strip().splitlines():
                            parts = line.split()
                            if len(parts) >= 3:
                                try:
                                    atype = int(parts[2])
                                    type_count[atype] = type_count.get(atype, 0) + 1
                                except ValueError:
                                    pass
                        if type_count:
                            total_mass_g = sum(
                                type_count.get(t, 0) * m / 6.022e23
                                for t, m in type_mass.items()
                            )
            except Exception:
                pass

        if total_mass_g is None:
            total_mass_g = n_atoms * ClassicalMDHandler._AVG_MASS_PER_ATOM / 6.022e23

        if vol_cm3 <= 0:
            return None
        return total_mass_g / vol_cm3

    @staticmethod
    def _check_dump_structure(dump_path: Path, n_sample: int = 300) -> tuple[bool, str]:
        """Check atomic structure from last frame of a LAMMPS dump file.

        Returns (ok, reason_if_bad).  Checks:
          1. Coordinates are finite (no NaN/inf).
          2. No severe intermolecular overlap (min inter-mol dist > 2.0 Å).
          3. No wildly short same-molecule bond-like distances.
        Reads only a sample of atoms for speed; reads the file in reverse to
        find the last frame without loading the whole file.
        """
        import math as _math
        import re as _re

        MIN_INTER_DIST = 2.0   # Å -- below this is severe overlap
        MIN_BOND_DIST  = 0.5   # Å -- below this is clearly wrong

        try:
            # Read last 512 kB to find last frame without loading entire file
            with open(dump_path, "rb") as fh:
                fh.seek(0, 2)
                fsize = fh.tell()
                read_start = max(0, fsize - 524288)
                fh.seek(read_start)
                tail = fh.read().decode("utf-8", errors="replace")
        except Exception as e:
            return True, ""   # can't read → accept conservatively

        # Find last TIMESTEP header in tail
        frames = list(_re.finditer(r"ITEM: TIMESTEP\n", tail))
        if not frames:
            return True, ""
        block = tail[frames[-1].start():]

        # Parse box
        box_m = _re.search(
            r"BOX BOUNDS.*?\n"
            r"([\-\d.eE+]+)\s+([\-\d.eE+]+)\n"
            r"([\-\d.eE+]+)\s+([\-\d.eE+]+)\n"
            r"([\-\d.eE+]+)\s+([\-\d.eE+]+)",
            block,
        )
        box = None
        if box_m:
            lx = float(box_m.group(2)) - float(box_m.group(1))
            ly = float(box_m.group(4)) - float(box_m.group(3))
            lz = float(box_m.group(6)) - float(box_m.group(5))
            if lx > 0 and ly > 0 and lz > 0:
                box = (lx, ly, lz)

        # Parse atom columns
        atoms_m = _re.search(r"ITEM: ATOMS (.+)\n([\s\S]+)", block)
        if not atoms_m:
            return True, ""

        cols_str = atoms_m.group(1).split()
        col_idx = {c: i for i, c in enumerate(cols_str)}
        atoms: list[dict] = []
        for line in atoms_m.group(2).splitlines():
            parts = line.split()
            if len(parts) < len(cols_str):
                continue
            try:
                x = float(parts[col_idx.get("x", col_idx.get("xu", 0))])
                y = float(parts[col_idx.get("y", col_idx.get("yu", 1))])
                z = float(parts[col_idx.get("z", col_idx.get("zu", 2))])
                mol = int(parts[col_idx["mol"]]) if "mol" in col_idx else 0
                atoms.append({"x": x, "y": y, "z": z, "mol": mol})
            except (ValueError, KeyError, IndexError):
                continue
            if len(atoms) >= n_sample:
                break

        if not atoms:
            return True, ""

        # 1. Check finite coordinates
        for at in atoms[:50]:
            for c in ("x", "y", "z"):
                if not _math.isfinite(at[c]):
                    return False, f"NaN/inf coordinate in dump last frame"

        # 2. Check minimum intermolecular distances (sample first 60 atoms)
        def mic(a, b):
            """Minimum-image-convention distance between two atom dicts."""
            dx, dy, dz = a["x"]-b["x"], a["y"]-b["y"], a["z"]-b["z"]
            if box:
                dx -= box[0]*round(dx/box[0])
                dy -= box[1]*round(dy/box[1])
                dz -= box[2]*round(dz/box[2])
            return _math.sqrt(dx*dx + dy*dy + dz*dz)

        half = atoms[:60]
        min_inter = float("inf")
        for i, a in enumerate(half):
            for b in half[i+1:i+10]:
                if a["mol"] != b["mol"] and a["mol"] > 0 and b["mol"] > 0:
                    d = mic(a, b)
                    if d < min_inter:
                        min_inter = d

        if min_inter < MIN_INTER_DIST:
            return False, (
                f"severe intermolecular overlap: min dist {min_inter:.2f} Å "
                f"< {MIN_INTER_DIST} Å in last dump frame"
            )

        # 3. Check for absurdly short intra-molecular distances
        by_mol: dict[int, list] = {}
        for at in atoms[:200]:
            by_mol.setdefault(at["mol"], []).append(at)
        too_short = 0
        checked = 0
        for mol_ats in list(by_mol.values())[:30]:
            for i, a in enumerate(mol_ats):
                for b in mol_ats[i+1:i+4]:
                    d = mic(a, b)
                    if 0 < d < MIN_BOND_DIST:
                        too_short += 1
                    checked += 1
        if checked > 0 and too_short / checked > 0.05:
            return False, (
                f"many very-short intramolecular distances "
                f"({too_short}/{checked} pairs < {MIN_BOND_DIST} Å)"
            )

        return True, ""

    @staticmethod
    def _validate_nvt_dump(nvt_dir: Path) -> bool:
        """Return True if the NVT dump has physically reasonable density.

        Checks dump_unwrapped.lmp (or dump.lmp) first frame against _RHO_MIN.
        A dump from a gas-phase NPT starting point will have density < _RHO_MIN.
        """
        for dname in ("dump_unwrapped.lmp", "dump.lmp", "dump_nvt.lmp"):
            dp = nvt_dir / dname
            if dp.exists() and dp.stat().st_size >= _DUMP_MIN_SIZE:
                # Look for system.data three levels up (nvt/ → T/ → comb/ → system.data)
                sys_data = nvt_dir.parent.parent / "system.data"
                rho = ClassicalMDHandler._dump_density(dp, sys_data)
                if rho is None:
                    return True   # can't check -- accept
                rmin = ClassicalMDHandler._RHO_MIN
                if rho < rmin:
                    log.warning(
                        "[h05_cmd] NVT dump gas-phase: ρ≈%.3f g/cm³ < %.1f in %s",
                        rho, rmin, nvt_dir,
                    )
                    return False
                return True
        return False  # no dump found

    @staticmethod
    def _validate_npt(npt_dir: Path, T_target: int = 300) -> bool:
        """Return True if the NPT output is physically reasonable.

        Checks (in order):
          1. minimized_structure.dat exists and is >10 kB
          2. No 'ERROR' in LAMMPS log tail
          3. Final density is within _RHO_MIN … _RHO_MAX g/cm³
          4. Final temperature within _T_TOL K of T_target
        """
        dat = npt_dir / "minimized_structure.dat"
        if not dat.exists() or dat.stat().st_size < 10_000:
            return False

        log_path: Path | None = None
        for lname in ("log.lammps", "lammps.out", "out"):
            lf = npt_dir / lname
            if lf.exists():
                log_path = lf
                break

        if log_path is None:
            return True  # no log to verify -- accept conservatively

        try:
            tail = log_path.read_text()[-8192:]
            if "ERROR" in tail:
                log.warning("[h05_cmd] NPT log has ERROR: %s", log_path)
                return False
            # Charge neutrality warning signals wrong counterion count in system.data.
            if "System is not charge neutral" in tail:
                log.warning("[h05_cmd] NPT system not charge neutral: %s", log_path)
                return False
        except Exception:
            pass

        density, temp = ClassicalMDHandler._parse_npt_thermo(log_path)

        if density is not None:
            rmin = ClassicalMDHandler._RHO_MIN
            rmax = ClassicalMDHandler._RHO_MAX
            if density < rmin:
                log.warning("[h05_cmd] NPT gas-phase/underdense: ρ=%.4f g/cm³ < %.1f in %s",
                            density, rmin, npt_dir)
                return False
            if density > rmax:
                log.warning("[h05_cmd] NPT unphysical density: ρ=%.4f g/cm³ in %s",
                            density, npt_dir)
                return False

        if temp is not None and abs(temp - T_target) > ClassicalMDHandler._T_TOL:
            log.warning("[h05_cmd] NPT temperature drift: T_final=%.1f vs T_target=%d in %s",
                        temp, T_target, npt_dir)
            return False

        # Check atomic structure from dump file: density + positions + overlaps.
        # Skip dump-density check when log already confirmed good density: the dump
        # records the initial trajectory (low density from an oversized starting box),
        # while minimized_structure.dat holds the final converged state.
        log_density_ok = density is not None and ClassicalMDHandler._RHO_MIN <= density <= ClassicalMDHandler._RHO_MAX
        for dname in ("dump_npt_unwrapped.lmp", "dump_npt.lmp"):
            dp = npt_dir / dname
            if dp.exists() and dp.stat().st_size >= _DUMP_MIN_SIZE:
                # 1. Volume-based density from first frame — skip if log confirmed convergence
                if not log_density_ok:
                    rho_dump = ClassicalMDHandler._dump_density(dp)
                    if rho_dump is not None and rho_dump < ClassicalMDHandler._RHO_MIN:
                        log.warning(
                            "[h05_cmd] NPT dump density: ρ≈%.3f g/cm³ < %.1f (gas/foam) in %s",
                            rho_dump, ClassicalMDHandler._RHO_MIN, npt_dir,
                        )
                        return False
                # 2. Structural checks on last frame: NaN, overlaps, bond distances
                struct_ok, reason = ClassicalMDHandler._check_dump_structure(dp)
                if not struct_ok:
                    log.warning(
                        "[h05_cmd] NPT structure anomaly: %s in %s", reason, npt_dir,
                    )
                    return False
                break  # only check one dump file

        return True

    @staticmethod
    def _validate_nvt(nvt_dir: Path, T_target: int = 300) -> bool:
        """Return True if the NVT production output is physically reasonable.

        Checks:
          1. dump_unwrapped.lmp exists and is ≥ _DUMP_MIN_SIZE
          2. No LAMMPS ERROR in log
          3. No charge-neutrality warning (bad counterion ratio in system.data)
          4. Final temperature within _T_TOL K of target
          5. Density from dump first frame within [_RHO_MIN, _RHO_MAX]
          6. Structural sanity on last dump frame (NaN, overlaps, short bonds)
        """
        dump = nvt_dir / "dump_unwrapped.lmp"
        if not dump.exists() or dump.stat().st_size < _DUMP_MIN_SIZE:
            return False

        log_path: Path | None = None
        for lname in ("log.lammps", "lammps.out", "out"):
            lf = nvt_dir / lname
            if lf.exists():
                log_path = lf
                break

        if log_path is not None:
            try:
                tail = log_path.read_text()[-8192:]
                if "ERROR" in tail:
                    log.warning("[h05_cmd] NVT log has ERROR: %s", log_path)
                    return False
                if "System is not charge neutral" in tail:
                    log.warning("[h05_cmd] NVT system not charge neutral: %s", log_path)
                    return False
                # Temperature check: last thermo line has step time temp press pe vol
                temp = None
                for line in reversed(tail.splitlines()):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            int(parts[0])
                            t = float(parts[2])
                            if t > 0:
                                temp = t
                                break
                        except (ValueError, IndexError):
                            continue
                if temp is not None and abs(temp - T_target) > ClassicalMDHandler._T_TOL:
                    log.warning("[h05_cmd] NVT temperature drift: T=%.1f vs target=%d in %s",
                                temp, T_target, nvt_dir)
                    return False
            except Exception:
                pass

        # Density + structure from dump
        sys_data = nvt_dir.parent.parent / "system.data"
        rho = ClassicalMDHandler._dump_density(dump, sys_data if sys_data.exists() else None)
        if rho is not None:
            if rho < ClassicalMDHandler._RHO_MIN:
                log.warning("[h05_cmd] NVT dump density too low: ρ=%.3f g/cm³ in %s",
                            rho, nvt_dir)
                return False
            if rho > ClassicalMDHandler._RHO_MAX:
                log.warning("[h05_cmd] NVT dump density too high: ρ=%.3f g/cm³ in %s",
                            rho, nvt_dir)
                return False

        struct_ok, reason = ClassicalMDHandler._check_dump_structure(dump)
        if not struct_ok:
            log.warning("[h05_cmd] NVT structure anomaly: %s in %s", reason, nvt_dir)
            return False

        return True

    @staticmethod
    def _cleanup_slurm_files(job_dir: Path) -> None:
        """Remove old SLURM stdout/stderr files from a job directory.

        Called before resubmitting a failed job so stale output doesn't accumulate.
        Matches %J.stdout, %J.stderr naming pattern (JobID digits).
        """
        import re as _re
        try:
            for f in job_dir.iterdir():
                if _re.match(r"^\d+\.(stdout|stderr)$", f.name):
                    try:
                        f.unlink()
                    except Exception:
                        pass
        except Exception:
            pass

    # ── ABC interface ─────────────────────────────────────────────────────────

    @staticmethod
    def _min_dump_size(project_dir: Path) -> int:
        """Minimum dump file size for SLURM runs."""
        return _DUMP_MIN_SIZE

    # ── Standard workflow ─────────────────────────────────────────────────────

    def _parse_composition_v2(self, project_dir: Path, yaml: dict) -> list[tuple[str, int]]:
        """Return (mol_name, count) pairs from molecule_counts_cmd for dielectric calc."""
        sim = yaml.get("simulation", {})
        mc  = sim.get("molecule_counts_cmd", {})
        if mc:
            return list(mc.items())
        components: list[tuple[str, int]] = []
        n = max(1, sim.get("n_molecules_cmd", 50))
        for s in sim.get("solvents", []):
            components.append((s["name"], n // 2))
        salt = sim.get("salt", "")
        if salt:
            components.append((salt, max(1, n // 4)))
        return components

    def _submit(self, project_dir: Path, yaml: dict,
                   state: "ProjectState") -> str | None:
        """CMD workflow: cmd/npt/ (300 K) → cmd/nvt/{T}/ (canonical temperatures).

        Source structure priority:
          1. preopt/contcar_cmd_preopt.vasp (MACE pre-optimized)
          2. designed_structures/system_cmd.data (OPLS-AA force-field data)

        Cross-ref:
          hpca/core/paths.py                    — contcar_preopt(), designed_structures()
          hpca/config/platform.yaml             — limits.cmd_npt_ps, limits.cmd_nvt_ns,
                                                   limits.nvt_temperatures
          hpca/orchestrator/handlers/h00_design — writes both source files
        """
        sim          = yaml.get("simulation", {})
        project_name = yaml.get("name", project_dir.name)

        # LAMMPS force-field parameters from platform.yaml
        timestep_fs  = self.plat("lammps_md", "timestep_fs_cmd",    2.0)
        temp_damp    = self.plat("lammps_md", "npt_temp_damp_cmd",  200.0)
        press_damp   = self.plat("lammps_md", "npt_press_damp_cmd", 2000.0)
        nvt_damp     = self.plat("lammps_md", "nvt_temp_damp_cmd",  100.0)

        # Simulation parameters from platform limits or project.yaml override (SLURM lane)
        npt_ps     = yaml.get("cmd_npt_ps", self.sim_limit("slurm", "cmd_npt_ps", 200))
        nvt_ns     = yaml.get("cmd_nvt_ns", self.sim_limit("slurm", "cmd_nvt_ns", 2))
        npt_opt_steps  = int(npt_ps * 1000 / timestep_fs)
        prod_steps     = int(nvt_ns * 1_000_000 / timestep_fs)
        hpc_cfg        = self.platform_config().get("hpc", {})
        lmp_ntasks     = int(hpc_cfg.get("lammps_cpu_ntasks_slurm", 104))
        lmp_ntasks_nvt = int(hpc_cfg.get("lammps_cpu_ntasks_slurm", 104))

        # Canonical temperature list from platform.yaml
        cmd_temps = self.nvt_temperatures()

        # Source structure: preopt CONTCAR preferred; preopted_system_cmd.data fallback
        preopt_vasp = contcar_preopt(project_dir, "cmd")
        system_data = preopt(project_dir) / "preopted_system_cmd.data"
        if not preopt_vasp.exists() and not system_data.exists():
            log.error("[h05_cmd] No source: preopt/contcar_cmd_preopt.vasp or preopted_system_cmd.data")
            return None

        components = self._parse_composition_v2(project_dir, yaml)
        epsilon    = (self._compute_dielectric(components)
                      if len(components) > 1
                      else self._get_dielectric().get(components[0][0], self._DEFAULT_DIELECTRIC)
                      if components else self._DEFAULT_DIELECTRIC)

        category    = yaml.get("category", "")
        system_type = yaml.get("system_type", "")
        is_solid    = _cat_is_crystalline(category)

        gate_T = 300 if 300 in cmd_temps else sorted(cmd_temps)[0]

        handler_state  = state.get_handler(self.name)
        submitted_jobs: dict = dict(handler_state.get("jobs", {}))
        first_job: str | None = None

        # ── NPT ──────────────────────────────────────────────────────────────
        npt_dir   = cmd_npt(project_dir)
        npt_dir.mkdir(parents=True, exist_ok=True)
        npt_final = npt_dir / "minimized_structure.dat"
        npt_start = npt_dir / "nvt_start.dat"
        npt_done  = npt_start.exists() or (
            npt_final.exists() and self._validate_npt(npt_dir, gate_T))

        if not npt_done:
            npt_key      = "cmd/npt"
            retry_count  = handler_state.get("npt_retry_count", 0)
            npt_fix      = handler_state.get("npt_fix", "")
            excl_nodes   = handler_state.get("excluded_nodes", [])
            # Corrective actions applied from retry 4 onwards:
            #   exclude_nodes  → add --exclude to sub.sh so SLURM avoids bad nodes
            #   reduce_timestep → halve timestep to survive bad-contact LAMMPS crashes
            eff_timestep = (timestep_fs / 2.0
                            if npt_fix == "reduce_timestep" else timestep_fs)
            sub_exclude  = excl_nodes if npt_fix == "exclude_nodes" and excl_nodes else None
            if not self.job_alive(submitted_jobs.get(npt_key)):
                rel_data = "../../preopt/preopted_system_cmd.data"
                self._write_npt_input(
                    npt_dir / "in.lammps", gate_T, npt_opt_steps,
                    data_file=rel_data, is_solid=is_solid, epsilon=epsilon,
                    timestep_fs=eff_timestep, temp_damp=temp_damp, press_damp=press_damp)
                self._write_sub_sh(
                    npt_dir / "sub.sh", f"{project_name}_cmd_npt",
                    n_tasks=lmp_ntasks, n_omp=1,
                    time=self.slurm_time("cmd_npt"), exclude=sub_exclude)
                self._cleanup_slurm_files(npt_dir)
                jid = self.sbatch(npt_dir / "sub.sh", cwd=npt_dir)
                if jid:
                    submitted_jobs[npt_key] = jid
                    first_job = jid
                    log.info("[h05_cmd] v2: NPT submitted → %s (retry=%d%s)",
                             jid, retry_count,
                             f", fix={npt_fix}" if npt_fix else "")
                state.set_stage(self.name, "RUNNING", jobs=submitted_jobs)
                return first_job
            else:
                # NPT job is alive. Update sentinel "job" so orchestrator doesn't
                # keep re-entering the dead-sentinel → auto_fix loop every poll.
                alive_jid = submitted_jobs.get(npt_key)
                state.set_stage(self.name, "RUNNING", jobs=submitted_jobs,
                                **({"job": alive_jid} if alive_jid else {}))
                return None  # NPT job alive, wait

        # Build nvt_start.dat from NPT output (charge-corrected)
        if not npt_start.exists() and npt_final.exists():
            self._fix_npt_charges(npt_final, system_data, npt_start)

        # Parse element string once from preopted_system_cmd.data (has OPLS comments) so
        # dump_modify gets correct element labels instead of all-"C" defaults.
        npt_start_path = cmd_npt(project_dir) / "nvt_start.dat"
        n_types_nvt    = _count_atom_types(npt_start_path)
        elems_v2       = _parse_elements_from_data(system_data, n_types=n_types_nvt)
        elem_str_v2    = " ".join(elems_v2) if elems_v2 else "C H O N S F Li"
        log.info("[h05_cmd] v2: element mapping for dump_modify: %s", elem_str_v2)

        # ── NVT at each temperature ───────────────────────────────────────────
        min_sz = ClassicalMDHandler._min_dump_size(project_dir)
        nvt_todo_v2: list[tuple[Path, int]] = []
        for T in cmd_temps:
            nvt_dir = cmd_nvt(project_dir, T)
            nvt_dir.mkdir(parents=True, exist_ok=True)
            if self._nvt_run_complete(nvt_dir, min_sz):
                continue

            nvt_key = f"cmd/nvt/{T}K"
            existing_jid = submitted_jobs.get(nvt_key)
            if self.job_alive(existing_jid):
                # No new submission is necessary, but return a live sentinel so
                # the orchestrator preserves RUNNING instead of interpreting
                # the deliberate wait as an sbatch failure.
                first_job = first_job or existing_jid
                continue

            # Relative path from cmd/nvt/{T}K/ to cmd/npt/nvt_start.dat
            nvt_data_file = "../../../cmd/npt/nvt_start.dat"

            self._write_nvt_input(
                nvt_dir / "in.lammps", T, prod_steps,
                data_file=nvt_data_file, epsilon=epsilon,
                elem_str=elem_str_v2, timestep_fs=timestep_fs, temp_damp=nvt_damp)
            self._write_sub_sh(
                nvt_dir / "sub.sh", f"{project_name}_cmd_{T}K_nvt",
                n_tasks=lmp_ntasks_nvt, n_omp=1,
                time=self.slurm_time("cmd_nvt"))
            self._cleanup_slurm_files(nvt_dir)
            jid = self.sbatch(nvt_dir / "sub.sh", cwd=nvt_dir)
            if jid:
                submitted_jobs[nvt_key] = jid
                first_job = first_job or jid
                log.info("[h05_cmd] v2: NVT submitted T=%dK ε=%.1f → %s", T, epsilon, jid)

        state.set_stage(self.name, "RUNNING", jobs=submitted_jobs)
        return first_job

    # ── Public interface ───────────────────────────────────────────────────────

    def _ensure_cmd_poscar(self, project_dir: Path, yaml_data: dict) -> bool:
        """Pack CMD box inline if poscar_cmd/system_cmd.data/preopt contcar are missing.

        Called by submit() so h05_cmd is self-contained for its CMD setup.
        h00_design._build_designed_structures_liquid skips DFT+MLMD tiers (their
        preopt contcars already exist) and packs only the missing CMD tier.
        """
        cmd_poscar  = designed_structures(project_dir) / "poscar_cmd.vasp"
        system_data = preopt(project_dir) / "preopted_system_cmd.data"
        preopt_out  = contcar_preopt(project_dir, "cmd")
        if cmd_poscar.exists() and system_data.exists() and preopt_out.exists():
            return True
        log.info("[h05_cmd] CMD box/preopted_system_cmd.data/preopt missing — packing CMD tier inline")
        from hpca.orchestrator.handlers.h00_design import MaterialsDesignHandler
        design = MaterialsDesignHandler()
        # Packs only CMD (DFT+MLMD skip because their preopt contcars exist).
        design._build_designed_structures_liquid(project_dir, yaml_data)
        # Preopts only CMD (DFT+MLMD skip because their contcars exist).
        design._run_preopt_all(project_dir, yaml_data)
        return cmd_poscar.exists() and system_data.exists()

    def can_run(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True for molecular projects; CMD packing is self-contained in submit()."""
        yaml = self.read_project_yaml(project_dir)
        from hpca.core.categories import is_molecular as _is_mol
        if not _is_mol(yaml.get("category", "")):
            return False
        # h05_cmd is self-contained: submit() packs CMD inline if needed.
        # h00_design no longer gates CMD; returning True lets submit() decide.
        return True

    def is_complete(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True only after every canonical NVT run finished cleanly.

        A production dump reaches ``_DUMP_MIN_SIZE`` early in a long LAMMPS
        run.  Treating size alone as completion lets downstream analysis read a
        file that is still being appended to.  Require both final artifacts
        written after ``run`` completes as well as the adequately sized dump.
        """
        # All canonical NVT temperatures must have a complete, finalised run.
        # Cross-ref: hpca/config/platform.yaml limits.nvt_temperatures
        temps  = self.nvt_temperatures()
        min_sz = ClassicalMDHandler._min_dump_size(project_dir)
        for T in temps:
            nvt_dir = cmd_nvt(project_dir, T)
            if not self._nvt_run_complete(nvt_dir, min_sz):
                return False
        return True

    @staticmethod
    def _nvt_run_complete(nvt_dir: Path, min_dump_size: int) -> bool:
        """Check final NVT artifacts without mistaking a growing dump for completion."""
        dump = nvt_dir / "dump_unwrapped.lmp"
        final_data = nvt_dir / "after_nvt_.dat"
        log_file = nvt_dir / "log.lammps"
        if not (dump.exists() and dump.stat().st_size >= min_dump_size
                and final_data.exists() and final_data.stat().st_size > 0
                and log_file.exists()):
            return False
        try:
            tail = log_file.read_bytes()[-4096:].decode("utf-8", errors="replace")
        except OSError:
            return False
        return "Total wall time:" in tail

    def submit(self, project_dir: Path, state: "ProjectState") -> str | None:
        """Ensure CMD box is packed, then submit NPT and NVT jobs via _submit()."""
        yaml = self.read_project_yaml(project_dir)
        if not self._ensure_cmd_poscar(project_dir, yaml):
            log.warning("[h05_cmd] CMD packing/preopt failed — retry next poll")
            return None
        return self._submit(project_dir, yaml, state)

    @staticmethod
    def _get_alive_jobs() -> set[str]:
        """Set of alive job IDs for this user.

        Reads from hpca.core.slurm_submit's shared, TTL-cached squeue
        snapshot rather than shelling out here directly — this handler's
        check_progress() runs once per RUNNING h05_cmd project per poll
        cycle, and with dozens of combinatorial sub-projects (e.g. LYC's ~67
        doping variants) advancing in parallel, an uncached call here was a
        major contributor to the scheduler RPC storm flagged by NREL on
        2026-08-12 (see slurm_submit.py's module docstring for the full
        writeup). One real squeue call per TTL window now serves everyone.
        """
        import os
        from hpca.core.slurm_submit import alive_job_ids
        user = (os.environ.get("USER")
                or os.environ.get("SLURM_JOB_USER")
                or "")
        if not user:
            return set()
        return alive_job_ids(user)

    # Max NPT resubmissions per check_progress() call -- keeps us under QOS burst.
    _RESUBMIT_CAP = 200

    def check_progress(self, project_dir: Path, state: "ProjectState") -> None:
        """Invalidate bad NPT/NVT outputs and resubmit dead NPT jobs within the resubmit cap."""
        handler_state  = state.get_handler(self.name)
        submitted_jobs = dict(handler_state.get("jobs", {}))

        # One squeue call to get all alive job IDs for this user.
        alive = self._get_alive_jobs()

        # ── Pass 1a: invalidate NPT "completions" whose density is out of range ──
        # Deleting minimized_structure.dat causes them to be requeued in Pass 2.
        n_invalid_npt = 0
        for npt_key in list(submitted_jobs):
            if not npt_key.endswith("/npt"):
                continue
            base = npt_key[:-4]
            rel_dir, _, T_str = base.rpartition("/")
            npt_dir = project_dir / rel_dir / T_str / "npt"
            dat = npt_dir / "minimized_structure.dat"
            if not dat.exists():
                continue
            try:
                T_val = int(T_str)
            except ValueError:
                continue
            if not self._validate_npt(npt_dir, T_val):
                try:
                    dat.unlink()
                    n_invalid_npt += 1
                    log.info("[h05_cmd] invalidated bad NPT completion: %s", npt_dir)
                except Exception as exc:
                    log.warning("[h05_cmd] could not remove %s: %s", dat, exc)

        if n_invalid_npt:
            log.info("[h05_cmd] %d NPT completions invalidated (density/T out of range)",
                     n_invalid_npt)

        # ── Pass 1b: invalidate NVT dumps from gas-phase NPT starting points ────
        n_invalid_nvt = 0
        for nvt_key in list(submitted_jobs):
            if not nvt_key.endswith("/nvt"):
                continue
            base = nvt_key[:-4]
            rel_dir, _, T_str = base.rpartition("/")
            nvt_dir = project_dir / rel_dir / T_str / "nvt"
            if not self._validate_nvt_dump(nvt_dir):
                # validate_nvt_dump already warned; delete the bad dump
                for dname in ("dump_unwrapped.lmp", "dump.lmp", "dump_nvt.lmp"):
                    dp = nvt_dir / dname
                    if dp.exists():
                        try:
                            dp.unlink()
                            n_invalid_nvt += 1
                        except Exception as exc:
                            log.warning("[h05_cmd] could not remove %s: %s", dp, exc)

        if n_invalid_nvt:
            log.info("[h05_cmd] %d NVT dump files invalidated (gas-phase density)",
                     n_invalid_nvt)

        # ── Pass 2: resubmit dead NPT jobs whose output is missing ──────────────
        # Load gate info per cmd_dir so we don't bypass the 300K gate here.
        yaml       = self.read_project_yaml(project_dir)
        sim        = yaml.get("simulation", {})
        cmd_temps  = sorted(sim.get("cmd_temps", [300]))
        gate_T     = self._T_GATE if self._T_GATE in cmd_temps else (cmd_temps[0] if cmd_temps else 300)

        n_resubmit = 0
        for npt_key, old_jid in list(submitted_jobs.items()):
            if not npt_key.endswith("/npt"):
                continue
            if old_jid and old_jid in alive:
                continue  # still PENDING/RUNNING
            # alive set can be empty when squeue --user fails on compute nodes;
            # fall back to per-job check before treating as dead.
            if old_jid and self.job_alive(old_jid):
                continue
            # key format: "rel_dir/T/npt"
            base = npt_key[:-4]          # strip trailing "/npt"
            rel_dir, _, T_str = base.rpartition("/")
            npt_dir   = project_dir / rel_dir / T_str / "npt"
            npt_final = npt_dir / "minimized_structure.dat"
            if npt_final.exists():
                continue  # valid completion (passed validate in Pass 1)
            sub_sh = npt_dir / "sub.sh"
            if not sub_sh.exists():
                continue
            # Respect the 300K gate: only resubmit non-gate-T if gate is done.
            try:
                T_val = int(T_str)
            except ValueError:
                T_val = gate_T  # unparseable → treat as gate to avoid blocking
            if T_val != gate_T:
                cmd_root = project_dir / rel_dir
                gate_npt = cmd_root / str(gate_T) / "npt" / "minimized_structure.dat"
                gate_done = (gate_npt.exists() and
                             self._validate_npt(cmd_root / str(gate_T) / "npt", gate_T))
                if not gate_done:
                    log.debug("[h05_cmd] resubmit holding %s T=%dK -- gate T=%dK not done",
                              rel_dir, T_val, gate_T)
                    continue
            if n_resubmit >= self._RESUBMIT_CAP:
                log.debug("[h05_cmd] resubmit cap (%d) reached -- continuing next cycle",
                          self._RESUBMIT_CAP)
                break
            # Remove stale SLURM stdout/stderr before resubmitting.
            self._cleanup_slurm_files(npt_dir)
            jid = self.sbatch(sub_sh, cwd=npt_dir)
            if jid:
                submitted_jobs[npt_key] = jid
                n_resubmit += 1

        if n_resubmit:
            log.info("[h05_cmd] auto-resubmit: %d failed NPT jobs requeued", n_resubmit)

        state.set_handler(self.name, {"jobs": submitted_jobs})

    _NPT_RETRY_LIMIT = 4   # blind retries before corrective action kicks in

    def auto_fix(self, project_dir: Path, state: "ProjectState") -> bool:
        """Called when the sentinel job dies.

        Retries 1–3: log reason, resubmit as-is.
        Retry 4+   : apply corrective action before next submit():
          - OFI/hardware → accumulate bad nodes; submit() adds --exclude
          - LAMMPS ERROR → submit() halves timestep in in.lammps
        Metadata (retry count, excluded nodes, fix type) is merged into handler
        state so submit() reads it without any extra round-trips.
        """
        handler_state = state.get_handler(self.name)
        reason        = self._diagnose_failure(project_dir)

        # NPT finished normally but orchestrator missed it — go straight to NVT
        if "completed normally" in reason:
            log.info("[h05_cmd] auto_fix: NPT completed normally — advancing to NVT")
            state.set_stage(self.name, "PENDING")
            return True

        retry_count   = handler_state.get("npt_retry_count", 0) + 1

        update: dict = {"npt_retry_count": retry_count}

        is_ofi    = "OFI" in reason or "Libfabric" in reason or "hardware" in reason
        is_lammps = reason.startswith("LAMMPS ERROR")

        # Always accumulate bad nodes (useful for --exclude even before threshold)
        if is_ofi:
            bad_node = self._extract_failed_node(project_dir)
            if bad_node:
                excluded = list(set(handler_state.get("excluded_nodes", [])) | {bad_node})
                update["excluded_nodes"] = excluded

        if retry_count < self._NPT_RETRY_LIMIT:
            log.info("[h05_cmd] auto_fix attempt %d/%d: %s",
                     retry_count, self._NPT_RETRY_LIMIT, reason)
        else:
            # Apply corrective action — submit() reads npt_fix to adjust script/sub.sh
            if is_ofi or not is_lammps:
                update["npt_fix"] = "exclude_nodes"
                log.warning(
                    "[h05_cmd] auto_fix attempt %d: %s — corrective action: "
                    "adding --exclude=%s to sub.sh",
                    retry_count, reason,
                    ",".join(update.get("excluded_nodes", handler_state.get("excluded_nodes", []))))
            else:
                update["npt_fix"] = "reduce_timestep"
                log.warning(
                    "[h05_cmd] auto_fix attempt %d: %s — corrective action: "
                    "halving timestep in in.lammps",
                    retry_count, reason)

        state.set_handler(self.name, update)
        state.set_stage(self.name, "PENDING")
        return True

    def _diagnose_failure(self, project_dir: Path) -> str:
        """Read LAMMPS log and stderr; return a short failure-reason string."""
        npt_dir = cmd_npt(project_dir)
        for lname in ("log.lammps", "lammps.out", "out"):
            lp = npt_dir / lname
            if lp.exists():
                try:
                    tail = lp.read_bytes()[-4096:].decode("utf-8", errors="replace")
                    for line in tail.splitlines():
                        if "ERROR" in line:
                            return f"LAMMPS ERROR: {line.strip()[:120]}"
                    if "Total wall time" in tail:
                        return "completed normally (missed by orchestrator)"
                except Exception:
                    pass
        try:
            stderr_files = sorted(npt_dir.glob("*.stderr"), key=lambda p: p.stat().st_mtime)
            if stderr_files:
                tail = stderr_files[-1].read_bytes()[-2048:].decode("utf-8", errors="replace")
                if "OFI" in tail or "Libfabric" in tail:
                    return "OFI Libfabric node failure (hardware) — resubmit to new node"
                if "signal" in tail.lower() or "killed" in tail.lower():
                    return "job killed by signal (OOM or walltime)"
                if tail.strip():
                    return f"stderr: {tail.splitlines()[-1].strip()[:120]}"
        except Exception:
            pass
        return "sentinel dead (no log/stderr found)"

    def _extract_failed_node(self, project_dir: Path) -> str | None:
        """Parse 'Local host: <node>' from the most recent NPT stderr file."""
        import re
        npt_dir = cmd_npt(project_dir)
        try:
            stderr_files = sorted(npt_dir.glob("*.stderr"), key=lambda p: p.stat().st_mtime)
            if stderr_files:
                text = stderr_files[-1].read_text(errors="replace")
                m = re.search(r"Local host:\s+(\S+)", text)
                if m:
                    return m.group(1)
        except Exception:
            pass
        return None

    # ── System builder ────────────────────────────────────────────────────────

    def _build_system_data(
        self,
        project_dir: Path,
        cmd_root:    Path,
        yaml:        dict,
        n_mol:       int,
    ) -> bool:
        """Parse composition from cmd_root path, build LAMMPS data file."""
        from hpca.sim.forcefield import MolData, build_mixed_system

        components = self._parse_composition(cmd_root, project_dir, yaml)
        if not components:
            log.warning("[h05_cmd] Could not resolve composition for %s", cmd_root)
            return False

        # Scale counts so total ≈ n_mol
        total_raw = sum(c for _, c in components)
        scale = max(1, round(n_mol / total_raw))
        mol_data: list[MolData] = []

        ff_dir = cmd_root / "ff"
        ff_dir.mkdir(parents=True, exist_ok=True)

        for (mol_name, raw_count) in components:
            count = max(1, raw_count * scale)
            vasp_path = self._find_molecule_file(mol_name, project_dir)
            lmp_path  = ff_dir / f"{mol_name}.lmp"

            if vasp_path and vasp_path.exists():
                try:
                    md = MolData.from_file(vasp_path, name=mol_name, count=count)
                    from hpca.sim.forcefield import write_lmp
                    write_lmp(md.atoms, md.bonds, md.types, lmp_path, mol_name)
                    mol_data.append(md)
                    log.info("[h05_cmd] FF assigned: %s (%d molecules)", mol_name, count)
                    continue
                except Exception as exc:
                    log.warning("[h05_cmd] FF failed for %s from VASP: %s", mol_name, exc)

            # Fallback: built-in geometry library
            from hpca.sim.forcefield import MOLECULES
            mol_key = mol_name.upper()
            if mol_key in MOLECULES:
                try:
                    md = MolData.from_builtin(mol_key, count=count)
                    from hpca.sim.forcefield import write_lmp
                    write_lmp(md.atoms, md.bonds, md.types, lmp_path, mol_name)
                    mol_data.append(md)
                    log.info("[h05_cmd] FF assigned (builtin): %s (%d molecules)", mol_name, count)
                    continue
                except Exception as exc:
                    log.warning("[h05_cmd] Builtin FF failed for %s: %s", mol_name, exc)

            # Third fallback: polymer chain builder (PVDF_HFP, PEO50, PMMA50, etc.)
            try:
                from hpca.sim.polymer import PolymerMolData
                # Accept names like PVDF_HFP, PVDF_HFP50, PEO, PMMA, PTFEP
                poly_key = mol_name.upper().replace("-", "_")
                if "PVDF" in poly_key and "HFP" in poly_key:
                    pm = PolymerMolData.pvdf_hfp(n_units=50, hfp_fraction=0.20, count=count)
                    md = pm.to_mol_data()
                    from hpca.sim.forcefield import write_lmp
                    write_lmp(md.atoms, md.bonds, md.types, lmp_path, mol_name)
                    mol_data.append(md)
                    log.info("[h05_cmd] FF assigned (polymer builder): %s (%d chains)", mol_name, count)
                    continue
            except Exception as exc:
                log.warning("[h05_cmd] Polymer builder failed for %s: %s", mol_name, exc)

            log.error("[h05_cmd] Cannot find molecule %s -- skipping composition", mol_name)
            return False

        if not mol_data:
            return False

        try:
            cmd_root.mkdir(parents=True, exist_ok=True)
            L_box = build_mixed_system(mol_data, cmd_root / "system.data")
            log.info("[h05_cmd] system.data written  box=%.1f Å  total_atoms=%d",
                     L_box, sum(len(m.atoms) * m.count for m in mol_data))
            return True
        except Exception as exc:
            log.error("[h05_cmd] build_mixed_system failed: %s", exc)
            return False

    # ── Composition parser ────────────────────────────────────────────────────

    @staticmethod
    def _parse_composition(
        cmd_root:    Path,
        project_dir: Path,
        yaml:        dict,
    ) -> list[tuple[str, int]]:
        """Derive (mol_name, count) pairs for building this cmd dir's system.data.

        Priority:
          1. cmd_combinations[combo_name] with molecule_counts_cmd (new per-combo format)
          2. Legacy combinations[] section
          3. Regex parse of the directory name
        """
        sim  = yaml.get("simulation", {})
        mol_counts_cmd: dict = sim.get("molecule_counts_cmd", {})

        # ── New format: cmd/combo_name  (combo_name = last path component) ─────
        combo_name = cmd_root.name   # e.g. "DMB__LiFSI__PEO__PVDF-HFP"

        for combo in yaml.get("cmd_combinations", []):
            if combo.get("name", "") != combo_name:
                continue
            # Collect all species in this combination
            species_in_combo: list[str] = []
            for cat_data in combo.get("components", {}).values():
                for c in cat_data.get("components", []):
                    species_in_combo.append(c["name"])
            if not species_in_combo:
                continue
            # Use per-tier molecule counts if available, filtered to this combo
            if mol_counts_cmd:
                result = [(sp, mol_counts_cmd[sp])
                          for sp in species_in_combo
                          if sp in mol_counts_cmd]
                if result:
                    return result
            # Fallback: use combination ratios directly
            result = [(c["name"], c["ratio"])
                      for cat_data in combo.get("components", {}).values()
                      for c in cat_data.get("components", [])]
            if result:
                return result

        # ── Legacy format: project.yaml combinations[] section ───────────────
        # Walk up the path to find the combination name (parent of "cmd")
        parts = cmd_root.parts
        try:
            cmd_idx = list(p.lower() for p in parts).index("cmd")
            cname   = parts[cmd_idx - 1]
        except (ValueError, IndexError):
            cname = cmd_root.parent.name

        for combo in yaml.get("combinations", []):
            if combo.get("name", "") == cname:
                result = []
                solvents = combo.get("solvents", [])
                ratios   = combo.get("ratios",   [1] * len(solvents))
                salt     = combo.get("salt", "")
                for sv, r in zip(solvents, ratios):
                    result.append((sv, int(r)))
                if salt:
                    result.append((salt, 1))
                if result:
                    return result

        # ── Fallback: parse ratio suffix from directory name ─────────────────
        m = re.match(r"^(.+?)_(\d+(?:-\d+)+)$", combo_name)
        if m:
            names  = m.group(1).split("_")
            ratios = [int(r) for r in m.group(2).split("-")]
            if len(names) == len(ratios):
                return list(zip(names, ratios))

        first = combo_name.split("_")[0]
        return [(first, 1)] if first else []

    @staticmethod
    def _find_molecule_file(mol_name: str, project_dir: Path) -> Path | None:
        """Search for a VASP/PDB/XYZ file matching mol_name in project_dir."""
        for suffix in (".vasp", ".poscar", ".pdb", ".xyz", ""):
            for fname in [mol_name + suffix,
                          mol_name.upper() + suffix,
                          mol_name.lower() + suffix]:
                p = project_dir / fname
                if p.exists():
                    return p
        # Check input_structures/ then structures/ subdirectory
        for sub in ("input_structures", "structures"):
            sdir = project_dir / sub
            if sdir.is_dir():
                for suffix in (".vasp", ".pdb", ".xyz"):
                    p = sdir / (mol_name + suffix)
                    if p.exists():
                        return p
        return None

    # ── LAMMPS input writers ──────────────────────────────────────────────────

    @staticmethod
    def _write_npt_input(
        path:        Path,
        T:           int,
        equil_steps: int,
        data_file:   str        = "system.data",
        timestep_fs: float      = 2.0,
        dump_every:  int | None = None,
        is_solid:    bool       = False,
        epsilon:     float      = 1.0,
        temp_damp:   float      = 200.0,
        press_damp:  float      = 2000.0,
    ) -> None:
        """NPT equilibration -- single-stage for liquids/gels, multi-stage for solids."""
        if is_solid:
            ClassicalMDHandler._write_solid_npt(
                path, T, equil_steps, data_file, timestep_fs, dump_every or 5_000)
        else:
            ClassicalMDHandler._write_liquid_npt(
                path, T, equil_steps, data_file, timestep_fs, dump_every,
                epsilon=epsilon, temp_damp=temp_damp, press_damp=press_damp)

    @staticmethod
    def _write_liquid_npt(
        path:        Path,
        T:           int,
        equil_steps: int,
        data_file:   str        = "system.data",
        timestep_fs: float      = 2.0,
        dump_every:  int | None = None,
        epsilon:     float      = 1.0,
        temp_damp:   float      = 200.0,
        press_damp:  float      = 2000.0,
    ) -> None:
        """NPT equilibration -- AtomicAI variable-block format, OPLS-AA pair style."""
        # Get element list from data file for dump_modify
        data_abs = path.parent / data_file
        elems    = _parse_elements_from_data(data_abs)
        elem_str = " ".join(elems) if elems else "C H O N S F Li"

        lmd = load_platform_config().get("lammps_md", {})
        if dump_every is None:
            dump_every = int(lmd.get("cmd_npt_dump_freq", 500))
        s_npt    = lmd.get("cmd_npt_npt_steps", equil_steps)
        lj_cut   = lmd.get("cmd_lj_cutoff_A",       14.0)
        ksp_acc  = lmd.get("cmd_kspace_accuracy",   1.0e-4)
        neigh_sk = lmd.get("cmd_neighbor_skin_A",   2.0)
        comm_cut = lmd.get("cmd_comm_cutoff_A",     35.0)
        thermo_f = int(lmd.get("cmd_thermo_freq",   10))

        script = f"""\
# LAMMPS NPT equilibration -- liquid/gel (OPLS-AA, generated by hpca h05_cmd)
#
# Protocol:
#   Stage 0 : CG minimization  (fixed volume -- removes bad contacts from PACKMOL packing)
#   Stage 1 : NPT at T, 1 atm  (1 ns -- box compresses from oversized to target density)
#
# Box starts oversized (~0.1–0.2 g/cm³); barostat at T=300 K safely compresses to ~1 g/cm³.

# Structure
units           real
boundary        p p p
atom_style      full

# Variables
variable read_data_file string "{data_file}"
variable dump_file1 string "dump_npt_unwrapped.lmp"
variable dump_file2 string "dump_npt.lmp"

variable T_target equal {T}
variable run_npt equal {s_npt}
variable timestep equal {timestep_fs}
variable thermo_freq equal {thermo_f}
variable dump_freq equal {dump_every}
variable temp_damp equal {temp_damp}
variable press_damp equal {press_damp}

# Force field
pair_style      lj/cut/coul/long {lj_cut} {lj_cut}
kspace_style    ewald {ksp_acc}
bond_style      harmonic
angle_style     harmonic
dihedral_style  opls
improper_style  cvff
# special_bonds BEFORE read_data: avoids ntopo rebuild with active OMP threads.
special_bonds lj/coul 0.0 0.0 0.5 coul 0.0 0.0 1.0 angle yes dihedral yes
atom_modify map array
variable nthreads getenv OMP_NUM_THREADS
package omp ${{nthreads}} neigh no

read_data       ${{read_data_file}}

# Mixture-averaged relative permittivity (mole-fraction weighted).
dielectric {epsilon:.2f}

neighbor        {neigh_sk} bin
neigh_modify    every 1 delay 0 check yes one 4096 page 200000
comm_modify cutoff {comm_cut}

timestep        ${{timestep}}

thermo ${{thermo_freq}}
thermo_style custom step time temp press pe vol density cella cellb cellc

# Stage 0: CG minimization -- remove bad contacts from PACKMOL packing
min_style    cg
min_modify   dmax 0.05
minimize     0.0 1.0e4 3000 30000
min_style    cg
min_modify   dmax 0.1
minimize     1.0e-4 1.0e-6 5000 50000
reset_timestep 0

velocity all create ${{T_target}} 87287 loop geom

# Dumps defined after reset_timestep -- LAMMPS forbids reset_timestep with active dumps.
dump 1 all custom ${{dump_freq}} ${{dump_file1}} id mol type element xu yu zu
dump_modify 1 element {elem_str}
dump 2 all custom ${{dump_freq}} ${{dump_file2}} id mol type element x y z
dump_modify 2 element {elem_str}

# Stage 1: NPT at T_target, 1 atm
fix npt all npt temp ${{T_target}} ${{T_target}} ${{temp_damp}} iso 1.0 1.0 ${{press_damp}}
run ${{run_npt}}
unfix npt

write_data minimized_structure.dat
"""
        path.write_text(script)

    @staticmethod
    def _write_solid_npt(
        path:        Path,
        T:           int,
        equil_steps: int,
        data_file:   str   = "system.data",
        timestep_fs: float = 1.0,
        dump_every:  int   = 2_000,
    ) -> None:
        """4-stage NPT for inorganic solids (SSEs, electrodes, coatings).

        Stage sequence:
          1. Minimize  -- remove bad contacts from design/MLIP pre-opt
          2. NVT heat  -- 20% of steps at T_high (2×T, max 2000 K)
          3. NPT relax -- 30% of steps at T_high, 0 bar → equilibrate volume
          4. NPT cool  -- 50% of steps, cool T_high → T at 0 bar
        Timestep 1 fs (smaller than liquids; solids have stiffer bonds).
        """
        T_high    = min(2 * T, 2000)
        s_heat    = max(1, int(equil_steps * 0.20))
        s_relax   = max(1, int(equil_steps * 0.30))
        s_cool    = max(1, equil_steps - s_heat - s_relax)
        T_damp    = 100.0   # fs (tighter thermostat for solids)
        P_damp    = 1000.0  # fs

        script = f"""\
# ── LAMMPS NPT equilibration -- solid (hpca h05_cmd) ─────────────────────────
# 4-stage: minimize → NVT heat → NPT relax at T_high → NPT cool to T
# T_high = {T_high} K,  T_target = {T} K

units        metal
atom_style   atomic
boundary     p p p

read_data    {data_file}

pair_style   lj/cut/coul/long 8.0
kspace_style ewald 1.0e-4

neighbor     1.0 bin
neigh_modify every 1 delay 0 check yes one 4096 page 200000

timestep     {timestep_fs * 0.001}
thermo_style custom step temp pe etotal vol press density
thermo       {dump_every}

# ── Stage 1: minimize to remove bad contacts ─────────────────────────────────
minimize 1.0e-4 1.0e-6 2000 20000
reset_timestep 0

# ── Stage 2: NVT heat to T_high ──────────────────────────────────────────────
velocity all create {T_high} 87287 loop geom
fix nvt_heat all nvt temp {T_high} {T_high} {T_damp}
run {s_heat}
unfix nvt_heat

# ── Stage 3: NPT relax at T_high, 0 bar ──────────────────────────────────────
fix npt_hi all npt temp {T_high} {T_high} {T_damp} iso 0.0 0.0 {P_damp}
run {s_relax}
unfix npt_hi
write_data system_hot.data nocoeff

# ── Stage 4: NPT cool from T_high to T ───────────────────────────────────────
fix npt_cool all npt temp {T_high} {T} {T_damp} iso 0.0 0.0 {P_damp}
run {s_cool}
unfix npt_cool

write_data system_final.data nocoeff
"""
        path.write_text(script)

    @staticmethod
    def _fix_npt_charges(src: Path, system_data: Path, dst: Path) -> None:
        """Restore per-atom charges corrupted by LAMMPS write_data type-collapse.

        LAMMPS write_data merges atom types with identical LJ (ε,σ)/mass and assigns
        one charge per merged type, overwriting per-atom charges.  This restores the
        correct per-atom charges from system.data (matched by atom ID) and writes
        the result to dst (nvt_start.dat) so NVT sees correct electrostatics.
        """
        if not src.exists() or not system_data.exists():
            log.warning("[h05_cmd] _fix_npt_charges: missing src=%s or system.data=%s", src, system_data)
            return

        # Read correct per-atom charges from system.data
        atom_charge: dict[int, float] = {}
        in_atoms = False
        for line in system_data.read_text().splitlines():
            s = line.strip()
            if s.startswith("Atoms"):
                in_atoms = True
                continue
            if in_atoms:
                if not s or s.startswith("#"):
                    if atom_charge:
                        break
                    continue
                parts = s.split()
                if len(parts) < 7:
                    break
                atom_charge[int(parts[0])] = float(parts[3])

        if not atom_charge:
            log.warning("[h05_cmd] _fix_npt_charges: no atoms read from %s", system_data)
            return

        lines = src.read_text().splitlines(keepends=True)
        atoms_start = next(
            (i + 2 for i, l in enumerate(lines) if l.strip().startswith("Atoms")), None
        )
        if atoms_start is None:
            log.warning("[h05_cmd] _fix_npt_charges: Atoms section not found in %s", src)
            return

        # Find end of Atoms section
        atoms_end = atoms_start
        for i in range(atoms_start, len(lines)):
            s = lines[i].strip()
            if not s or (s and s[0].isalpha() and not s[0].isdigit()):
                atoms_end = i
                break

        patched = list(lines)
        q_after = 0.0
        for i in range(atoms_start, atoms_end):
            parts = lines[i].split()
            if len(parts) < 7:
                continue
            aid = int(parts[0])
            q = atom_charge.get(aid, float(parts[3]))
            q_after += q
            parts[3] = f"{q:.6f}"
            patched[i] = "  ".join(parts) + "\n"

        patched[0] = "LAMMPS data file -- nvt_start (charges fixed from system.data)\n"
        dst.write_text("".join(patched))
        log.info("[h05_cmd] _fix_npt_charges: wrote %s  q_sum=%+.4f e", dst.name, q_after)

    @staticmethod
    def _write_nvt_input(
        path:        Path,
        T:           int,
        prod_steps:  int,
        data_file:   str        = "../npt/nvt_start.dat",
        dump_every:  int | None = None,
        timestep_fs: float      = 2.0,
        epsilon:     float      = 1.0,
        elem_str:    str | None = None,
        temp_damp:   float      = 100.0,
    ) -> None:
        """NVT production run -- AtomicAI variable-block format, dual dumps."""
        if elem_str is None:
            # LAMMPS write_data collapses identical-LJ types and drops OPLS comments, so
            # minimized_structure.dat / nvt_start.dat have fewer types than system.data.
            # Read element labels from system.data (which has # OPLS_TYPE comments), but
            # truncate to the actual type count in the NVT data file so dump_modify element
            # receives exactly N labels for N atom types.
            nvt_data_abs = path.parent / data_file
            n_types      = _count_atom_types(nvt_data_abs)
            system_data  = path.parent.parent.parent / "system.data"
            elems        = _parse_elements_from_data(system_data, n_types=n_types)
            if not elems:
                elems = _parse_elements_from_data(nvt_data_abs)
            elem_str = " ".join(elems) if elems else "C H O N S F Li"

        _lmd     = load_platform_config().get("lammps_md", {})
        if dump_every is None:
            dump_every = int(_lmd.get("cmd_nvt_dump_freq", 1_000))
        lj_cut   = _lmd.get("cmd_lj_cutoff_A",     14.0)
        ksp_acc  = _lmd.get("cmd_kspace_accuracy",  1.0e-4)
        neigh_sk = _lmd.get("cmd_neighbor_skin_A",  2.0)
        thermo_f = int(_lmd.get("cmd_thermo_freq",  10))

        script = f"""\
# LAMMPS NVT production (OPLS-AA, generated by hpca h05_cmd)

# Structure
units           real
boundary        p p p
atom_style      full

# Variables
variable read_data_file string "{data_file}"
variable dump_file1 string "dump_unwrapped.lmp"
variable dump_file2 string "dump.lmp"

# Numeric Variables
variable T equal {T}
variable run_nvt equal {prod_steps}
variable timestep equal {timestep_fs}
variable thermo_freq equal {thermo_f}
variable dump_freq equal {dump_every}
variable temp_damp equal {temp_damp}

# Force field
pair_style      lj/cut/coul/long {lj_cut} {lj_cut}
kspace_style    ewald {ksp_acc}
bond_style      harmonic
angle_style     harmonic
dihedral_style  opls
improper_style  cvff
# special_bonds BEFORE read_data: avoids ntopo rebuild that races with OMP threads.
special_bonds lj/coul 0.0 0.0 0.5 coul 0.0 0.0 1.0 angle yes dihedral yes
# atom_modify map array: thread-safe atom-ID lookup for OMP runs.
atom_modify map array
variable nthreads getenv OMP_NUM_THREADS
package omp ${{nthreads}} neigh no

read_data       ${{read_data_file}}

# Mixture-averaged relative permittivity (mole-fraction weighted).
dielectric {epsilon:.2f}

neighbor        {neigh_sk} bin
neigh_modify    every 1 delay 0 check yes one 4096 page 200000

timestep        ${{timestep}}

thermo ${{thermo_freq}}
thermo_style custom step time temp press pe vol

dump 1 all custom ${{dump_freq}} ${{dump_file1}} id mol type element xu yu zu
dump_modify 1 element {elem_str}
dump 2 all custom ${{dump_freq}} ${{dump_file2}} id mol type element x y z
dump_modify 2 element {elem_str}

velocity all create ${{T}} 87287 loop geom

fix 1 all nvt temp ${{T}} ${{T}} ${{temp_damp}}
run ${{run_nvt}}
unfix 1

write_data after_nvt_.dat
"""
        path.write_text(script)

    @classmethod
    def _write_sub_sh(
        cls,
        path:      Path,
        job_name:  str,
        n_tasks:   int = 1,
        n_omp:     int = 104,
        time:      str = _CPU_TIME_LIMIT,
        exclude:   "list[str] | None" = None,
    ) -> None:
        """Write CPU LAMMPS SLURM script using OPLS-AA DeepMD-LAMMPS 2023 environment."""
        hpc      = cls.platform_config().get("hpc", {})
        lmp_bin  = hpc.get("lammps_cpu_bin", "")
        mpirun   = hpc.get("mpirun_bin", "mpirun")
        venv     = hpc.get("deepmd_lammps_venv_2023", "")
        account  = hpc.get("accounts", {}).get("standard") or _account_fallback()
        work_dir = str(path.parent)
        exclude_line = f"#SBATCH --exclude={','.join(exclude)}\n" if exclude else ""
        script = (
            "#!/bin/bash\n"
            f"#SBATCH --account={account}\n"
            f"{exclude_line}"
            "#SBATCH --nodes=1\n"
            f"#SBATCH --ntasks-per-node={n_tasks}\n"
            "#SBATCH --cpus-per-task=1\n"
            "#SBATCH --mem=0\n"
            f"#SBATCH --time={time}\n"
            f"#SBATCH --job-name={job_name}\n"
            f"#SBATCH --error={work_dir}/%J.stderr\n"
            f"#SBATCH --output={work_dir}/%J.stdout\n"
            "module purge\n"
            f"source {venv}/bin/activate\n"
            f"export LD_LIBRARY_PATH={Path(lmp_bin).parent}/lammps/src:$LD_LIBRARY_PATH\n"
            "export DP_DISABLE_CUDA=1\n"
            "export OMP_NUM_THREADS=1\n"
            "export DP_INTRA_OP_PARALLELISM_THREADS=1\n"
            "export DP_INTER_OP_PARALLELISM_THREADS=1\n"
            f"cd {work_dir}\n"
            f"{mpirun} -np {n_tasks} {lmp_bin} -in in.lammps\n"
        )
        path.write_text(script)
        path.chmod(0o755)
