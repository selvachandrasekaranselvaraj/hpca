"""
h08_echem.py — Electrochemistry handler.
Computes OCV, electrochemical window, energy above hull (EAH), formation energy.
Mix of daemon-local (pymatgen) and SLURM (DFT statics for missing reference energies).

For inorganic_sse: one shared echem/ folder at the parent LYC project level collects
ALL sub-project DFT opt structures and runs the electrochemical window analysis together.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from .base import SimulationHandler
from hpca.core.paths import dft_opt, load_platform_config

if TYPE_CHECKING:
    from hpca.orchestrator.state_tracker import ProjectState

log = logging.getLogger("hpca.orch")
# Layout: see hpca/core/paths.py; HPC paths: hpca/config/platform.yaml

# MP API key read from env var MP_API_KEY or ~/.config/mprester.yaml
MP_API_KEY = ""
_CLADUE_SITE = load_platform_config().get("hpc", {}).get(
    "cladue_site_packages", ""
)

# Nernst-Einstein constant
kB_eV = 8.617333e-5  # eV/K
eV_per_J = 6.241509e18


class EchemHandler(SimulationHandler):
    """Handler: formation energy, EAH, electrochemical window, OCV."""

    name = "h08_echem"
    is_daemon = True  # runs in-process; SLURM statics path not used in polymer mode

    # ── SSE shared-folder helpers ────────────────────────────────────────────

    def _is_sse(self, project_dir: Path) -> bool:
        """Return True if project.yaml category is 'inorganic_sse'."""
        try:
            return self.read_project_yaml(project_dir).get("category", "") == "inorganic_sse"
        except Exception:
            return False

    def _echem_root(self, project_dir: Path) -> Path:
        """For SSE: shared folder at parent level; for others: per-project results/echem/."""
        if self._is_sse(project_dir):
            return project_dir.parent / "echem"
        return project_dir / "results" / "echem"

    def _collect_sibling_opts(self, project_dir: Path) -> list[tuple[str, dict, float]]:
        """SSE: scan sibling sub-projects for completed DFT opt → [(name, comp, E_total)]."""
        results = []
        parent = project_dir.parent
        for sub_dir in sorted(parent.iterdir()):
            if not sub_dir.is_dir() or sub_dir.name.startswith("."):
                continue
            opt_dir = dft_opt(sub_dir)
            outcar = opt_dir / "OUTCAR"
            if not outcar.exists():
                continue
            E = self._read_toten(outcar)
            if E is None:
                continue
            poscar = opt_dir / "CONTCAR"
            if not poscar.exists():
                poscar = opt_dir / "POSCAR"
            comp = _read_poscar_composition(poscar) if poscar.exists() else {}
            if comp:
                results.append((sub_dir.name, comp, E))
        return results

    # ── Public interface ─────────────────────────────────────────────────────

    def can_run(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True when a completed DFT opt OUTCAR or Arrhenius CSV is available."""
        if self._is_sse(project_dir):
            # Ready as soon as at least one sibling sub-project has a completed opt
            return bool(self._collect_sibling_opts(project_dir))
        return (
            (dft_opt(project_dir) / "OUTCAR").exists()
            or (project_dir / "Analysis" / "arrhenius.csv").exists()
        )

    def is_complete(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True when echem_summary.json exists and no newer sibling opt OUTCAR was written."""
        echem_dir = self._echem_root(project_dir)
        summary_path = echem_dir / "echem_summary.json"
        if not summary_path.exists():
            return False
        if self._is_sse(project_dir):
            # Re-run if a sibling's opt OUTCAR is newer than the summary
            # (a new sub-project finished DFT opt while echem was already done)
            summary_mtime = summary_path.stat().st_mtime
            parent = project_dir.parent
            for sub_dir in sorted(parent.iterdir()):
                if not sub_dir.is_dir() or sub_dir.name.startswith("."):
                    continue
                outcar = dft_opt(sub_dir) / "OUTCAR"
                if outcar.exists() and outcar.stat().st_mtime > summary_mtime:
                    log.info("[h08_echem] SSE: new opt data in %s — re-running echem",
                             sub_dir.name)
                    return False
        return True

    def submit(self, project_dir: Path, state: "ProjectState") -> str | None:
        """Run formation energy, EAH, electrochemical window, and OCV calculations in-process."""
        echem_dir = self._echem_root(project_dir)
        echem_dir.mkdir(parents=True, exist_ok=True)
        _ensure_cladue_env()

        if self._is_sse(project_dir):
            return self._submit_sse(project_dir, echem_dir, state)
        return self._submit_standard(project_dir, echem_dir, state)

    # ── SSE: shared multi-structure echem run ────────────────────────────────

    def _submit_sse(self, project_dir: Path, echem_dir: Path,
                    state: "ProjectState") -> str | None:
        """Collect all sibling DFT opt structures and run electrochemical analysis once."""
        lock_path = echem_dir / ".running"

        # Atomic lock: skip if another sub-project orchestrator is already running this
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
        except FileExistsError:
            log.info("[h08_echem] SSE echem already running (lock exists) — skipping")
            return None

        try:
            sibling_opts = self._collect_sibling_opts(project_dir)
            if not sibling_opts:
                log.warning("[h08_echem] SSE: no sibling opt data found")
                return None

            log.info("[h08_echem] SSE echem: collecting %d sub-projects from %s",
                     len(sibling_opts), project_dir.parent)

            summary: dict = {"n_structures": len(sibling_opts)}

            # Write structure inventory CSV
            inv_rows = ["sub_project,formula,E_total_eV,n_atoms,E_per_atom_eV"]
            for name, comp, E in sibling_opts:
                n = sum(comp.values())
                formula = "".join(f"{el}{cnt}" for el, cnt in sorted(comp.items()))
                inv_rows.append(f"{name},{formula},{E:.6f},{n},{E/n:.6f}")
            (echem_dir / "structure_inventory.csv").write_text(
                "\n".join(inv_rows) + "\n"
            )
            log.info("[h08_echem] Wrote structure_inventory.csv (%d structures)", len(sibling_opts))

            # Electrochemical window: use host (pure / undoped) structure
            # Host is identified as the sub-project with the fewest unique elements
            host_name, host_comp, host_E = min(
                sibling_opts, key=lambda x: len(x[1])
            )
            log.info("[h08_echem] Host structure: %s (%s)", host_name,
                     "".join(f"{el}{cnt}" for el, cnt in sorted(host_comp.items())))

            # Formation energies: each sub-project relative to host + dopant references
            fe_rows = self._compute_sse_formation_energies(
                sibling_opts, host_name, host_comp, host_E, echem_dir
            )
            if fe_rows:
                summary["formation_energies"] = fe_rows

            # Electrochemical window for EVERY sub-project → "species" list for multi-bar plot
            species_list: list[dict] = []
            for name, comp, _E in sibling_opts:
                sub_dir = project_dir.parent / name
                win = self._compute_echem_window(sub_dir, echem_dir)
                if win and "V_ox" in win:
                    species_list.append({
                        "name": name,
                        "V_red": win.get("V_red", 0.0),
                        "V_ox":  win["V_ox"],
                        "window_V": win.get("window_V", win["V_ox"] - win.get("V_red", 0.0)),
                    })
                    log.info("[h08_echem] %s: V_red=%.2f V  V_ox=%.2f V",
                             name, win.get("V_red", 0), win["V_ox"])
            if species_list:
                summary["species"] = species_list
                log.info("[h08_echem] SSE: built ECW species list (%d compounds)", len(species_list))

            # EAH for each sub-project structure
            eah_results = self._compute_sse_eah(sibling_opts, echem_dir)
            if eah_results:
                summary["eah_per_structure"] = eah_results

            # Write to handler-local path and canonical registry path
            summary_path = echem_dir / "echem_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2))
            canonical = project_dir / "results" / "data" / "echem.json"
            canonical.parent.mkdir(parents=True, exist_ok=True)
            canonical.write_text(json.dumps(summary, indent=2))
            log.info("[h08_echem] SSE: wrote %s  +  %s", summary_path, canonical)

        finally:
            lock_path.unlink(missing_ok=True)

        state.set_stage("h08_echem", "COMPLETE", summary={"sse_echem_dir": str(echem_dir)})
        return None

    def _compute_sse_formation_energies(
        self,
        sibling_opts: list[tuple[str, dict, float]],
        host_name: str,
        host_comp: dict,
        host_E: float,
        echem_dir: Path,
    ) -> list[dict]:
        """Formation energy of each doped structure relative to host + element references."""
        rows: list[dict] = []
        try:
            from mp_api.client import MPRester
            all_elements = set()
            for _, comp, _ in sibling_opts:
                all_elements.update(comp.keys())

            ref_energies: dict[str, float] = {}
            with MPRester(MP_API_KEY) as mpr:
                for el in all_elements:
                    entries = mpr.get_entries(el)
                    if entries:
                        ref_energies[el] = min(
                            e.energy / e.composition.num_atoms for e in entries
                        )

            host_n = sum(host_comp.values())
            csv_rows = ["sub_project,formula,n_atoms,E_total_eV,E_form_eV_per_atom"]

            for name, comp, E in sibling_opts:
                n = sum(comp.values())
                formula = "".join(f"{el}{cnt}" for el, cnt in sorted(comp.items()))
                E_ref = sum(comp.get(el, 0) * ref_energies.get(el, 0.0) for el in comp)
                E_form = (E - E_ref) / n if n else None
                entry = {"sub_project": name, "formula": formula,
                         "E_form_eV_per_atom": round(E_form, 6) if E_form else None}
                rows.append(entry)
                ef_str = f"{E_form:.6f}" if E_form is not None else "N/A"
                csv_rows.append(f"{name},{formula},{n},{E:.6f},{ef_str}")

            (echem_dir / "formation_energies.csv").write_text("\n".join(csv_rows) + "\n")
            log.info("[h08_echem] SSE: wrote formation_energies.csv")
        except Exception as exc:
            log.warning("[h08_echem] SSE formation energy calc failed: %s", exc)
        return rows

    def _compute_sse_eah(
        self,
        sibling_opts: list[tuple[str, dict, float]],
        echem_dir: Path,
    ) -> list[dict]:
        """Energy above hull for each SSE sub-project structure."""
        results: list[dict] = []
        try:
            from pymatgen.core import Composition
            from pymatgen.entries.computed_entries import ComputedEntry
            from pymatgen.analysis.phase_diagram import PhaseDiagram
            from mp_api.client import MPRester

            all_elements = set()
            for _, comp, _ in sibling_opts:
                all_elements.update(comp.keys())

            with MPRester(MP_API_KEY) as mpr:
                ref_entries = mpr.get_entries_in_chemsys(list(all_elements))

            if not ref_entries:
                return results

            eah_rows = ["sub_project,formula,EAH_eV_per_atom"]
            for name, comp, E in sibling_opts:
                try:
                    pmg_comp = Composition(comp)
                    our_entry = ComputedEntry(pmg_comp, E)
                    all_entries = list(ref_entries) + [our_entry]
                    pd = PhaseDiagram(all_entries)
                    eah = pd.get_e_above_hull(our_entry)
                    formula = pmg_comp.reduced_formula
                    results.append({"sub_project": name, "formula": formula,
                                    "EAH_eV_per_atom": round(eah, 6)})
                    eah_rows.append(f"{name},{formula},{eah:.6f}")
                    log.info("[h08_echem] EAH %s: %.4f eV/atom", name, eah)
                except Exception as exc:
                    log.debug("[h08_echem] EAH failed for %s: %s", name, exc)

            (echem_dir / "eah_all_structures.csv").write_text("\n".join(eah_rows) + "\n")
            log.info("[h08_echem] SSE: wrote eah_all_structures.csv")
        except Exception as exc:
            log.warning("[h08_echem] SSE EAH calc failed: %s", exc)
        return results

    # ── Standard (non-SSE) submit ────────────────────────────────────────────

    def _submit_standard(self, project_dir: Path, echem_dir: Path,
                         state: "ProjectState") -> str | None:
        """Run all echem analyses for a non-SSE project and write echem_summary.json."""
        summary: dict = {}

        # Formation energy
        ef = self._compute_formation_energy(project_dir, echem_dir)
        if ef is not None:
            summary["formation_energy_eV_per_atom"] = ef
            log.info("[h08_echem] Formation energy: %.4f eV/atom", ef)

        # Energy above hull
        eah = self._compute_eah(project_dir, echem_dir)
        if eah is not None:
            summary["energy_above_hull_eV_per_atom"] = eah
            log.info("[h08_echem] EAH: %.4f eV/atom", eah)

        # Electrochemical window
        window = self._compute_echem_window(project_dir, echem_dir)
        if window:
            summary.update(window)
            log.info("[h08_echem] Window: V_red=%.3f V  V_ox=%.3f V  W=%.3f V",
                     window.get("V_red", 0), window.get("V_ox", 0), window.get("window_V", 0))

        # OCV
        ocv = self._compute_ocv(project_dir, echem_dir)
        if ocv:
            summary.update(ocv)

        # Write summary to handler-local path and canonical registry path
        summary_path = echem_dir / "echem_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        canonical = project_dir / "results" / "data" / "echem.json"
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_text(json.dumps(summary, indent=2))
        log.info("[h08_echem] Wrote %s  +  %s", summary_path, canonical)

        state.set_stage("h08_echem", "COMPLETE", summary=summary)
        return None

    # ── Formation energy ────────────────────────────────────────────────────

    def _compute_formation_energy(self, project_dir: Path, echem_dir: Path) -> float | None:
        """Query MP for elemental reference energies and return formation energy in eV/atom."""
        outcar = dft_opt(project_dir) / "OUTCAR"
        if not outcar.exists():
            return None

        E_total = self._read_toten(outcar)
        if E_total is None:
            return None

        poscar = dft_opt(project_dir) / "CONTCAR"
        if not poscar.exists():
            poscar = dft_opt(project_dir) / "POSCAR"
        if not poscar.exists():
            return None

        composition = _read_poscar_composition(poscar)
        if not composition:
            return None

        n_atoms = sum(composition.values())

        try:
            from mp_api.client import MPRester
            ref_energies: dict[str, float] = {}

            with MPRester(MP_API_KEY) as mpr:
                for el in composition:
                    entries = mpr.get_entries(el)
                    if entries:
                        # Use lowest energy entry per atom
                        e_per_atom = min(
                            e.energy / e.composition.num_atoms for e in entries
                        )
                        ref_energies[el] = e_per_atom

            E_ref_total = sum(
                composition[el] * ref_energies.get(el, 0.0) for el in composition
            )
            E_form = (E_total - E_ref_total) / n_atoms

            # Save CSV
            rows = ["element,n_atoms,ref_energy_eV_per_atom"]
            for el, n in composition.items():
                rows.append(f"{el},{n},{ref_energies.get(el, 'N/A')}")
            rows.append(f"TOTAL,{n_atoms},—")
            rows.append(f"E_formation_eV_per_atom,{E_form:.6f},—")
            (echem_dir / "formation_energy.csv").write_text("\n".join(rows) + "\n")

            return E_form

        except Exception as exc:
            log.warning("[h08_echem] Formation energy via MP failed: %s", exc)
            return None

    # ── Energy above hull ───────────────────────────────────────────────────

    def _compute_eah(self, project_dir: Path, echem_dir: Path) -> float | None:
        """Compute energy above hull using pymatgen PhaseDiagram + MP competing phases."""
        outcar = dft_opt(project_dir) / "OUTCAR"
        poscar = dft_opt(project_dir) / "CONTCAR"
        if not poscar.exists():
            poscar = dft_opt(project_dir) / "POSCAR"

        if not (outcar.exists() and poscar.exists()):
            return None

        E_total = self._read_toten(outcar)
        if E_total is None:
            return None

        composition = _read_poscar_composition(poscar)
        if not composition:
            return None

        try:
            from pymatgen.core import Composition
            from pymatgen.entries.computed_entries import ComputedEntry
            from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
            from mp_api.client import MPRester

            comp = Composition(composition)
            n_atoms = comp.num_atoms

            # Query all competing phases
            el_str = "-".join(sorted(composition.keys()))
            with MPRester(MP_API_KEY) as mpr:
                entries = mpr.get_entries_in_chemsys(list(composition.keys()))

            if not entries:
                return None

            # Add our computed entry
            our_entry = ComputedEntry(comp, E_total)
            all_entries = list(entries) + [our_entry]
            pd = PhaseDiagram(all_entries)
            eah = pd.get_e_above_hull(our_entry)

            # Save CSV
            (echem_dir / "eah.csv").write_text(
                f"formula,EAH_eV_per_atom\n{comp.reduced_formula},{eah:.6f}\n"
            )
            return eah

        except Exception as exc:
            log.warning("[h08_echem] EAH via MP failed: %s", exc)
            return None

    # ── Electrochemical window ──────────────────────────────────────────────

    # Oxidation potentials of halides vs Li/Li+ (V) — literature consensus
    _HAL_VOX: dict[str, float] = {"F": 6.0, "Cl": 4.3, "Br": 3.5, "I": 2.9}

    def _compute_echem_window(self, project_dir: Path, echem_dir: Path) -> dict | None:
        """Estimate V_red/V_ox from halide composition and optionally MP GrandPotentialPhaseDiagram."""
        poscar = dft_opt(project_dir) / "CONTCAR"
        if not poscar.exists():
            poscar = dft_opt(project_dir) / "POSCAR"
        if not poscar.exists():
            return None

        composition = _read_poscar_composition(poscar)
        if not composition:
            return None

        yaml = self.read_project_yaml(project_dir)

        # Manual override takes precedence
        V_red = yaml.get("V_red_manual", 0.0)   # Li plating = 0 V by definition
        V_ox  = yaml.get("V_ox_manual", None)

        # V_ox is set by the weakest (lowest-potential) halide channel in the structure;
        # the material oxidises at whichever anion decomposes first.
        if V_ox is None:
            present = {h: v for h, v in self._HAL_VOX.items() if composition.get(h, 0) > 0}
            if present:
                V_ox = round(min(present.values()), 3)
                log.info("[h08_echem] V_ox = %.3f V (weakest halide channel: %s)",
                         V_ox, present)

        # Try MP GrandPotentialPhaseDiagram for a physics-based window
        try:
            _ensure_cladue_env()
            from pymatgen.core import Composition as PMGComp
            from pymatgen.entries.computed_entries import ComputedEntry
            from pymatgen.analysis.phase_diagram import PhaseDiagram, GrandPotentialPhaseDiagram
            from mp_api.client import MPRester

            with MPRester(MP_API_KEY) as mpr:
                entries = mpr.get_entries_in_chemsys(list(composition.keys()))

            if entries:
                pd = PhaseDiagram(entries)
                # Sweep Li chemical potential: find stability boundaries
                working_ion = yaml.get("mobile_ion", "Li")
                comp_formula = PMGComp(composition).reduced_formula
                matching = [e for e in entries
                            if e.composition.reduced_formula == comp_formula]
                if matching:
                    target = min(matching, key=lambda e: e.energy_per_atom)
                    eah = pd.get_e_above_hull(target)
                    log.info("[h08_echem] EAH (MP) = %.4f eV/atom", eah)

                    # GrandPotentialPD: reduction potential where material becomes unstable
                    # at mu_Li = 0 (Li/Li+ reference)
                    gppd = GrandPotentialPhaseDiagram(entries, {working_ion: 0.0})
                    # Decomposition products at mu_Li = -V_red = 0
                    # Use the known Li+ reference: V_red = 0 V by convention
                    log.info("[h08_echem] GPPD computed for %s", comp_formula)

        except Exception as exc:
            log.debug("[h08_echem] MP window query failed (using halide estimate): %s", exc)

        result: dict = {"V_red": V_red}
        if V_ox is not None:
            result["V_ox"]    = V_ox
            result["window_V"] = round(V_ox - V_red, 3)

        # Read NEB barriers to annotate (Ea correlates with Li+ mobility, not window)
        neb_file = project_dir / "results" / "neb_barriers.json"
        if neb_file.exists():
            try:
                neb_data = json.loads(neb_file.read_text())
                min_ea = min(
                    (v["Ea_meV"] for v in neb_data.values()
                     if isinstance(v, dict) and v.get("Ea_meV") is not None),
                    default=None,
                )
                if min_ea is not None:
                    result["min_Ea_meV"] = min_ea
                    log.info("[h08_echem] min NEB barrier = %.1f meV", min_ea)
            except Exception:
                pass

        if result:
            rows = [f"{k},{v}" for k, v in result.items()]
            (echem_dir / "echem_window.csv").write_text(
                "quantity,value\n" + "\n".join(rows) + "\n"
            )

        return result or None

    # ── OCV ─────────────────────────────────────────────────────────────────

    def _compute_ocv(self, project_dir: Path, echem_dir: Path) -> dict | None:
        """Compute OCV from echem_static DFT energies and ionic conductivity via Nernst-Einstein."""
        yaml = self.read_project_yaml(project_dir)
        result: dict = {}

        # Nernst-Einstein: σ = D * q² * c * N / (kB * T)
        # h06 writes Analysis/{variant}/arrhenius.csv; prefer aimd, then any variant
        _arrh_candidates = sorted(project_dir.glob("Analysis/*/arrhenius.csv"))
        _aimd_csv = project_dir / "Analysis" / "aimd" / "arrhenius.csv"
        arrh_csv = _aimd_csv if _aimd_csv.exists() else (_arrh_candidates[0] if _arrh_candidates else None)
        if arrh_csv and arrh_csv.exists():
            try:
                import numpy as np
                data = np.loadtxt(str(arrh_csv), delimiter=",", skiprows=1)
                if data.size == 0:
                    raise ValueError("arrhenius.csv is empty")
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                T_arr = data[:, 0]
                D_arr = data[:, 1]
                Ea_eV = data[0, 4] if data.shape[1] > 4 else None

                # Conductivity at 300 K
                T0 = 300.0
                if len(D_arr) > 0:
                    idx = np.argmin(np.abs(T_arr - T0))
                    D_300 = D_arr[idx]
                    # σ ≈ D * q² * c_carrier / (kB * T) [S/m, rough]
                    q = 1.602e-19
                    kB = 1.381e-23
                    c = yaml.get("carrier_density_m3", 1e27)  # Li density
                    sigma = D_300 * q * q * c / (kB * T0)
                    result["D_300K_m2s"] = float(D_300)
                    result["sigma_300K_S_m"] = float(sigma)
                    if Ea_eV is not None:
                        result["Ea_eV"] = float(Ea_eV)
                    log.info("[h08_echem] σ(300K) = %.3e S/m", sigma)
            except Exception as exc:
                log.warning("[h08_echem] Nernst-Einstein calc failed: %s", exc)

        # OCV from Li insertion/removal energetics
        # E_Li_metal per atom (standard DFT-PBE reference)
        E_Li_metal = yaml.get("E_Li_metal_eV", -1.908)
        mobile_ion  = yaml.get("mobile_ion", "Li")
        # Collect all (n_Li, E_total) pairs from OUTCAR + POSCAR
        echem_states: list[tuple[int, float]] = []
        for candidate in sorted(project_dir.glob("echem_static*")):
            try:
                outcar = candidate / "OUTCAR"
                poscar = candidate / "POSCAR"
                if not outcar.exists():
                    continue
                E = self._read_toten(outcar)
                if E is None:
                    continue
                n_li = 0
                if poscar.exists():
                    comp = _read_poscar_composition(poscar)
                    n_li = comp.get(mobile_ion, 0)
                else:
                    n_li = yaml.get("n_extracted_ions", 0)
                echem_states.append((n_li, E))
            except Exception as exc:
                log.debug("[h08_echem] echem_static parse failed: %s", exc)
        # Add fully-lithiated reference (from DFT opt)
        opt_outcar = dft_opt(project_dir) / "OUTCAR"
        opt_poscar = dft_opt(project_dir) / "POSCAR"
        if opt_outcar.exists():
            E_full = self._read_toten(opt_outcar)
            if E_full is not None:
                comp_full = _read_poscar_composition(opt_poscar) if opt_poscar.exists() else {}
                n_li_full = comp_full.get(mobile_ion, yaml.get("n_extracted_ions", 1))
                echem_states.append((n_li_full, E_full))
        # Sort by n_Li descending (lithiated → delithiated)
        echem_states.sort(key=lambda x: -x[0])
        if len(echem_states) >= 2:
            try:
                import csv as _csv
                voltages = []
                for i in range(len(echem_states) - 1):
                    n_hi, E_hi = echem_states[i]
                    n_lo, E_lo = echem_states[i + 1]
                    dn = n_hi - n_lo
                    if dn <= 0:
                        continue
                    # V = -(E_hi - E_lo - dn * E_Li_metal) / dn
                    V = -(E_hi - E_lo - dn * E_Li_metal) / dn
                    x_avg = 0.5 * (n_hi + n_lo)
                    voltages.append((x_avg, V, n_hi, n_lo))
                if voltages:
                    vp_path = echem_dir / "voltage_profile.csv"
                    with vp_path.open("w", newline="") as fh:
                        writer = _csv.writer(fh)
                        writer.writerow(["x_Li_avg", "voltage_V", "n_hi", "n_lo"])
                        for row in voltages:
                            writer.writerow([f"{v:.4f}" for v in row])
                    avg_ocv = sum(v[1] for v in voltages) / len(voltages)
                    result["OCV_V"] = float(avg_ocv)
                    result["n_voltage_steps"] = len(voltages)
                    log.info("[h08_echem] OCV = %.3f V (%d steps)", avg_ocv, len(voltages))

                    # Capacity vs voltage
                    self._compute_capacity_vs_voltage(
                        project_dir, echem_dir, voltages, yaml
                    )
            except Exception as exc:
                log.debug("[h08_echem] OCV voltage profile failed: %s", exc)
        elif len(echem_states) <= 1:
            log.info("[h08_echem] OCV skipped — no echem_static*/OUTCAR found "
                     "(need at least one delithiated structure alongside dft/opt/)")

        # NEB migration Ea
        neb_file = project_dir / "results" / "neb_barriers.json"
        if neb_file.exists():
            try:
                neb_data = json.loads(neb_file.read_text())
                if neb_data:
                    barriers = [v["Ea_meV"] for v in neb_data.values()
                                if isinstance(v, dict) and v.get("Ea_meV")]
                    if barriers:
                        result["Ea_migration_meV"] = float(min(barriers))
            except Exception:
                pass

        if result:
            (echem_dir / "ocv.json").write_text(json.dumps(result, indent=2))

        return result or None

    # ── Capacity vs voltage ─────────────────────────────────────────────────

    # Atomic masses (g/mol) for common SSE elements
    _ATOMIC_MASS: dict[str, float] = {
        "Li": 6.941,  "Na": 22.990, "Mg": 24.305, "Ca": 40.078,
        "Y":  88.906, "La": 138.905, "Zr": 91.224, "Ta": 180.948,
        "Nb": 92.906, "In": 114.818,
        "F":  18.998, "Cl": 35.453, "Br": 79.904,  "I":  126.904,
        "O":  15.999, "S":  32.065, "P":  30.974,
    }
    _FARADAY = 96485.0  # C/mol

    def _compute_capacity_vs_voltage(
        self,
        project_dir: Path,
        echem_dir: Path,
        voltages: list[tuple],   # [(x_avg, V, n_hi, n_lo), ...]
        yaml: dict,
    ) -> None:
        """Write capacity_vs_voltage.csv for plotting C (mAh/g) vs voltage."""
        if not voltages:
            return

        poscar = dft_opt(project_dir) / "CONTCAR"
        if not poscar.exists():
            poscar = dft_opt(project_dir) / "POSCAR"
        if not poscar.exists():
            return

        composition = _read_poscar_composition(poscar)
        if not composition:
            return

        # Total formula-unit mass (g/mol of the full supercell)
        M_total = sum(n * self._ATOMIC_MASS.get(el, 50.0)
                      for el, n in composition.items())
        if M_total <= 0:
            return

        import csv as _csv
        rows = []
        cumulative_Li = 0
        for x_avg, V, n_hi, n_lo in voltages:
            dn = int(round(n_hi - n_lo))
            if dn <= 0:
                continue
            cumulative_Li += dn
            # mAh/g = (n_removed × F) / (M_total [g/mol] × 3600 [s/h]) × 1000 mA/A
            capacity_mAhg = (cumulative_Li * self._FARADAY) / (M_total * 3600.0) * 1000.0
            rows.append((cumulative_Li, round(capacity_mAhg, 4), round(V, 4), round(x_avg, 4)))

        if not rows:
            return

        cv_path = echem_dir / "capacity_vs_voltage.csv"
        with cv_path.open("w", newline="") as fh:
            writer = _csv.writer(fh)
            writer.writerow(["n_Li_removed", "capacity_mAhg", "voltage_V", "x_Li_avg"])
            for row in rows:
                writer.writerow(row)

        max_cap = rows[-1][1]
        log.info("[h08_echem] Wrote capacity_vs_voltage.csv: %d steps, max %.1f mAh/g",
                 len(rows), max_cap)

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _read_toten(outcar_path: Path) -> float | None:
        """Read last TOTEN = value from OUTCAR."""
        last_e = None
        try:
            with open(outcar_path, errors="replace") as fh:
                for line in fh:
                    if "TOTEN" in line and "=" in line:
                        try:
                            val = float(line.split("=")[1].strip().split()[0])
                            last_e = val
                        except (IndexError, ValueError):
                            pass
        except Exception as exc:
            log.warning("[h08_echem] OUTCAR read failed: %s", exc)
        return last_e


# ── Module-level helpers ─────────────────────────────────────────────────────────

def _ensure_cladue_env() -> None:
    """Prepend the cladue site-packages directory to sys.path if not already present."""
    if _CLADUE_SITE not in sys.path:
        sys.path.insert(0, _CLADUE_SITE)


def _read_poscar_composition(poscar_path: Path) -> dict[str, int]:
    """Return {element: count} from POSCAR lines 6-7."""
    lines = poscar_path.read_text().splitlines()
    if len(lines) < 7:
        return {}
    try:
        species = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
        return {sp: cnt for sp, cnt in zip(species, counts)}
    except (ValueError, IndexError):
        return {}
