"""
h07_electronic.py — Electronic characterization handler (daemon-local).
Parses Bader charges from ACF.dat, DOS from DOSCAR, band gap from vasprun.xml.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .base import SimulationHandler
from hpca.core.paths import dft_opt, dft_base

if TYPE_CHECKING:
    from hpca.orchestrator.state_tracker import ProjectState

log = logging.getLogger("hpca.orch")
# Layout: see hpca/core/paths.py

# PAW valence electron table (for charge transfer = PAW - Bader)
def _paw_valence() -> dict[str, int]:
    """Lazy-load paw_valence from hpca/data/paw_valence.json."""
    from hpca.data import load
    return load("paw_valence")

NREL_COLORS = ["#0079C2", "#F7A11A", "#5E9732", "#D1495B", "#6A0572",
               "#00B4D8", "#F4A261", "#6B6B6B"]


class ElectronicHandler(SimulationHandler):
    """Daemon-local handler: Bader charges + DOS parsing + band gap extraction."""

    name = "h07_electronic"
    is_daemon = True

    def can_run(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True for non-pure-molecular projects with ACF.dat (Bader) or a DOSCAR present."""
        yaml = self.read_project_yaml(project_dir)
        category = yaml.get("category", "")
        from hpca.core.categories import is_sse as _is_sse, is_molecular as _is_mol
        if _is_mol(category) and not _is_sse(category):
            return False
        return (
            (project_dir / "bader" / "ACF.dat").exists()
            or (project_dir / "dos" / "nonscf" / "DOSCAR").exists()
        )

    def is_complete(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True when at least one CSV exists in results/electronic/."""
        elec_dir = project_dir / "results" / "electronic"
        if not elec_dir.is_dir():
            return False
        csvs = list(elec_dir.glob("*.csv"))
        return len(csvs) >= 1

    def submit(self, project_dir: Path, state: "ProjectState") -> str | None:
        """Parse Bader charges, DOS, and band gap in-process; write CSVs and mark COMPLETE."""
        elec_dir = project_dir / "results" / "electronic"
        elec_dir.mkdir(parents=True, exist_ok=True)

        results: dict = {}

        if (project_dir / "bader" / "ACF.dat").exists():
            bader_result = self._parse_bader(project_dir, elec_dir)
            results["bader"] = bader_result

        if (project_dir / "dos" / "nonscf" / "DOSCAR").exists():
            dos_result = self._parse_dos(project_dir, elec_dir)
            results["dos"] = dos_result

        # Also try vasprun.xml for band gap
        vasprun = project_dir / "dos" / "nonscf" / "vasprun.xml"
        if not vasprun.exists():
            vasprun = dft_opt(project_dir) / "vasprun.xml"
        if vasprun.exists():
            gap = self._parse_bandgap_vasprun(vasprun)
            if gap is not None:
                results["band_gap_eV"] = gap
                log.info("[h07_electronic] Band gap: %.3f eV", gap)
                work_dir = vasprun.parent
                hl = self._compute_homo_lumo(work_dir, gap_eV=gap)
                if hl:
                    results["homo_lumo"] = hl

        state.set_stage("h07_electronic", "COMPLETE", results=results)
        log.info("[h07_electronic] COMPLETE for %s", project_dir.name)
        return None

    def _compute_homo_lumo(self, work_dir: Path, gap_eV: float = 0.0) -> dict | None:
        """Extract HOMO/LUMO energies from vasprun.xml or DOSCAR."""
        import json
        vasprun = work_dir / "vasprun.xml"
        doscar  = work_dir / "DOSCAR"
        try:
            if vasprun.exists():
                import xml.etree.ElementTree as ET
                tree = ET.parse(str(vasprun))
                root = tree.getroot()
                efermi = None
                for elem in root.iter("i"):
                    if elem.get("name") == "efermi":
                        efermi = float(elem.text)
                        break
                if efermi is not None:
                    result = {
                        "efermi_eV": efermi,
                        "homo_eV":   efermi - gap_eV / 2,
                        "lumo_eV":   efermi + gap_eV / 2,
                        "gap_eV":    gap_eV,
                    }
                    (work_dir / "homo_lumo.json").write_text(json.dumps(result, indent=2))
                    log.info("[h07] HOMO=%.3f eV  LUMO=%.3f eV  gap=%.3f eV",
                             result["homo_eV"], result["lumo_eV"], gap_eV)
                    return result
            if doscar.exists():
                lines = doscar.read_text().splitlines()
                efermi = float(lines[5].split()[3])
                result = {"efermi_eV": efermi, "homo_eV": efermi, "lumo_eV": efermi, "gap_eV": 0.0}
                (work_dir / "homo_lumo.json").write_text(json.dumps(result, indent=2))
                return result
        except Exception as exc:
            log.debug("[h07] HOMO/LUMO extraction failed: %s", exc)
        return None

    # ── Bader ─────────────────────────────────────────────────────────────────

    def _parse_bader(self, project_dir: Path, elec_dir: Path) -> dict:
        """Read ACF.dat, match atoms to POSCAR elements, compute charge transfer, and write CSV."""
        acf_path = project_dir / "bader" / "ACF.dat"
        poscar = self._find_poscar(project_dir)

        # Read ACF.dat
        atoms: list[dict] = []
        with open(acf_path) as fh:
            lines = fh.readlines()

        # Skip first 2 header lines; stop at "---" separator
        for line in lines[2:]:
            if line.strip().startswith("---") or line.strip().startswith("VACUUM"):
                break
            parts = line.split()
            if len(parts) >= 5:
                try:
                    atoms.append({
                        "idx": int(parts[0]),
                        "x": float(parts[1]),
                        "y": float(parts[2]),
                        "z": float(parts[3]),
                        "bader_charge": float(parts[4]),
                    })
                except ValueError:
                    continue

        # Get element sequence from POSCAR
        elements_seq: list[str] = []
        if poscar and poscar.exists():
            elements_seq = self._read_poscar_elements(poscar)

        # Pad elements if mismatch
        if len(elements_seq) < len(atoms):
            elements_seq.extend(["X"] * (len(atoms) - len(elements_seq)))

        # Build output
        rows: list[dict] = []
        for i, atom in enumerate(atoms):
            el = elements_seq[i] if i < len(elements_seq) else "X"
            paw = _paw_valence().get(el, 0)
            ct = paw - atom["bader_charge"]
            rows.append({
                "atom_idx": atom["idx"],
                "element": el,
                "bader_charge": atom["bader_charge"],
                "paw_valence": paw,
                "charge_transfer": ct,
            })
            log.debug("[h07_electronic] Atom %d (%s): Bader=%.3f CT=%.3f",
                      atom["idx"], el, atom["bader_charge"], ct)

        # Write CSV
        csv_path = elec_dir / "bader_charges.csv"
        header = "atom_idx,element,bader_charge,paw_valence,charge_transfer"
        lines_out = [header]
        for r in rows:
            lines_out.append(
                f"{r['atom_idx']},{r['element']},{r['bader_charge']:.4f},"
                f"{r['paw_valence']},{r['charge_transfer']:.4f}"
            )
        csv_path.write_text("\n".join(lines_out) + "\n")
        log.info("[h07_electronic] Wrote %s (%d atoms)", csv_path, len(rows))

        # Bar chart
        self._plot_bader(rows, elec_dir)

        return {"n_atoms": len(rows), "csv": str(csv_path)}

    def _plot_bader(self, rows: list[dict], elec_dir: Path) -> None:
        """Save a bar chart of per-atom charge transfer coloured by element to elec_dir."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            elements = [r["element"] for r in rows]
            ct_vals = [r["charge_transfer"] for r in rows]
            idxs = [r["atom_idx"] for r in rows]

            unique_el = list(dict.fromkeys(elements))
            color_map = {el: NREL_COLORS[i % len(NREL_COLORS)]
                         for i, el in enumerate(unique_el)}
            colors = [color_map[el] for el in elements]

            fig, ax = plt.subplots(figsize=(max(8, len(rows) // 4), 6))
            ax.bar(idxs, ct_vals, color=colors)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Atom index")
            ax.set_ylabel("Charge transfer (e⁻)")
            ax.set_title("Bader Charge Transfer")

            # Legend
            from matplotlib.patches import Patch
            handles = [Patch(color=color_map[el], label=el) for el in unique_el]
            ax.legend(handles=handles, loc="best")

            fig.tight_layout()
            fig.savefig(str(elec_dir / "bader_charges.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            log.debug("[h07_electronic] Bader plot failed: %s", exc)

    # ── DOS ──────────────────────────────────────────────────────────────────

    def _parse_dos(self, project_dir: Path, elec_dir: Path) -> dict:
        """Parse DOSCAR to extract total DOS, compute band gap, write CSV and PNG."""
        doscar_path = project_dir / "dos" / "nonscf" / "DOSCAR"

        with open(doscar_path) as fh:
            lines = fh.readlines()

        # Header: line 1: nions nkpts nedos ...; line 6: Emax Emin nedos Efermi
        try:
            header6 = lines[5].split()
            emax = float(header6[0])
            emin = float(header6[1])
            nedos = int(header6[2])
            efermi = float(header6[3])
        except (IndexError, ValueError) as exc:
            log.error("[h07_electronic] DOSCAR header parse failed: %s", exc)
            return {}

        # Parse total DOS (nedos lines after header)
        dos_start = 6
        energies: list[float] = []
        dos_up: list[float] = []
        dos_dn: list[float] = []

        for line in lines[dos_start: dos_start + nedos]:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    energies.append(float(parts[0]))
                    dos_up.append(float(parts[1]))
                    dos_dn.append(float(parts[2]))
                except ValueError:
                    continue
            elif len(parts) == 2:
                try:
                    energies.append(float(parts[0]))
                    dos_up.append(float(parts[1]))
                    dos_dn.append(0.0)
                except ValueError:
                    continue

        # Find band gap
        band_gap = self._compute_bandgap(energies, dos_up, dos_dn, efermi)
        log.info("[h07_electronic] DOS efermi=%.3f eV  band_gap=%.3f eV",
                 efermi, band_gap if band_gap else -1)

        # Save CSV
        try:
            import numpy as np
            csv_path = elec_dir / "dos_total.csv"
            data = np.column_stack([energies, dos_up, dos_dn])
            np.savetxt(str(csv_path), data, delimiter=",",
                       header="energy_eV,dos_up,dos_down", comments="")
        except Exception as exc:
            log.warning("[h07_electronic] DOS CSV save failed: %s", exc)

        # Plot DOS
        self._plot_dos(energies, dos_up, dos_dn, efermi, band_gap, elec_dir)

        # Attempt PDOS if LORBIT=11
        incar = dft_opt(project_dir) / "INCAR"
        if self._incar_has_lorbit11(incar) and len(lines) > dos_start + nedos + 2:
            self._parse_pdos(lines, dos_start + nedos, nedos, energies, elec_dir)

        return {
            "efermi": efermi, "band_gap_eV": band_gap, "nedos": nedos,
            "csv": str(elec_dir / "dos_total.csv"),
        }

    def _compute_bandgap(
        self, energies: list, dos_up: list, dos_dn: list,
        efermi: float, threshold: float = 0.01
    ) -> float | None:
        """Find gap: last energy below efermi with total DOS > threshold, first above."""
        try:
            combined = [u + d for u, d in zip(dos_up, dos_dn)]
            below = [e for e, d in zip(energies, combined) if e <= efermi and d > threshold]
            above = [e for e, d in zip(energies, combined) if e >= efermi and d > threshold]
            if not below or not above:
                return None
            vbm = max(below)
            cbm = min(above)
            gap = cbm - vbm
            return gap if gap > 0.01 else None
        except Exception:
            return None

    def _plot_dos(
        self, energies: list, dos_up: list, dos_dn: list,
        efermi: float, band_gap, elec_dir: Path
    ) -> None:
        """Save a filled total DOS plot aligned to Fermi level to elec_dir."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            e_arr = np.array(energies) - efermi  # shift to Fermi level
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.fill_between(e_arr, dos_up, alpha=0.6, color=NREL_COLORS[0], label="DOS up")
            if any(d != 0 for d in dos_dn):
                ax.fill_between(e_arr, [-d for d in dos_dn], alpha=0.6,
                                color=NREL_COLORS[1], label="DOS down")
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, label="Efermi")
            ax.set_xlabel("Energy - Efermi (eV)")
            ax.set_ylabel("DOS (states/eV)")
            ax.set_title("Total DOS")
            ax.legend()
            if band_gap:
                ax.annotate(f"Gap = {band_gap:.2f} eV", xy=(0.05, 0.9),
                            xycoords="axes fraction", fontsize=11)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(str(elec_dir / "dos_total.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            log.debug("[h07_electronic] DOS plot failed: %s", exc)

    def _parse_pdos(
        self, all_lines: list, pdos_start: int, nedos: int,
        energies: list, elec_dir: Path
    ) -> None:
        """Parse per-atom PDOS blocks (LORBIT=11) and aggregate by element.

        DOSCAR PDOS layout: after the total DOS block, each atom has a 1-line
        header followed by nedos data lines with columns:
          energy  s  py  pz  px  dxy  dyz  dz2  dxz  dx2  [spin-up then spin-down]
        We sum s/p/d channels per atom, then average per element and save CSVs.
        """
        try:
            import numpy as np
        except ImportError:
            return

        poscar = self._find_poscar(elec_dir.parent.parent)
        elements_seq: list[str] = []
        if poscar and poscar.exists():
            elements_seq = self._read_poscar_elements(poscar)

        n_atoms = len(elements_seq) if elements_seq else 0
        if n_atoms == 0:
            log.debug("[h07_electronic] PDOS: no element sequence — skipping")
            return

        # Detect number of columns in first PDOS atom data line
        cursor = pdos_start
        # Skip header line of first atom block
        cursor += 1
        if cursor >= len(all_lines):
            return
        sample = all_lines[cursor].split()
        n_cols = len(sample)
        is_spin = n_cols >= 19  # spin-polarised has ~19 cols (energy + 9 up + 9 dn)

        # Accumulate per-element PDOS
        element_dos: dict[str, dict] = {}  # el → {s_up, p_up, d_up, s_dn, p_dn, d_dn}

        cursor = pdos_start
        for atom_i in range(n_atoms):
            cursor += 1  # skip per-atom header line
            s_up = np.zeros(nedos); p_up = np.zeros(nedos); d_up = np.zeros(nedos)
            s_dn = np.zeros(nedos); p_dn = np.zeros(nedos); d_dn = np.zeros(nedos)

            for e_i in range(nedos):
                if cursor >= len(all_lines):
                    break
                parts = all_lines[cursor].split()
                cursor += 1
                try:
                    if is_spin and len(parts) >= 19:
                        s_up[e_i] = float(parts[1])
                        p_up[e_i] = float(parts[2]) + float(parts[3]) + float(parts[4])
                        d_up[e_i] = (float(parts[5]) + float(parts[6]) + float(parts[7])
                                     + float(parts[8]) + float(parts[9]))
                        s_dn[e_i] = float(parts[10])
                        p_dn[e_i] = float(parts[11]) + float(parts[12]) + float(parts[13])
                        d_dn[e_i] = (float(parts[14]) + float(parts[15]) + float(parts[16])
                                     + float(parts[17]) + float(parts[18]))
                    elif len(parts) >= 10:
                        s_up[e_i] = float(parts[1])
                        p_up[e_i] = float(parts[2]) + float(parts[3]) + float(parts[4])
                        d_up[e_i] = (float(parts[5]) + float(parts[6]) + float(parts[7])
                                     + float(parts[8]) + float(parts[9]))
                except (ValueError, IndexError):
                    pass

            el = elements_seq[atom_i] if atom_i < len(elements_seq) else "X"
            if el not in element_dos:
                element_dos[el] = {
                    "s_up": np.zeros(nedos), "p_up": np.zeros(nedos), "d_up": np.zeros(nedos),
                    "s_dn": np.zeros(nedos), "p_dn": np.zeros(nedos), "d_dn": np.zeros(nedos),
                    "count": 0,
                }
            for k, arr in [("s_up", s_up), ("p_up", p_up), ("d_up", d_up),
                           ("s_dn", s_dn), ("p_dn", p_dn), ("d_dn", d_dn)]:
                element_dos[el][k] += arr
            element_dos[el]["count"] += 1

        # Write one CSV per element (summed over atoms, not averaged)
        e_arr = np.array(energies)
        for el, d in element_dos.items():
            csv_path = elec_dir / f"dos_pdos_{el}.csv"
            header = "energy_eV,s_up,p_up,d_up,s_dn,p_dn,d_dn"
            data = np.column_stack([
                e_arr, d["s_up"], d["p_up"], d["d_up"],
                d["s_dn"], d["p_dn"], d["d_dn"]
            ])
            np.savetxt(str(csv_path), data, delimiter=",", header=header, comments="")
            log.info("[h07_electronic] PDOS %s: %d atoms → %s", el, d["count"], csv_path)

        # Stacked PDOS plot
        self._plot_pdos(energies, element_dos, elec_dir)

    def _plot_pdos(self, energies: list, element_dos: dict, elec_dir: Path) -> None:
        """Save a stacked PDOS plot with one filled curve per element to elec_dir."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            e_arr = np.array(energies)
            # Use efermi = 0 (energies already shifted in calling code if needed)
            fig, ax = plt.subplots(figsize=(9, 6))
            for i, (el, d) in enumerate(element_dos.items()):
                color = NREL_COLORS[i % len(NREL_COLORS)]
                total_up = d["s_up"] + d["p_up"] + d["d_up"]
                total_dn = d["s_dn"] + d["p_dn"] + d["d_dn"]
                ax.fill_between(e_arr, total_up, alpha=0.45, color=color, label=f"{el} ↑")
                if total_dn.any():
                    ax.fill_between(e_arr, -total_dn, alpha=0.45, color=color,
                                    linestyle="--", label=f"{el} ↓")
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Energy (eV)")
            ax.set_ylabel("PDOS (states/eV)")
            ax.set_title("Projected DOS by Element")
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(str(elec_dir / "dos_pdos.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            log.debug("[h07_electronic] PDOS plot failed: %s", exc)

    # ── Band gap from vasprun.xml ──────────────────────────────────────────

    @staticmethod
    def _parse_bandgap_vasprun(vasprun_path: Path) -> float | None:
        """Extract band gap from vasprun.xml eigenvalues using streaming XML.

        Collects all (eigenvalue, occupation) pairs across all k-points and spins,
        then computes VBM (max e where occ > 0.5) and CBM (min e where occ < 0.5
        and e > VBM).  Returns 0.0 for metals.  Returns None on parse failure.
        """
        try:
            import xml.etree.ElementTree as ET
            occupied: list[float] = []   # eigenvalues with occ > 0.5
            empty: list[float] = []      # eigenvalues with occ < 0.5

            in_eigenvalues = False
            for event, elem in ET.iterparse(str(vasprun_path), events=("start", "end")):
                if event == "start" and elem.tag == "eigenvalues":
                    in_eigenvalues = True
                if event == "end" and elem.tag == "eigenvalues":
                    in_eigenvalues = False
                if in_eigenvalues and event == "end" and elem.tag == "r":
                    parts = (elem.text or "").split()
                    if len(parts) == 2:
                        try:
                            e, occ = float(parts[0]), float(parts[1])
                            if occ > 0.5:
                                occupied.append(e)
                            else:
                                empty.append(e)
                        except ValueError:
                            pass
                    elem.clear()  # release memory

            if not occupied or not empty:
                return None

            vbm = max(occupied)
            candidates = [e for e in empty if e > vbm]
            if not candidates:
                return 0.0  # metal
            cbm = min(candidates)
            gap = max(0.0, cbm - vbm)
            return round(gap, 4)
        except Exception:
            return None

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _find_poscar(project_dir: Path) -> Path | None:
        """Return the first existing POSCAR/CONTCAR from bader/ or dft/opt/, or None."""
        for candidate in [
            project_dir / "bader" / "POSCAR",
            dft_opt(project_dir) / "CONTCAR",
            dft_opt(project_dir) / "POSCAR",
        ]:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _read_poscar_elements(poscar_path: Path) -> list[str]:
        """Read element sequence from POSCAR (line 6 = species, line 7 = counts)."""
        lines = poscar_path.read_text().splitlines()
        if len(lines) < 7:
            return []
        species = lines[5].split()
        try:
            counts = [int(x) for x in lines[6].split()]
        except ValueError:
            return []
        result = []
        for sp, cnt in zip(species, counts):
            result.extend([sp] * cnt)
        return result

    @staticmethod
    def _incar_has_lorbit11(incar_path: Path) -> bool:
        """Return True if INCAR contains both 'LORBIT' and '11' (projected DOS enabled)."""
        if not incar_path.exists():
            return False
        return "LORBIT" in incar_path.read_text() and "11" in incar_path.read_text()
