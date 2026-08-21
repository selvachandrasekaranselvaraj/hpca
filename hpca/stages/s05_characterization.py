"""
Stage 05 — Static DFT characterization.
Orchestrates: Bader, DOS/PDOS, NEB extraction, charge density diff.
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

from hpca.analysis.electronic import (
    parse_doscar, parse_bader_acf, compute_charge_transfer,
    parse_neb_energies, plot_dos, plot_bader_charges, plot_neb_profile,
    project_pdos, find_band_gap, run_electronic_analysis,
)

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Common valence electron counts (PAW PBE standard potentials)
VALENCE = {
    "Li": 1, "Na": 1, "K": 1,
    "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2,
    "Al": 3, "Ga": 3, "In": 3,
    "Si": 4, "Ge": 4, "Sn": 4,
    "N":  5, "P":  5, "As": 5, "Sb": 5,
    "O":  6, "S":  6, "Se": 6, "Te": 6,
    "F":  7, "Cl": 7, "Br": 7, "I":  7,
    "V":  5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10,
    "Cu": 11, "Zn": 12,
    "Y":  11, "Zr": 4, "Nb": 5, "Mo": 6,
}


def run(project, output_base: Path, tasks: list[str] = None,
        submit_missing: bool = False) -> dict:
    """
    Stage 05 entry point.
    tasks: ['bader', 'dos', 'neb', 'charge_diff', 'all']
    """
    if tasks is None or "all" in tasks:
        tasks = ["bader", "dos", "neb"]

    out_dir = Path(output_base) / "characterization" / project.name
    out_dir.mkdir(parents=True, exist_ok=True)
    proj_dir = Path(project.root)
    results  = {"project": project.name, "tasks": tasks}

    # ── DOS ───────────────────────────────────────────────────────────────────
    if "dos" in tasks:
        dos_dir = proj_dir / "dos" / "nonscf"
        doscar  = dos_dir / "DOSCAR"
        if doscar.exists():
            vr = dos_dir / "vasprun.xml"
            dos_data = parse_doscar(doscar, vr if vr.exists() else None)
            bg = find_band_gap(dos_data.get("energies", []),
                                dos_data.get("dos_total", []))
            results["dos"] = {"band_gap": bg}

            # PDOS by species — try to get atom_species from POSCAR
            atom_species = _read_poscar_species(proj_dir / "dos" / "nonscf" / "POSCAR")
            if atom_species and dos_data.get("pdos"):
                pdos_sp = project_pdos(dos_data["pdos"], atom_species)
                if HAS_PLOTLY:
                    fig = plot_dos(dos_data, project.name, pdos_sp)
                    _save_fig(fig, out_dir, f"{project.name}_dos")
                    results["dos"]["figure"] = str(out_dir / f"{project.name}_dos.html")
            elif HAS_PLOTLY and "energies" in dos_data:
                fig = plot_dos(dos_data, project.name)
                _save_fig(fig, out_dir, f"{project.name}_dos")
                results["dos"]["figure"] = str(out_dir / f"{project.name}_dos.html")
        elif submit_missing:
            print(f"  DOS DOSCAR not found — check {dos_dir}")

    # ── Bader ─────────────────────────────────────────────────────────────────
    if "bader" in tasks:
        acf = proj_dir / "bader" / "ACF.dat"
        if acf.exists():
            bader_data = parse_bader_acf(acf)
            atom_species = _read_poscar_species(proj_dir / "bader" / "POSCAR")
            if atom_species and "charges" in bader_data:
                ct = compute_charge_transfer(bader_data["charges"],
                                              atom_species, VALENCE)
                results["bader"] = {
                    sp: {"mean_transfer": info["mean"], "std": info["std"]}
                    for sp, info in ct["by_species"].items()
                }
                if HAS_PLOTLY:
                    fig = plot_bader_charges(bader_data,
                                              list(set(atom_species)), atom_species,
                                              project.name)
                    _save_fig(fig, out_dir, f"{project.name}_bader")
                    results["bader"]["figure"] = str(out_dir / f"{project.name}_bader.html")

    # ── NEB ───────────────────────────────────────────────────────────────────
    if "neb" in tasks:
        neb_base = proj_dir / "neb"
        if neb_base.exists():
            neb_results = {}
            for path_dir in sorted(neb_base.iterdir()):
                if not (path_dir.is_dir() and "path" in path_dir.name):
                    continue
                neb_data = parse_neb_energies(path_dir)
                if not neb_data:
                    continue
                neb_results[path_dir.name] = {
                    "barrier_fwd_eV": neb_data["barrier_fwd_eV"],
                    "barrier_rev_eV": neb_data["barrier_rev_eV"],
                    "reaction_energy_eV": neb_data["reaction_energy_eV"],
                }
                if HAS_PLOTLY:
                    fig = plot_neb_profile(neb_data, project.name, path_dir.name)
                    fname = f"{project.name}_neb_{path_dir.name}"
                    _save_fig(fig, out_dir, fname)
                    neb_results[path_dir.name]["figure"] = \
                        str(out_dir / f"{fname}.html")
            results["neb"] = neb_results

    # ── Charge density difference ─────────────────────────────────────────────
    if "charge_diff" in tasks:
        from hpca.analysis.electronic import compute_charge_density_diff
        chg_ab = proj_dir / "bader" / "CHGCAR"
        chg_a  = proj_dir / "bader" / "A" / "CHGCAR"
        chg_b  = proj_dir / "bader" / "B" / "CHGCAR"
        if all(p.exists() for p in [chg_ab, chg_a, chg_b]):
            diff_res = compute_charge_density_diff(
                chg_ab, chg_a, chg_b,
                output_path=out_dir / "CHGCAR_diff"
            )
            results["charge_diff"] = diff_res

    return results


def _read_poscar_species(poscar_path: Path) -> list[str] | None:
    """Read per-atom species list from VASP POSCAR line 5+6."""
    if not poscar_path.exists():
        return None
    try:
        lines = poscar_path.read_text().splitlines()
        species_line = lines[5].split()  # element symbols
        counts_line  = lines[6].split()  # counts per element
        # Distinguish: if counts_line is numeric, line 5 has symbols; else line 6 has symbols
        if not counts_line[0].isdigit():
            species_line = lines[6].split()
            counts_line  = lines[7].split()
        atom_species = []
        for sp, cnt in zip(species_line, counts_line):
            atom_species.extend([sp] * int(cnt))
        return atom_species
    except Exception:
        return None


def _save_fig(fig, out_dir: Path, stem: str):
    """Save a Plotly figure to both HTML and PNG files under out_dir using stem as the base name."""
    if fig is None:
        return
    html = out_dir / f"{stem}.html"
    png  = out_dir / f"{stem}.png"
    try:
        fig.write_html(str(html))
        fig.write_image(str(png), width=900, height=520)
    except Exception:
        try:
            fig.write_html(str(html))
        except Exception:
            pass
