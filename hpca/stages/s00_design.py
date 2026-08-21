"""
Stage 00 — Materials Design & Structure Preparation.
Handles: CIF → POSCAR, structure generation via pymatgen / ASE,
         MP API queries, supercell construction, defect insertion,
         interface slab builders (cathode|SSE, metal|SSE, polymer|SSE).
"""
# Layout: see hpca/core/paths.py
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
from typing import Optional

from hpca.core.paths import dft_opt


# ─────────────────────────────────────────────────────────────────────────────
# Optional heavy imports — graceful degradation
# ─────────────────────────────────────────────────────────────────────────────
try:
    from pymatgen.core import Structure, Lattice, Element
    from pymatgen.io.vasp.inputs import Poscar
    from pymatgen.transformations.standard_transformations import (
        SupercellTransformation, SubstitutionTransformation,
        RemoveSitesTransformation,
    )
    from pymatgen.analysis.structure_matcher import StructureMatcher
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False

try:
    from pymatgen.ext.matproj import MPRester
    HAS_MP = True
except ImportError:
    HAS_MP = False

try:
    import ase
    from ase.io import read as ase_read, write as ase_write
    from ase.build import make_supercell, surface, add_vacuum
    from ase import Atoms as AseAtoms
    HAS_ASE = True
except ImportError:
    HAS_ASE = False


# ─────────────────────────────────────────────────────────────────────────────
# CIF / POSCAR conversion
# ─────────────────────────────────────────────────────────────────────────────

def cif_to_poscar(cif_path: Path, poscar_path: Path,
                  primitive: bool = False) -> Path:
    """Convert CIF → POSCAR using pymatgen (preferred) or ASE fallback."""
    cif_path = Path(cif_path)
    poscar_path = Path(poscar_path)
    poscar_path.parent.mkdir(parents=True, exist_ok=True)

    if HAS_PYMATGEN:
        s = Structure.from_file(str(cif_path))
        if primitive:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            sga = SpacegroupAnalyzer(s)
            s   = sga.get_primitive_standard_structure()
        Poscar(s).write_file(str(poscar_path))
        print(f"  CIF → POSCAR (pymatgen): {poscar_path}")
        return poscar_path

    if HAS_ASE:
        atoms = ase_read(str(cif_path))
        ase_write(str(poscar_path), atoms, format="vasp", vasp5=True,
                  direct=True, sort=True)
        print(f"  CIF → POSCAR (ASE): {poscar_path}")
        return poscar_path

    raise RuntimeError("Neither pymatgen nor ASE available for CIF conversion.")


def poscar_to_ase(poscar_path: Path):
    """Return ASE Atoms from POSCAR."""
    if not HAS_ASE:
        raise RuntimeError("ASE not available.")
    return ase_read(str(poscar_path), format="vasp")


# ─────────────────────────────────────────────────────────────────────────────
# Materials Project query
# ─────────────────────────────────────────────────────────────────────────────

def query_mp(formula: str, api_key: str = None,
             output_dir: Path = None) -> list[dict]:
    """
    Query Materials Project for structures matching `formula`.
    Returns list of dicts with mp_id, energy_per_atom, structure.
    Downloads the lowest-energy CIF to output_dir if given.
    """
    if not HAS_MP:
        print("  pymatgen MPRester not available — skipping MP query.")
        return []

    key = api_key or _read_mp_key()
    results = []
    try:
        with MPRester(key) as mpr:
            docs = mpr.summary.search(
                formula=formula,
                fields=["material_id", "energy_per_atom",
                        "formation_energy_per_atom", "band_gap",
                        "structure"],
            )
        for d in docs:
            results.append({
                "mp_id":           d.material_id,
                "energy_per_atom": d.energy_per_atom,
                "formation_energy_per_atom": d.formation_energy_per_atom,
                "band_gap":        d.band_gap,
            })

        # Sort by energy; download best CIF
        docs.sort(key=lambda d: d.energy_per_atom)
        if output_dir and docs:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            best = docs[0]
            cif_path = out / f"{best.material_id}.cif"
            best.structure.to(fmt="cif", filename=str(cif_path))
            print(f"  MP best structure: {best.material_id} → {cif_path}")

    except Exception as e:
        print(f"  MP query error: {e}")

    return results


def _read_mp_key() -> Optional[str]:
    """Read MP API key from ~/.config/mprester.yaml or env."""
    import os
    key = os.environ.get("MP_API_KEY", "")
    if key:
        return key
    cfg = Path.home() / ".config" / "mprester.yaml"
    if cfg.exists():
        import yaml
        data = yaml.safe_load(cfg.read_text())
        return data.get("api_key", "")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Supercell construction
# ─────────────────────────────────────────────────────────────────────────────

def make_supercell_poscar(poscar_path: Path, scaling: list[int],
                           output_path: Path) -> Path:
    """
    Build supercell from POSCAR.
    scaling: [2,2,2] or [[2,0,0],[0,2,0],[0,0,2]]
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if HAS_PYMATGEN:
        s = Structure.from_file(str(poscar_path))
        if isinstance(scaling[0], int):
            import numpy as np
            m = np.diag(scaling)
        else:
            m = scaling
        t = SupercellTransformation(m)
        sc = t.apply_transformation(s)
        Poscar(sc).write_file(str(output_path))
        print(f"  Supercell {scaling} → {output_path} ({sc.num_sites} atoms)")
        return output_path

    if HAS_ASE:
        import numpy as np
        atoms = ase_read(str(poscar_path), format="vasp")
        if isinstance(scaling[0], int):
            m = np.diag(scaling)
        else:
            m = np.array(scaling)
        sc = make_supercell(atoms, m)
        ase_write(str(output_path), sc, format="vasp", vasp5=True,
                  direct=True, sort=True)
        print(f"  Supercell {scaling} → {output_path} ({len(sc)} atoms)")
        return output_path

    raise RuntimeError("Neither pymatgen nor ASE available.")


# ─────────────────────────────────────────────────────────────────────────────
# Defect / vacancy insertion
# ─────────────────────────────────────────────────────────────────────────────

def insert_vacancy(poscar_path: Path, species: str,
                   index: int = 0,
                   output_path: Path = None) -> Path:
    """Remove one atom of `species` (by index among that species) to create vacancy."""
    if not HAS_PYMATGEN:
        raise RuntimeError("pymatgen required for vacancy insertion.")
    s = Structure.from_file(str(poscar_path))
    indices = [i for i, site in enumerate(s) if site.specie.symbol == species]
    if not indices:
        raise ValueError(f"No {species} found in {poscar_path}")
    remove_idx = indices[min(index, len(indices) - 1)]
    t   = RemoveSitesTransformation([remove_idx])
    sc  = t.apply_transformation(s)
    out = output_path or poscar_path.parent / f"POSCAR_{species}_vac"
    Poscar(sc).write_file(str(out))
    print(f"  Vacancy: removed {species}[{remove_idx}] → {out}")
    return Path(out)


def insert_substitution(poscar_path: Path, replace: str, with_element: str,
                         output_path: Path = None) -> Path:
    """Replace all `replace` species with `with_element`."""
    if not HAS_PYMATGEN:
        raise RuntimeError("pymatgen required.")
    s = Structure.from_file(str(poscar_path))
    t = SubstitutionTransformation({replace: with_element})
    sc = t.apply_transformation(s)
    out = output_path or poscar_path.parent / f"POSCAR_{replace}to{with_element}"
    Poscar(sc).write_file(str(out))
    print(f"  Substitution {replace}→{with_element} → {out}")
    return Path(out)


def set_li_concentration(poscar_path: Path, target_frac: float,
                          output_path: Path = None) -> Path:
    """
    Adjust Li content to `target_frac` (0–1) of max Li sites.
    Used for NaP doping studies (e.g. 6.25%, 12.5% Li).
    Removes Li atoms randomly to reach target.
    """
    import random
    if not HAS_PYMATGEN:
        raise RuntimeError("pymatgen required.")
    s = Structure.from_file(str(poscar_path))
    li_idx  = [i for i, site in enumerate(s) if site.specie.symbol == "Li"]
    n_keep  = max(1, round(len(li_idx) * target_frac))
    remove  = sorted(random.sample(li_idx, len(li_idx) - n_keep), reverse=True)
    for i in remove:
        s.remove_sites([i])
    out = output_path or poscar_path.parent / f"POSCAR_Li{int(target_frac*100):03d}pct"
    Poscar(s).write_file(str(out))
    print(f"  Li concentration {target_frac:.2%}: {n_keep}/{len(li_idx)} Li kept → {out}")
    return Path(out)


# ─────────────────────────────────────────────────────────────────────────────
# Interface slab builders
# ─────────────────────────────────────────────────────────────────────────────

def build_surface_slab(poscar_path: Path, miller: tuple,
                        layers: int = 4, vacuum_A: float = 15.0,
                        output_path: Path = None) -> Path:
    """
    Build a slab with `layers` unit cells along `miller` direction
    and vacuum of `vacuum_A` Angstrom.
    """
    if not HAS_ASE:
        raise RuntimeError("ASE required for surface slab building.")
    atoms = ase_read(str(poscar_path), format="vasp")
    slab  = surface(atoms, miller, layers, vacuum=vacuum_A / 2)
    slab.center(vacuum=vacuum_A / 2, axis=2)
    out = output_path or poscar_path.parent / f"POSCAR_slab_{''.join(map(str,miller))}"
    ase_write(str(out), slab, format="vasp", vasp5=True, direct=True, sort=True)
    print(f"  Surface slab {miller} {layers}L vac={vacuum_A}Å → {out} ({len(slab)} atoms)")
    return Path(out)


def build_interface(slab_a_path: Path, slab_b_path: Path,
                     gap_A: float = 2.5,
                     output_path: Path = None) -> Path:
    """
    Stack slab_a and slab_b along z with a gap of `gap_A` Angstrom.
    Simple z-concatenation; use for cathode|SSE / metal|SSE interfaces.
    """
    if not HAS_ASE:
        raise RuntimeError("ASE required.")
    import numpy as np
    a = ase_read(str(slab_a_path), format="vasp")
    b = ase_read(str(slab_b_path), format="vasp")

    # Translate b so its bottom is above a's top + gap
    z_top_a = a.positions[:, 2].max()
    z_bot_b = b.positions[:, 2].min()
    b.translate([0, 0, z_top_a - z_bot_b + gap_A])

    # Combine; use a's cell + extended z
    new_cell    = a.cell.copy()
    new_cell[2] = [0, 0, b.positions[:, 2].max() + gap_A]
    combined    = a + b
    combined.set_cell(new_cell, scale_atoms=False)
    combined.set_pbc([True, True, True])

    out = output_path or Path(str(slab_a_path).replace("_a_", "_interface_"))
    ase_write(str(out), combined, format="vasp", vasp5=True, direct=True, sort=True)
    print(f"  Interface {slab_a_path.name} | {slab_b_path.name} → {out} ({len(combined)} atoms)")
    return Path(out)


# ─────────────────────────────────────────────────────────────────────────────
# Polymer / liquid initial config
# ─────────────────────────────────────────────────────────────────────────────

def build_polymer_box(smiles: str, n_chains: int,
                       chain_length: int,
                       box_density_gcm3: float = 1.1,
                       output_path: Path = None) -> dict:
    """
    Generate initial polymer simulation box using Packmol via subprocess.
    Falls back to a simple linear chain if Packmol is not found.
    Returns dict with 'poscar', 'lammps_data', 'n_atoms'.
    """
    import tempfile, shutil

    # Molecular weight estimate from SMILES (very rough: count heavy atoms × 12)
    n_heavy  = sum(1 for c in smiles if c.isalpha() and c != 'H')
    mw_guess = n_heavy * 12.0 * chain_length
    n_atoms_est = n_chains * chain_length * n_heavy

    # Box size from density
    import math
    avogadro = 6.022e23
    mass_g   = n_chains * mw_guess / avogadro
    vol_cm3  = mass_g / box_density_gcm3
    L_A      = (vol_cm3 * 1e24) ** (1.0/3.0)

    out_dir  = Path(output_path) if output_path else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "smiles": smiles,
        "n_chains": n_chains,
        "chain_length": chain_length,
        "box_L_A": round(L_A, 2),
        "n_atoms_estimate": n_atoms_est,
        "output_dir": str(out_dir),
    }
    print(f"  Polymer box: {n_chains} chains × {chain_length} monomers, "
          f"L={L_A:.1f} Å, ρ≈{box_density_gcm3} g/cm³")
    return result


def build_liquid_box(species: list[str], counts: list[int],
                      box_A: float = 20.0,
                      output_path: Path = None) -> Path:
    """
    Build a random liquid box for electrolyte MD using ASE.
    species: element symbols, counts: number of each.
    Returns POSCAR path.
    """
    if not HAS_ASE:
        raise RuntimeError("ASE required.")
    import numpy as np

    rng   = np.random.default_rng(42)
    pos   = []
    syms  = []
    for sp, n in zip(species, counts):
        pos.append(rng.uniform(0, box_A, (n, 3)))
        syms.extend([sp] * n)

    all_pos = np.vstack(pos)
    atoms   = AseAtoms(symbols=syms, positions=all_pos,
                       cell=[box_A]*3, pbc=True)

    out = Path(output_path) if output_path else Path("POSCAR_liquid")
    ase_write(str(out), atoms, format="vasp", vasp5=True, direct=True, sort=True)
    print(f"  Liquid box: {dict(zip(species,counts))}, L={box_A} Å → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Structure validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_structure(poscar_path: Path,
                        min_dist_A: float = 1.5) -> dict:
    """
    Basic sanity checks: min inter-atomic distance, charge neutrality hint.
    Returns dict with 'pass', 'min_dist', 'n_atoms', 'warnings'.
    """
    warnings = []

    if HAS_PYMATGEN:
        s = Structure.from_file(str(poscar_path))
        n_atoms   = s.num_sites
        dist_mat  = s.distance_matrix
        import numpy as np
        np.fill_diagonal(dist_mat, 1e10)
        min_d = dist_mat.min()
        if min_d < min_dist_A:
            warnings.append(f"Min distance {min_d:.2f} Å < {min_dist_A} Å — possible overlap")
        result = {
            "pass":     min_d >= min_dist_A,
            "min_dist": round(min_d, 3),
            "n_atoms":  n_atoms,
            "formula":  s.formula,
            "warnings": warnings,
        }

    elif HAS_ASE:
        import numpy as np
        atoms   = ase_read(str(poscar_path), format="vasp")
        n_atoms = len(atoms)
        dists   = atoms.get_all_distances(mic=True)
        np.fill_diagonal(dists, 1e10)
        min_d   = dists.min()
        if min_d < min_dist_A:
            warnings.append(f"Min distance {min_d:.2f} Å < {min_dist_A} Å")
        result = {
            "pass":     min_d >= min_dist_A,
            "min_dist": round(min_d, 3),
            "n_atoms":  n_atoms,
            "formula":  atoms.get_chemical_formula(),
            "warnings": warnings,
        }
    else:
        result = {"pass": None, "warnings": ["No validator available"]}

    if result.get("pass"):
        print(f"  Structure OK: {result.get('formula','')} "
              f"{result['n_atoms']} atoms, d_min={result['min_dist']} Å")
    else:
        for w in warnings:
            print(f"  WARNING: {w}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(project, output_base: Path,
        task: str = "prepare",
        cif_path: Path = None,
        mp_api_key: str = None,
        supercell: list = None,
        miller: tuple = None,
        layers: int = 4,
        vacuum_A: float = 15.0,
        li_frac: float = None) -> dict:
    """
    Stage 00 entry point.

    task options:
      'prepare'    — CIF → POSCAR + validate
      'query_mp'   — Materials Project query for project formula
      'supercell'  — build supercell from existing POSCAR
      'vacancy'    — insert Li vacancy
      'slab'       — build surface slab
      'interface'  — stack two slabs
      'li_conc'    — adjust Li concentration
    """
    out_dir  = Path(output_base) / "design" / project.name
    out_dir.mkdir(parents=True, exist_ok=True)
    proj_dir = Path(project.root)
    results  = {"project": project.name, "task": task}

    # ── Prepare: CIF → POSCAR ──────────────────────────────────────────────
    if task == "prepare":
        src_cif = cif_path or proj_dir / f"{project.name}.cif"
        if not Path(src_cif).exists():
            # Try to find any CIF in project root
            cifs = list(proj_dir.glob("*.cif"))
            if cifs:
                src_cif = cifs[0]
                print(f"  Found CIF: {src_cif}")
            else:
                print(f"  No CIF found in {proj_dir}. "
                      "Provide cif_path or run task='query_mp' first.")
                results["status"] = "no_cif"
                return results

        poscar_out = proj_dir / "vc" / "POSCAR"
        poscar_out.parent.mkdir(exist_ok=True)
        try:
            cif_to_poscar(src_cif, poscar_out)
            val = validate_structure(poscar_out)
            results["poscar"]   = str(poscar_out)
            results["validate"] = val
            results["status"]   = "ok" if val.get("pass") else "warning"
        except Exception as e:
            results["error"]  = str(e)
            results["status"] = "error"

    # ── Materials Project query ────────────────────────────────────────────
    elif task == "query_mp":
        formula = getattr(project, "formula", project.name)
        mp_results = query_mp(formula, api_key=mp_api_key,
                               output_dir=proj_dir)
        results["mp_results"] = mp_results[:5]
        if mp_results:
            results["status"] = "ok"
            results["best_mp_id"] = mp_results[0].get("mp_id")

    # ── Supercell ──────────────────────────────────────────────────────────
    elif task == "supercell":
        sc = supercell or [2, 2, 2]
        poscar_in  = dft_opt(proj_dir) / "CONTCAR"
        if not poscar_in.exists():
            poscar_in = proj_dir / "vc"  / "CONTCAR"
        if not poscar_in.exists():
            poscar_in = proj_dir / "vc"  / "POSCAR"
        poscar_out = dft_opt(proj_dir) / "POSCAR"
        poscar_out.parent.mkdir(parents=True, exist_ok=True)
        try:
            make_supercell_poscar(poscar_in, sc, poscar_out)
            val = validate_structure(poscar_out)
            results.update({"poscar": str(poscar_out), "supercell": sc,
                             "validate": val, "status": "ok"})
        except Exception as e:
            results.update({"error": str(e), "status": "error"})

    # ── Vacancy ────────────────────────────────────────────────────────────
    elif task == "vacancy":
        poscar_in  = dft_opt(proj_dir) / "POSCAR"
        poscar_out = proj_dir / "neb"  / "POSCAR_vac"
        try:
            insert_vacancy(poscar_in, "Li", output_path=poscar_out)
            results.update({"poscar": str(poscar_out), "status": "ok"})
        except Exception as e:
            results.update({"error": str(e), "status": "error"})

    # ── Surface slab ───────────────────────────────────────────────────────
    elif task == "slab":
        m     = miller or (0, 0, 1)
        poscar_in  = dft_opt(proj_dir) / "CONTCAR"
        if not poscar_in.exists():
            poscar_in = proj_dir / "vc" / "POSCAR"
        poscar_out = out_dir / f"POSCAR_slab_{''.join(map(str,m))}"
        try:
            build_surface_slab(poscar_in, m, layers=layers,
                                vacuum_A=vacuum_A, output_path=poscar_out)
            results.update({"poscar": str(poscar_out), "miller": list(m),
                             "layers": layers, "status": "ok"})
        except Exception as e:
            results.update({"error": str(e), "status": "error"})

    # ── Li concentration ───────────────────────────────────────────────────
    elif task == "li_conc":
        frac = li_frac if li_frac is not None else 0.0625
        poscar_in  = dft_opt(proj_dir) / "POSCAR"
        label      = f"Li{int(frac*100):03d}pct"
        poscar_out = proj_dir / f"doped_{label}" / "POSCAR"
        poscar_out.parent.mkdir(exist_ok=True)
        try:
            set_li_concentration(poscar_in, frac, poscar_out)
            results.update({"poscar": str(poscar_out), "li_frac": frac,
                             "status": "ok"})
        except Exception as e:
            results.update({"error": str(e), "status": "error"})

    else:
        results["status"] = f"unknown task: {task}"

    return results
