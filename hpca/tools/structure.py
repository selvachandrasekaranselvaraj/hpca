"""
Structure tool: load, convert, supercell, validate, search Materials Project,
build from spacegroup.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .base import Tool, ToolResult


class StructureTool(Tool):
    """AI tool for loading, converting, analysing, and building crystal structures."""

    name = "structure"
    description = (
        "Load and analyse crystal structures (POSCAR, CIF, XYZ, JSON), "
        "convert between formats, build supercells, validate geometries, "
        "search the Materials Project API, and build structures from spacegroup."
    )

    def _parameters(self) -> dict:
        """Return JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "load", "convert", "supercell", "info",
                        "validate", "search_mp", "build_from_spacegroup",
                    ],
                },
                "path":          {"type": "string"},
                "input_path":    {"type": "string"},
                "output_path":   {"type": "string"},
                "matrix":        {
                    "type": "array",
                    "description": "3×3 supercell matrix or 3-element diagonal",
                    "items": {},
                },
                "formula":       {"type": "string"},
                "max_results":   {"type": "integer"},
                "min_dist":      {"type": "number"},
                "spacegroup_num":{"type": "integer"},
                "species":       {"type": "array", "items": {"type": "string"}},
                "coords":        {"type": "array"},
                "lattice_abc":   {"type": "array", "items": {"type": "number"}},
            },
            "required": ["action"],
        }

    # ── Public methods ────────────────────────────────────────────────────────

    def load(self, path: str) -> dict:
        """
        Load a structure file and return summary dict:
        {formula, spacegroup, a, b, c, alpha_deg, beta_deg, gamma_deg,
         n_atoms, density, elements, n_per_element}
        """
        from pymatgen.core import Structure
        s = Structure.from_file(path)
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        try:
            sga = SpacegroupAnalyzer(s)
            sg = sga.get_space_group_symbol()
        except Exception:
            sg = "unknown"

        elem_counts: dict[str, int] = {}
        for site in s.sites:
            el = str(site.specie.symbol)
            elem_counts[el] = elem_counts.get(el, 0) + 1

        lp = s.lattice
        return {
            "formula":     s.composition.reduced_formula,
            "spacegroup":  sg,
            "a":           round(lp.a, 4),
            "b":           round(lp.b, 4),
            "c":           round(lp.c, 4),
            "alpha_deg":   round(lp.alpha, 3),
            "beta_deg":    round(lp.beta, 3),
            "gamma_deg":   round(lp.gamma, 3),
            "n_atoms":     len(s),
            "density":     round(s.density, 4),
            "elements":    sorted(elem_counts.keys()),
            "n_per_element": elem_counts,
        }

    def convert(self, input_path: str, output_path: str) -> ToolResult:
        """
        Convert between POSCAR, CIF, XYZ, JSON.
        Format is auto-detected from file extension.
        """
        try:
            from pymatgen.core import Structure
            s = Structure.from_file(input_path)

            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)

            ext = out.suffix.lower()
            fmt_map = {
                ".vasp": "poscar", "": "poscar",
                ".cif":  "cif",
                ".xyz":  "xyz",
                ".json": "json",
                ".poscar": "poscar",
            }
            fmt = fmt_map.get(ext, "poscar")
            if out.name.upper() in ("POSCAR", "CONTCAR"):
                fmt = "poscar"

            s.to(fmt=fmt, filename=str(out))
            return ToolResult(f"Converted {input_path} → {output_path} (fmt={fmt})")
        except Exception as exc:
            return ToolResult(str(exc), success=False)

    def supercell(
        self,
        poscar_path: str,
        matrix,
        output_path: Optional[str] = None,
    ) -> Path:
        """
        Build a supercell from POSCAR using a 3×3 matrix or [nx, ny, nz] diagonal.
        Writes POSCAR to output_path (or poscar_path + '_super').
        Returns the output Path.
        """
        from pymatgen.core import Structure
        import numpy as np

        s = Structure.from_file(poscar_path)

        # Accept [nx,ny,nz] shorthand
        m = matrix
        if isinstance(m, (list, tuple)) and len(m) == 3 and not isinstance(m[0], (list, tuple)):
            m = [[m[0], 0, 0], [0, m[1], 0], [0, 0, m[2]]]

        s.make_supercell(m)

        if output_path is None:
            output_path = str(Path(poscar_path).parent / "POSCAR_super")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        s.to(fmt="poscar", filename=str(out))
        return out

    def info(self, path: str) -> str:
        """Return a formatted human-readable summary of a structure."""
        try:
            d = self.load(path)
            lines = [
                f"Formula:    {d['formula']}",
                f"Spacegroup: {d['spacegroup']}",
                f"Lattice:    a={d['a']} b={d['b']} c={d['c']} Å",
                f"Angles:     α={d['alpha_deg']} β={d['beta_deg']} γ={d['gamma_deg']}°",
                f"N atoms:    {d['n_atoms']}",
                f"Density:    {d['density']} g/cm³",
                f"Elements:   {', '.join(d['elements'])}",
                "Per element: "
                + ", ".join(f"{k}:{v}" for k, v in d["n_per_element"].items()),
            ]
            return "\n".join(lines)
        except Exception as exc:
            return f"Error loading {path}: {exc}"

    def validate(self, path: str, min_dist: float = 1.5) -> dict:
        """
        Validate a structure: check minimum interatomic distance.
        Returns {valid, min_distance, issue}.
        """
        try:
            from pymatgen.core import Structure
            s = Structure.from_file(path)
            dm = s.distance_matrix
            import numpy as np
            np.fill_diagonal(dm, 1e10)
            actual_min = float(dm.min())
            valid = actual_min >= min_dist
            return {
                "valid": valid,
                "min_distance": round(actual_min, 4),
                "issue": None if valid else f"Min distance {actual_min:.3f} Å < threshold {min_dist} Å",
            }
        except Exception as exc:
            return {"valid": False, "min_distance": None, "issue": str(exc)}

    def search_mp(self, formula: str, max_results: int = 5) -> list[dict]:
        """
        Search the Materials Project for structures matching formula.
        Requires MP_API_KEY environment variable.
        Falls back to empty list if API unavailable.
        """
        api_key = os.environ.get("MP_API_KEY", "")
        results = []

        # Try mp-api (new client)
        try:
            from mp_api.client import MPRester
            with MPRester(api_key) as mpr:
                docs = mpr.summary.search(
                    formula=formula,
                    fields=["material_id", "formula_pretty", "energy_above_hull",
                            "symmetry", "structure"],
                )
                for doc in docs[:max_results]:
                    results.append({
                        "mp_id":             doc.material_id,
                        "formula":           doc.formula_pretty,
                        "energy_above_hull": doc.energy_above_hull,
                        "spacegroup":        doc.symmetry.symbol if doc.symmetry else "?",
                    })
            return results
        except ImportError:
            pass
        except Exception:
            pass

        # Try legacy pymatgen MPRester
        try:
            from pymatgen.ext.matproj import MPRester as LegacyMPRester
            with LegacyMPRester(api_key) as mpr:
                entries = mpr.get_entries(formula)
                for e in entries[:max_results]:
                    results.append({
                        "mp_id":             e.entry_id,
                        "formula":           e.composition.reduced_formula,
                        "energy_above_hull": None,
                        "spacegroup":        "?",
                    })
            return results
        except Exception:
            pass

        return results

    def build_from_spacegroup(
        self,
        spacegroup_num: int,
        species: list[str],
        coords: list,
        lattice_abc: list[float],
    ) -> Path:
        """
        Build a structure from spacegroup, Wyckoff positions, and lattice params.
        lattice_abc: [a, b, c, alpha, beta, gamma] in Å and degrees.
        coords: list of [x, y, z] fractional coordinates.
        Writes POSCAR to /tmp/built_structure.vasp; returns Path.
        """
        from pymatgen.core import Structure, Lattice
        from pymatgen.symmetry.groups import SpaceGroup

        if len(lattice_abc) == 3:
            latt = Lattice.cubic(lattice_abc[0]) if lattice_abc[0] == lattice_abc[1] == lattice_abc[2] \
                   else Lattice.orthorhombic(*lattice_abc)
        elif len(lattice_abc) == 6:
            latt = Lattice.from_parameters(*lattice_abc)
        else:
            raise ValueError("lattice_abc must have 3 or 6 elements.")

        from pymatgen.core import PeriodicSite
        from pymatgen.symmetry.structure import SymmetrizedStructure
        # Use Structure directly with given coords, let pymatgen apply SG symmetry
        s = Structure.from_spacegroup(spacegroup_num, latt, species, coords)

        out = Path("/tmp/built_structure.vasp")
        s.to(fmt="poscar", filename=str(out))
        return out

    # ── execute() dispatch ─────────────────────────────────────────────────────

    def execute(self, action: str, **kwargs) -> ToolResult:
        """Execute the tool action and return a ToolResult."""
        try:
            if action == "load":
                d = self.load(kwargs["path"])
                text = "\n".join(f"{k}: {v}" for k, v in d.items())
                return ToolResult(text, metadata=d)

            elif action == "convert":
                return self.convert(kwargs["input_path"], kwargs["output_path"])

            elif action == "supercell":
                p = self.supercell(
                    kwargs["path"],
                    kwargs["matrix"],
                    output_path=kwargs.get("output_path"),
                )
                return ToolResult(f"Supercell written: {p}")

            elif action == "info":
                return ToolResult(self.info(kwargs["path"]))

            elif action == "validate":
                d = self.validate(
                    kwargs["path"],
                    min_dist=kwargs.get("min_dist", 1.5),
                )
                text = "\n".join(f"{k}: {v}" for k, v in d.items())
                return ToolResult(text, metadata=d)

            elif action == "search_mp":
                results = self.search_mp(
                    kwargs["formula"],
                    max_results=kwargs.get("max_results", 5),
                )
                if not results:
                    return ToolResult("No MP results (check MP_API_KEY).")
                lines = [
                    f"{r['mp_id']:15s}  {r['formula']:20s}  "
                    f"E_hull={r['energy_above_hull']}  sg={r['spacegroup']}"
                    for r in results
                ]
                return ToolResult("\n".join(lines), metadata={"results": results})

            elif action == "build_from_spacegroup":
                p = self.build_from_spacegroup(
                    kwargs["spacegroup_num"],
                    kwargs["species"],
                    kwargs["coords"],
                    kwargs["lattice_abc"],
                )
                return ToolResult(f"Structure written: {p}")

            else:
                return ToolResult(f"Unknown action: {action}", success=False)

        except Exception as exc:
            return ToolResult(str(exc), success=False)
