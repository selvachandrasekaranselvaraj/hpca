"""
np_builder.py — Pure-Python nanoparticle carver and NP-on-substrate builder.

Replaces the external /home/user/bin/build_substrate_np.py script so that
the HPCA package is fully standalone with no external-script dependencies.

Only stdlib + numpy are required (numpy is lazy-imported inside each function).
ASE is used opportunistically for slab operations when available.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger("hpca.np_builder")


# ---------------------------------------------------------------------------
# Low-level POSCAR helpers  (same approach as neb_tools._read_poscar)
# ---------------------------------------------------------------------------

def _read_poscar(path: Path) -> dict:
    """Parse a VASP5 POSCAR and return a plain dict.

    Keys:
        comment  : str
        scale    : float
        lattice  : np.ndarray shape (3,3)  — unscaled row vectors (Å when scale=1)
        elements : list[str]
        counts   : list[int]
        coords   : np.ndarray shape (N,3)  — always fractional
        selective: bool
        flags    : list[tuple[str,str,str]]
    """
    import numpy as _np

    path = Path(path)
    lines = path.read_text().splitlines()

    comment = lines[0]
    scale = float(lines[1].split()[0])

    lattice = _np.array(
        [list(map(float, lines[i].split()[:3])) for i in range(2, 5)],
        dtype=float,
    )

    # VASP4 vs VASP5: if line 5 starts with a digit it is counts (no species line)
    vasp4 = lines[5].split()[0][0].isdigit()
    if vasp4:
        elements: list[str] = []
        counts = list(map(int, lines[5].split()))
        body_start = 6
    else:
        elements = lines[5].split()
        counts = list(map(int, lines[6].split()))
        body_start = 7

    n_atoms = sum(counts)

    selective = False
    if lines[body_start].strip().lower().startswith("s"):
        selective = True
        body_start += 1

    coord_mode = lines[body_start].strip().lower()
    is_direct = coord_mode.startswith("d")
    body_start += 1

    coords_raw = _np.zeros((n_atoms, 3), dtype=float)
    flags: list[tuple[str, str, str]] = []
    for i in range(n_atoms):
        parts = lines[body_start + i].split()
        coords_raw[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
        if selective and len(parts) >= 6:
            flags.append((parts[3], parts[4], parts[5]))
        else:
            flags.append(("T", "T", "T"))

    if is_direct:
        coords = coords_raw
    else:
        # Cartesian → fractional: frac = cart @ inv(lat)
        # (scale cancels: physical_cart = scale*raw, physical_lat = scale*lat)
        inv_lat = _np.linalg.inv(lattice)
        coords = coords_raw @ inv_lat

    return dict(
        comment=comment,
        scale=scale,
        lattice=lattice,
        elements=elements,
        counts=counts,
        n_atoms=n_atoms,
        coords=coords,
        selective=selective,
        flags=flags,
    )


def _write_poscar(data: dict, title: str = "HPCA NP") -> str:
    """Serialise a POSCAR dict (fractional coords) to a string.

    The dict must contain at minimum: lattice, elements, counts, coords.
    Optional: scale (default 1.0), selective (default False), flags.
    """
    scale    = data.get("scale", 1.0)
    lattice  = data["lattice"]
    elements = data.get("elements", [])
    counts   = data["counts"]
    coords   = data["coords"]
    selective = data.get("selective", False)
    flags    = data.get("flags", [("T", "T", "T")] * len(coords))

    lines: list[str] = []
    lines.append(title)
    lines.append(f"  {scale:.10f}")

    for row in lattice:
        lines.append("  " + "  ".join(f"{v:20.16f}" for v in row))

    if elements:
        lines.append("  " + "  ".join(elements))

    lines.append("  " + "  ".join(str(c) for c in counts))

    if selective:
        lines.append("Selective dynamics")

    lines.append("Direct")

    for i, coord in enumerate(coords):
        coord_str = "  ".join(f"{v:20.16f}" for v in coord)
        if selective:
            fx, fy, fz = flags[i]
            lines.append(f"  {coord_str}  {fx}  {fy}  {fz}")
        else:
            lines.append(f"  {coord_str}")

    return "\n".join(lines) + "\n"


def _direct_to_cart(frac: "np.ndarray", lat: "np.ndarray") -> "np.ndarray":
    """Convert fractional → Cartesian coordinates.  cart = frac @ lat."""
    import numpy as _np
    return _np.asarray(frac, dtype=float) @ _np.asarray(lat, dtype=float)


def _cart_to_direct(cart: "np.ndarray", lat: "np.ndarray") -> "np.ndarray":
    """Convert Cartesian → fractional coordinates.  frac = cart @ inv(lat)."""
    import numpy as _np
    inv_lat = _np.linalg.inv(_np.asarray(lat, dtype=float))
    return _np.asarray(cart, dtype=float) @ inv_lat


# ---------------------------------------------------------------------------
# Nanoparticle carver
# ---------------------------------------------------------------------------

def carve_nanoparticle(
    bulk_poscar: "Path | str",
    radius_A: float = 5.0,
    *,
    center_element: "str | None" = None,
    shape: str = "sphere",
) -> str:
    """Carve a nanoparticle from a bulk crystal by cutting a sphere.

    Args:
        bulk_poscar   : Path to bulk POSCAR (VASP5 format).
        radius_A      : Nanoparticle radius in Å.
        center_element: Element symbol to use as the center atom.
                        Defaults to the geometric centroid of all atoms.
        shape         : "sphere" (only supported value for now).

    Returns:
        POSCAR string of the nanoparticle placed in a cubic vacuum box
        with side = 2*(radius_A + 5) Å.
    """
    import numpy as _np

    if shape != "sphere":
        raise ValueError(f"carve_nanoparticle: unsupported shape {shape!r}; use 'sphere'")

    bulk = _read_poscar(Path(bulk_poscar))
    lat  = bulk["lattice"] * bulk["scale"]   # physical lattice (Å)
    frac = bulk["coords"]
    cart = _direct_to_cart(frac, lat)

    # Determine center point
    if center_element and bulk["elements"]:
        # Build per-atom element list
        atom_elements: list[str] = []
        for el, cnt in zip(bulk["elements"], bulk["counts"]):
            atom_elements.extend([el] * cnt)
        idx_center = [i for i, e in enumerate(atom_elements) if e == center_element]
        if idx_center:
            center = cart[idx_center].mean(axis=0)
        else:
            log.warning("carve_nanoparticle: element %r not found; using centroid", center_element)
            center = cart.mean(axis=0)
    else:
        center = cart.mean(axis=0)

    # Keep atoms within radius
    disp   = cart - center
    dist   = _np.linalg.norm(disp, axis=1)
    mask   = dist <= radius_A
    n_kept = int(mask.sum())

    if n_kept == 0:
        raise ValueError(
            f"carve_nanoparticle: no atoms within radius {radius_A} Å of center. "
            "Try increasing radius_A."
        )

    log.info("carve_nanoparticle: kept %d/%d atoms within %.1f Å", n_kept, len(cart), radius_A)

    # Rebuild element / count arrays for kept atoms
    atom_elements_all: list[str] = []
    if bulk["elements"] and bulk["counts"]:
        for el, cnt in zip(bulk["elements"], bulk["counts"]):
            atom_elements_all.extend([el] * cnt)
    else:
        atom_elements_all = ["X"] * len(cart)

    kept_elements_raw = [atom_elements_all[i] for i in range(len(cart)) if mask[i]]
    kept_cart         = cart[mask]

    # Build ordered element / count lists preserving original ordering
    seen_order: list[str] = []
    for el in kept_elements_raw:
        if el not in seen_order:
            seen_order.append(el)
    new_elements: list[str] = []
    new_counts: list[int] = []
    new_cart_ordered: list["np.ndarray"] = []
    for el in seen_order:
        idxs = [i for i, e in enumerate(kept_elements_raw) if e == el]
        new_elements.append(el)
        new_counts.append(len(idxs))
        new_cart_ordered.extend(kept_cart[idxs])

    new_cart = _np.array(new_cart_ordered, dtype=float)

    # Build cubic vacuum box: side = 2*(radius_A + 5)
    box_half = radius_A + 5.0
    box_size = 2.0 * box_half
    box_lat  = _np.diag([box_size, box_size, box_size])

    # Center NP in box
    np_center = new_cart.mean(axis=0)
    shifted   = new_cart - np_center + _np.array([box_half, box_half, box_half])
    frac_out  = _cart_to_direct(shifted, box_lat)

    poscar_data = dict(
        scale=1.0,
        lattice=box_lat,
        elements=new_elements,
        counts=new_counts,
        n_atoms=sum(new_counts),
        coords=frac_out,
        selective=False,
        flags=[("T", "T", "T")] * sum(new_counts),
    )
    return _write_poscar(poscar_data, title="Nanoparticle (HPCA np_builder)")


# ---------------------------------------------------------------------------
# NP-on-substrate builder  — main public API
# ---------------------------------------------------------------------------

def build_substrate_with_nanoparticles(
    substrate_file: "str | Path",
    bulk_file: "str | Path",
    np_radius: float = 5.0,
    distances: "list[float] | None" = None,
    n_x: int = 1,
    n_y: int = 1,
    z_gap: float = 2.0,
    boundary_pad: float = 5.0,
    z_reps: int = 1,
    vacuum: float = 15.0,
    save_np: bool = True,
    out_dir: "Path | None" = None,
) -> list[str]:
    """Place nanoparticle(s) on a substrate surface.

    Compatible API with the external build_substrate_np.py script so that
    h00_design.py can call this function with the same keyword arguments.

    Args:
        substrate_file: POSCAR of the substrate slab.
        bulk_file     : POSCAR of the bulk material to carve the NP from.
        np_radius     : Radius of nanoparticle in Å.
        distances     : List of z-gaps (Å) from top substrate atom to NP bottom.
                        Defaults to [3.0].
        n_x, n_y      : Number of NP replicas in x and y directions.
        z_gap         : Extra clearance (Å) added between substrate surface and NP.
        boundary_pad  : Lateral padding (Å) around the NP tiling footprint — added
                        to the lateral cell dimensions to avoid periodic image clashes.
        z_reps        : Number of vertical repetitions of the substrate unit cell
                        (applied before placing the NP so you can thicken the slab
                        on-the-fly; set to 1 to leave the slab as-is).
        vacuum        : Vacuum thickness (Å) added above the topmost NP atom.
        save_np       : If True, also write nanoparticle.vasp to out_dir.
        out_dir       : Directory for output files.  Defaults to substrate_file.parent.

    Returns:
        List of output file names relative to out_dir,
        e.g. ["substrate_np_d3.vasp", "substrate_np_d5.vasp"].
    """
    import numpy as _np

    substrate_file = Path(substrate_file)
    bulk_file      = Path(bulk_file)

    if distances is None:
        distances = [3.0]
    distances = [float(d) for d in distances]

    if out_dir is None:
        out_dir = substrate_file.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Carve nanoparticle ────────────────────────────────────────────────
    log.info("np_builder: carving NP from %s, radius=%.1f Å", bulk_file.name, np_radius)
    np_poscar_str = carve_nanoparticle(bulk_file, np_radius)

    # Parse NP data
    import tempfile, os as _os
    _tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".vasp", delete=False)
    _tmp.write(np_poscar_str)
    _tmp.close()
    np_data = _read_poscar(Path(_tmp.name))
    _os.unlink(_tmp.name)

    np_lat  = np_data["lattice"] * np_data["scale"]   # cubic vacuum box (Å)
    np_frac = np_data["coords"]
    np_cart = _direct_to_cart(np_frac, np_lat)

    # NP dimensions in Cartesian
    np_z_min = float(np_cart[:, 2].min())
    np_z_max = float(np_cart[:, 2].max())
    np_x_span = float(np_cart[:, 0].max() - np_cart[:, 0].min())
    np_y_span = float(np_cart[:, 1].max() - np_cart[:, 1].min())
    np_center_x = float(np_cart[:, 0].mean())
    np_center_y = float(np_cart[:, 1].mean())

    if save_np:
        np_path = out_dir / "nanoparticle.vasp"
        np_path.write_text(np_poscar_str)
        log.info("np_builder: wrote %s", np_path)

    # ── 2. Parse and optionally thicken substrate ─────────────────────────────
    log.info("np_builder: reading substrate %s", substrate_file.name)
    sub_data = _read_poscar(substrate_file)

    if z_reps > 1:
        sub_data = _tile_z(sub_data, z_reps)

    sub_lat  = sub_data["lattice"] * sub_data["scale"]
    sub_frac = sub_data["coords"]
    sub_cart = _direct_to_cart(sub_frac, sub_lat)
    sub_z_max = float(sub_cart[:, 2].max())
    sub_z_min = float(sub_cart[:, 2].min())

    # Substrate lateral dimensions
    sub_x = float(sub_lat[0, 0])   # a-vector x component (Å)
    sub_y = float(sub_lat[1, 1])   # b-vector y component (Å)
    # (handles orthogonal cells; for non-orthogonal, use vector norms)
    sub_a_len = float(_np.linalg.norm(sub_lat[0]))
    sub_b_len = float(_np.linalg.norm(sub_lat[1]))

    # ── 3. Build tiling layout for NPs ──────────────────────────────────────
    # Each NP occupies (np_x_span + 2*boundary_pad) × (np_y_span + 2*boundary_pad)
    # laterally.  We tile n_x × n_y NPs.
    cell_x = n_x * (np_x_span + 2.0 * boundary_pad)
    cell_y = n_y * (np_y_span + 2.0 * boundary_pad)

    # Use the larger of substrate or required NP footprint
    new_a = max(sub_a_len, cell_x)
    new_b = max(sub_b_len, cell_y)

    # Scale substrate to fit laterally (uniform xy scale per axis)
    scale_a = new_a / sub_a_len if sub_a_len > 0 else 1.0
    scale_b = new_b / sub_b_len if sub_b_len > 0 else 1.0

    # Build per-distance output
    output_files: list[str] = []

    for dist in distances:
        out_fname = f"substrate_np_d{int(dist)}.vasp"
        out_path  = out_dir / out_fname

        # ── 4. Assemble combined structure for this distance ─────────────────

        # Scale substrate cell laterally
        new_sub_lat = sub_lat.copy()
        new_sub_lat[0] *= scale_a
        new_sub_lat[1] *= scale_b
        # Substrate fractional coords are unchanged (they refer to the same
        # fractional positions; the cell is just stretched).

        # Where to place NP bottom: z_max_substrate + z_gap + dist
        z_np_bottom_target = sub_z_max + z_gap + dist

        # Shift NP so its bottom sits at z_np_bottom_target
        np_z_shift = z_np_bottom_target - np_z_min

        # Collect all NP atom positions (Cartesian, new cell)
        np_atoms_cart: list["np.ndarray"] = []
        np_atoms_el:   list[str] = []

        atom_elements_np: list[str] = []
        for el, cnt in zip(np_data["elements"], np_data["counts"]):
            atom_elements_np.extend([el] * cnt)

        for ix in range(n_x):
            for iy in range(n_y):
                # Lateral center for this replica
                cx = (ix + 0.5) * (new_a / n_x)
                cy = (iy + 0.5) * (new_b / n_y)

                for i_atom in range(np_data["n_atoms"]):
                    ax = np_cart[i_atom, 0] - np_center_x + cx
                    ay = np_cart[i_atom, 1] - np_center_y + cy
                    az = np_cart[i_atom, 2] + np_z_shift
                    np_atoms_cart.append(_np.array([ax, ay, az]))
                    np_atoms_el.append(atom_elements_np[i_atom])

        # z_top of NP in absolute Cartesian
        np_z_top = max(v[2] for v in np_atoms_cart)

        # New c-vector length: substrate + NP + vacuum
        # We keep substrate atoms in z from sub_z_min..sub_z_max,
        # NP sits above that, then vacuum on top.
        # Total c length needed:
        total_z = np_z_top + vacuum
        # But also ensure total_z > current sub c length
        current_c = float(_np.linalg.norm(sub_lat[2]))
        total_z = max(total_z, current_c + 2.0 * np_radius + vacuum)

        new_sub_lat[2] = _np.array([0.0, 0.0, total_z])

        # Convert substrate atoms to Cartesian in new (scaled) cell
        # Fractional coords are unchanged; multiply by new lat
        new_sub_cart = _direct_to_cart(sub_frac, new_sub_lat)

        # Combine substrate + NP atoms
        all_cart:  list["np.ndarray"] = list(new_sub_cart)
        all_cart  += np_atoms_cart

        sub_atom_els: list[str] = []
        for el, cnt in zip(sub_data["elements"], sub_data["counts"]):
            sub_atom_els.extend([el] * cnt)
        all_els = sub_atom_els + np_atoms_el

        # Rebuild element/count lists preserving substrate-first ordering
        # then NP elements
        ordered_els: list[str] = []
        for el in sub_data["elements"]:
            if el not in ordered_els:
                ordered_els.append(el)
        for el in (np_data["elements"] or []):
            if el not in ordered_els:
                ordered_els.append(el)

        new_elements:  list[str] = []
        new_counts:    list[int] = []
        new_cart_all:  list["np.ndarray"] = []
        for el in ordered_els:
            idxs = [i for i, e in enumerate(all_els) if e == el]
            if idxs:
                new_elements.append(el)
                new_counts.append(len(idxs))
                new_cart_all.extend(all_cart[i] for i in idxs)

        combined_cart = _np.array(new_cart_all, dtype=float)
        combined_frac = _cart_to_direct(combined_cart, new_sub_lat)

        # Wrap fractional coords into [0,1)
        combined_frac = combined_frac % 1.0

        poscar_data = dict(
            scale=1.0,
            lattice=new_sub_lat,
            elements=new_elements,
            counts=new_counts,
            n_atoms=sum(new_counts),
            coords=combined_frac,
            selective=False,
            flags=[("T", "T", "T")] * sum(new_counts),
        )

        poscar_str = _write_poscar(
            poscar_data,
            title=f"Substrate+NP r={np_radius:.1f}A d={dist:.1f}A (HPCA np_builder)",
        )
        out_path.write_text(poscar_str)
        output_files.append(out_fname)
        log.info(
            "np_builder: wrote %s (%d atoms, dist=%.1f Å)",
            out_fname, sum(new_counts), dist,
        )

    return output_files


# ---------------------------------------------------------------------------
# Internal helper: tile substrate in z
# ---------------------------------------------------------------------------

def _tile_z(sub_data: dict, z_reps: int) -> dict:
    """Repeat a substrate slab z_reps times along the c-axis.

    Returns a new data dict with tiled atoms and scaled c-vector.
    """
    import numpy as _np

    lat  = sub_data["lattice"] * sub_data["scale"]
    frac = sub_data["coords"]
    n    = sub_data["n_atoms"]

    new_frac = _np.zeros((n * z_reps, 3), dtype=float)
    for k in range(z_reps):
        new_frac[k * n:(k + 1) * n, :2] = frac[:, :2]
        new_frac[k * n:(k + 1) * n, 2]  = (frac[:, 2] + k) / z_reps

    new_lat       = lat.copy()
    new_lat[2, 2] = lat[2, 2] * z_reps   # extend c

    new_counts  = [c * z_reps for c in sub_data["counts"]]
    new_n_atoms = sum(new_counts)

    return dict(
        comment=sub_data.get("comment", "tiled"),
        scale=1.0,
        lattice=new_lat,
        elements=sub_data["elements"],
        counts=new_counts,
        n_atoms=new_n_atoms,
        coords=new_frac,
        selective=False,
        flags=[("T", "T", "T")] * new_n_atoms,
    )
