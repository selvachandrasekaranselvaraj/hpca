"""hpca/core/neb/image_tools.py — Selective dynamics, XYZ trajectory, and NEB chain generation."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    from pymatgen.core import Structure
except ImportError:
    import sys
    print("ERROR: pymatgen is required. Install with: pip install pymatgen")
    sys.exit(1)

from .poscar_io import read_structure, write_poscar
from .path_finder import (
    analyze_migration_sites,
    build_anchor_chain,
    build_nonlinear_chained_path,
    build_chained_path_linear,
)


# -----------------------------------------------------------------------------
# Selective Dynamics
# -----------------------------------------------------------------------------

def apply_constrained_path_dynamics(
    structure: Structure,
    migrating_atom_index: int,
    neighbor_radius: float = 4.0,
) -> Structure:
    """
    Add selective dynamics flags for a constrained NEB (cNEB) calculation.

    The migrating atom is set to F F F (fixed). Atoms within the neighbour sphere
    are set to T T T (free). All other atoms are F F F (fixed).

    Args:
        structure (Structure): Pymatgen Structure (original).
        migrating_atom_index (int): 0‑based index of the migrating atom.
        neighbor_radius (float): Radius (Å) within which atoms are allowed to relax.

    Returns:
        Structure: New Structure with a site property "selective_dynamics".
    """
    new_struct = structure.copy()
    coords = new_struct.cart_coords
    center = coords[migrating_atom_index]
    sel_dyn = []
    for i, pos in enumerate(coords):
        dist = np.linalg.norm(pos - center)
        if i == migrating_atom_index:
            sel_dyn.append([False, False, False])  # migrating atom held at interpolated position
        elif dist <= neighbor_radius:
            sel_dyn.append([True, True, True])
        else:
            sel_dyn.append([False, False, False])
    new_struct.add_site_property("selective_dynamics", sel_dyn)
    return new_struct


# -----------------------------------------------------------------------------
# XYZ Trajectory Writer
# -----------------------------------------------------------------------------

def write_xyz_trajectory(mech_dir: Path, output_xyz: Path, n_images: int) -> None:
    """
    Write an XYZ trajectory file from all POSCARs in a mechanism directory.

    The files are read from mech_dir/00, mech_dir/01, ..., mech_dir/N+1 and
    concatenated into a single XYZ file suitable for OVITO or other visualisation.

    Args:
        mech_dir (Path): Mechanism directory (e.g., neb/vacancy).
        output_xyz (Path): Output XYZ file path.
        n_images (int): Number of intermediate images.
    """
    n_end = n_images + 1
    all_structs = []
    for idx in range(0, n_end + 1):
        poscar_path = mech_dir / f"{idx:02d}" / "POSCAR"
        if not poscar_path.exists():
            continue
        struct = read_structure(poscar_path)
        all_structs.append((idx, struct))

    if not all_structs:
        raise RuntimeError(f"No POSCARs found in {mech_dir}")

    lines = []
    for idx, struct in all_structs:
        lines.append(f"{len(struct)}")
        lines.append(f"Mechanism: {mech_dir.name}, image {idx:02d}")
        for site in struct:
            symbol = site.species_string
            if " " in symbol:
                symbol = symbol.split()[0]
            coords = site.coords
            lines.append(f"{symbol} {coords[0]:.8f} {coords[1]:.8f} {coords[2]:.8f}")

    output_xyz.write_text("\n".join(lines) + "\n")
    print(f"[XYZ] Trajectory written to {output_xyz}")


# -----------------------------------------------------------------------------
# Main Generator (single mechanism)
# -----------------------------------------------------------------------------

def generate_neb_chain(
    poscar_path: Path,
    output_dir: Path,
    template_dir: Optional[Path] = None,
    mobile_element: str = "Li",
    mechanism: str = "vacancy",
    dopant_element: Optional[str] = None,
    directions: Optional[List[str]] = None,
    n_images: Optional[int] = None,
    spacing: float = 0.2,
    hop_distance: float = 2.8,
    neighbor_radius: float = 4.0,
    void_cutoff: float = 2.5,
    anchors: Optional[List[List[float]]] = None,
    perturb: bool = False,
    perturb_strength: float = 0.02,
    cutoff_radius: float = 2.0,
    spring_constant: float = 1.0,
    repulsion_scale: float = 100.0,
) -> Path:
    """
    Generate all input files for a single NEB mechanism.

    This is the main orchestrator function. It:
        1. Reads the POSCAR and detects voids.
        2. Builds anchor points (nodes) using crystallographic directions.
        3. Generates intermediate images using a non‑linear path predictor.
        4. Writes POSCARs with selective dynamics for each image and endpoint.
        5. Copies INCAR, KPOINTS, POTCAR (or builds POTCAR) into each image dir.
        6. Saves an XYZ trajectory and metadata.

    Important: This function does NOT run VASP; it only generates inputs.

    Args:
        poscar_path (Path): Path to the initial relaxed POSCAR.
        output_dir (Path): Root output directory (e.g., ./neb).
        template_dir (Optional[Path]): Directory containing INCAR, KPOINTS, POTCAR.
        mobile_element (str): Symbol of the mobile ion.
        mechanism (str): One of "vacancy", "interstitial", or "dopant".
        dopant_element (Optional[str]): Element for dopant (if mechanism == "dopant").
        directions (Optional[List[str]]): Crystallographic directions.
        n_images (Optional[int]): Override automatic number of images.
        spacing (float): Target spacing between images (Å).
        hop_distance (float): Step distance along directions (Å).
        neighbor_radius (float): Selective dynamics sphere radius (Å).
        void_cutoff (float): Void detection cutoff (Å).
        anchors (Optional[List[List[float]]]): Explicit anchor coordinates.
        perturb (bool): If True, add random perturbation to images.
        perturb_strength (float): Perturbation strength (Å).
        cutoff_radius (float): Repulsion cutoff for non‑linear predictor (Å).
        spring_constant (float): Spring constant for non‑linear predictor.
        repulsion_scale (float): Repulsion strength for non‑linear predictor.

    Returns:
        Path: Path to the generated mechanism directory (e.g., neb/vacancy).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    struct = read_structure(poscar_path)
    lattice = struct.lattice.matrix

    # Detect voids and mobile sites
    site_data = analyze_migration_sites(
        struct, mobile_element=mobile_element, void_cutoff=void_cutoff
    )
    mobile_sites = site_data["mobile_sites"]
    void_sites = site_data["void_sites"]

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    with open(analysis_dir / "void_sites.yaml", "w") as f:
        json.dump(site_data, f, indent=2)

    if not mobile_sites:
        raise RuntimeError(f"No {mobile_element} sites found in POSCAR.")

    if mechanism == "vacancy":
        if not void_sites:
            raise RuntimeError("No void sites detected for vacancy mechanism.")
    elif mechanism in ("interstitial", "dopant"):
        if len(void_sites) < 2:
            raise RuntimeError(f"Need at least 2 void sites for {mechanism}, found {len(void_sites)}.")
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    # Determine if we insert a new atom (dopant)
    if mechanism == "dopant":
        insert_dopant = True
        migrating_element = dopant_element if dopant_element else mobile_element
        migrating_idx = 0  # inserted at index 0
        initial_frac = np.array(void_sites[0]["fractional"])
    else:
        insert_dopant = False
        migrating_element = mobile_element
        migrating_idx = mobile_sites[0]["index"]
        initial_frac = np.array(mobile_sites[0]["fractional"])

    # Build anchors
    if anchors is None:
        if directions is None:
            directions = ["[100]", "[010]", "[001]", "[110]", "[101]", "[011]", "[111]"]
        anchor_fracs = build_anchor_chain(
            initial_frac, void_sites, directions, hop_distance, lattice, max_anchors=10
        )
    else:
        anchor_fracs = [np.array(a) for a in anchors]

    # Compute total straight-line length (for spacing)
    total_length = 0.0
    for i in range(len(anchor_fracs) - 1):
        diff = anchor_fracs[i+1] - anchor_fracs[i]
        diff -= np.round(diff)
        total_length += np.linalg.norm(diff @ lattice)

    if n_images is None:
        n_images = max(5, int(total_length / spacing))
        print(f"Auto‑spacing: total length = {total_length:.2f} Å, spacing = {spacing:.2f} Å → {n_images} intermediate images")
    else:
        print(f"Manual override: {n_images} intermediate images")

    # Build initial and final structures
    if mechanism in ("vacancy", "interstitial"):
        i_struct = struct.copy()
        f_struct = struct.copy()
        f_struct.replace(migrating_idx, migrating_element, anchor_fracs[-1])
    else:  # dopant
        i_struct = struct.copy()
        i_struct.insert(0, migrating_element, anchor_fracs[0])
        f_struct = struct.copy()
        f_struct.insert(0, migrating_element, anchor_fracs[-1])

    mech_dir = output_dir / mechanism
    mech_dir.mkdir(exist_ok=True)

    write_poscar(i_struct, mech_dir / "i.vasp")
    write_poscar(f_struct, mech_dir / "f.vasp")

    # ---------- Non‑linear path generation ----------
    # Framework atoms: all atoms except the migrating one (or all non-mobile)
    if mechanism == "dopant":
        framework_fracs = np.array([struct.frac_coords[i] for i in range(len(struct))])
    else:
        framework_indices = [i for i in range(len(struct)) if i != migrating_idx]
        framework_fracs = np.array([struct.frac_coords[i] for i in framework_indices])

    coords_nonlinear = build_nonlinear_chained_path(
        anchor_fracs, framework_fracs, lattice, n_images,
        spacing=spacing, cutoff_radius=cutoff_radius
    )

    # Fallback to linear if the predictor fails
    if len(coords_nonlinear) != n_images:
        print(f"Warning: non-linear predictor returned {len(coords_nonlinear)} images, expected {n_images}. Falling back to linear.")
        coords_nonlinear = build_chained_path_linear(anchor_fracs, n_images, lattice)

    path_coords = coords_nonlinear
    # ------------------------------------------------

    def get_migrating_pos(struct: Structure) -> np.ndarray:
        """Return the Cartesian position of the migrating atom in struct."""
        return struct.cart_coords[migrating_idx]

    n_end = n_images + 1

    # Determine template files (INCAR, KPOINTS, POTCAR)
    if template_dir is None:
        template_dir = poscar_path.parent
    template_dir = Path(template_dir)

    # POTCAR source: supplied template first, otherwise canonical configured builder.
    potcar_source = template_dir / "POTCAR" if (template_dir / "POTCAR").exists() else None
    if potcar_source is None:
        from hpca.core.potcar import build_potcar
        from hpca.core.vasp_job import read_poscar_elements
        generated_potcar = output_dir / "POTCAR"
        try:
            build_potcar(read_poscar_elements(poscar_path), generated_potcar)
            potcar_source = generated_potcar
            print(f"[INFO] Built configured POTCAR: {potcar_source}")
        except (FileNotFoundError, KeyError, OSError) as exc:
            print(f"[WARNING] POTCAR unavailable: {exc}")

    def copy_template_files(img_dir: Path) -> None:
        """Copy INCAR, KPOINTS, and POTCAR into an image directory."""
        for fname in ["INCAR", "KPOINTS"]:
            src = template_dir / fname
            dst = img_dir / fname
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        if potcar_source and potcar_source.exists():
            dst = img_dir / "POTCAR"
            if not dst.exists():
                shutil.copy2(potcar_source, dst)

    # Endpoint 00
    ep0_dir = mech_dir / "00"
    ep0_dir.mkdir(exist_ok=True)
    write_poscar(i_struct, ep0_dir / "POSCAR")
    copy_template_files(ep0_dir)
    with open(ep0_dir / "metadata.yaml", "w") as f:
        json.dump({
            "image_index": 0,
            "target_fractional": i_struct.frac_coords[migrating_idx].tolist(),
            "target_cartesian": get_migrating_pos(i_struct).tolist(),
            "mechanism": mechanism,
            "migrating_atom_index": migrating_idx,
            "is_endpoint": True,
        }, f, indent=2)

    # Endpoint N+1
    epn_dir = mech_dir / f"{n_end:02d}"
    epn_dir.mkdir(exist_ok=True)
    write_poscar(f_struct, epn_dir / "POSCAR")
    copy_template_files(epn_dir)
    with open(epn_dir / "metadata.yaml", "w") as f:
        json.dump({
            "image_index": n_end,
            "target_fractional": f_struct.frac_coords[migrating_idx].tolist(),
            "target_cartesian": get_migrating_pos(f_struct).tolist(),
            "mechanism": mechanism,
            "migrating_atom_index": migrating_idx,
            "is_endpoint": True,
        }, f, indent=2)

    # Intermediate images
    for img_idx, frac in enumerate(path_coords, start=1):
        img_dir = mech_dir / f"{img_idx:02d}"
        img_dir.mkdir(exist_ok=True)

        if mechanism in ("vacancy", "interstitial"):
            img_struct = struct.copy()
            img_struct.replace(migrating_idx, migrating_element, frac)
        else:  # dopant
            img_struct = struct.copy()
            img_struct.insert(0, migrating_element, frac)

        img_struct = apply_constrained_path_dynamics(
            img_struct, migrating_idx, neighbor_radius
        )

        if perturb:
            pos_cart = img_struct.cart_coords[migrating_idx]
            perturb_vec = np.random.normal(0, perturb_strength, 3)
            new_pos_cart = pos_cart + perturb_vec
            new_frac = new_pos_cart @ np.linalg.inv(lattice)
            if mechanism in ("vacancy", "interstitial"):
                img_struct.replace(migrating_idx, migrating_element, new_frac % 1.0)
            else:
                img_struct.remove_sites([0])
                img_struct.insert(0, migrating_element, new_frac % 1.0)

        write_poscar(img_struct, img_dir / "POSCAR")
        copy_template_files(img_dir)
        with open(img_dir / "metadata.yaml", "w") as f:
            json.dump({
                "image_index": img_idx,
                "target_fractional": frac.tolist(),
                "target_cartesian": (frac @ lattice).tolist(),
                "actual_fractional": img_struct.frac_coords[migrating_idx].tolist(),
                "actual_cartesian": img_struct.cart_coords[migrating_idx].tolist(),
                "mechanism": mechanism,
                "migrating_atom_index": migrating_idx,
                "perturbed": perturb,
            }, f, indent=2)

    # Global metadata
    with open(mech_dir / "migration_metadata.yaml", "w") as f:
        json.dump({
            "mechanism": mechanism,
            "mobile_element": mobile_element,
            "insert_dopant": insert_dopant,
            "dopant_element": dopant_element if insert_dopant else None,
            "n_images": n_images,
            "anchors": [a.tolist() for a in anchor_fracs],
            "directions": directions,
            "hop_distance": hop_distance,
            "neighbor_radius": neighbor_radius,
            "void_cutoff": void_cutoff,
            "migrating_atom_index": migrating_idx,
            "perturb_applied": perturb,
            "perturb_strength": perturb_strength,
            "total_length_A": total_length,
            "image_spacing": spacing if n_images is None else None,
            "nonlinear_prediction": True,
            "cutoff_radius": cutoff_radius,
            "spring_constant": spring_constant,
            "repulsion_scale": repulsion_scale,
        }, f, indent=2)

    # Verification table (prints displacement of the migrating atom)
    print("\n" + "=" * 80)
    print(f"VERIFICATION for {mechanism.upper()} mechanism")
    print("=" * 80)
    print(f"{'Image':<8} {'Target coord (frac)':<30} {'Distance from prev (Å)':<20}")
    print("-" * 80)

    all_positions = []
    for idx in range(0, n_end + 1):
        if idx == 0:
            s = i_struct
        elif idx == n_end:
            s = f_struct
        else:
            meta_path = mech_dir / f"{idx:02d}" / "metadata.yaml"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                pos = np.array(meta["actual_cartesian"])
                target_frac = np.array(meta["target_fractional"])
                all_positions.append((idx, pos, target_frac))
                continue
            else:
                s = read_structure(mech_dir / f"{idx:02d}" / "POSCAR")
        pos = s.cart_coords[migrating_idx]
        target_frac = s.frac_coords[migrating_idx]
        all_positions.append((idx, pos, target_frac))

    prev_pos = None
    total_dist = 0.0
    for idx, pos, target_frac in all_positions:
        if prev_pos is not None:
            dist = np.linalg.norm(pos - prev_pos)
            total_dist += dist
            print(f"{idx:02d}      {target_frac[0]:6.3f} {target_frac[1]:6.3f} {target_frac[2]:6.3f}   {dist:8.4f}")
        else:
            print(f"{idx:02d}      {target_frac[0]:6.3f} {target_frac[1]:6.3f} {target_frac[2]:6.3f}   {'--':>8}")
        prev_pos = pos

    print("-" * 80)
    print(f"Total displacement (end to start): {total_dist:.4f} Å")
    print(f"Direct distance (start → end):     {np.linalg.norm(all_positions[-1][1] - all_positions[0][1]):.4f} Å")
    print("=" * 80)

    print(f"[SUCCESS] {mechanism} chain generated in {mech_dir}")
    print(f"  - {n_images} intermediate images")
    print(f"  - Anchors: {len(anchor_fracs)}")
    print(f"  - Total path length: {total_length:.2f} Å")
    print(f"  - Average spacing: {total_length/(n_images+1):.3f} Å")

    # Write XYZ trajectory
    traj_file = output_dir / f"{mechanism}_trajectory.xyz"
    write_xyz_trajectory(mech_dir, traj_file, n_images)

    return mech_dir
