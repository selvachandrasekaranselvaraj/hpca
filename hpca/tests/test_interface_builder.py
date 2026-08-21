"""Unit tests for hpca.core.interface_builder — the electrode-slab +
PACKMOL-electrolyte sandwich builder used for HC-Na | electrolyte | HC-Na
interface projects.

The full PACKMOL packing step is exercised separately as an integration test
(skipped if the packmol binary isn't available); everything else here is pure
geometry/arithmetic and runs unconditionally.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from hpca.core.paths import load_platform_config
from hpca.core.interface_builder import (
    ElectrodeBox,
    read_lammps_dump,
    read_poscar_electrode,
    slice_slab_gap,
    open_gap_by_extending_box,
    electrolyte_counts_for_gap,
    build_molecule_ids,
    min_intermolecular_distance,
    write_sandwich_poscar,
    _NATOMS,
)


def _make_dump(tmp_path: Path, box=(20.0, 20.0, 20.0)) -> Path:
    """A small synthetic LAMMPS dump: a uniform grid of C atoms + a few Na,
    spanning the full box in z so slicing has real atoms to remove/keep."""
    rng = np.random.default_rng(0)
    n_c, n_na = 400, 40
    pts_c = rng.random((n_c, 3)) * box
    pts_na = rng.random((n_na, 3)) * box

    lines = [
        "ITEM: TIMESTEP", "0", "ITEM: NUMBER OF ATOMS", str(n_c + n_na),
        "ITEM: BOX BOUNDS xy xz yz pp pp pp",
        f"0.0 {box[0]} 0.0", f"0.0 {box[1]} 0.0", f"0.0 {box[2]} 0.0",
        "ITEM: ATOMS id type x y z element",
    ]
    i = 1
    for x, y, z in pts_c:
        lines.append(f"{i} 1 {x} {y} {z} C"); i += 1
    for x, y, z in pts_na:
        lines.append(f"{i} 2 {x} {y} {z} Na"); i += 1

    p = tmp_path / "dump.lmp"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_read_lammps_dump_roundtrip(tmp_path):
    path = _make_dump(tmp_path)
    box = read_lammps_dump(path)
    assert isinstance(box, ElectrodeBox)
    assert len(box.elements) == 440
    assert box.elements.count("C") == 400
    assert box.elements.count("Na") == 40
    np.testing.assert_allclose(box.box_lengths, [20.0, 20.0, 20.0])
    # all coordinates within the box
    assert (box.xyz >= 0).all() and (box.xyz <= 20.0).all()


def test_slice_slab_gap_centers_window_and_conserves_atoms(tmp_path):
    box = read_lammps_dump(_make_dump(tmp_path))
    kept_el, kept_xyz, z_lo, z_hi = slice_slab_gap(box, gap_thickness=8.0, axis=2)

    assert z_hi - z_lo == pytest.approx(8.0)
    assert z_lo == pytest.approx(6.0)   # centered: (20-8)/2
    assert z_hi == pytest.approx(14.0)

    n_removed = len(box.elements) - len(kept_el)
    assert n_removed > 0
    # every kept atom's z must be outside [z_lo, z_hi)
    assert not ((kept_xyz[:, 2] >= z_lo) & (kept_xyz[:, 2] < z_hi)).any()
    assert len(kept_el) == len(kept_xyz)


def test_slice_slab_gap_rejects_gap_larger_than_box(tmp_path):
    box = read_lammps_dump(_make_dump(tmp_path))
    with pytest.raises(ValueError):
        slice_slab_gap(box, gap_thickness=25.0, axis=2)


_TOY_POSCAR = """toy electrode
1.0
  10.0  0.0  0.0
  0.0  10.0  0.0
  0.0  0.0  10.0
  C  Na
  4  1
Direct
  0.10  0.10  0.10
  0.60  0.10  0.10
  0.10  0.60  0.10
 -0.05  0.30  0.30
  0.50  0.50  0.50
"""


def test_read_poscar_electrode_parses_direct_coords(tmp_path):
    p = tmp_path / "CONTCAR"
    p.write_text(_TOY_POSCAR)
    box = read_poscar_electrode(p)
    assert box.elements == ["C", "C", "C", "C", "Na"]
    np.testing.assert_allclose(box.box_lengths, [10.0, 10.0, 10.0])
    # atom 0: frac (0.1,0.1,0.1) * 10 A cell -> cartesian (1,1,1)
    np.testing.assert_allclose(box.xyz[0], [1.0, 1.0, 1.0])
    # atom 3: frac x=-0.05 -> cartesian x=-0.5 (unwrapped at this stage)
    np.testing.assert_allclose(box.xyz[3], [-0.5, 3.0, 3.0])


def test_open_gap_by_extending_box_preserves_atoms_and_wraps_coords(tmp_path):
    p = tmp_path / "CONTCAR"
    p.write_text(_TOY_POSCAR)
    box = read_poscar_electrode(p)

    els, xyz, new_box, z_lo, z_hi = open_gap_by_extending_box(box, gap_thickness=6.0, axis=2)

    assert els == box.elements
    assert len(xyz) == len(els)
    # only the z axis grows; x/y unchanged
    np.testing.assert_allclose(new_box, [10.0, 10.0, 16.0])
    assert z_lo == pytest.approx(10.0)
    assert z_hi == pytest.approx(16.0)
    # the out-of-range atom (x=-0.5) must be wrapped into [0, 10)
    assert (xyz[:, 0] >= 0).all() and (xyz[:, 0] < 10.0).all()
    np.testing.assert_allclose(xyz[3, 0], 9.5)  # -0.5 wrapped -> 9.5
    # all z coordinates stay within the ORIGINAL length, not the new box
    assert (xyz[:, 2] >= 0).all() and (xyz[:, 2] < 10.0).all()


def test_electrolyte_counts_scale_with_molarity_and_ratio():
    prev_salt = 0
    for molarity in (0.1, 0.5, 1.0, 2.0):
        counts = electrolyte_counts_for_gap(50.0, 50.0, 20.0, molarity)
        assert counts["NaPF6"] >= prev_salt
        prev_salt = counts["NaPF6"]
        assert counts["EC"] > 0 and counts["DEC"] > 0

    # 2:1 EC:DEC ratio should roughly double EC vs DEC
    counts_ratio = electrolyte_counts_for_gap(50.0, 50.0, 20.0, 1.0,
                                               solvent_ratio=(2.0, 1.0))
    assert counts_ratio["EC"] > counts_ratio["DEC"]


def test_build_molecule_ids_groups_electrode_as_single_id():
    mol_id = build_molecule_ids(n_electrode=100, mol_counts={"NaPF6": 2, "EC": 3},
                                 mol_natoms=_NATOMS)
    assert (mol_id[:100] == 0).all()               # electrode: one frozen body
    assert len(mol_id) == 100 + 2 * 8 + 3 * 10      # NaPF6=8 atoms, EC=10 atoms
    # each electrolyte molecule gets a distinct, contiguous id
    ids_after = mol_id[100:]
    assert len(np.unique(ids_after)) == 5           # 2 NaPF6 + 3 EC = 5 molecules


def test_min_intermolecular_distance_ignores_bonded_pairs():
    """A tight 'molecule' (bond-length spacing) next to a well-separated
    second molecule: the reported minimum must be the INTERmolecular gap,
    not the short intramolecular bond."""
    box_lengths = np.array([20.0, 20.0, 20.0])
    # molecule 0 (electrode stand-in): two atoms 1.1 A apart (a "bond")
    # molecule 1: a single atom placed 3.0 A from the nearest electrode atom
    xyz = np.array([
        [5.0, 5.0, 5.0],
        [6.1, 5.0, 5.0],   # 1.1 A from atom 0 — intramolecular, must be ignored
        [9.1, 5.0, 5.0],   # 3.0 A from atom 1 — the real intermolecular gap
    ])
    elements = ["C", "C", "O"]
    mol_id = np.array([0, 0, 1])

    dmin, i, j = min_intermolecular_distance(elements, xyz, box_lengths, mol_id)
    assert dmin == pytest.approx(3.0, abs=1e-6)
    assert {i, j} == {1, 2}


def test_write_sandwich_poscar_roundtrip(tmp_path):
    elements = ["C", "C", "Na", "O", "H", "H"]
    xyz = np.array([[i, i, i] for i in range(6)], dtype=float)
    box_lengths = np.array([30.0, 30.0, 30.0])
    out = tmp_path / "POSCAR"
    write_sandwich_poscar(elements, xyz, box_lengths, out)

    lines = out.read_text().splitlines()
    species = lines[5].split()
    counts = [int(c) for c in lines[6].split()]
    assert species == ["C", "Na", "O", "H"]
    assert counts == [2, 1, 1, 2]
    assert sum(counts) == len(elements)


def _resolve_packmol_bin() -> str:
    return (shutil.which("packmol")
            or load_platform_config().get("hpc", {}).get("packmol_bin", ""))


@pytest.mark.skipif(not _resolve_packmol_bin(), reason="packmol binary not available")
def test_pack_electrolyte_gap_produces_non_overlapping_cell(tmp_path):
    """End-to-end smoke test: small synthetic electrode + a handful of
    molecules, verifying PACKMOL's pbc-aware fixed-obstacle packing leaves no
    real intermolecular overlap."""
    from hpca.core.interface_builder import pack_electrolyte_gap
    from hpca.sim.structure_fetch import fetch_structure

    packmol_bin = _resolve_packmol_bin()

    box = read_lammps_dump(_make_dump(tmp_path, box=(20.0, 20.0, 20.0)))
    kept_el, kept_xyz, z_lo, z_hi = slice_slab_gap(box, gap_thickness=8.0, axis=2)
    n_electrode = len(kept_el)

    mol_counts = {"NaPF6": 1, "EC": 2}
    mol_vasp = {}
    for name in mol_counts:
        p = tmp_path / f"{name}.vasp"
        assert fetch_structure(name, "solvent" if name != "NaPF6" else "salt", tmp_path)
        mol_vasp[name] = p

    result = pack_electrolyte_gap(
        kept_el, kept_xyz, box.box_lengths, z_lo, z_hi,
        mol_counts, mol_vasp, packmol_bin=packmol_bin, tolerance=2.0, timeout=120,
    )
    assert result is not None
    els, xyz = result
    assert len(els) == n_electrode + 1 * 8 + 2 * 10

    mol_id = build_molecule_ids(n_electrode, mol_counts, _NATOMS)
    dmin, _, _ = min_intermolecular_distance(els, xyz, box.box_lengths, mol_id)
    assert dmin >= 1.9   # PACKMOL tolerance was 2.0; allow small numerical slack


@pytest.mark.skipif(not _resolve_packmol_bin(), reason="packmol binary not available")
def test_pack_electrolyte_gap_with_no_electrode_builds_pure_bulk_liquid(tmp_path):
    """CMD-tier lane: same packer, zero electrode atoms — a plain bulk
    electrolyte box (used for classical-force-field transport properties,
    where there is no reactive electrode chemistry to represent)."""
    from hpca.core.interface_builder import pack_electrolyte_gap
    from hpca.sim.structure_fetch import fetch_structure

    packmol_bin = _resolve_packmol_bin()
    box_lengths = np.array([20.0, 20.0, 20.0])
    mol_counts = {"NaPF6": 1, "EC": 2}
    mol_vasp = {}
    for name in mol_counts:
        p = tmp_path / f"{name}.vasp"
        assert fetch_structure(name, "solvent" if name != "NaPF6" else "salt", tmp_path)
        mol_vasp[name] = p

    result = pack_electrolyte_gap(
        [], np.empty((0, 3)), box_lengths, 0.0, 20.0,
        mol_counts, mol_vasp, packmol_bin=packmol_bin, tolerance=2.0, timeout=120,
    )
    assert result is not None
    els, xyz = result
    assert len(els) == 1 * 8 + 2 * 10

    mol_id = build_molecule_ids(0, mol_counts, _NATOMS)
    dmin, _, _ = min_intermolecular_distance(els, xyz, box_lengths, mol_id)
    assert dmin >= 1.9
