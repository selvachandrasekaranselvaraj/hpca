"""Regression tests for diethyl carbonate (DEC) solvent registration.

DEC (EC:DEC electrolytes) was entirely absent from HPCA's molecule library —
no PubChem CID, no OPLS-AA geometry, no MW/density/atom-count entries. These
tests guard the fix: DEC's fallback geometry must be internally consistent
(declared bonds == geometry-derived bonds) and resolve to known OPLS-AA atom
types, and NaPF6/EC:DEC composition math must produce sane, increasing salt
counts with molarity.
"""
from __future__ import annotations


def test_dec_pubchem_cid_registered():
    from hpca.sim.structure_fetch import _PUBCHEM_CIDS, _ALIASES
    assert _PUBCHEM_CIDS["DEC"] == 7766
    assert _ALIASES["dec"] == "DEC"


def test_dec_geometry_matches_declared_bonds():
    """The hand-specified DEC bond list must agree with covalent-radius bond
    guessing on the same coordinates — otherwise the geometry and topology
    have drifted apart (e.g. a copy/paste error in atom ordering)."""
    from hpca.sim.forcefield import MOLECULES, _guess_bonds

    mol = MOLECULES["DEC"]
    atoms = mol["atoms"]
    assert len(atoms) == 18  # C5H10O3
    els = [a[0] for a in atoms]
    assert els.count("C") == 5
    assert els.count("H") == 10
    assert els.count("O") == 3

    assert sorted(mol["bonds"]) == sorted(_guess_bonds(atoms))


def test_dec_atom_types_all_recognized():
    from hpca.sim.forcefield import MOLECULES, assign_atom_types, OPLS_ATOM_TYPES

    mol = MOLECULES["DEC"]
    types = assign_atom_types(mol["atoms"], mol["bonds"])
    assert all(t in OPLS_ATOM_TYPES for t in types)
    # Symmetric molecule: both ethyl arms resolve to the same alkyl types.
    assert types.count("CT_C") == 2   # both -O-CH2- carbons
    assert types.count("CT_M") == 2   # both terminal -CH3 carbons
    assert types.count("OS_E") == 2   # both ester-bridging oxygens


def test_dec_registered_in_h00_design_tables():
    from hpca.orchestrator.handlers import h00_design as h00
    assert h00._NATOMS_PER_MOL["DEC"] == 18
    assert h00._MW["DEC"] == 118.13
    assert h00._DENSITY["DEC"] == 0.975


def test_napf6_ec_dec_molecule_counts_scale_with_molarity():
    """1:1 EC:DEC + NaPF6 composition math should yield roughly balanced
    EC/DEC counts and a NaPF6 count that never decreases as molarity rises."""
    from hpca.orchestrator.handlers.h00_design import MaterialsDesignHandler as H

    comp = {
        "solvents": [{"name": "EC", "ratio": 1.0}, {"name": "DEC", "ratio": 1.0}],
        "salts": [{"name": "NaPF6"}],
    }
    prev_salt = 0
    for molarity in (0.1, 0.3, 0.5, 1.0, 2.0):
        comp["salt_molarity"] = molarity
        counts = H._auto_mol_counts_from_comp(comp, natoms_target=250)
        assert counts["NaPF6"] >= prev_salt
        prev_salt = counts["NaPF6"]
        # Roughly 1:1 EC:DEC (within rounding) at every concentration.
        assert abs(counts["EC"] - counts["DEC"]) <= 3
