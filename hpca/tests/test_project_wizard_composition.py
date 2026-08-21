"""Regression tests for wizard composition resolution."""


def _gel_comp():
    return {
        "solvents": [{"name": "DME", "ratio": 1.0}],
        "salts": [{"name": "LiFSI"}, {"name": "LiPF6"}],
        "polymers": [{"monomer": "PEO", "n_chains": 20,
                      "n_monomers": 20, "vol_pct": 20.0}],
        "copolymers": [{"monomer": "PVDF-HFP", "n_chains": 40,
                        "n_monomers_per_chain": 20, "vol_pct": 5.0}],
        "salt_molarity": 0.2,
    }


def test_individual_polymer_volume_fractions_control_chain_allocation():
    from hpca.tools.project_wizard import _compute_mlmd_box
    spec = _compute_mlmd_box(_gel_comp(), {"polymer": 20.0, "copolymer": 5.0})
    # The old implementation preserved the input 1:2 chain ratio and yielded
    # about 3 PEO : 7 PVDF-HFP chains.  A 20:5 volume request must be PEO-rich.
    assert spec["chains"]["PEO"] > spec["chains"]["PVDF-HFP"]


def test_mixed_salt_molarity_preserves_requested_ratio():
    from hpca.orchestrator.handlers.h00_design import MaterialsDesignHandler
    comp = {
        "salt_molarity": 0.7,
        "solvents": [{"name": "DME", "ratio": 1}],
        "salts": [{"name": "LiFSI", "ratio": 2},
                  {"name": "LiPF6", "ratio": 1}],
    }
    counts = MaterialsDesignHandler._auto_mol_counts_from_comp(comp, 50_000)
    assert counts["LiFSI"] > counts["LiPF6"]
    assert abs(counts["LiFSI"] / counts["LiPF6"] - 2.0) < 0.15


def test_lithium_silicon_is_classified_as_alloy_electrode():
    from hpca.tools.project_wizard import _guess_role, _suggest_system

    class FakeStructure:
        species = ["Li", "Si"]

    role = _guess_role(FakeStructure())
    assert role == "alloy_electrode"
    assert _suggest_system([{"role": role}]) == "bulk_electrode"


def test_bulk_electrode_defaults_enable_dos_without_neb():
    from hpca.tools.project_wizard import _stages_block
    stages = _stages_block("bulk_electrode", [], True)
    assert stages["neb"] is False
    assert stages["echem"] is True
    assert stages["dft"]["dos_scf"] is True
    assert stages["dft"]["dos_nonscf"] is True


def test_doping_variant_name_uses_actual_concentration():
    from hpca.tools.project_wizard import _add_mono_variants
    variants, elements = [], []
    _add_mono_variants(variants, elements, "Si", "Si", 30, "C", [5.0])
    assert variants[0]["name"] == "Si_C6p7"
    assert variants[0]["requested_pct"] == 5.0
    assert round(variants[0]["actual_pct"], 3) == 6.667


def test_solid_workload_estimate_includes_dataset_boxes():
    from hpca.tools.project_wizard import _solid_workload_estimate
    doc = {
        "crystal_doping_variants": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
        "simulation": {"aimd_temps": [300, 500, 800], "aimd_steps": 3000,
                       "aimd_dataset_steps": 3000},
        "stages": {"aimd": True, "neb": False,
                   "dft": {"vc_relax": True, "opt": True}},
    }
    estimate = _solid_workload_estimate(doc)
    assert estimate["aimd_jobs"] == 3 * (3 + 16)
    assert estimate["vasp_ionic_steps"] == 3 * 19 * 3000


def test_supercell_parser():
    from hpca.orchestrator.handlers.h00_design import MaterialsDesignHandler
    assert MaterialsDesignHandler._supercell_factors("2x3x1") == [2, 3, 1]
    factors = MaterialsDesignHandler._target_supercell_factors(32, 5000)
    assert abs(32 * factors[0] * factors[1] * factors[2] - 5000) <= 200
