"""Tests for hpca.core.project_schema validate() and migrate()."""
import pytest


def _valid():
    return {
        "name": "test",
        "category": "inorganic_sse",
        "mobile_ion": "Li",
        "T_ref": 300,
    }


def test_valid_minimal():
    from hpca.core.project_schema import validate
    errs = validate(_valid())
    assert errs == []


def test_missing_name():
    from hpca.core.project_schema import validate
    d = _valid(); del d["name"]
    errs = validate(d)
    assert any("name" in e.lower() for e in errs)


def test_missing_category():
    from hpca.core.project_schema import validate
    d = _valid(); del d["category"]
    errs = validate(d)
    assert any("category" in e.lower() for e in errs)


def test_invalid_category():
    from hpca.core.project_schema import validate
    d = _valid(); d["category"] = "bogus_category"
    errs = validate(d)
    assert any("category" in e.lower() for e in errs)


def test_transport_float_fields():
    from hpca.core.project_schema import validate
    d = _valid(); d["D_aimd"] = "not_a_number"
    errs = validate(d)
    assert any("D_aimd" in e for e in errs)


def test_transport_float_ok():
    from hpca.core.project_schema import validate
    d = _valid(); d["D_aimd"] = 1.5e-9
    assert validate(d) == []


def test_migrate_mobile_species():
    from hpca.core.project_schema import migrate
    d = {"name": "x", "mobile_species": "Li", "category": "inorganic_sse", "T_ref": 300}
    out = migrate(d)
    assert out.get("mobile_ion") == "Li"
    assert "mobile_species" not in out


def test_migrate_project_root():
    from hpca.core.project_schema import migrate
    d = {"name": "x", "project_root": "/some/path", "category": "inorganic_sse",
         "mobile_ion": "Li", "T_ref": 300}
    out = migrate(d)
    assert out.get("root") == "/some/path"
    assert "project_root" not in out


def test_execution_lane_rejects_unsafe_override():
    from hpca.core.project_schema import validate
    d = {"name": "x", "category": "inorganic_sse", "mobile_ion": "Li", "T_ref": 300,
         "execution": {"stages": {"h01_dft": "daemon"}}}
    assert any("h01_dft" in error and "unsupported" in error for error in validate(d))


def test_all_workflow_stage_keys_are_recognized():
    from hpca.core.project_schema import validate
    d = {"name": "x", "category": "inorganic_sse", "mobile_ion": "Li", "T_ref": 300,
         "stages": {"design": True, "classical_md": False, "active_learning": True,
                    "chaai": True}}
    assert validate(d) == []


def test_invalid_autonomy_budget_is_rejected():
    from hpca.core.project_schema import validate
    d = _valid()
    d["autonomy"] = {"mode": "unattended", "max_total_submissions": 0}
    assert any("limits must be positive" in error for error in validate(d))


def test_gel_polymer_uses_structured_polymers_block():
    from hpca.core.project_schema import validate
    d = {
        "name": "gel", "category": "polymer", "system_type": "gel",
        "T_ref": 300,
        "polymers": [{"monomer": "PEO", "chain_length": 20, "n_chains": 2}],
        "simulation": {"solvents": [{"name": "DME", "ratio": 1}]},
    }
    assert validate(d) == []


def test_liquid_accepts_nested_wizard_solvents():
    from hpca.core.project_schema import validate
    d = {
        "name": "liq", "category": "liquid_electrolyte", "T_ref": 300,
        "simulation": {"solvents": [{"name": "DME", "ratio": 1}]},
    }
    assert validate(d) == []
