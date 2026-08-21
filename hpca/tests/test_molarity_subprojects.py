from pathlib import Path

import yaml

from hpca.core.combinations import (
    aimd_dataset_combination, is_combinatorial_parent, production_combinations,
)
from hpca.orchestrator.handlers.h00_design import MaterialsDesignHandler
from hpca.orchestrator.handlers.h11_manuscript import ManuscriptHandler
from hpca.orchestrator.hpca_orchestrator import _discover_for_root


def _combo(name: str, solvent: str, molarity: float) -> dict:
    return {
        "name": name,
        "label": f"{solvent} + LiFSI {molarity} M",
        "salt_molarity": molarity,
        "components": {
            "solvent": {"components": [{"name": solvent, "ratio": 1}]},
            "salt": {"components": [{"name": "LiFSI", "ratio": 1}]},
        },
    }


def _parent_yaml() -> dict:
    cmd = [
        _combo(f"{solvent}_LiFSI_{tag}M", solvent, molarity)
        for solvent in ("DMB", "DME")
        for tag, molarity in (("1p0", 1.0), ("1p5", 1.5), ("2p0", 2.0),
                              ("2p5", 2.5), ("3p0", 3.0))
    ]
    return {
        "name": "Fluorine_free_solvent",
        "category": "liquid_electrolyte",
        "system_type": "liquid_electrolyte",
        "mobile_ion": "Li",
        "T_ref": 300,
        "aimd_combinations": [
            {"name": "DMB_LiFSI", "label": "DMB + LiFSI"},
            {"name": "DME_LiFSI", "label": "DME + LiFSI"},
        ],
        "cmd_combinations": cmd,
        "simulation": {
            "salt_molarity": 1.0,
            "aimd_temps": [300, 400],
            "mlmd_temps": [300, 400],
            "target_density_gcm3": 1.0,
            "tier_cmd": {"density_gcm3": 1.0},
            "comp_spec": {
                "solvents": [{"name": "DMB", "ratio": 1}, {"name": "DME", "ratio": 1}],
                "salts": [{"name": "LiFSI", "ratio": 1}],
            },
        },
        "stages": {"design": True, "cmd": True},
    }


def test_production_policy_resolves_each_molarity_and_shared_aimd_key():
    parent = _parent_yaml()
    production = production_combinations(parent)
    assert len(production) == 10
    assert is_combinatorial_parent(parent)
    assert aimd_dataset_combination(parent, production[0])["name"] == "DMB_LiFSI"
    assert aimd_dataset_combination(parent, production[-1])["name"] == "DME_LiFSI"


def test_design_creates_and_discovers_ten_independent_children(tmp_path: Path):
    parent = _parent_yaml()
    parent["root"] = str(tmp_path)
    (tmp_path / "project.yaml").write_text(yaml.safe_dump(parent, sort_keys=False))

    MaterialsDesignHandler()._build_combinatorial_subprojects(tmp_path, parent)
    expected = [item["name"] for item in parent["cmd_combinations"]]
    assert sorted(path.name for path in tmp_path.iterdir() if (path / "project.yaml").exists()) == sorted(expected)

    for item in parent["cmd_combinations"]:
        child = yaml.safe_load((tmp_path / item["name"] / "project.yaml").read_text())
        assert child["name"] == item["name"]
        assert child["simulation"]["salt_molarity"] == item["salt_molarity"]
        assert child["simulation"]["comp_spec"]["salt_molarity"] == item["salt_molarity"]
        assert [entry["name"] for entry in child["cmd_combinations"]] == [item["name"]]
        assert "composition_variants" not in child
        assert child["aimd_dataset_key"] in {"DMB_LiFSI", "DME_LiFSI"}

    discovered = _discover_for_root(tmp_path)
    assert discovered[0] == tmp_path
    assert {path.name for path in discovered[1:]} == set(expected)

    # Obsolete pair-level directories must not enter consolidated reporting.
    for legacy in ("DMB_LiFSI", "DME_LiFSI"):
        path = tmp_path / legacy
        path.mkdir()
        (path / "project.yaml").write_text("name: legacy\n")
    selected = ManuscriptHandler()._find_sub_project_dirs(tmp_path, parent)
    assert {path.name for path in selected} == set(expected)
