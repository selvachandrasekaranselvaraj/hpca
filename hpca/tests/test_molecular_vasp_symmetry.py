"""Molecular VASP cells must not be subjected to Bravais classification."""
from pathlib import Path

from hpca.registry.incar import build_incar
from hpca.orchestrator.handlers.h01_dft import DFTHandler
from hpca.orchestrator.state_tracker import ProjectState


def test_molecular_incar_disables_symmetry_completely():
    incar = build_incar("vc_relax", natoms=200,
                        project_yaml={"category": "liquid_electrolyte"})
    assert incar["ISYM"] == -1
    assert "SYMPREC" not in incar


def test_crystal_incar_preserves_registered_symmetry_policy():
    incar = build_incar("vc_relax", natoms=200,
                        project_yaml={"category": "bulk_sse"})
    assert incar.get("ISYM") != -1


def test_exhausted_molecular_sick_job_gets_one_stronger_retry(tmp_path: Path):
    (tmp_path / "project.yaml").write_text("category: liquid_electrolyte\n")
    work = tmp_path / "dft" / "vc"
    work.mkdir(parents=True)
    (work / "INCAR").write_text("ISYM = 0\nSYMPREC = 1E-4\n")
    state = ProjectState(tmp_path)
    state.set_handler("h01_dft.aimd_relax", {"stage": "COMPLETE"})
    state.set_handler("h01_dft.vc_relax", {
        "stage": "FAILED", "error": "FIX_BUDGET_EXHAUSTED",
        "fixed": "SICK_JOB_SYMPREC", "fix_count": 3,
    })
    handler = DFTHandler()
    handler._enabled_subtasks = lambda _: ["vc_relax"]

    assert handler.migrate_molecular_sick_job(
        tmp_path, state, {"category": "liquid_electrolyte"}) == ["h01_dft.vc_relax"]
    assert state.get_stage("h01_dft.vc_relax") == "PENDING"
    assert state.get_handler("h01_dft.vc_relax")["fix_count"] == 0
    text = (work / "INCAR").read_text()
    assert "ISYM" in text and "-1" in text
    assert "SYMPREC" not in text
    assert handler.migrate_molecular_sick_job(
        tmp_path, state, {"category": "liquid_electrolyte"}) == []
