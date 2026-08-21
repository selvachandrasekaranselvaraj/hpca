from pathlib import Path

from hpca.core.preoptimization import decide_preoptimization
from hpca.core.project_schema import validate


def _write_poscar(path: Path, elements: str, counts: str, coords: list[str]) -> Path:
    path.write_text(
        "test\n1.0\n10 0 0\n0 10 0\n0 0 10\n"
        f"{elements}\n{counts}\nDirect\n" + "\n".join(coords) + "\n"
    )
    return path


def _platform(**overrides) -> dict:
    values = {
        "mode": "auto", "severe_overlap_A": 0.8,
        "repair_min_distance_A": 1.0, "crystal_min_distance_A": 1.0,
        "max_runtime_s": 1800, "seconds_per_atom_step": 0.002, "steps": 1000,
    }
    values.update(overrides)
    return {"preoptimization": values}


def test_reasonable_crystal_skips_mace(tmp_path):
    poscar = _write_poscar(tmp_path / "POSCAR", "Li Si", "1 1", ["0 0 0", ".25 .25 .25"])
    decision = decide_preoptimization(poscar, "solid", {}, _platform())
    assert not decision.run_mace
    assert decision.reason == "crystal_already_physically_reasonable"


def test_generated_molecular_structure_selects_optional_mace(tmp_path):
    poscar = _write_poscar(tmp_path / "POSCAR", "C H O", "1 1 1", ["0 0 0", ".2 .2 .2", ".4 .4 .4"])
    decision = decide_preoptimization(
        poscar, "polymer", {}, _platform(), generated_structure=True,
    )
    assert decision.run_mace
    assert decision.model == "mace_off"
    assert decision.reason == "generated_molecular_structure"


def test_runtime_guard_skips_large_structure(tmp_path):
    coords = [f"{i / 100:.3f} 0 0" for i in range(50)]
    poscar = _write_poscar(tmp_path / "POSCAR", "C", "50", coords)
    decision = decide_preoptimization(
        poscar, "polymer", {}, _platform(max_runtime_s=1), generated_structure=True,
    )
    assert not decision.run_mace
    assert decision.reason == "estimated_runtime_exceeds_limit"


def test_runtime_guard_runs_before_dense_distance_analysis(tmp_path, monkeypatch):
    """Large CMD/MLMD inputs must never allocate pairwise distance matrices."""
    poscar = _write_poscar(tmp_path / "POSCAR", "C H", "30000 30000", [])

    def dense_distance_must_not_run(_path):
        raise AssertionError("dense minimum-distance analysis was reached")

    monkeypatch.setattr(
        "hpca.core.preoptimization.min_distance_poscar", dense_distance_must_not_run
    )
    decision = decide_preoptimization(
        poscar, "polymer", {}, _platform(), generated_structure=True,
    )
    assert not decision.run_mace
    assert decision.reason == "estimated_runtime_exceeds_limit"
    assert decision.atom_count == 60000
    assert decision.minimum_distance_A is None


def test_unsupported_element_skips_before_model_execution(tmp_path):
    poscar = _write_poscar(tmp_path / "POSCAR", "U", "1", ["0 0 0"])
    decision = decide_preoptimization(poscar, "solid", {"preoptimization": {"mode": "mace"}}, _platform())
    assert not decision.run_mace
    assert decision.reason == "unsupported_elements:U"


def test_severe_overlap_is_repaired_deterministically(tmp_path):
    poscar = _write_poscar(tmp_path / "POSCAR", "C H", "1 1", ["0 0 0", ".01 0 0"])
    decision = decide_preoptimization(
        poscar, "polymer", {}, _platform(), generated_structure=True,
    )
    assert decision.overlap_repaired
    assert decision.minimum_distance_A >= 1.0


def test_project_schema_rejects_invalid_preoptimization_policy():
    project = {"name": "x", "category": "solid", "T_ref": 300,
               "preoptimization": {"mode": "always", "max_runtime_s": -1}}
    errors = validate(project)
    assert "preoptimization.mode must be auto, mace, or none" in errors
    assert "preoptimization.max_runtime_s must be a positive number" in errors
