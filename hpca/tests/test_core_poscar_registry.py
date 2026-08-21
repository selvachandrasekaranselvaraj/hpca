"""Tests for hpca.core.poscar_registry."""
import pytest
import tempfile
from pathlib import Path


def _make_poscar(path: Path):
    path.write_text(
        "Li6PS5Cl\n1.0\n9.86 0 0\n0 9.86 0\n0 0 9.86\nLi P S Cl\n4 1 5 1\nDirect\n"
        + "0.244 0.0 0.244\n0.756 0.0 0.756\n0.0 0.244 0.244\n0.0 0.756 0.756\n"
        + "0.0 0.0 0.0\n0.119 0.119 0.119\n0.881 0.881 0.119\n0.881 0.119 0.881\n"
        + "0.119 0.881 0.881\n0.381 0.381 0.381\n0.0 0.5 0.5\n"
    )


def test_get_poscar_source_opt():
    from hpca.registry.poscar import get_poscar_source
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        opt_dir = project_dir / "dft" / "opt"
        opt_dir.mkdir(parents=True)
        contcar = opt_dir / "CONTCAR"
        _make_poscar(contcar)
        result = get_poscar_source("dft_followup", project_dir)
        assert result is not None


def test_get_poscar_source_missing():
    from hpca.registry.poscar import get_poscar_source
    with tempfile.TemporaryDirectory() as tmp:
        result = get_poscar_source("dft_followup", Path(tmp))
        assert result is None


def test_dft_preopt_is_inside_dft_tree():
    from hpca.core.paths import contcar_preopt
    from hpca.registry.folder import contcar_dft_preopt
    root = Path("/project")
    expected = root / "dft" / "preopt" / "CONTCAR"
    assert contcar_preopt(root, "dft") == expected
    assert contcar_dft_preopt(root) == expected
    assert contcar_preopt(root, "mlmd") == root / "preopt" / "contcar_mlmd_preopt.vasp"


def test_legacy_dft_preopt_is_migrated(tmp_path):
    from hpca.orchestrator.handlers.h00_design import MaterialsDesignHandler
    legacy = tmp_path / "preopt" / "contcar_dft_preopt.vasp"
    legacy.parent.mkdir()
    legacy.write_text("legacy structure")

    MaterialsDesignHandler._migrate_legacy_dft_preopt(tmp_path)

    target = tmp_path / "dft" / "preopt" / "CONTCAR"
    assert target.read_text() == "legacy structure"
    assert not legacy.exists()
