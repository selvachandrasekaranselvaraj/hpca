from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_proven_dead_submission_modules_are_removed():
    assert not (ROOT / "hpca" / "core" / "slurm_factory.py").exists()
    assert not (ROOT / "hpca" / "core" / "aimd_job.py").exists()


def test_compatibility_shims_are_import_only():
    shims = [
        ROOT / "hpca" / "core" / "folder_registry.py",
        ROOT / "hpca" / "core" / "incar_registry.py",
        ROOT / "hpca" / "core" / "poscar_registry.py",
        ROOT / "hpca" / "core" / "submission_registry.py",
        ROOT / "hpca" / "orchestrator" / "dependency_graph.py",
        ROOT / "hpca" / "core" / "neb_tools.py",
    ]
    for path in shims:
        text = path.read_text()
        assert "subprocess" not in text
        assert "def " not in text
