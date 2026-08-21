from pathlib import Path


def test_handlers_use_package_qualified_autofix_imports():
    handlers = Path(__file__).parents[1] / "orchestrator" / "handlers"
    for relative in ("h01_dft.py", "h04_mlip.py"):
        text = (handlers / relative).read_text()
        assert "from auto_fix import" not in text
        assert "from hpca.orchestrator.auto_fix import" in text


def test_qualified_autofix_module_is_importable():
    from hpca.orchestrator import auto_fix
    assert callable(auto_fix.detect_vasp_error)
    assert callable(auto_fix.detect_deepmd_error)
