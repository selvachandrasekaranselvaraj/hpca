from pathlib import Path

from hpca.core.neb import apply_selective_dynamics, find_migrating_atom, make_neb_images


def test_neb_domain_package_exposes_primary_and_fallback_algorithms():
    assert callable(make_neb_images)
    assert callable(apply_selective_dynamics)
    assert callable(find_migrating_atom)


def test_neb_domain_algorithms_do_not_depend_on_orchestrator_or_scheduler():
    source = (Path(__file__).parents[1] / "core" / "neb" / "linear.py").read_text()
    assert "hpca.orchestrator" not in source
    assert "hpca.scheduler" not in source
    assert "subprocess" not in source
