from hpca.registry import validate_registries


def test_canonical_registries_are_internally_consistent():
    assert validate_registries() == ()


def test_registry_modules_do_not_own_runtime_operations():
    import hpca.registry.folder as folder
    import hpca.registry.incar as incar
    import hpca.registry.stage as stage

    for module in (folder, incar, stage):
        text = open(module.__file__, encoding="utf-8").read()
        assert "subprocess." not in text
        assert "from hpca.scheduler" not in text
