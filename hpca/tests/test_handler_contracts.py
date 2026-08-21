from hpca.orchestrator.handlers import ALL_HANDLERS, validate_handler_contracts


def test_all_handlers_satisfy_canonical_contract():
    assert len(ALL_HANDLERS) == 15
    assert validate_handler_contracts() == ()


def test_handler_lane_comes_from_registry():
    lanes = {handler.name: handler.execution_lane for handler in ALL_HANDLERS}
    assert lanes["h00_design"] == "daemon"
    assert lanes["h01_dft"] == "slurm"
