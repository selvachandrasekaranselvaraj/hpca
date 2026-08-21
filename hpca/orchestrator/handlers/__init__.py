"""HPCA simulation-type handlers."""
from .base import SimulationHandler
from .h00_design import MaterialsDesignHandler
from .h01_dft import DFTHandler
from .h02_aimd import AIMDHandler
from .h03_neb import NEBHandler
from .h04_mlip import MLIPHandler
from .h05_cmd import ClassicalMDHandler
from .h05_lammps import LAMMPSHandler
from .h06_analysis import AnalysisHandler
from .h07_electronic import ElectronicHandler
from .h08_echem import EchemHandler
from .h09_continuum import ContinuumHandler
from .h10_plotting import PlottingHandler
from .h11_manuscript import ManuscriptHandler
from .h12_chaai import CHAAIHandler
from .h13_active_learning import ActiveLearningHandler

ALL_HANDLERS: list[SimulationHandler] = [
    MaterialsDesignHandler(),
    DFTHandler(),
    AIMDHandler(),
    NEBHandler(),
    MLIPHandler(),
    ActiveLearningHandler(),
    ClassicalMDHandler(),
    LAMMPSHandler(),
    AnalysisHandler(),
    ElectronicHandler(),
    EchemHandler(),
    ContinuumHandler(),
    PlottingHandler(),
    ManuscriptHandler(),
    CHAAIHandler(),
]


def validate_handler_contracts(handlers=ALL_HANDLERS) -> tuple[str, ...]:
    """Return structural handler-contract errors without running handlers."""
    errors: list[str] = []
    names = [handler.name for handler in handlers]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        errors.append(f"duplicate handler names: {duplicates}")
    for handler in handlers:
        if not handler.name:
            errors.append(f"{type(handler).__name__} has no name")
            continue
        try:
            definition = handler.stage_definition
        except KeyError:
            errors.append(f"handler {handler.name!r} is not in the stage registry")
            continue
        if definition.handler != handler.name:
            errors.append(f"handler {handler.name!r} registry target is {definition.handler!r}")
        expected_daemon = definition.lane.value == "daemon"
        if bool(handler.is_daemon) != expected_daemon:
            errors.append(f"handler {handler.name!r} lane disagrees with is_daemon")
        for method in ("can_run", "submit", "is_complete"):
            if not callable(getattr(handler, method, None)):
                errors.append(f"handler {handler.name!r} has no callable {method}")
    return tuple(errors)

__all__ = [
    "SimulationHandler",
    "MaterialsDesignHandler",
    "DFTHandler",
    "AIMDHandler",
    "NEBHandler",
    "MLIPHandler",
    "ActiveLearningHandler",
    "ClassicalMDHandler",
    "LAMMPSHandler",
    "AnalysisHandler",
    "ElectronicHandler",
    "EchemHandler",
    "ContinuumHandler",
    "PlottingHandler",
    "ManuscriptHandler",
    "CHAAIHandler",
    "ALL_HANDLERS",
    "validate_handler_contracts",
]
