"""HPCA consolidated tool layer."""
from .base import Tool, ToolResult
from .slurm import SlurmTool
from .shell import ShellTool
from .files import FilesTool
from .vasp import VASPTool
from .lammps import LAMMPSTool
from .deepmd import DeepMDTool
from .structure import StructureTool
from .diagnoser import DiagnoserTool

ALL_TOOLS = [
    SlurmTool,
    ShellTool,
    FilesTool,
    VASPTool,
    LAMMPSTool,
    DeepMDTool,
    StructureTool,
    DiagnoserTool,
]

_REGISTRY: dict[str, type[Tool]] = {
    T.name: T for T in ALL_TOOLS if hasattr(T, "name") and T.name
}


def get_tool(name: str) -> Tool:
    """Instantiate and return a tool by name."""
    cls = _REGISTRY.get(name)
    if cls is None:
        raise KeyError(
            f"Unknown tool: {name!r}. Available: {list(_REGISTRY)}"
        )
    return cls()


def list_tools() -> list[str]:
    """Return sorted list of registered tool names."""
    return sorted(_REGISTRY.keys())


def all_schemas() -> list[dict]:
    """Return OpenAI-compatible JSON schemas for all tools."""
    return [cls().to_schema() for cls in ALL_TOOLS]


__all__ = [
    "Tool", "ToolResult",
    "SlurmTool", "ShellTool", "FilesTool",
    "VASPTool", "LAMMPSTool", "DeepMDTool",
    "StructureTool", "DiagnoserTool",
    "ALL_TOOLS", "get_tool", "list_tools", "all_schemas",
]
