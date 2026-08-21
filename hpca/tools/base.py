"""Base classes for HPCA tools."""
from dataclasses import dataclass, field
from typing import Any
import json


@dataclass
class ToolResult:
    """Unified return type carrying output text, success flag, and optional metadata."""

    output: str
    success: bool = True
    metadata: dict = field(default_factory=dict)

    def __str__(self):
        """Return the output string representation of this result."""
        return self.output


class Tool:
    """Abstract base for all HPCA tools."""
    name: str = ""
    description: str = ""

    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool action and return a ToolResult."""
        raise NotImplementedError

    def to_schema(self) -> dict:
        """OpenAI-compatible function schema for vLLM tool_choice."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._parameters(),
            }
        }

    def _parameters(self) -> dict:
        """Return JSON schema for this tool's parameters."""
        return {"type": "object", "properties": {}, "required": []}
