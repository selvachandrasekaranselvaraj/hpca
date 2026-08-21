"""Canonical expansion policy for combinatorial production projects."""
from __future__ import annotations

from typing import Any


def production_combinations(project_yaml: dict[str, Any]) -> list[dict]:
    """Return independently simulated combinations in deterministic input order.

    Molarity-resolved ``cmd_combinations`` are production units. Projects without
    that list retain the historical ``aimd_combinations`` behavior.
    """
    cmd = [dict(item) for item in project_yaml.get("cmd_combinations", [])
           if isinstance(item, dict) and item.get("name")]
    if cmd:
        return cmd
    return [dict(item) for item in project_yaml.get("aimd_combinations", [])
            if isinstance(item, dict) and item.get("name")]


def aimd_dataset_combination(project_yaml: dict[str, Any], production: dict) -> dict:
    """Return the solvent/salt AIMD dataset descriptor for one production unit."""
    aimd = [dict(item) for item in project_yaml.get("aimd_combinations", [])
            if isinstance(item, dict) and item.get("name")]
    name = str(production.get("name", ""))
    exact = next((item for item in aimd if item["name"] == name), None)
    if exact:
        return exact
    matches = [item for item in aimd if name.startswith(f"{item['name']}_")]
    if matches:
        return max(matches, key=lambda item: len(item["name"]))
    return {"name": name, "label": production.get("label", name)}


def is_combinatorial_parent(project_yaml: dict[str, Any]) -> bool:
    return len(production_combinations(project_yaml)) > 1
