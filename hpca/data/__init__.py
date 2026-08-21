"""hpca.data — Static reference data (element tables, FF params, molecular properties)."""
from __future__ import annotations
import json
from pathlib import Path

_DATA_DIR = Path(__file__).parent


def load(name: str) -> dict | list:
    """Load a JSON data file from hpca/data/ by stem name."""
    path = _DATA_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"hpca.data: '{name}.json' not found in {_DATA_DIR}")
    return json.loads(path.read_text(encoding="utf-8"))
