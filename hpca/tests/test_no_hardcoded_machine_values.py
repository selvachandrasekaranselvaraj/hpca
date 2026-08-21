"""Contract test: no machine-specific values hardcoded in package code.

Cluster paths, SLURM account names, and environment-module names must live
only in hpca/config/platform.yaml.  This test walks every .py file's AST and
inspects string constants in executable code (docstrings and comments are
exempt — they may cite paths as examples).

To grant a documented exception, add "relative/path.py" to _WHITELIST with a
justification comment.
"""
from __future__ import annotations

import ast
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]

# Substrings that indicate a machine-specific value
_BANNED = ("/projects/", "/kfs2/", "/nopt/", "/home/", "nmclps")

# Files exempt from the check, relative to the hpca package root
_WHITELIST = {
    "tests/test_no_hardcoded_machine_values.py",  # this file names the patterns
    "chaai/gen_training.py",  # synthetic training-conversation text, not executed code
}


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Return id()s of Constant nodes that are docstrings."""
    doc_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                doc_ids.add(id(body[0].value))
    return doc_ids


def _violations_in(path: Path) -> list[str]:
    """Return 'line: excerpt' for each banned string constant in executable code."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as exc:
        return [f"unparseable: {exc}"]
    doc_ids = _docstring_nodes(tree)
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc_ids:
            s = node.value
            if any(b in s for b in _BANNED) or (
                "module load " in s and s.split("module load ", 1)[1][:1].isalpha()
            ):
                excerpt = s.strip().replace("\n", "\\n")[:80]
                out.append(f"line {node.lineno}: {excerpt!r}")
    return out


def test_no_hardcoded_machine_values():
    """Package .py files must not embed cluster paths, accounts, or module names."""
    failures: list[str] = []
    for path in sorted(PKG_ROOT.rglob("*.py")):
        rel = path.relative_to(PKG_ROOT).as_posix()
        if "__pycache__" in rel or rel in _WHITELIST:
            continue
        for v in _violations_in(path):
            failures.append(f"{rel}: {v}")
    assert not failures, (
        "Machine-specific values found outside platform.yaml "
        f"({len(failures)}):\n  " + "\n  ".join(failures)
        + "\nMove them to hpca/config/platform.yaml (or whitelist with justification)."
    )
