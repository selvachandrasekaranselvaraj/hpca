"""Generate one mkdocstrings page per production HPCA module and complete site nav."""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def discover_modules(root: Path = REPO_ROOT) -> list[tuple[str, Path]]:
    """Return ``(module_name, source_path)`` for every non-test HPCA module."""
    modules: list[tuple[str, Path]] = []
    for source in sorted((root / "hpca").rglob("*.py")):
        relative = source.relative_to(root)
        if "tests" in relative.parts or "__pycache__" in relative.parts:
            continue
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        if not parts:
            continue
        modules.append((".".join(parts), source))
    return modules


def api_doc_path(module: str, source: Path) -> Path:
    """Return generated Markdown path for one module."""
    parts = module.split(".")
    if source.name == "__init__.py":
        return Path("reference/api", *parts, "index.md")
    return Path("reference/api", *parts[:-1], f"{parts[-1]}.md")


def main() -> None:
    """Write generated reference pages and the complete literate navigation file."""
    import mkdocs_gen_files

    manual_nav = """* [Home](index.md)
* Getting started
    * [Installation](getting-started/installation.md)
    * [First autonomous project](getting-started/first-project.md)
* User guide
    * [Project types](user-guide/project-types.md)
    * [Project control](user-guide/project-control.md)
    * [Examples](user-guide/examples.md)
* Scientific workflow
    * [End-to-end workflow](workflow/end-to-end.md)
    * [Boxed flowcharts](workflow/boxed-flowcharts.md)
    * [Stage contracts](workflow/stages.md)
    * [Files and directories](workflow/files.md)
* Operations
    * [Daemon and SLURM](operations/daemon.md)
    * [Monitoring and recovery](operations/recovery.md)
    * [Kestrel profile](operations/kestrel.md)
* Reference
    * [project.yaml](reference/project-yaml.md)
    * [Commands](reference/cli.md)
    * [Registries](reference/registries.md)
    * [Configuration](reference/configuration.md)
    * [Mathematical formulas](reference/formulas.md)
* Development
    * [Architecture](development/architecture.md)
    * [Extending HPCA](development/extending.md)
    * [Documentation](development/documentation.md)
* Python API
"""
    nav_lines = [manual_nav]
    for module, source in discover_modules():
        doc_path = api_doc_path(module, source)
        with mkdocs_gen_files.open(doc_path, "w") as page:
            page.write(f"# `{module}`\n\n::: {module}\n")
        mkdocs_gen_files.set_edit_path(doc_path, source.relative_to(REPO_ROOT))
        nav_lines.append(f"    * [`{module}`]({doc_path.as_posix()})\n")

    nav_lines.extend([
        "* [Historical records](archive/index.md)\n",
        "* [Changelog](changelog.md)\n",
    ])
    with mkdocs_gen_files.open("SUMMARY.md", "w") as nav_file:
        nav_file.writelines(nav_lines)


if __name__ == "__main__":
    main()
