"""
Stage 07 — Manuscript generation.
Aggregates all pipeline outputs and calls ManuscriptGenerator.
"""
from __future__ import annotations
import json
from pathlib import Path


def run(project, output_base: Path, title: str = None,
        authors: list[str] = None, **kwargs) -> dict:
    """
    Stage 07 entry point. Scans output_base for results from all prior stages,
    loads benchmark CSVs, and generates a full .docx manuscript.
    """
    out_dir = Path(output_base) / "manuscripts" / project.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect results from prior stages
    pipeline_results = _collect_results(project, output_base)

    # Build manuscript
    try:
        from hpca.manuscript.generator import generate_manuscript
        doc_path = generate_manuscript(
            project=project,
            results=pipeline_results,
            output_dir=out_dir,
            title=title or f"{project.full_name} — Computational Study",
            authors=authors or ["Dr. Selva Chandrasekaran Selvaraj"],
        )
        return {"status": "complete", "manuscript": str(doc_path)}
    except ImportError as e:
        return {"status": "error", "error": str(e)}


def _collect_results(project, output_base: Path) -> dict:
    """Scan output directories for analysis results, figures, benchmarks."""
    import pandas as pd

    base = Path(output_base)
    results = {"project": project.name, "params": {}, "analysis": {},
                "continuum": {}, "benchmark": None, "figures": {}}

    # Analysis outputs (MSD, RDF, etc.)
    analysis_dir = base / "analysis" / project.name
    if analysis_dir.exists():
        for csv_f in analysis_dir.rglob("*.csv"):
            key = csv_f.stem
            try:
                results["analysis"][key] = pd.read_csv(csv_f).to_dict()
            except Exception:
                pass
        for html_f in analysis_dir.rglob("*.html"):
            results["figures"][html_f.stem] = html_f

    # Continuum outputs
    cont_dir = base / "continuum" / project.name
    if cont_dir.exists():
        for csv_f in cont_dir.rglob("*.csv"):
            key = csv_f.stem
            try:
                results["continuum"][key] = pd.read_csv(csv_f).to_dict()
            except Exception:
                pass

    # Benchmark master CSV (look in project results/data/ first, then output_base)
    proj_dir = Path(project.root)
    bm_csv = proj_dir / "results" / "data" / f"{project.name}_benchmark_all.csv"
    if not bm_csv.exists():
        bm_csv = base / "analysis" / project.name / f"{project.name}_benchmark_all.csv"
    if bm_csv.exists():
        try:
            results["benchmark"] = pd.read_csv(bm_csv)
        except Exception:
            pass

    # PNG figures for embedding
    for png_f in list(analysis_dir.rglob("*.png")) + list(cont_dir.rglob("*.png")) if (analysis_dir.exists() or cont_dir.exists()) else []:
        results["figures"][png_f.stem] = png_f

    return results
