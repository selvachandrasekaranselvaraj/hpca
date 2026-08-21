"""Build a single-file HPCA manual without requiring MkDocs.

The output contains the complete maintained narrative manual. Generated Python API pages remain
the responsibility of MkDocs; this file includes an API module inventory and links to sources.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUTPUT = ROOT / "site" / "hpca-complete.html"

PAGES = [
    "index.md",
    "getting-started/installation.md",
    "getting-started/first-project.md",
    "user-guide/project-types.md",
    "user-guide/project-control.md",
    "user-guide/examples.md",
    "workflow/end-to-end.md",
    "workflow/boxed-flowcharts.md",
    "workflow/stages.md",
    "workflow/files.md",
    "operations/daemon.md",
    "operations/recovery.md",
    "operations/kestrel.md",
    "reference/project-yaml.md",
    "reference/cli.md",
    "reference/registries.md",
    "reference/configuration.md",
    "reference/formulas.md",
    "development/architecture.md",
    "development/architecture-implementation.md",
    "development/extending.md",
    "development/documentation.md",
    "archive/index.md",
    "changelog.md",
]


def slug(path: str) -> str:
    """Stable section anchor for a documentation path."""
    return "page-" + re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-")


def title_of(text: str, fallback: str) -> str:
    """Return the first H1 title from Markdown."""
    match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else fallback


def rewrite_doc_links(text: str, source: Path) -> str:
    """Rewrite links between manual pages to their single-file section anchors."""
    def replace(match: re.Match[str]) -> str:
        label, raw = match.group(1), match.group(2)
        target, _, fragment = raw.partition("#")
        if not target.endswith(".md"):
            return match.group(0)
        resolved = (source.parent / target).resolve()
        try:
            relative = resolved.relative_to(DOCS).as_posix()
        except ValueError:
            return match.group(0)
        anchor = slug(relative)
        return f"[{label}](#{anchor})"
    return re.sub(r"\[([^]]+)\]\(([^)]+)\)", replace, text)


def boxed_mermaid(html_text: str) -> str:
    """Keep Mermaid source visible as a boxed flow when JavaScript is unavailable."""
    return html_text.replace(
        '<pre><code class="language-mermaid">',
        '<pre class="mermaid"><code class="language-mermaid flow-source">',
    )


def build() -> Path:
    """Render the complete manual to :data:`OUTPUT`."""
    articles: list[str] = []
    toc: list[str] = []
    for relative in PAGES:
        source = DOCS / relative
        text = source.read_text(encoding="utf-8")
        title = title_of(text, relative)
        text = rewrite_doc_links(text, source)
        rendered = markdown.markdown(
            text,
            extensions=["tables", "fenced_code", "attr_list", "md_in_html", "toc"],
            output_format="html5",
        )
        rendered = boxed_mermaid(rendered)
        anchor = slug(relative)
        toc.append(f'<li><a href="#{anchor}">{html.escape(title)}</a></li>')
        articles.append(f'<article id="{anchor}" class="doc-section">{rendered}</article>')

    examples = []
    for yaml_path in sorted((DOCS / "examples").glob("*.yaml")):
        examples.append(
            f"<h3>{html.escape(yaml_path.name)}</h3>"
            f"<pre><code>{html.escape(yaml_path.read_text(encoding='utf-8'))}</code></pre>"
        )

    from gen_api_pages import discover_modules
    modules = discover_modules(ROOT)
    api_items = "".join(
        f"<li><code>{html.escape(name)}</code> — "
        f"<code>{html.escape(path.relative_to(ROOT).as_posix())}</code></li>"
        for name, path in modules
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HPCA Complete Documentation</title>
<style>
:root {{ --ink:#172033; --blue:#1769aa; --pale:#eef6fc; --line:#9fb9cc; --accent:#d96704; }}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{ margin:0; color:var(--ink); background:#f5f7fa; font:16px/1.58 Arial,Helvetica,sans-serif; }}
header {{ background:linear-gradient(120deg,#123a5a,#1769aa); color:white; padding:2.2rem max(5vw,2rem); }}
header h1 {{ margin:0 0 .4rem; font-size:2.2rem; }}
.layout {{ display:grid; grid-template-columns:minmax(240px,300px) minmax(0,1fr); gap:1.5rem; max-width:1500px; margin:auto; padding:1.5rem; }}
nav {{ position:sticky; top:1rem; align-self:start; max-height:94vh; overflow:auto; background:white; border:1px solid var(--line); border-radius:10px; padding:1rem; }}
nav ul {{ padding-left:1.2rem; }} nav a {{ color:var(--blue); text-decoration:none; }}
main {{ min-width:0; }}
.doc-section {{ background:white; border:1px solid var(--line); border-left:6px solid var(--blue); border-radius:10px; margin:0 0 1.5rem; padding:1.4rem 1.7rem; box-shadow:0 2px 8px #20304012; }}
h1,h2,h3 {{ line-height:1.25; }} h1 {{ color:#123a5a; }} h2 {{ color:var(--blue); border-bottom:2px solid #dbe8f2; padding-bottom:.25rem; }}
table {{ border-collapse:collapse; width:100%; display:block; overflow:auto; }} th,td {{ border:1px solid #bccbd6; padding:.5rem .65rem; vertical-align:top; }} th {{ background:var(--pale); }}
pre {{ overflow:auto; padding:1rem; background:#172033; color:#f3f7fa; border-radius:8px; }} code {{ font-family:Consolas,monospace; }}
pre:has(.flow-source) {{ white-space:pre-wrap; background:var(--pale); color:var(--ink); border:2px solid var(--blue); box-shadow:inset 0 0 0 4px white; }}
.math, .arithmatex, p:has(> script[type^="math/tex"]) {{ overflow:auto; border:1px solid var(--line); border-left:5px solid var(--accent); background:#fffaf4; padding:.7rem 1rem; }}
.examples,.api {{ background:white; border:1px solid var(--line); border-radius:10px; margin-bottom:1.5rem; padding:1.5rem; }}
.top {{ float:right; font-size:.85rem; }}
@media(max-width:900px) {{ .layout {{ grid-template-columns:1fr; }} nav {{ position:relative; max-height:none; }} }}
@media print {{ nav {{ display:none; }} .layout {{ display:block; max-width:none; }} .doc-section {{ break-inside:avoid; box-shadow:none; }} header {{ background:white; color:black; }} }}
</style>
<script>
window.MathJax={{tex:{{inlineMath:[['$','$']],displayMath:[['$$','$$']]}},svg:{{fontCache:'global'}}}};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js" onload="mermaid.initialize({{startOnLoad:true,securityLevel:'loose'}})"></script>
</head>
<body id="top">
<header><h1>HPCA Complete Documentation</h1><p>Users · HPC operators · developers · scientific formulas · boxed workflows</p></header>
<div class="layout">
<nav aria-label="Manual contents"><strong>Contents</strong><ul>{''.join(toc)}</ul><p><a href="#examples">Embedded YAML examples</a></p><p><a href="#api-inventory">API inventory</a></p></nav>
<main>{''.join(articles)}
<section id="examples" class="examples"><h1>Embedded project examples</h1>{''.join(examples)}</section>
<section id="api-inventory" class="api"><h1>Python API module inventory</h1><p>{len(modules)} production modules. Full symbol documentation is generated by MkDocs/mkdocstrings.</p><ul>{api_items}</ul></section>
</main></div></body></html>"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(document, encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path} ({path.stat().st_size:,} bytes)")
