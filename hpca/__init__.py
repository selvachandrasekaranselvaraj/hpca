"""
HPCA Pipeline — Comprehensive computational materials platform.

Entry points:
    CLI:    python /path/to/workspace/hpca/pipeline.py --help
    Python: from hpca.core.project import ProjectRegistry

Modules:
    core        — ProjectRegistry, MaterialProject dataclass
    stages      — s01_dft … s07_manuscript
    analysis    — trajectory, msd, rdf, sei, phase, electronic
    continuum   — ion transport, interface, mechanical, electrochemical
    viz         — Plotly theme, transport, SEI, comparison dashboards
    manuscript  — .docx generator with TOC and auto-numbered figures
    chaai       — Chaai training JSONL generation
"""
try:
    from importlib.metadata import version
    __version__ = version("hpca")
except ImportError:  # pragma: no cover - Python 3.9+ always provides importlib.metadata
    __version__ = "1.1.0"
except Exception:  # editable source tree without installed metadata
    __version__ = "1.1.0"
