# HPCA

HPCA is an autonomous, restartable control plane for computational-materials workflows on HPC systems. It turns a validated project specification into design, electronic-structure, molecular-dynamics, MLIP, analysis, visualization, and reporting tasks while reducing routine human intervention.

> **Research software:** Validate inputs, scheduler settings, physical assumptions, and generated results before relying on them.

## Highlights

- Durable stop, resume, recovery, lease, and event state.
- SLURM-aware orchestration for VASP, LAMMPS, DeepMD, MACE, and analysis.
- Explicit stage contracts and artifact provenance.
- Extensible scheduler, tool, registry, and workflow boundaries.

## Installation

```bash
git clone https://github.com/YOUR-ACCOUNT/hpca.git
cd hpca
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional integrations: `python -m pip install -e '.[dft,monitoring]'`.

## Quick start

```bash
mkdir my-hpca-project && cd my-hpca-project
hpca new
hpca start . --slurm
hpca status .
```

Before submission, configure scheduler accounts, partitions, executables, pseudopotential roots, and resource limits for your cluster. Never commit credentials, proprietary pseudopotentials, or calculation data.

```mermaid
flowchart LR
 A[project.yaml] --> B[Validate] --> C[Reconcile state] --> D[Submit ready work] --> E[Validate evidence] --> C
```

## Documentation

Start with [installation](docs/getting-started/installation.md), [workflow stages](docs/workflow/stages.md), [configuration](docs/reference/configuration.md), [recovery](docs/operations/recovery.md), and [architecture](docs/development/architecture.md). Preview with `mkdocs serve`.

## Development

```bash
python -m pip install -e '.[dev,docs]'
python -m pytest
python -m compileall -q hpca
mkdocs build --strict
```

HPCA is pre-1.0 research software. Use [CITATION.cff](CITATION.cff) when citing it. No open-source license has yet been selected; see [LICENSE](LICENSE).

