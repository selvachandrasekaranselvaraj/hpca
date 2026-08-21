# Installation

## Requirements

- Python 3.9 or newer.
- A filesystem visible to the daemon and SLURM compute nodes.
- SLURM commands (`sbatch`, `squeue`, `sacct`, `scancel`) for queued execution.
- Scientific executables required by selected stages: VASP, LAMMPS, PACKMOL, DeepMD/MACE,
  and Bader analysis as applicable.
- Valid licenses, modules, pseudopotentials, and scheduler accounts supplied by the site.

Install the package for development:

```bash
cd /path/to/workspace/hpca
python -m pip install -e .
hpca --help
hpca-daemon --help
```

Optional feature groups are `dft`, `manuscript`, `viz`, `monitoring`, `dev`, and `docs`.

```bash
python -m pip install -e '.[dft,manuscript,viz,monitoring,dev,docs]'
```

## Documentation build

```bash
mkdocs build --strict
mkdocs serve --dev-addr 127.0.0.1:8000
```

The documentation dependencies are build-time only. The static site contains no remote
runtime dependency and can be opened from `site/index.html` or served by any local web server.

## Site configuration

Review `hpca/config/platform.yaml` before production use. Paths, accounts, modules, binaries,
POTCAR locations, limits, and stage defaults in the bundled file describe the Kestrel profile;
they are not portable assumptions.
