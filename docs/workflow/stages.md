# Stage contracts

The stage registry is authoritative for names, execution lanes, category routing, and
dependencies. Handlers implement the operational adapter for each stage.

| Stage | Lane | Main input | Validated output |
|---|---|---|---|
| `h00_design` | daemon | `project.yaml`, structures/components | three designed tiers and sub-project YAML |
| `h01_dft` | SLURM | preoptimized DFT POSCAR | converged relax/static outputs |
| `h02_aimd` | SLURM | `dft/opt/CONTCAR` | NPT/NVT and dataset VASP trajectories |
| `h03_neb` | SLURM | relaxed endpoints/migration definition | converged image energies and barrier |
| `h04_mlip` | SLURM | validated AIMD frames | accepted DeepMD/MACE model and metrics |
| `h05_cmd` | SLURM | force-field LAMMPS data | equilibrated and production trajectories |
| `h05_lammps` | SLURM | accepted MLIP and MLMD cell | NPT/NVT MLMD trajectories |
| `h13_active_learning` | daemon coordinator | exploratory MLMD and model | frozen augmented dataset/model |
| `h06_analysis` | daemon | validated CMD/MLMD/AIMD trajectories | transport and structural result tables |
| `h07_electronic` | daemon coordinator | DFT electronic outputs | Bader/DOS/electronic summaries |
| `h08_echem` | daemon coordinator | optimized/static data | electrochemical summaries |
| `h09_continuum` | daemon | analysis/echem parameters | continuum-model results |
| `h10_plotting` | daemon | validated result tables | figures and plotting data |
| `h11_manuscript` | daemon | results, figures, metadata | DOCX manuscript and FAIR package |
| `h12_chaai` | daemon | redacted workflow events | CHAAI training artifacts |

“Daemon” means bounded coordination or local scientific post-processing. A daemon handler may
coordinate a SLURM subcalculation; it must not run unbounded MPI/GPU work on the daemon node.

Use the [generated stage-registry API](../reference/api/hpca/registry/stage.md) for exact
dependency and category tables.
