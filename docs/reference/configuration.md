# Configuration reference

`hpca/config/platform.yaml` is the runtime configuration source. Its major sections are:

| Section | Content |
|---|---|
| `category_defaults` | Physics/model defaults by category |
| `mlip_registry` | Supported model backends, categories, environments and GPU policy |
| `hpc` | Site executables, environments, accounts, modules, pseudopotentials and libraries |
| `limits.slurm` | Tier sizes, VASP/AIMD/MD steps and production durations |
| `slurm_time`, `vasp_nodes` | Wall-times and resource presets |
| `aimd_dataset` | Deformation/random scales, temperatures and frame selection |
| `lammps_md`, `mlip_defaults` | MD and training defaults/acceptance thresholds |
| `analysis_defaults` | Units, cutoffs, sampling and fitting controls |
| `orchestrator`, `handler_timeouts` | Polling, handoff, inbox and bounded operation timeouts |
| `project_schema` | Required project fields and category-specific constraints |

Precedence for supported simulation settings is project override, then platform default, then
the documented handler fallback. A project override does not bypass schema, lane, autonomy,
or scientific validation constraints.

Changes to the stage DAG, filesystem layout, VASP templates, or SLURM templates belong in
their canonical registries, not in `platform.yaml`.
