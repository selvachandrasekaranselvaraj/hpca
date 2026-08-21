# Registries

A registry is an authoritative catalog of stable definitions or lookup rules. It does not
submit jobs, mutate workflow state, or execute a scientific stage.

| Registry | Owns | Does not own |
|---|---|---|
| `folder` | Canonical directory and artifact paths | File creation or stage completion |
| `incar` | Named VASP parameter templates and overrides | Running VASP |
| `poscar` | Stage-specific structure-source resolution | Structure optimization |
| `submission` | SLURM script templates and resource selection | `sbatch` lifecycle state |
| `stage` | Stage identity, lane, ordering, category routing and dependencies | Stage execution |

## Component boundary

- **Stage:** declarative work definition: name, dependencies, lane, inputs, outputs, policy.
- **Handler:** operational adapter for one stage: prepare, submit/execute, check, validate, and
  bounded recovery.
- **Orchestrator:** global dependency evaluation, reconciliation, idempotent dispatch, and
  transition recording.
- **Daemon:** persistent request, lease, lifecycle, and project-orchestrator supervision.

Adding a second path table, INCAR template collection, stage graph, or submission-template
switch inside a handler creates drift and is prohibited. Extend the canonical registry first.
