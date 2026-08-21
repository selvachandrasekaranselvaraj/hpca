# `project.yaml` reference

`project.yaml` is the canonical scientific request. It is validated by the wizard, daemon,
orchestrator, and status tooling. Older field names are migrated to workflow schema version 2.

## Required fields

| Field | Meaning |
|---|---|
| `name` | Stable project identifier |
| `category` | Registered material category |
| `mobile_ion` | Transport ion/species |
| `T_ref` | Reference temperature; currently fixed to 300 K by the wizard |
| Category-specific fields | For example `polymer_type`, `solvents`, or a structure source |

## Main sections

| Section | Purpose |
|---|---|
| `simulation` | AIMD/NVT temperatures, step counts, composition and cell choices |
| `stages` | Enables scientific capabilities by stable stage aliases |
| `execution` | Default lane and supported per-stage lane overrides |
| `autonomy` | Unattended mode, approvals, allowlists, and bounded attempt/submission policy |
| `doping` / combinations | Declarative sub-project expansion |
| manuscript metadata | Title/authors and report inputs |

Allowed execution lanes are `auto`, `daemon`, and `slurm`. A stage override cannot contradict
the lane registered for that handler. Temperature arrays must be non-empty positive numeric
lists when present. MLIP backends accepted by the current schema are `deepmd`, `mace`, `both`,
and `uma`.

Use [validated examples](../user-guide/examples.md) rather than copying fields from archived
plans. For exact current validation and migration rules, see the
[`project_schema` API](api/hpca/core/project_schema.md).
