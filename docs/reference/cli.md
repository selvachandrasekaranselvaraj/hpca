# Command reference

## `hpca`

| Command | Purpose |
|---|---|
| `new [DIR]` / `init [DIR]` | Interactive project design and YAML generation |
| `start [DIR] [--slurm]` | Start or resume autonomous orchestration |
| `resume [DIR] [--slurm]` | Alias of start for an existing project |
| `stop [DIR]` | Stop future orchestration for the project |
| `status [DIR] [--all]` | Project/stage/scheduler status |
| `validate [DIR] [--json]` | Validate `project.yaml`; invalid input exits 2 |
| `health [DIR] [--json]` | Read-only control/state health snapshot |
| `log [DIR] [-n N] [-f]` | Tail orchestrator logs |
| `run --project NAME --stages ...` | Execute selected legacy pipeline stages |
| `analyze`, `plot`, `continuum`, `manuscript` | Run bounded post-processing capabilities |
| `benchmark`, `train-chaai` | MLIP benchmark and CHAAI training controls |

Always use `hpca COMMAND --help` for exact arguments from the installed version.

## `hpca-daemon`

`init`, `link`, `project-start`, `project-stop`, `project-update`, `project-status`, `run`, `status`, and
`migrate-legacy` manage the persistent inbox/control plane. `migrate-legacy` is a read-only
preview unless `--apply` is supplied.

After changing a registered `project.yaml`, stop the project, wait until its inbox request is
`paused`, and run `hpca-daemon project-update PROJECT/project.yaml`. HPCA validates the new file,
archives the superseded request, and enqueues a new checksum-bound request. Updates are rejected
while the project is running or active.

## Other installed commands

- `hpca-orch`: low-level status, single-poll advance, stage reset, start/stop, logs, and CHAAI.
- `hpca-status`: simulation status scanner.
- `hpca-neb`: NEB preparation/analysis command.
- `hpca-monitor`: monitoring dashboard.

Low-level commands are intended for operators and developers. Prefer `hpca` and
`hpca-daemon` for normal project lifecycle control.

`hpca stop DIR` changes only `DIR/.hpca/control.json`. The daemon reconciles that request and
terminates only the matching project orchestrator; the command does not cancel jobs by a shared
SLURM name.
