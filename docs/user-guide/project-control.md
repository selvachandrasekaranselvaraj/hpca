# Project control

Project control is deliberately local to the project directory.

| Intent | Command | Durable effect |
|---|---|---|
| Start or resume | `hpca start . --slurm` | Desired state becomes running; orchestrator is submitted |
| Inspect stages | `hpca status .` | Read-only stage and scheduler summary |
| Inspect desired state | `hpca-daemon project-status .` | Reads `.hpca/control.json` |
| Follow logs | `hpca log . -f` | Read-only log stream |
| Stop dispatch | `hpca stop .` | Desired state becomes stopped; future dispatch pauses |
| Resume dispatch | `hpca resume . --slurm` | Reconciliation precedes new dispatch |

The daemon inbox is an implementation detail under the HPCA repository. Users control a
project through its directory and do not edit inbox request files.

## Safe operating rules

- Do not delete state or output files to force a retry. Use the supported reset/recovery path.
- Do not assume a missing local process means a SLURM job is absent.
- Do not modify `project.yaml` while work is being dispatched; stop, edit, validate, then resume.
- A stopped project may still have running scientific jobs. Cancellation is a separate,
  explicit scheduler operation.
