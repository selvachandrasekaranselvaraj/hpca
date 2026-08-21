# First autonomous project

Create a dedicated project directory and run the interactive design wizard:

```bash
mkdir -p /path/to/workspace/test/Si_test
cd /path/to/workspace/test/Si_test
hpca new
```

The wizard scans local structure files, asks only questions relevant to the selected material,
writes `project.yaml`, validates it, creates project-local control state, and optionally
registers it in HPCA's repository-local daemon inbox.

## Start and inspect

```bash
hpca start . --slurm
hpca status .
hpca log . --lines 100
hpca-daemon project-status .
```

`hpca start . --slurm` is the normal production path. Omit `--slurm` only for an explicitly
local orchestrator process; scientific compute stages still follow their registered lane.

## Stop and resume from the same directory

```bash
hpca stop .
hpca resume . --slurm
```

The desired state is stored in `.hpca/control.json`. Stopping pauses future orchestration; it
does not silently cancel already running compute jobs. On resume, HPCA reconciles process,
state, lease, and SLURM evidence before deciding whether anything must be submitted.

## Before leaving the session

Confirm the daemon or project orchestrator was submitted, not merely started as a shell
background process. Terminal history is not workflow state: `project.yaml`, `.hpca/`, logs,
and scheduler jobs persist independently of the login session.
