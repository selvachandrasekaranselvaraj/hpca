# Daemon and SLURM

The HPCA daemon is the persistent control plane above per-project orchestrators. Its default
inbox is resolved from the installed repository (`daemon_inbox/` beside the `hpca` package),
not from a separate `/path/to/workspace/daemon_inbox` deployment.

## Initialize and run

```bash
hpca-daemon init --account your_account
sbatch daemon_inbox/hpca-daemon.sbatch
hpca-daemon status
```

For a bounded foreground validation, use:

```bash
hpca-daemon run --once
```

Register and control a project explicitly:

```bash
hpca-daemon project-start /path/to/workspace/test/Si_test/project.yaml
hpca-daemon project-status /path/to/workspace/test/Si_test
hpca-daemon project-stop /path/to/workspace/test/Si_test
```

Requests progress through `incoming`, `active`, and terminal `archived` or `failed` areas.
Immutable request metadata includes the canonical YAML path and content hash. Allowed roots
prevent a request from redirecting the daemon to an arbitrary filesystem location.

## Execution lanes

- Daemon: validation, dependency evaluation, reconciliation, script generation, bounded
  analysis, plotting, and reporting.
- SLURM: VASP, AIMD, NEB, MLIP training, CMD/MLMD production, MPI/GPU work, and expensive
  scientific subprocesses.

The stage registry declares the normal lane. Per-project execution settings may select `auto`
or the registered lane; they cannot move a SLURM-only stage onto the daemon node.

## Ten-day handoff

The daemon requests 240 hours and submits its successor after 220 elapsed hours, leaving a
20-hour overlap. The old process remains leader until lock/lease handoff. The successor must
acquire the singleton lease and reconcile active projects and SLURM jobs before dispatching.
Running scientific jobs survive daemon replacement.

Do not operate two independent inbox daemons against the same state without the repository's
lease mechanism.
