# Monitoring and recovery

Start with read-only evidence:

```bash
hpca status /absolute/project/path
hpca-daemon project-status /absolute/project/path
hpca log /absolute/project/path --lines 200
squeue -u "$USER"
sacct -u "$USER" --starttime today
```

## Recovery sequence

1. Confirm the project's desired state in `.hpca/control.json`.
2. Validate and hash the canonical `project.yaml`.
3. Compare HPCA state with local PIDs, leases, `squeue`, and `sacct`.
4. Inspect the handler's required scientific outputs and completion gate.
5. Resume only after reconciliation; reset a stage only when its evidence proves retry is safe.

## Failure classes

| Symptom | Likely class | Safe response |
|---|---|---|
| Invalid project is never submitted | Schema/preflight failure | Stop, correct YAML, validate, resume |
| Job ID exists but no queue entry | Scheduler history/reconciliation | Check `sacct`; do not immediately resubmit |
| Output exists but stage remains running | Scientific gate incomplete | Inspect convergence/frame/integrity evidence |
| Retry budget exhausted | Autonomous policy terminal state | Review cause and explicitly revise policy/state |
| Daemon reaches wall-time | Expected handoff | Confirm successor lease and heartbeat |
| Project is stopped but jobs run | Stop semantics | Decide explicitly whether scheduler cancellation is intended |

Never delete durable state, fake completion sentinels, or remove scheduler evidence as a
recovery technique. Live job submission/cancellation and production inbox migration are
operational changes and require explicit authorization.
