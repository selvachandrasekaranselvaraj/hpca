# HPCA Package Architecture

The canonical maintained architecture guide is
[`docs/development/architecture.md`](../docs/development/architecture.md). This compatibility
page remains for existing links.

HPCA separates declarative scientific workflow from execution and supervision.

| Component | Responsibility | Must not do |
|---|---|---|
| `hpca.registry` | Authoritative paths, templates, submission definitions, stage metadata | Execute stages or mutate workflow state |
| Stage definition | Declare identity, dependencies, lane, inputs, outputs, validation policy | Submit jobs or inspect processes |
| Handler | Prepare, execute/submit, monitor and validate one stage | Choose global order or duplicate registries |
| `hpca.orchestrator` | Evaluate the DAG, reconcile state, dispatch handlers idempotently | Reimplement scientific calculations |
| `hpca.daemon` | Register projects, hold leases, supervise orchestrators and handoff | Implement handlers or scientific stages |

Unattended execution is governed by `hpca.core.autonomy.AutonomyPolicy`. It may advance
only allowed stages with validated design evidence and durable per-stage/project budgets.
Budget exhaustion and missing evidence fail closed; unattended does not mean unrestricted.

The historical `hpca.stages` and `hpca.sim` modules remain scientific APIs until their
callers can be migrated deliberately. New workflow execution belongs only in handlers;
new declarative definitions belong only in the stage registry. Compatibility imports in
`hpca.core.*_registry` are scheduled for removal in HPCA 2.0 and contain no duplicate logic.
