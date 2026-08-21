# Architecture implementation rules (1–18)

This is the enforceable architecture map. Each numbered box has one owner and one principal
rule; tests and the architecture audit turn these boundaries into release gates.

```mermaid
flowchart TB
  S01[1 Skill: concise router to maintained rules] --> S02[2 Audit: deterministic boundary baseline]
  S02 --> S03[3 NEB: science in core/neb; submission in registry]
  S03 --> S04[4 Submission registry: typed strict templates]
  S04 --> S05[5 Scheduler: submit/query/cancel adapter]
  S05 --> S06[6 Registries: declarations and lookup only]
  S06 --> S07[7 Configuration: read, migrate once, validate]
  S07 --> S08[8 Daemon: checksum request, lease, local control]
  S08 --> S09[9 Orchestrator: durable deterministic state machine]
  S09 --> S10[10 Handlers: uniform registered adapters]
  S10 --> S11[11 Domain: scientific algorithms without workflow state]
  S11 --> S12[12 Artifacts: canonical path and SHA-256 provenance]
  S12 --> S13[13 Formulas: units, domains, reference tests]
  S13 --> S14[14 Operations: health and bounded recovery]
  S14 --> S15[15 Interfaces: stable CLI and service exit contracts]
  S15 --> S16[16 Documentation: source-linked HTML and flowcharts]
  S16 --> S17[17 Cleanup: compatibility is explicit and temporary]
  S17 --> S18[18 Release: tests, audit, docs, package gates]
```

## Box contracts

| Box | Owner | Must | Must not |
|---:|---|---|---|
| 1 | `skills/hpca-production-workflow` | Route work to focused rule references | Duplicate the full manuals |
| 2 | architecture audit | Fail on new violations deterministically | Mutate source while auditing |
| 3 | `hpca.core.neb` | Generate/parse scientific NEB data | Render or submit SLURM jobs |
| 4 | `hpca.registry.submission` | Declare and validate template parameters | Silently ignore unknown settings |
| 5 | `hpca.scheduler` | Submit, query, cancel, express dependencies | Own scientific input templates |
| 6 | `hpca.registry` | Own paths, INCAR, stages, submissions | Orchestrate or inspect live jobs |
| 7 | project schema/I/O | Normalize once and report field paths | Maintain competing validators |
| 8 | `hpca.daemon` | Use immutable requests, leases, atomic transitions | Embed scientific stage behavior |
| 9 | orchestrator | Reconcile dependency/state evidence | Implement scientific equations |
| 10 | handlers | Translate stage services to workflow state | Bypass registry/scheduler boundaries |
| 11 | domain packages | Accept inputs and return scientific results | Read daemon inbox or mutate orchestration |
| 12 | artifact service | Record relative identity, hash, producer, format | Treat existence alone as scientific validity |
| 13 | `hpca.science` | State units and reject invalid domains | Hide conversions in handlers |
| 14 | monitor/recovery | Read health; retry classified transient faults | Retry permanent/configuration faults forever |
| 15 | CLI/services | Target one project and offer JSON output | Cancel unrelated user jobs |
| 16 | `docs/` | Match executable behavior and equations | Maintain hand-edited divergent HTML |
| 17 | compatibility shims | Point one-way to canonical modules with removal target | Receive new implementation logic |
| 18 | CI/release | Gate tests, strict audit, docs, package build | Publish from a dirty or failing tree |

## Runtime ownership flow

```mermaid
flowchart LR
  INPUT[Structure files and project.yaml] --> CONFIG[Schema boundary]
  CONFIG --> CONTROL[Daemon control plane]
  CONTROL --> MACHINE[Orchestrator state machine]
  MACHINE --> HANDLER[Registered handler]
  HANDLER --> DOMAIN[Scientific domain service]
  HANDLER --> TEMPLATE[Submission registry]
  TEMPLATE --> SCHED[Scheduler adapter]
  DOMAIN --> ART[Validated artifact]
  SCHED --> EXEC[VASP / LAMMPS / MLIP]
  EXEC --> ART
  ART --> PROV[Append-only provenance]
  PROV --> MACHINE
```
