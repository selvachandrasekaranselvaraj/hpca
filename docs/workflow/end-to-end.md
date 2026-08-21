# End-to-end workflow

```mermaid
flowchart TD
    A[Project directory] --> B[hpca new / project.yaml]
    B --> C[Schema, structure, policy validation]
    C --> D[Repository-local daemon inbox]
    D --> E[Lease and reconciliation]
    E --> H00[h00 design and sub-project expansion]
    H00 --> P[Material-agnostic preoptimization]
    P --> DFT[h01 DFT]
    H00 --> CMD[h05 CMD for molecular systems]
    DFT --> AIMD[h02 AIMD reference dataset]
    DFT --> NEB[h03 NEB where applicable]
    DFT --> EL[h07 electronic / h08 electrochemistry]
    AIMD --> MLIP[h04 MLIP training and validation]
    MLIP --> MLMD[h05 MLMD]
    MLMD --> AL[h13 active learning and model freeze]
    CMD --> ANA[h06 analysis]
    AL --> ANA
    ANA --> CONT[h09 continuum]
    ANA --> PLOT[h10 plotting]
    PLOT --> MAN[h11 manuscript and archive]
```

## Control flow

```mermaid
stateDiagram-v2
    [*] --> Incoming
    Incoming --> Active: validate + acquire project lease
    Active --> Stopped: desired_state=STOPPED
    Stopped --> Active: desired_state=RUNNING
    Active --> Active: reconcile then dispatch one eligible action
    Active --> Archived: all enabled stages validated
    Active --> Failed: terminal policy or unrecoverable validation failure
```

Every poll reconciles durable state with the local process table and SLURM before dispatch.
Submission attempts are consumed before execution, so a daemon crash cannot bypass autonomy
limits. Completion requires stage-specific output validation, not merely a zero exit status.

## DFT and AIMD ordering

```mermaid
flowchart LR
    DS[designed_structures/poscar_dft.vasp] --> PRE[dft/preopt]
    PRE --> Q{Doped solid?}
    Q -->|yes| AR[dft/aimd_relax]
    Q -->|no| VC[dft/vc]
    AR --> VC
    VC -->|ISIF=3| OPT[dft/opt]
    OPT -->|ISIF=2| DATA[aimd/dataset]
```

Preoptimization is material-type agnostic and never resides below `aimd/`. Molecular AIMD
generates diverse MLIP reference configurations independent of the full production-molarity
sweep. The reference NPT temperature is fixed at 300 K; production temperatures remain
configurable.
