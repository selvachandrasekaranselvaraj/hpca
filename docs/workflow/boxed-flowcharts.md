# Boxed workflow flowcharts

Each box below is an independently observable workflow section. Arrows represent required
data or control dependencies; parallel branches do not imply a shared execution lane.

## 1. Project and control plane

```mermaid
flowchart LR
    A[Input structures and scientific choices] --> B[hpca new]
    B --> C[project.yaml schema validation]
    C --> D[Project-local .hpca/control.json]
    C --> E[Repository-local daemon inbox request]
    E --> F[Request validation and content hash]
    F --> G[Project lease]
    G --> H[Reconcile process, state, and SLURM]
    H --> I[Dispatch one eligible action]
```

## 2. Design and preoptimization

```mermaid
flowchart LR
    A[Validated project definition] --> B[h00 sub-project expansion]
    B --> C[DFT cell about 200 atoms]
    B --> D[MLMD cell about 6000 atoms]
    B --> E[CMD cell about 60000 atoms]
    C --> F[dft/preopt]
    D --> G[preopt/contcar_mlmd_preopt.vasp]
    E --> H[preopt/contcar_cmd_preopt.vasp]
    E --> I[preopt/system_cmd.data]
```

## 3. DFT relaxation and characterization

```mermaid
flowchart LR
    A[dft/preopt/CONTCAR] --> B{Doped solid?}
    B -->|Yes| C[dft/aimd_relax]
    B -->|No| D[dft/vc ISIF=3]
    C --> D
    D --> E[dft/opt ISIF=2]
    E --> F[dft/bader]
    E --> G[dft/dos/scf]
    G --> H[dft/dos/nonscf]
    E --> I[dft/static and echem_static]
    E --> J[h02 AIMD]
    E --> K[h03 NEB]
```

## 4. AIMD reference dataset

```mermaid
flowchart LR
    A[dft/opt/CONTCAR] --> B[300 K NPT reference equilibration]
    B --> C[5 deformation scales x 2 temperatures]
    B --> D[3 random scales x 2 temperatures]
    C --> E[16 reference boxes]
    D --> E
    E --> F[Validated OUTCAR, XDATCAR, energies and forces]
    F --> G[Frame filtering and train/validation split]
```

For molecular/liquid/polymer systems, these boxes generate diverse training configurations;
they do not repeat every production molarity.

## 5. MLIP, production MD, and active learning

```mermaid
flowchart LR
    A[Validated AIMD frames] --> B[h04 DeepMD or MACE training]
    B --> C[Energy and force validation gates]
    C --> D[h05 MLMD NPT at 300 K]
    D --> E[Parallel NVT temperature sweep]
    E --> F[h13 uncertainty selection]
    F --> G[Additional AIMD labels]
    G --> B
    F --> H[Frozen dataset and accepted model]
    I[OPLS-AA system_cmd.data] --> J[h05 CMD NPT at 300 K]
    J --> K[CMD NVT temperature sweep]
```

## 6. Analysis, models, and publication outputs

```mermaid
flowchart LR
    A[CMD and MLMD trajectories] --> B[MSD and diffusion]
    A --> C[RDF and coordination]
    A --> D[Van Hove, VACF, VDOS]
    A --> E[Hopping, Haven ratio, phase metrics]
    B --> F[Arrhenius and conductivity]
    B --> G[Continuum and electrochemical models]
    C --> H[Validated result tables]
    D --> H
    E --> H
    F --> H
    G --> H
    H --> I[Figures]
    I --> J[DOCX manuscript]
    J --> K[FAIR result package]
```

The equations behind every analysis and model box are listed in the
[mathematical formula reference](../reference/formulas.md).
