# Files and directories

The folder registry is the source of truth. A representative sub-project layout is:

```text
project/
├── project.yaml
├── .hpca/control.json
│   └── artifacts.jsonl         # append-only checksummed provenance
├── designed_structures/
│   ├── poscar_dft.vasp
│   ├── poscar_mlmd.vasp
│   └── poscar_cmd.vasp
├── preopt/
│   ├── contcar_mlmd_preopt.vasp
│   ├── contcar_cmd_preopt.vasp
│   └── system_cmd.data
├── dft/
│   ├── preopt/
│   ├── aimd_relax/          # doped solids only
│   ├── vc/                  # ISIF=3
│   ├── opt/                 # ISIF=2
│   ├── bader/
│   ├── dos/{scf,nonscf}/
│   ├── static/
│   └── echem_static/
├── aimd/
│   ├── NPT/                 # crystalline 300 K reference equilibration
│   ├── dataset/
│   └── <temperature>/
├── mlmd/{mlff,npt,nvt}/
├── cmd/{npt,nvt}/
├── neb/
├── results/
├── figures/
├── manuscript/
└── logs/
```

Molecular NPT calculations may be stored below `aimd/<temperature>/NPT/`; crystalline NPT is
stored at `aimd/NPT/`. This does not make either directory a DFT preoptimization location.

Handlers must obtain paths from `hpca.registry.folder` and must not reproduce these strings
inline. Output existence is insufficient for completion when the relevant handler defines a
convergence, frame-count, density, integrity, or uncertainty gate.

## Key artifact contract

| Artifact | Canonical relative path |
|---|---|
| DFT design | `designed_structures/poscar_dft.vasp` |
| MLMD design | `designed_structures/poscar_mlmd.vasp` |
| CMD design | `designed_structures/poscar_cmd.vasp` |
| DFT preoptimization | `dft/preopt/CONTCAR` |
| Doped-solid equilibration | `dft/aimd_relax/` |
| MLMD preoptimization | `preopt/contcar_mlmd_preopt.vasp` |
| CMD preoptimization | `preopt/contcar_cmd_preopt.vasp` |
| CMD force-field system | `preopt/system_cmd.data` |
| Variable-cell result | `dft/vc/CONTCAR` |
| Atomic optimization result | `dft/opt/CONTCAR` |
| AIMD reference dataset | `aimd/dataset/` |

Every registered durable result records its project-relative path, producer, semantic kind,
file format, byte size, SHA-256 digest, UTC creation time, and optional metadata in
`.hpca/artifacts.jsonl`. Verification recomputes both size and digest; a matching filename is
not sufficient.
