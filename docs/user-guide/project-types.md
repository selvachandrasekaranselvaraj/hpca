# Project types

HPCA routes work from the canonical category registry rather than project names.

| Family | Representative categories | Design source | Typical branches |
|---|---|---|---|
| Molecular | `solvent`, `salt`, `liquid_electrolyte`, `polymer`, `copolymer` | Molecular library, SMILES, PACKMOL, chain builder | DFT/AIMD dataset, CMD, MLIP/MLMD, transport |
| Crystalline | `solid`, `inorganic`, `inorganic_sse` | CIF/POSCAR/structure source and supercell | DFT, AIMD, MLIP/MLMD, electronic; NEB/echem where enabled |
| Interface | Solid/solid or solid/liquid composition metadata | Supplied interface or supported builder | Category-selected DFT, AIMD, NEB, electronic and transport |

## Molecular and polymer projects

Concentration and component fractions define production compositions and combinatorial
sub-projects. AIMD is a reference-data generator: it samples representative distorted and
thermal configurations and is not required to reproduce each production molarity. CMD and
MLMD retain the production composition and temperature sweep.

## Crystalline and doped projects

The input structure is relaxed before downstream calculation. Each doping percentage becomes
a named sub-project with its own `project.yaml` and structure. HPCA records requested and
realized site fractions because integer substitution can differ from the requested percentage.

For a doped solid, a short `dft/aimd_relax` pre-equilibration precedes variable-cell relaxation.
Undoped solids enter variable-cell relaxation directly.

## Combinatorial projects

The top-level project expands selected solvent, salt, polymer, copolymer, concentration, or
doping combinations into independently restartable sub-projects. Selection of a smaller AIMD
subset limits reference-data cost without reducing the production CMD/MLMD sweep.
