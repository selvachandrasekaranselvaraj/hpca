# Validated examples

These files are tested with HPCA's real schema validator:

- [Molecular/polymer electrolyte](../examples/polymer-electrolyte.yaml)
- [Substitution-doped solid](../examples/doped-solid.yaml)

Copy an example into a new directory as `project.yaml`, replace the structure/composition
values, then validate through `hpca status .` before registration. The interactive `hpca new`
wizard is preferred because it calculates combinations, atom budgets, doping realizations,
and ENCUT inputs consistently.

Examples intentionally omit Kestrel absolute binary and pseudopotential paths. Those are
deployment configuration, not project data.
