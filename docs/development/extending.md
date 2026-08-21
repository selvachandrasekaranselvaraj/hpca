# Extending HPCA

## Add or change a category

1. Register its semantics in `hpca.core.categories`.
2. Add category physics/platform defaults only where applicable.
3. Update category stage routing and dependency overrides in the stage registry.
4. Implement or reuse design and scientific behavior through handlers/domain services.
5. Add schema, wizard, example, workflow, and category-routing tests.

## Add a stage

1. Define name, lane, description, dependencies, order, category enablement, and outputs in
   `hpca.registry.stage` and the folder registry.
2. Add a handler implementing preparation, dispatch, progress checks, scientific completion,
   and bounded recovery.
3. Register the handler with the orchestrator without duplicating dependency logic.
4. Add submission/INCAR/POSCAR definitions to their registries as needed.
5. Test fresh submission, running reconciliation, success, partial output, scheduler failure,
   retry exhaustion, restart, stop/resume, and category exclusion.
6. Update stage, file-contract, CLI/configuration, and API documentation.

## Compatibility

Retain an old import path only when it is a documented public compatibility requirement.
Compatibility modules must delegate to the canonical implementation and have a removal plan;
they must never evolve into a second registry.
