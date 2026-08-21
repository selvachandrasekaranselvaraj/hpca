# Maintaining the documentation

Install and build:

```bash
python -m pip install -e '.[dev,docs]'
mkdocs build --strict
python docs/build_complete_html.py
python -m pytest hpca/tests/test_documentation.py
```

## Source-of-truth rules

- Describe behavior implemented by the current code, not a proposed design.
- Link to generated API pages instead of copying signatures.
- Take names and dependencies from the stage registry, paths from the folder registry, VASP
  settings from the INCAR registry, and scheduler templates from the submission registry.
- Put deployment paths on the deployment-profile page, not in portable tutorials.
- Validate every example YAML with `hpca.core.project_schema.validate`.
- Move completed proposals to the archive and label them non-authoritative.

Any change to a console command, project field, category, stage, registry, output contract, or
daemon lifecycle must update the corresponding page and documentation contract test.

API pages are generated from every non-test `hpca/**/*.py` file by `docs/gen_api_pages.py`.
Do not hand-create competing API pages.

`docs/build_complete_html.py` produces the printable single-file manual at
`site/hpca-complete.html`. Its formulas remain visible as source text offline; MathJax enhances
their rendering when network access to the configured CDN is available.
