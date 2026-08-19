# Development

## Workspace

The project targets CPython 3.11–3.13 on macOS 14+ and Ubuntu 22.04/24.04; `.python-version`
selects Python 3.12 for local development. The root `pyproject.toml` owns shared Ruff, pytest,
and workspace configuration.

The canonical Make targets are `setup`, `check`, `test`, `skill-check`, and `release-check`.
They are implemented stable interfaces for workspace setup, static checks, tests, Skill
validation, and release-content validation. `make setup` installs the working `asset-mania`
console command into the workspace environment.

## Engineering rules

Use red-green-refactor for behavior changes, deterministic JSON for portable outputs, and
source-read-only inspection. Before changing a contract, search [the rules](../rules/README.md)
and update tests, docs, and the relevant schema together. Preserve the separate future Blender
GPL boundary and never introduce an unapproved network, model-download, or paid-compute path.
