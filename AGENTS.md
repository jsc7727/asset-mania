# Asset Mania Agent Guide

Asset Mania is a guide-first, pre-alpha workspace. Read the focused rules before changing
code or documentation.

## Start here

- Read [rules/README.md](rules/README.md) and use [rules/index.md](rules/index.md) to route
  the task.
- Search relevant guidance before implementation: `rg -n "<topic>" rules`.
- Inspect source assets read-only. Inspection must never modify, move, overwrite, or upload a
  source asset.

## Working boundaries

- Keep changes scoped and preserve the portable manifest, privacy, and provenance contracts.
- Do not silently substitute a provider, model, revision, quality, or workflow.
- Do not upload external data, download models, spend money, invoke remote generation, or use
  paid compute without fresh explicit approval for that operation.
- Keep detailed operational guidance in `rules/`; this file is only the entrypoint.

## Valid commands

The repository commands are `make setup`, `make check`, `make test`, `make skill-check`, and
`make release-check`. Use only the targets that exist for the task at hand and report their
results.
