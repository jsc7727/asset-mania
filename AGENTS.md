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
- Treat reconstruction engines as replaceable plugins. Do not hard-code the current model as
  permanent; define or change plugin interfaces only through a future approved design specification.
- Do not upload external data, download models, spend money, invoke remote generation, or use
  paid compute without fresh explicit approval for that operation.
- Keep detailed operational guidance in `rules/`; this file is only the entrypoint.

## Agent delegation

- Use **SOL HIGH** (`gpt-5.6-sol`, `reasoning_effort=high`) for supervision, architecture,
  implementation plans, security and privacy decisions, integration, and final review.
- Delegate concrete implementation, tests, documentation, and other bounded independent work to
  **SOL LIGHT** subagents (`gpt-5.6-sol`, `reasoning_effort=low`).
- Give each SOL LIGHT subagent one explicit, independently verifiable scope. The supervising SOL
  HIGH agent owns task decomposition, shared-state coordination, conflict resolution, verification,
  and the final result.
- Do not delegate user approvals, credential handling, licensing judgments, privacy policy, security
  boundaries, destructive actions, or final acceptance decisions to SOL LIGHT.
- When selecting SOL LIGHT explicitly, use `fork_turns="none"` or a limited positive turn count so
  the model and reasoning override can be applied; pass only the context needed for that task.

## Valid commands

The repository commands are `make setup`, `make check`, `make test`, `make skill-check`,
`make release-check`, `make license-check`, `make schema-check`, and `make publication-check`. Use only the targets that exist for the task at hand and report their
results.
