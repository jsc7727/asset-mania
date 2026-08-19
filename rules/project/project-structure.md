# Project structure

- `packages/contracts`: versioned portable contract package and schemas.
- `packages/cli`: deterministic offline inspection CLI.
- `skills/asset-mania`: thin agent routing that invokes the installed CLI.
- `docs/`: public, human-facing contract and roadmap guidance.
- `rules/`: concise operational guidance for agents and contributors.
- `tests/` and package-local tests: fixtures, compatibility coverage, and release validation.
- `scripts/`: repository-owned validation and release utilities.

Do not duplicate CLI logic in the Skill or use documentation as an alternate implementation.
