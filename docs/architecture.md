# Architecture

The root project is a non-package `uv` workspace. Future `packages/contracts` will own the
versioned schema, stable diagnostics, and portable data types. Future `packages/cli` will own
local inspection and command behavior while depending on contracts through the workspace.

The root Skill will orchestrate the installed CLI rather than duplicate inspection logic. Docs
own human guidance and `rules/` owns concise agent instructions. This separation makes future
local, BYOK, custom-remote, and cloud providers additive instead of coupling them to v0.1.

Provider operations will use the conceptual boundary `preflight`, `plan`, `run`, `status`,
`cancel`, `fetch`, and `validate`. v0.1 deliberately implements only local inspection and
planning; it has no provider execution path.

See [the roadmap](roadmap.md), [security and privacy](security-and-privacy.md), and
[the manifest contract](concepts/run-manifest.md).
