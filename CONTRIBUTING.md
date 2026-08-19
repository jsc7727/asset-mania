# Contributing to Asset Mania

Thank you for helping build the inspection and planning foundation.

## Before you start

Read [AGENTS.md](AGENTS.md), [the rules index](rules/index.md), and the relevant documentation.
The current repository is pre-alpha, with a working v0.1 inspection CLI and locked workspace.
Run `make setup` before making a change and use the verified quickstart for a local smoke test.

## Contribution workflow

1. Keep one focused change per branch and use Conventional Commits.
2. Write behavior changes red-green-refactor, including deterministic and privacy assertions.
3. Update public documentation and rule guidance when the user-facing contract changes.
4. Run `make check`, `make test`, `make skill-check`, and `make release-check`, then report the
   exact evidence in the pull request.

Do not add real-person imagery, downloaded weights, credentials, opaque binaries, or uploaded
source data. See [security reporting](SECURITY.md) for vulnerabilities and
[third-party notices](THIRD_PARTY_NOTICES.md) for attribution requirements.
