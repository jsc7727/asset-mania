# Contributing to Asset Mania

Thank you for helping build the inspection and planning foundation.

## Before you start

Read [AGENTS.md](AGENTS.md), [the rules index](rules/index.md), and the relevant documentation.
The current repository is pre-alpha: the intended workspace commands are being established, but
the inspection CLI is not yet available for an executable quickstart.

## Contribution workflow

1. Keep one focused change per branch and use Conventional Commits.
2. Write behavior changes red-green-refactor, including deterministic and privacy assertions.
3. Update public documentation and rule guidance when the user-facing contract changes.
4. Run the applicable canonical checks once their implementation is present, then report the
   exact evidence in the pull request.

Do not add real-person imagery, downloaded weights, credentials, opaque binaries, or uploaded
source data. See [security reporting](SECURITY.md) for vulnerabilities and
[third-party notices](THIRD_PARTY_NOTICES.md) for attribution requirements.
