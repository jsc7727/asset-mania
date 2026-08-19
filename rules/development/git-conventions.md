# Git conventions

- Work on a scoped branch; use Conventional Commits and the repository-local author identity.
- Inspect `git status` before and after work. Preserve unrelated or user-owned changes.
- Do not amend, force-push, or rewrite history unless explicitly requested.
- Before committing, run the relevant checks, inspect the staged diff, and verify the commit
  contains no private inputs, weights, datasets, credentials, or opaque binaries.
- Keep commits focused: workspace/docs, contracts, inspectors, Skill distribution, and release
  automation are separate implementation boundaries.
