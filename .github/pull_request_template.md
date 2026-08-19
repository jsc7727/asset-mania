## Summary

Describe the user-visible or repository-visible outcome and the scope intentionally left out.

## Verification

- [ ] I added or updated focused tests and observed the required red-green-refactor cycle for behavior changes.
- [ ] `make check` passes.
- [ ] `make test` passes.
- [ ] `make skill-check` passes.
- [ ] `make release-check` passes.
- [ ] I updated documentation, schemas, and Skill references when their contracts changed.

## Safety and release contents

- [ ] Source inputs remain read-only and portable output contains no basename or absolute source path.
- [ ] This change performs no undeclared upload, network request, model download, Blender invocation, or paid action.
- [ ] No credential, cookie, token, private prompt, real-person image, user run output, weight, cache, or dataset is included.
- [ ] New third-party files are inventoried in `THIRD_PARTY_NOTICES.md` with license and redistribution evidence.
- [ ] New binary fixtures are synthetic or redistributable and inventoried in `tests/fixtures/PROVENANCE.md`.
- [ ] Any provider, model, revision, quality, or workflow limitation is disclosed without silent fallback.
