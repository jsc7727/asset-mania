# Tech stack boundaries

- Use the root non-package `uv` workspace and its CPython 3.11–3.13 boundary.
- Keep portable schemas, diagnostics, and serialisation in `packages/contracts`; keep CLI
  orchestration and inspection in `packages/cli`.
- Root development dependencies are JSON Schema, pytest, pytest-cov, and Ruff. Add a dependency
  only when the task requires it and record its licensing implications.
- v0.1 is offline and must not invoke Blender, a provider, model download, renderer, or GPU job.
- A future Blender `bpy` add-on is separately packaged GPL-3.0-or-later; do not mix it into the
  Apache-2.0 CLI/core distribution.
