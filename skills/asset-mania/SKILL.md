---
name: asset-mania
description: Use when a request involves Asset Mania preflight, image-to-3D or Blender-to-image eligibility, or planning an Asset Mania generation workflow.
---

# Asset Mania

Asset Mania v0.1 performs deterministic, local, source-read-only inspection. It cannot generate
images or 3D assets.

## Inspect

When an input is available, run:

```text
python3 <skill-directory>/scripts/inspect.py inspect <input> [options]
```

- Use `--workflow image-to-3d` for images and `--kind object|character|face-head` when declared.
- Use `--workflow scene-to-image` for `.blend` inputs.
- Report the manifest status, diagnostics, warnings, and capability limits. A successful preflight
  still reports `WORKFLOW_NOT_IMPLEMENTED` in v0.1.
- Never modify, move, overwrite, embed, or upload the source.

Read [the CLI contract](references/cli-contract.md) for options, streams, run files, and exit codes.
Use [the manifest schema](references/manifest-v1.schema.json) when machine-readable field details
matter.

## Stop at the v0.1 boundary

For any request to generate images or 3D, upload data, download a model, use a remote provider, or
spend paid compute:

1. Local inspection may run when an input is available.
2. State that the requested execution is unavailable in v0.1.
3. Stop without requesting approval and without any network, upload, download, Blender, GPU, or
   paid action. Do not substitute another provider, model, quality, or workflow.

Read [safety and licenses](references/safety-and-licenses.md) only when explaining current or
future external-action, face-rights, privacy, provenance, or licensing boundaries. Maintainers can
use [the fixed evaluations](references/evals.md) to forward-test this skill; structural validation
does not replace those behavior checks.
