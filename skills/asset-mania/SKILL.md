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

## Machine-readable contracts

Use [the v1 manifest schema](references/manifest-v1.schema.json) for the schema of a run this
skill can actually produce today.

The remaining schemas are published contracts for the planned v0.2 execution stages. They describe
field shapes only; no stage below is executable, so read them to answer a question about the format
and never to imply the workflow runs:

- [run manifest v2](references/manifest-v2.schema.json)
- [workflow plan v1](references/workflow-plan-v1.schema.json)
- [conditioning bundle v1](references/conditioning-bundle-v1.schema.json)
- [view v1](references/view-v1.schema.json)
- [provider evidence v1](references/provider-evidence-v1.schema.json)
- [provider plan v1](references/provider-plan-v1.schema.json)
- [approval receipt v1](references/approval-receipt-v1.schema.json)

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
