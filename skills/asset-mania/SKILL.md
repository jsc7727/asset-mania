---
name: asset-mania
description: Use when a request involves Asset Mania preflight, image-to-3D or Blender-to-image eligibility, or planning an Asset Mania generation workflow.
---

# Asset Mania

Asset Mania performs deterministic, local, source-read-only work: inspection, scene preflight,
conditioning, aligned-view ingest, reprojection and bake, and validated export. It does not
generate images itself. External generation stays behind exact, plan-bound approval.

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
- [engine clearance v1](references/engine-clearance-v1.schema.json)
- [reconstruction plan v1](references/reconstruction-plan-v1.schema.json)

## Local stage routing

These stages run locally and touch no network. Route a request to the one that matches, and
report the manifest status, diagnostics, and capability limits it returns:

| Request | Stage |
| --- | --- |
| "what is in this .blend?" | `scene preflight` |
| "set up a render for this camera and frame" | `scene plan`, then `scene condition` |
| "use this image as the texture source" | `view ingest` |
| "bake that view into the UVs" | `texture bake` |
| "give me a GLB / FBX / editable file" | `export` |

Rules that hold for every stage:

- Never modify, move, overwrite, embed, or upload the source.
- A subject category is a user declaration. `unknown` is blocked; never infer it from pixels
  or geometry.
- `real_person` needs a rights receipt bound to the exact plan digest before anything runs.
- An alignment claim needs the exact `CONDITION_SHA256:VIEW_SHA256` string. There is no
  boolean shortcut, and a same-sized image is not evidence of alignment.
- Report low coverage as incomplete. Do not present an incomplete bake as a finished asset.

## Stop at the external-action boundary

For any request to upload data, download a model, use a remote provider, or spend paid
compute:

1. Local stages may run when an input is available.
2. State that external generation requires an explicit, plan-bound approval for each gate:
   external egress, paid compute, and face rights where the subject is a real person.
3. Do not request or imply approval on the user's behalf, and take no network, upload,
   download, GPU, or paid action without it. Never substitute another provider, model,
   revision, quality, or workflow.

The GPT Image 2 adapter is **experimental and contract-verified only**: it has been exercised
against a fake transport with sockets denied and has never made a live call. Do not describe
it as working or live-verified.

Read [safety and licenses](references/safety-and-licenses.md) only when explaining current or
future external-action, face-rights, privacy, provenance, or licensing boundaries. Maintainers can
use [the fixed evaluations](references/evals.md) to forward-test this skill; structural validation
does not replace those behavior checks.
