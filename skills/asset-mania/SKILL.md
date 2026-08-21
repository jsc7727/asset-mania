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

## Generic image-to-3D runs, but not through this skill

For any request to turn a photo or a single image into a 3D model, mesh, or figure —
including "make this person 3D":

1. State plainly that the stages this skill drives do not generate 3D geometry. They project a
   supplied, camera-aligned image onto a mesh the user already owns; they require the 3D model
   as an **input**.
2. Separately, an engine adapter exists and has produced a mesh from an image on a developer
   machine. Say so — it is true, and claiming otherwise to stay on the safe side is its own
   inaccuracy. Then say what stops it here: there is **no cleared engine** in the user's
   installation. No engine code or weight ships in any wheel, and the adapter refuses to run
   until the user issues an `engine-clearance-v1` artifact covering the engine code, the
   weights, the architecture config TripoSR fetches at runtime, and every runtime dependency.
3. Never issue that clearance, draft it as though it were issued, or run the adapter with a
   clearance the user did not author. Accepting third-party licence terms is the user's
   decision, and `cleared_by` accepts `user` for exactly that reason.
4. Do not describe the remaining work as larger than it is. The blocker is clearance and
   acquisition, which the user can do by reading `scripts/acquire_engine_assets.py` output and
   deciding — not missing implementation.
5. A mesh from that path closes only after a bounded repair pass, and the pass refuses to cap
   an opening wider than a tenth of the mesh. So the result is `closed` on a clean subject and
   **`open` when the model failed to reconstruct a region** — report whichever state the run
   returned, and never present `open` as usable where a closed volume is required.
6. If the user has a UV-mapped mesh, offer the local stages above instead, which do work.

## Faces and heads

`asset_kind: face_head` is gated twice over, and both gates matter for different reasons:

1. A `real_person` subject needs a rights receipt bound to the exact plan digest. That receipt
   is a user assertion. Never issue or imply one on the user's behalf.
2. `face_head` with `non_person` is refused with `SUBJECT_KIND_INCOHERENT`. Before this gate
   existed the combination sealed a plan with no receipt at all, which made it the way around
   the rights gate rather than an odd declaration. If a user offers it, say what it means and
   ask which subject actually applies -- do not pick one for them, and do not suggest
   `non_person` as a way to proceed.

Every `face_head` mesh carries a `likeness-disclosure-v1` recording the source image, the plan,
the engine, and what has been measured. Report its contents when describing such a mesh.

**No face accuracy has been measured.** The one accuracy figure this project has -- a symmetric
mean surface distance of 6.0% of the subject's longest axis -- comes from a rendered geometric
subject compared against its own source mesh. It does not transfer to a human face. Never quote
it as a likeness figure, and never describe an output as an identification-grade likeness, a
biometric record, or a match to a specific person: a single view underdetermines the geometry
behind a face, and nothing here has measured whether such a claim would hold.

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
