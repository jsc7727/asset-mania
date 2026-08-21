# Asset Mania v0.3 Generic Image-to-3D Design

> **Status.** Design frozen 2026-08-21. No engine, model weight, or preprocessing model is
> bundled, downloaded, or executed by this design. Nothing here claims that generic
> image-to-3D works in Asset Mania; it defines the contract and the gates that a working
> engine would have to pass.

## Outcome

v0.2 delivered a real local Blender round trip that projects a supplied, camera-aligned
image onto a mesh the user already owns. v0.3 addresses the capability users actually ask
for first: turn a single image into a mesh.

The hard part is not calling an inference engine. It is that every candidate engine arrives
with a dependency closure whose licenses differ from its headline license, plus model
weights licensed separately from code, plus a preprocessing step that silently pulls its own
model. v0.1 documentation got this wrong once already: it described stock TRELLIS and
InstantMesh as permissively licensed when their closures are not. That correction landed in
v0.2, as prose. v0.3 turns it into a gate.

So the deliverable is inverted from the obvious one:

- **primary:** a closed engine-clearance contract and a fail-closed gate, so an engine
  cannot be planned or executed until its entire closure is recorded and cleared;
- **secondary:** the provider port, the reconstruction plan, and the input contract that a
  cleared engine plugs into;
- **explicitly not in this milestone:** running TripoSR, downloading any weight, or claiming
  that image-to-3D works.

## Binding Decisions

- The first engine target is `triposr-local`: single image to mesh, executed locally.
- No weight, checkpoint, or preprocessing model is bundled or downloaded by any stage. A
  user acquires them, and the acquisition is recorded before anything runs.
- An engine is unusable until its **clearance artifact** records, for the engine code, the
  model weights, the preprocessing model, and **every runtime dependency**: an immutable
  revision, a content digest, a license identifier, a license URL, and a download receipt.
  A single missing or uncleared entry fails the run.
- License identifiers are recorded from a verified source at acquisition time. This design
  asserts no license fact about any third-party engine, because asserting one from memory is
  how the v0.1 error happened.
- Background removal is never implicit. Either the input is already masked, or the run names
  an audited background-removal model pinned by digest. An unpinned `rembg` default is
  rejected outright.
- Subject and asset-kind declarations carry over unchanged from v0.2: they are user
  declarations, `unknown` is blocked, and nothing is inferred from pixels. A `real_person`
  subject additionally requires the plan-bound `face_rights` receipt.
- A generated mesh is `content_origin: generated` and stays so transitively. It is never
  presented as observed geometry.
- Engine execution is an injected port. The adapter constructs no subprocess, socket, or
  model loader of its own, so "nothing ran before clearance" is testable rather than
  promised.
- A reconstruction output has **no camera correspondence**. It therefore cannot enter the
  v0.2 conditioning or bake path, which requires an aligned view for a known camera. Mixing
  the two would produce a texture that looks plausible and is wrong.

## Command Surface

```text
asset-mania engine clearance verify ENGINE_CLEARANCE
    [--out RUNS]

asset-mania image reconstruct SOURCE_IMAGE
    --engine triposr-local
    --clearance ENGINE_CLEARANCE
    --asset-kind object|character|face-head
    --subject non-person|synthetic-person|real-person|unknown
    [--mask SOURCE_MASK]
    [--background-removal AUDITED_MODEL_CLEARANCE]
    [--rights-receipt RIGHTS_RECEIPT]
    [--out RUNS]
```

`engine clearance verify` is offline and reads only the clearance artifact. `image
reconstruct` fails closed before touching the engine when clearance, input, or declaration
requirements are unmet.

## Execution Contracts

### `engine-clearance-v1`

The artifact that makes an engine usable. Closed, and every component is required:

- `schema_id: asset-mania/engine-clearance`, `schema_version: 1.0`;
- `engine`: a portable identifier such as `triposr-local`;
- `components`: an array ordered by `role`, one entry per role, covering at minimum
  `engine_code`, `model_weights`, and `preprocessing_model`;
- `runtime_dependencies`: a name-sorted array covering **every** runtime dependency;
- `cleared_by`: `user` only. A maintainer cannot clear an engine on a user's behalf;
- `cleared_at`, `expires_at`;
- `clearance_sha256`.

A component and a dependency share the same closed record:

```json
{
  "role": "model_weights",
  "name": "portable-label",
  "revision": "an immutable revision or version",
  "content_sha256": "64 lowercase hex",
  "license_spdx": "an SPDX identifier or CUSTOM",
  "license_url": "https://…",
  "commercial_use": "cleared" | "prohibited" | "unknown",
  "download_receipt_sha256": "64 lowercase hex"
}
```

`commercial_use` is the field that does the work. `prohibited` and `unknown` both fail: an
uncleared license is not a smaller problem than a forbidding one, because the reason v0.1
was wrong is that nobody had checked.

### `reconstruction-plan-v1`

Immutable, sealed, and bound to the clearance digest:

- source image digest, decoded dimensions, and colour space;
- mask digest, or the background-removal clearance digest, never neither;
- `engine`, `clearance_sha256`, `engine_profile`;
- user-declared `asset_kind` and `subject`;
- `expected_output`: mesh format, whether a texture is expected, and the declared unit
  scale;
- `overwrite_policy: create_only`;
- `plan_sha256`.

### Diagnostics

New closed codes, added to the v2 enum:

| Code | Meaning |
| --- | --- |
| `ENGINE_NOT_CLEARED` | clearance is absent, incomplete, expired, or not user-issued |
| `ENGINE_LICENSE_UNCLEARED` | a component or dependency is `prohibited` or `unknown` |
| `ENGINE_UNAVAILABLE` | the engine port is not installed or refused to start |
| `MASK_REQUIRED` | neither a mask nor an audited background-removal clearance was given |
| `BACKGROUND_REMOVAL_UNPINNED` | a background-removal model has no pinned digest |
| `RECONSTRUCTION_FAILED` | the engine ran and produced no usable mesh |
| `RECONSTRUCTION_UNVERIFIED` | a mesh was produced but failed structural validation |

## Input Contract

The reconstruction input is a still image, decoded fully before use, under the same
normalization rules v0.2 applies to a view: 8-bit sRGB, RGB or straight-alpha RGBA, no
implicit resize, rotation, or colour conversion, EXIF orientation absent or 1, metadata
stripped from the normalized copy, and hidden RGB zeroed.

Two differences from a v0.2 view:

- there is no conditioning resolution to match, so an arbitrary size is accepted within the
  decompression limits; and
- a foreground mask is mandatory, because a single-image reconstructor given a full scene
  reconstructs the scene.

## Output Contract

A reconstruction produces a mesh with no camera correspondence and no authored UVs. It is
therefore validated on its own terms:

- structural validation only: finite vertices, non-degenerate triangles, a closed or
  explicitly open manifold state recorded rather than assumed, and a recorded triangle and
  vertex count;
- `content_origin: generated`, `sensitivity: user-content`, `upload_eligible: false`;
- the mesh is **not** eligible for the v0.2 bake path. Feeding it there is rejected, because
  bake requires non-overlapping authored UVs and an aligned view for a known camera, and a
  reconstruction has neither.

## What This Milestone Does Not Claim

- No engine has been executed. No weight has been downloaded. No license has been cleared.
- The clearance gate is verified against synthetic clearance artifacts and a fake engine
  port with subprocesses and sockets denied.
- Any public claim that generic image-to-3D works requires a real cleared engine, a real
  run, and recorded evidence at the publication SHA. Until then the capability table says
  `Planned`, and the Skill refuses the request.

## Primary References

Engine, weight, and dependency license facts are recorded in the clearance artifact from
sources verified at acquisition time. They are deliberately absent from this document.
