# Asset Mania GPT Image 2 Turntable and Multi-view Reconstruction Design

**Status:** approved design, frozen before implementation  
**Date:** 2026-08-22  
**Scope:** one real-person face/head photograph to an eight-view generated turntable and one
locally fused, provenance-carrying GLB

## Goal

Add a real end-to-end workflow that keeps the observed front photograph at yaw `0`, uses the
pinned `gpt-image-2-2026-04-21` snapshot to generate seven additional views at 45-degree yaw
increments, audits the set without claiming biometric identity consistency, reconstructs one
TripoSR mesh per view, and fuses the eight meshes into a closed neutral GLB by yaw-aware voxel
consensus.

The workflow is research-grade. Generated side and rear views are inferences, not observations.
The output must never be described as an exact likeness, an identity match, a biometric record,
or observed rear geometry.

## Scope boundaries

### Included

- A closed turntable plan and viewset contract.
- One observed `0`-degree view plus seven generated views at `45`, `90`, `135`, `180`, `225`,
  `270`, and `315` degrees.
- GPT Image 2 generation through the existing approval-gated OpenAI provider boundary.
- A deterministic structural consistency audit and a human-review contact sheet.
- Local foreground-mask derivation for generated white-background views.
- Eight independent, clearance-gated TripoSR reconstructions.
- Yaw normalization and voxel-majority fusion into a neutral GLB.
- A sealed multi-view reconstruction record and face likeness disclosure.
- Fake-transport, synthetic-fusion, and opt-in live E2E verification.

### Excluded

- Claims that generated views reveal the subject's real unseen appearance.
- Face recognition, face embeddings, identity scoring, or biometric matching.
- AliceVision/COLMAP photogrammetry in this milestone. Those tools assume photographs with
  camera-consistent features; generated views do not provide that evidence.
- InstantMesh, TRELLIS, FLAME, DECA, MICA, or new model-weight downloads.
- Automatic paid retries. A retry is a new plan and a new approval.
- Texture-atlas fusion. Version 1 delivers vertex-colour source meshes for evidence and a
  neutral fused geometry; it does not hide cross-view texture seams.

## Architecture

```text
observed front photograph (yaw 0)
        |
        +--> turntable plan + face/external/paid approvals
        |
        +--> GPT Image 2 edits at yaw 45..315 (seven calls, no retry)
        |         |
        |         +--> eight-view viewset + masks + contact sheet
        |                          |
        |                          +--> structural consistency audit
        |                                      |
        +--------------------------------------+
                                               |
                                  TripoSR x 8, locally and offline
                                               |
                                  yaw normalization about +Z
                                               |
                                  voxel occupancy majority vote
                                               |
                                  closed neutral GLB + disclosure
```

The system keeps provider generation, viewset validation, per-view reconstruction, and mesh
fusion as separate units. Each unit has a closed input and output contract and can fail without
publishing a later-stage artifact.

## Contract additions

### `turntable-plan-v1`

The plan is immutable and self-sealed. Required fields are:

- `schema_id: asset-mania/turntable-plan`, `schema_version: 1.0`;
- the observed source image digest, dimensions, and supplied foreground-mask digest;
- `asset_kind: face_head`, `subject: real_person | synthetic_person`;
- `provider: openai`, endpoint `/v1/images/edits`, and model snapshot
  `gpt-image-2-2026-04-21`;
- fixed yaw schedule `[0,45,90,135,180,225,270,315]`, pitch `0`, roll `0`;
- generation controls: `1024x1024`, `medium`, PNG, opaque white background, `n: 1` per call;
- one prompt digest and a prompt-template revision;
- fresh policy/pricing evidence and aggregate seven-call estimated and maximum cost;
- required gates `[face_rights, external_egress, paid_compute]` for `real_person`;
- `overwrite_policy: create_only` and `plan_sha256`.

One plan authorizes exactly seven provider calls in ascending yaw order. The receipt scope is one
turntable run, not an open-ended provider allowance. Receipts are consumed before the first call.
If any call fails, later calls are not attempted and no viewset is published. Partial bytes remain
quarantined and local-sensitive.

### `turntable-viewset-v1`

The viewset contains exactly eight records sorted by yaw:

- yaw `0` has `origin: observed`; every other record has `origin: generated`;
- image and normalized-mask SHA-256, byte size, media type, width, and height;
- provider request ID and reported usage for generated views only;
- `target_yaw`, `pitch`, `roll`, and portable label `view-1` through `view-8`;
- structural audit metrics and status;
- `identity_consistency: unmeasured`;
- aggregate actual cost, separated from the preflight estimate;
- `viewset_sha256`.

The viewset never contains the source filename, absolute path, prompt text, credential, or image
bytes. The run directory contains those private bytes and remains local-sensitive.

### `multiview-reconstruction-v1`

The final record contains:

- the turntable viewset digest and eight input-image digests;
- eight per-view mesh digests, triangle counts, vertex counts, and manifold states;
- normalization, yaw, voxel-grid, and vote-threshold parameters;
- fused GLB digest, byte size, triangle count, vertex count, and manifold state;
- `content_origin: generated`, `sensitivity: user-content`, `upload_eligible: false`;
- a `likeness-disclosure-v1` with `views: 8` and face accuracy still unmeasured;
- `record_sha256`.

## GPT Image 2 generation profile

The existing provider adapter remains the only network boundary. A new turntable request builder
does not change the current scene-conditioning profile.

Each provider call receives one private RGBA cutout derived from the approved observed image and
mask. The prompt fixes:

- the target yaw and zero pitch/roll;
- neutral expression, open eyes, closed mouth, unchanged hairstyle and visible clothing;
- centered head and upper neck, level camera, long-lens studio portrait perspective;
- flat white background, even diffuse light, no text, no watermark, no accessories added;
- preservation of facial proportions while stating that unseen geometry must be inferred.

The seven calls are issued sequentially. There is no provider retry and no fallback model,
snapshot, size, quality, or background. Each response is decoded, bounded, normalized, and placed
in quarantine before the next call. Only all seven valid replies can form a viewset.

## Structural consistency audit

The audit is deterministic and deliberately non-biometric. Every view must satisfy:

- PNG, `1024x1024`, eight-bit sRGB;
- one foreground component after edge-connected white-background removal;
- foreground coverage between `0.20` and `0.75` of all pixels;
- foreground centroid within `0.10` of normalized image centre on each axis;
- foreground pixels on the image border below `0.01` of foreground pixels;
- adjacent-view foreground-area ratio between `0.65` and `1.35`;
- no byte-identical or decoded-pixel-identical generated views;
- all eight yaws present exactly once and sorted.

The audit emits a four-by-two contact sheet for human review. Passing means the files are a
structurally usable turntable candidate. It does not mean the face identity, unseen hairstyle, or
rear anatomy is correct. That remains `identity_consistency: unmeasured`.

If fewer than eight views pass, if an angle is missing, or if a metric is outside the fixed
limits, the run ends with `VIEWSET_INCONSISTENT`; TripoSR is never launched.

## Per-view TripoSR reconstruction

Every audited view is reconstructed through the existing clearance-gated adapter. The production
profile uses:

- engine `triposr-local` and profile `triposr-local-cpu-v1`;
- offline weights and DINO configuration already covered by engine clearance;
- supplied normalized mask, never `rembg`;
- marching-cubes resolution `256` and threshold `25.0`;
- vertex colours retained in the per-view evidence meshes.

The deterministic test profile uses resolution `32`; the live E2E may use `128` for a first run
and must use `256` before a result is labeled the final research artifact. Each mesh must contain
triangles and vertices and report `closed`, `open`, or `unknown`. Fusion requires at least six
closed, winding-consistent meshes and rejects fewer with `MULTIVIEW_INSUFFICIENT_MESHES`.

## Yaw-aware voxel fusion

Fusion is implemented inside the optional TripoSR engine package, not the common pipeline.

1. Load each mesh without implicit repair.
2. Merge coincident vertices, remove degenerate faces, repair only holes within the existing
   bounded repair policy, and require consistent winding.
3. Centre each mesh on its bounding-box centre and scale its longest extent to `1.0`.
4. Rotate generated mesh `i` by `-target_yaw` degrees about `+Z`; yaw `0` is the reference.
5. Translate each mesh to the median centroid. Do not run unconstrained rotational ICP.
6. Voxelize each closed mesh into the common cube `[-0.6,0.6]^3` at `192^3` final resolution
   (`48^3` deterministic test profile) and fill its interior.
7. Mark a voxel occupied when at least four of eight meshes vote occupied. If only six or seven
   meshes are eligible, require `ceil(count / 2)` votes.
8. Extract the consensus isosurface with the installed CPU torchmcubes extension.
9. Transform vertices back to normalized world coordinates, orient faces outward, and apply the
   existing bounded small-hole repair.
10. Require a non-empty, watertight, positive-volume mesh and export a neutral GLB.

The fused mesh does not inherit one input's vertex colours because that would misrepresent one
generated angle as a coherent 360-degree texture. Texture fusion is a later, separately measured
milestone.

## Orchestration and commands

The executable maintainer path is:

```text
python scripts/run_turntable_multiview_e2e.py plan ...
python scripts/run_turntable_multiview_e2e.py generate ...
python scripts/run_turntable_multiview_e2e.py reconstruct ...
python scripts/run_turntable_multiview_e2e.py verify ...
```

`plan` is offline. `generate` is the only networked/paid phase and refuses without the exact three
plan-bound receipts and a credential resolved at call time. `reconstruct` is local and offline.
`verify` checks all contracts, source integrity, GLB structure, manifold state, disclosure, and
renders a Blender contact preview.

The public `asset-mania` CLI is not expanded until the maintainer E2E proves the workflow. This
keeps pre-alpha execution behind an explicit script while contracts and behavior stabilize.

## Failure and recovery semantics

- Invalid input or plan: exit `2`, no run created.
- Contract, audit, or provider response failure: exit `3`, terminal failed run with stable code.
- Internal failure: exit `4`, sanitized terminal run.
- Missing approval: exit `5`, no provider credential resolution or request.
- User cancellation: exit `6`, no unissued call and no automatic continuation.
- Storage failure: exit `73`, no replacement of an existing run.

Provider calls are paid and non-idempotent. A failed or timed-out call is never retried. A new
attempt requires a new plan because the approved maximum cost and remaining yaw schedule changed.
Local TripoSR and fusion stages may be rerun only into a new create-only run directory while reusing
the immutable viewset.

## Privacy, approvals, and provenance

- The observed photograph, masks, generated views, prompts, per-view meshes, and fused mesh are
  local-sensitive user content and never enter fixtures, logs, commits, telemetry, or galleries.
- The observed photograph and RGBA cutout are the only image content sent to OpenAI. Local TripoSR
  meshes and run records are not uploaded.
- A real-person plan needs exact `face_rights`, `external_egress`, and `paid_compute` receipts.
- Prompt text and credentials are private and absent from portable artifacts.
- GPT-generated views are always `generated`; they never become observed evidence.
- Source bytes are hashed before and after the workflow and must be identical.
- No output is automatically published or copied outside `.asset-mania/`.

## Testing strategy

### Contract tests

- Closed schemas, canonical self-digests, fixed yaw order, origin rules, and no path leakage.
- A plan altered in yaw, prompt digest, model snapshot, control, cost, or attachment fails.
- A viewset with missing, duplicate, or reordered angles fails.

### Provider tests

- Fake transport receives exactly seven sequential requests at yaws `45..315`.
- All three approvals are consumed before the first secret resolution and request.
- No retry occurs after HTTP, timeout, moderation, cancellation, or response validation failure.
- Partial responses stay quarantined and no viewset is published.
- Reported usage and actual cost stay separate from aggregate preflight estimates.

### Audit tests

- Synthetic white-background portraits cover every metric boundary.
- Missing angles, duplicate pixels, off-centre masks, border contact, and area jumps fail with
  `VIEWSET_INCONSISTENT`.
- Passing never changes `identity_consistency` from `unmeasured`.

### Fusion tests

- Eight rotated synthetic ellipsoids with bounded noise fuse into a positive-volume watertight
  mesh near the analytic extent.
- One outlier is rejected by majority voting.
- Fewer than six valid meshes, inconsistent winding, empty consensus, and large openings fail.
- The fused GLB reimports in a fresh Blender process.

### E2E tests

1. Deterministic offline E2E with fake provider images and `32`/`48` reconstruction/fusion profiles.
2. Real GPT Image 2 E2E using the approved face photograph, exact provider snapshot, seven paid
   calls, eight local TripoSR meshes, `192^3` fusion, GLB validation, and Blender preview.
3. The real E2E remains opt-in and cannot run from the ordinary unit-test target.

## Acceptance criteria

- The three new schemas validate and are distributed consistently.
- Existing single-view provider and TripoSR contracts remain unchanged and pass their tests.
- The deterministic fake-provider E2E passes without network or model downloads.
- The live run produces eight provenance-labeled views or stops honestly before reconstruction.
- An audited live viewset produces at least six valid per-view meshes and one watertight,
  positive-volume fused GLB at the final profile.
- The GLB reimports in a fresh Blender process and has a four-view preview.
- The output record carries an eight-view likeness disclosure with face accuracy unmeasured.
- Source bytes remain unchanged, private content stays ignored, and `git diff --check`, relevant
  tests, `make check`, `make test`, `make skill-check`, and `make release-check` pass on a supported
  platform. Windows-specific skips or failures must be reported separately rather than hidden.

