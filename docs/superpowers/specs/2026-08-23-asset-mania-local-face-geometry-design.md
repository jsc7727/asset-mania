# Asset Mania Local Face Geometry Design

Status: proposed; requires user review before implementation
Date: 2026-08-23
Scope: authorized real-person `face_head` input processed locally with MICA and DECA

## Goal

Produce a face-only 3D geometry candidate whose untextured Blender clay render carries more of the
authorized portrait's visible facial proportions than the current DAD-3DHeads mesh. Geometry must
pass before any fixed head, external hair, or generated texture work begins.

This phase does not promise scan accuracy, biometric identification, or recovered unseen anatomy.
A single frontal image underdetermines side and rear geometry. The output is a local research
estimate with explicit provenance and an honest manual visual verdict.

## Why the current result is not the baseline

The corrected-axis DAD clay audit produced a valid 5,023-vertex FLAME mesh, but the recognizable
identity came primarily from texture. The generated yaw images improved only texture coverage;
they did not change the observed-front DAD geometry. DAD remains eligible later as a head and neck
carrier, but it is no longer the authoritative face geometry.

## Selected architecture

Use two independent local face-geometry plugins and one original fusion stage:

1. **MICA identity base** — predicts the neutral metric FLAME face shape from the authorized source.
2. **DECA detail source** — predicts expression-aware local detail and a coarse comparison mesh.
3. **Asset Mania fusion** — keeps MICA positions as the identity authority and applies only bounded
   DECA normal displacement inside the FLAME face region with a smooth boundary taper.
4. **Clay verification** — exports neutral-material GLBs and renders MICA, DECA, fusion, and the
   corrected-axis DAD baseline under identical Blender cameras.

MICA and DECA remain replaceable external plugins. The common local contract does not name either
model as permanent, and no later head, hair, or texture stage may depend on their private Python
APIs.

## Explicitly deferred subprojects

The following are separate specs and implementation plans, opened only if face geometry passes:

- fixed Blender head, ear, and neck template fitting;
- weighted face-to-head boundary fusion;
- external licensed hair asset catalog and automatic fitting;
- Stable Diffusion UV completion conditioned on Blender depth, normals, and UV position;
- final PBR material and avatar export.

No generated yaw image, Stable Diffusion output, DAD side prediction, or external hair mesh may
alter the clay face geometry in this phase.

## Privacy and v0.4 policy amendment

This design narrowly amends the v0.4 prohibition on face detectors, landmarks, and identity
features. The amendment applies only when all of these conditions hold:

- `asset_kind=face_head` and `subject=real_person` are declared;
- a plan-bound `face_rights` receipt is atomically consumed before source open;
- inference is local on a user-controlled GPU;
- the plugin process starts with network denied;
- MICA's identity feature and any detector or landmark tensors exist only in process memory;
- no identity embedding, detector crop, landmark array, FLAME parameter vector, or face-derived
  feature is written to disk, stdout, stderr, logs, manifests, test fixtures, or telemetry;
- the process exits after writing only the closed geometry inventory;
- no identity comparison score, biometric template, or face-recognition claim is produced.

The subject remains user-declared. A detector may locate the already-declared face for local
alignment, but it may not decide whether the subject is real, identify the person, compare them to
another person, or weaken the rights gate.

Every exported clay mesh retains the existing `likeness-disclosure-v1` requirement. The disclosure
binds the source digest, plan digest, consumed rights receipt, engine/profile, and one observed view;
it records `ground_truth_available=false` and `face_benchmark=null`. The new geometry workflow does
not add a likeness score or reinterpret a manual verdict as measured biometric accuracy.

## Licensing and acquisition boundary

MICA, DECA, FLAME assets, and their model weights are non-commercial research dependencies and are
not part of the Apache-2.0 workspace. Public packages contain original adapters only.

- No implementation task downloads a model, dataset, source checkout, or FLAME asset.
- Live acquisition requires a fresh explicit user approval for each exact source and weight set.
- FLAME assets are user-supplied from the user's licensed local copy; credentials are never handled.
- Every external source revision and every weight or model file is SHA-256 sealed before inference.
- Missing, changed, unlicensed, or unhashed dependencies fail closed.
- External code, weights, private outputs, and real-person images stay under `.asset-mania/` and
  are rejected by publication and release checks.

## Plugin protocol

Add a new protocol rather than broadening the DAD-specific v0 protocol in place.

### Request

`asset-mania.face-geometry-plugin-request.v1` contains exactly:

```python
@dataclass(frozen=True, slots=True)
class FaceGeometryPluginRequest:
    schema: Literal["asset-mania.face-geometry-plugin-request.v1"]
    plugin: Literal["mica-local", "deca-local"]
    profile: Literal["identity-neutral-v1", "detail-displacement-v1"]
    plugin_revision: str
    source_image: Path
    output_directory: Path
    device: Literal["cuda"]
    checkpoint_sha256: str
    topology: Literal["flame-2020-5023"]
    face_rights_receipt_sha256: str
    network: Literal["denied-during-inference"]
```

Paths are absolute only inside the private request envelope. Portable records contain fixed labels
and digests, never paths or basenames.

### Result

`asset-mania.face-geometry-plugin-result.v1` contains exactly:

```python
@dataclass(frozen=True, slots=True)
class FaceGeometryPluginResult:
    schema: Literal["asset-mania.face-geometry-plugin-result.v1"]
    plugin: Literal["mica-local", "deca-local"]
    profile: Literal["identity-neutral-v1", "detail-displacement-v1"]
    status: Literal["succeeded", "incompatible_runtime", "invalid_output", "execution_failed"]
    geometry: Path | None
    vertex_count: int
    triangle_count: int
    elapsed_seconds: float
    device: Literal["cuda"]
    checkpoint_sha256: str
    topology: Literal["flame-2020-5023"]
    ephemeral_identity_feature_used: bool
    persisted_identity_feature_count: Literal[0]
```

A successful plugin directory contains one numeric-only, pickle-disabled `geometry.npz` and
nothing else. It stores only:

- `vertices`: finite `float32`, shape `(5023, 3)`;
- `faces`: integer, shape `(9976, 3)`;
- `source_projection`: finite `float32`, shape `(5023, 2)`;
- MICA: `detail_displacement` is all zeros, shape `(5023,)`;
- DECA: `detail_displacement` is finite signed metres, shape `(5023,)`.

No embedding, landmark, crop, texture, albedo, source pixel, model parameter vector, or external
path is allowed in the inventory.

## Runtime isolation

Each adapter uses a user-supplied Python executable and source checkout in its own private runtime.
The public workspace does not import torch, MICA, DECA, InsightFace, face-alignment, or FLAME.

Before source open, the launcher verifies:

- exact Git revision;
- checkpoint SHA-256;
- user-supplied FLAME asset SHA-256;
- CUDA availability and device type;
- an empty create-only output directory;
- a valid plan-bound rights receipt;
- sanitized environment with provider and credential variables removed.

The worker monkey-patches Python HTTP clients and sockets to refuse egress, uses an isolated home
and cache tree, captures subprocess output, and rejects any source path or basename in output.

## Geometry normalization and fusion

The fusion package is original Apache-2.0 code and consumes only validated numeric plugin results.

1. Verify exact FLAME face indices and triangle bytes against the sealed user-supplied topology.
2. Fit a similarity transform from DECA coarse vertices to MICA vertices using the stable inner-face
   vertex set supplied by the sealed FLAME indices.
3. Transform DECA per-vertex displacement into MICA metric space.
4. Reject non-finite values, absolute displacement above `0.003` metres, or RMS displacement above
   `0.0015` metres; never clamp or silently repair them.
5. Apply displacement along MICA vertex normals only where the sealed FLAME face mask is positive.
6. Use a two-ring cosine taper at the face-mask boundary; skull, ears, neck, and eye shells receive
   zero DECA displacement.
7. Preserve MICA triangle topology and export glTF Y-up metres with a neutral material.

The phase emits three comparison meshes: `mica-clay.glb`, `deca-clay.glb`, and
`mica-deca-clay.glb`. The corrected-axis DAD GLB is an input baseline, not a fusion source.

## Automated gates

All automated gates fail closed:

- exactly 5,023 vertices and 9,976 triangles for FLAME outputs;
- identical face topology across MICA, DECA coarse, and sealed FLAME topology;
- finite positions, projections, normals, and displacement;
- no degenerate or non-manifold triangles;
- consistent winding and positive extent on every axis;
- longest-axis extent between `0.15` and `0.30` metres after metric normalization;
- absolute detail displacement at most `0.003` metres;
- RMS detail displacement at most `0.0015` metres;
- face-region displacement coverage at least `0.90`;
- zero displacement outside the tapered face region;
- source bytes unchanged before and after every stage;
- zero persisted identity features and zero unexpected plugin output files;
- embedded neutral material, no texture image, and no external GLB URI.

Front reprojection is diagnostic, not a likeness score. The fitted front projection reports
normalized mean landmark-free vertex reprojection only over the sealed face contour subset; it may
compare versions of this pipeline but may not be described as biometric accuracy.

## Manual clay verdict

Automated topology cannot prove likeness without a reference scan. A sealed private manual review
must therefore record `passed` or `failed` after identical Blender rendering.

`passed` requires:

- the MICA clay front is visibly closer to the authorized source's face proportions than corrected
  DAD clay;
- both three-quarter renders remain anatomically plausible;
- fusion preserves MICA jaw width, cheek volume, nose projection, and eye spacing;
- DECA detail improves local relief without changing identity proportions;
- no texture, hair, lighting trick, or generated side image is used to support the verdict.

Failure ends this program increment. It does not trigger DAD substitution, generated-view shape
fusion, model fallback, weaker thresholds, or head/hair/texture work.

## Private E2E stages

The create-only private runner exposes:

```text
geometry-plan    seal source, rights, topology, runtimes, revisions, weights, gates
mica-run         produce and validate MICA numeric geometry
deca-run         produce and validate DECA numeric geometry and displacement
geometry-fuse    align, taper, fuse, measure, and export three clay GLBs
geometry-verify  render identical Blender front and eight-view comparisons
geometry-review  write a separate sealed manual verdict
```

`geometry-fuse` also writes one sealed `likeness-disclosure-v1` for each clay GLB. A mesh and its
disclosure share the same source, plan, rights, engine/profile, and artifact digest and cannot be
published independently.

No stage overwrites a prior result. A changed model, source, gate, runtime, or workflow creates a
new attempt and preserves earlier evidence.

## Synthetic tests

Public tests use only analytic non-human FLAME-shaped fixtures and solid numeric arrays. They cover:

- closed request/result field inventories;
- rights receipt binding and network-denied execution;
- rejection of embeddings, landmarks, crops, textures, and extra files;
- exact topology and numeric-array validation;
- known similarity-transform recovery;
- displacement bounds and two-ring taper behavior;
- neutral GLB readback and glTF axis orientation;
- create-only stage order, source integrity, redaction, and no fallback;
- release, license, schema, skill, and publication boundaries.

No real face, face-derived hash, model weight, external source, or private geometry appears in a
tracked fixture.

## Completion boundary

This design is complete only when the synthetic suite and repository gates pass and one approved
private run records an honest clay verdict. A passing result authorizes planning the fixed head
assembly subproject; it does not authorize head, hair, Stable Diffusion, or publication claims by
itself.
