# Asset Mania DAD-3DHeads Face Plugin Design

**Status:** approved for implementation and the fixed acquisition plan on 2026-08-23

## Purpose

Replace the failed TripoSR voxel-based face experiment with one explicit, face-specific research
plugin and run a private end-to-end comparison on the already-authorized source portrait.

The selected experiment uses DAD-3DHeads to predict a FLAME-family full-head mesh from the single
observed front image. It does not use the seven generated yaw views for geometry. This isolates the
model change from the previous view-generation and voxel-fusion failures.

The experiment is non-commercial research only. It must not claim exact likeness, biometric
identity, commercial clearance, production topology, hair reconstruction, or a watertight scan.

## Evidence behind the change

The existing private face pipeline failed twice at the model/geometry level:

- eight-view TripoSR occupancy fusion produced a valid but unrecognizable blob;
- the observed-front-anchor visual hull passed every numeric gate but replaced facial detail with
  stepped bands during voxel resurfacing;
- three bounded debugging variants did not restore the face;
- the research record therefore closes further TripoSR voxel tuning and recommends a
  face-specific model family.

DAD-3DHeads is selected because its official demo directly exposes a `3d_mesh` OBJ output from one
image and predicts a fixed-topology full-head mesh. The pinned source uses a 256-pixel input, a
TorchScript checkpoint, and 300 shape plus 100 expression parameters.

Official evidence used for this design:

- source: <https://github.com/PinataFarms/DAD-3DHeads>
- pinned revision: `68cc9b51974e2628f7a8f8ed2dadc5f73b3f8aa7`
- checkpoint URL:
  <https://media.pinatafarm.com/public/research/dad-3dheads/dad_3dheads.trcd>
- checkpoint HTTP content length observed on 2026-08-23: `132711657` bytes
- upstream license: CC BY-NC-SA 4.0
- original environment: Python 3.8, PyTorch 1.9.0, torchvision 0.10.0

The upstream loader retries downloads and writes into the user's home directory. Asset Mania must
not call that path. Acquisition is a separate, one-attempt, explicitly approved stage that writes
only into ignored project storage and verifies the resulting digest before inference.

## Goals

- Add an experimental process boundary through which a maintainer E2E can select a named face
  reconstruction plugin without hard-coding TripoSR as permanent.
- Execute DAD-3DHeads from an exact pinned source revision and an exact locally hashed checkpoint.
- Keep the source portrait local and read-only; no face bytes leave the machine.
- Use the RTX 4070 through the already installed PyTorch `2.13.0+cu130` runtime when compatibility
  tests prove the pinned TorchScript model works there.
- Produce a private raw OBJ, normalized GLB, front-projected vertex-color GLB, four-view Blender
  preview, comparison image, and machine-readable report.
- Compare the DAD result with the failed TripoSR anchor and face-hybrid outputs using identical
  Blender camera and material settings.
- Record an honest manual visual verdict while keeping identity consistency `unmeasured`.
- Preserve the existing TripoSR packages and profiles as historical experimental paths; no silent
  fallback or destructive replacement is allowed.

## Non-goals

- No commercial use or commercial-readiness claim.
- No redistribution of DAD source, checkpoint, FLAME assets, compatibility checkout, or patched
  files in Asset Mania releases.
- No publication of the source portrait, generated views, masks, meshes, renders, local paths, or
  private reports.
- No dataset download, retraining, fine-tuning, identity embedding, face recognition, or biometric
  score.
- No additional GPT Image call, upload, remote inference, paid API, or paid compute.
- No SMIRK, MICA, DECA, EMOCA, or automatic alternative-model fallback in this experiment.
- No hair, eyelashes, teeth, tongue, realistic eye, or rear-head texture generation.
- No immediate migration of every existing engine to the new plugin boundary.
- No promise that PyTorch 2.13 compatibility will succeed. Failure ends the run and is reported.

## Considered approaches

### Selected: private out-of-tree DAD plugin behind a small process protocol

Keep the pinned DAD checkout, checkpoint, environment, and any compatibility patch under ignored
`.asset-mania/dad3dheads/` storage. A repository-owned maintainer runner exchanges a closed JSON
request/result with a plugin process and performs post-processing independently.

This gives the experiment a real replaceable boundary without importing CC BY-NC-SA code into the
Apache packages or pretending that the dependency is commercially cleared.

### Rejected: vendor DAD code or weights into the Apache workspace

The DAD repository declares CC BY-NC-SA 4.0, and the dependency closure includes FLAME-family
assets whose redistribution and commercial status must not be inferred from the top-level file.
Mixing those files into the Apache distribution would make the public license boundary misleading.

### Deferred: SMIRK or MICA

SMIRK and MICA are credible face-specific alternatives, but both add separate FLAME access,
checkpoint, license, and compatibility questions. DAD has a direct full-head OBJ demo and an
official ungated checkpoint URL, making it the narrower first experiment. Failure does not trigger
an automatic switch; another model requires another design and approval.

## Public and private boundaries

### Public tracked content

- a versioned experimental plugin request/result contract;
- a maintainer-only E2E orchestrator;
- a DAD plugin launcher that contains only original integration code and imports an external
  user-supplied checkout at runtime;
- synthetic fake-plugin tests and non-human mesh fixtures;
- license, privacy, provenance, and failed/successful experiment documentation;
- checks that reject DAD source files, checkpoint extensions, faces, and private outputs from Git.

### Private ignored content

```text
.asset-mania/dad3dheads/
  source/                 pinned upstream checkout
  checkpoint/             dad_3dheads.trcd and receipt
  venv/                   isolated compatibility runtime
  patches/                exact compatibility diff if required
  runs/<timestamp>/
    input/source.png      derived working copy; original remains read-only
    raw/head.obj
    converted/head.glb
    converted/head-colored.glb
    verification/report.json
    verification/preview.png
    verification/comparison.png
```

Every write is create-only. The runner rejects an existing stage or output rather than replacing
it. Public records contain only stable identifiers, hashes, measurements, license classification,
and coarse runtime facts; they contain no private basename or absolute path.

## License classification

DAD-3DHeads is treated as **source-available non-commercial research software**, not as a
permissive or commercially usable open-source engine. CC BY-NC-SA permits non-commercial sharing
under attribution and share-alike conditions, but Asset Mania does not redistribute the upstream
materials in this profile.

The checkpoint has no independently discovered license file at its download URL. The experiment
therefore applies the upstream repository's non-commercial research restriction conservatively
and marks checkpoint redistribution as uncleared.

The plugin boundary does not relicense DAD, its checkpoint, FLAME assets, generated meshes, or the
user's input. `THIRD_PARTY_NOTICES.md` may record the optional external dependency and restriction,
but no clearance artifact may label it Apache-compatible or commercially cleared.

## Approval and acquisition gate

The actual checkout and checkpoint download require one fresh approval after this written design
is accepted. Before approval, the runner may only print or serialize the acquisition plan.

The fixed acquisition plan is:

| Field | Value |
| --- | --- |
| Source provider | GitHub, `PinataFarms/DAD-3DHeads` |
| Source revision | `68cc9b51974e2628f7a8f8ed2dadc5f73b3f8aa7` |
| Checkpoint provider | `media.pinatafarm.com` official URL above |
| Expected checkpoint bytes | `132711657` |
| Network egress | no user file or derived face data; only standard GET requests |
| Cost | no charge reported; no paid compute |
| Destination | ignored `.asset-mania/dad3dheads/` only |
| Overwrite | forbidden |
| Retry | none |
| License | CC BY-NC-SA 4.0; non-commercial research; redistribution uncleared |

After acquisition, compute SHA-256 for the checkout archive or tree receipt and checkpoint. Seal
those digests into a private acquisition receipt before environment installation or inference.

## Plugin process protocol

The experimental protocol is a local subprocess boundary, not an in-process Python import in the
portable CLI. Its schema version is `asset-mania.face-plugin-request.v0`.

The request is private and contains:

```json
{
  "schema": "asset-mania.face-plugin-request.v0",
  "plugin": "dad3dheads-local",
  "plugin_revision": "68cc9b51974e2628f7a8f8ed2dadc5f73b3f8aa7",
  "source_image": "<private absolute path>",
  "output_directory": "<new private absolute path>",
  "device": "cuda",
  "checkpoint_sha256": "<verified digest>",
  "network": "denied-during-inference"
}
```

The result is private and contains:

```json
{
  "schema": "asset-mania.face-plugin-result.v0",
  "plugin": "dad3dheads-local",
  "status": "succeeded",
  "raw_mesh": "<private absolute path>",
  "projection_data": "<private absolute path>",
  "vertex_count": 0,
  "triangle_count": 0,
  "elapsed_seconds": 0.0,
  "device": "cuda",
  "checkpoint_sha256": "<verified digest>"
}
```

Allowed status values are `succeeded`, `incompatible_runtime`, `invalid_output`, and
`execution_failed`. Unknown fields, unexpected output files, a mismatched plugin/revision/digest,
or a success response with a missing mesh fail closed.

The orchestrator accepts an explicit plugin executable. It never searches PATH for a substitute
and never maps a DAD failure to TripoSR.

## DAD compatibility runtime

The upstream environment pins Python 3.8 and PyTorch 1.9.0, which does not match the installed CUDA
13 stack. The approved experiment intentionally tests a compatibility execution on the existing
PyTorch `2.13.0+cu130` runtime.

Compatibility work follows these limits:

1. Create a new isolated environment; do not mutate the verified TripoSR environments.
2. Install only the minimum inference dependency closure after recording exact resolved versions.
3. Pre-place the verified checkpoint in an isolated home directory so upstream auto-download code
   is never reached.
4. Load the pinned TorchScript checkpoint and run one synthetic/non-human smoke input first.
5. If an upstream API incompatibility occurs, make the smallest explicit patch in the private
   checkout, save its unified diff and SHA-256, and rerun the synthetic smoke test.
6. Do not change architecture, weights, input size, FLAME constants, or output topology.
7. Do not fall back to CPU, PyTorch 1.9, another CUDA wheel, or another model without a new plan and
   approval.

Inference runs with the working directory set to the pinned checkout and an isolated home/cache.
The runner verifies CUDA availability before loading the face image and again after inference.

## Input handling

The original portrait remains read-only. Before any run, record its SHA-256 and compare it after
the run. A normalized working copy named `source.png` is created inside the new private run only
after the output directory is reserved.

Normalization applies EXIF orientation, converts to sRGB RGB, preserves the full square frame, and
writes deterministic PNG bytes. It does not retouch, crop, reshape, beautify, or call a detector.
DAD's fixed longest-side resize and padding then produce the 256-square model input.

The existing rights-and-consent statement authorizes private local processing. It does not
authorize publication of the portrait or derived outputs.

## Geometry and appearance post-processing

The raw DAD OBJ remains preserved as model evidence. A separate deterministic stage:

1. parses finite vertices and integer triangle indices;
2. rejects empty, degenerate, non-finite, or out-of-range geometry;
3. records connected components, boundary edges, winding consistency, signed volume when defined,
   bounds, vertex count, and triangle count;
4. normalizes by bounds centre and longest extent without changing proportions;
5. rotates only according to a fixed, synthetic-tested DAD-to-Blender axis transform;
6. exports a create-only GLB;
7. samples the observed source image at DAD's predicted projected-vertex coordinates for vertices
   that are front-visible and inside the image;
8. assigns neutral skin-gray color to unobserved vertices and exports a second GLB with vertex
   colors.

The front projection improves reviewability but is not a full texture reconstruction. It must not
stretch the observed face onto the rear head or infer hair. The report records observed-color
coverage separately from geometry metrics.

The converter does not require watertightness because FLAME-family head topology may contain a
neck boundary. It reports boundary loops and fails only on unexpected fragmentation, non-manifold
edges, inverted/zero-area majority, or invalid geometry.

## Maintainer E2E

`scripts/run_face_plugin_e2e.py` exposes these create-only stages:

```text
plan       seal plugin, revision, license, acquisition, runtime, and output plan
acquire    fetch the exact source/checkpoint after fresh approval and write hashes
smoke      prove the compatibility runtime on a synthetic non-human input
run        process the private observed portrait through the selected plugin
convert    validate OBJ and emit plain plus front-colored GLBs
verify     independently recompute metrics and render the Blender comparison
```

The public deterministic E2E injects a fake plugin executable that writes a small asymmetric
synthetic head mesh. It proves request/result validation, create-only outputs, no fallback,
redaction, source-integrity checks, conversion, and failure propagation without network, CUDA,
models, or real-person fixtures.

The private live E2E selects exactly `dad3dheads-local`. It may start only after the acquisition
receipt, synthetic smoke test, CUDA check, and source-integrity precheck all pass.

## Verification and acceptance

### Deterministic public gates

- malformed or mismatched plugin results fail closed;
- an absent executable does not trigger fallback;
- source and output collision tests prove create-only behavior;
- the fake OBJ converts to a readable GLB with the expected axis orientation;
- public reports contain no absolute path or source basename;
- release/publication checks reject DAD code, checkpoint, private mesh, private render, and face
  fixture patterns;
- current TripoSR tests remain unchanged and green at their supported layers.

### Private runtime gates

- pinned revision and checkpoint byte count match the approved plan;
- recorded SHA-256 values match every consumed external artifact;
- the synthetic smoke uses CUDA and produces a valid mesh;
- the face run makes no network request and uses the same checkpoint digest;
- source bytes are unchanged before and after;
- raw OBJ and both GLBs pass independent structural validation;
- Blender renders front, right, rear, and left views with identical comparison settings.

### Manual visual verdict

Numeric geometry validity is necessary but not sufficient. The private report marks
`visual_quality=passed` only when all of the following are visible in the DAD result:

- a coherent human head rather than the prior plate/blob or stepped-band surface;
- nose, lips, chin, cheek, and eye-socket relief visible from front and three-quarter views;
- no exploded triangles, severe axis inversion, or face projected onto the rear;
- a clear qualitative improvement over both prior private TripoSR outputs.

Hair and rear appearance are excluded from the pass criterion. `identity_consistency` remains
`unmeasured` regardless of the visual verdict.

## Failure handling

Stop the run without retry or substitution when:

- source or checkpoint acquisition differs from the approved URL, revision, length, or digest;
- the isolated runtime cannot import the minimum dependency closure;
- TorchScript loading or CUDA execution is incompatible with PyTorch 2.13;
- the plugin attempts an unapproved network request or writes outside its reserved directory;
- request/result fields or output inventory differ from the closed protocol;
- geometry validation, source-integrity validation, GLB readback, or Blender rendering fails;
- the output is geometrically valid but visually worse or not meaningfully better.

An incompatibility is reported as an experiment result, not repaired through a hidden PyTorch,
model, or device change.

## Documentation outcome

Before the private run, public documentation may say only that the DAD plugin experiment is
designed and not yet executed. After the run, `docs/research.md` records exact runtime and
structural measurements plus the manual visual verdict. README capability wording changes only if
the public fake-plugin E2E and the private live evidence both support the claim.

No release artifact includes the plugin dependency or private output. A future commercially usable
face plugin requires a separately cleared model, weights, training-data terms, dependencies, and
redistribution evidence.
