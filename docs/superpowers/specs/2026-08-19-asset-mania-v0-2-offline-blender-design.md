# Asset Mania v0.2 Offline Blender Pipeline Design

**Date:** 2026-08-19

**Status:** Approved for implementation

**Repository:** `jsc7727/asset-mania`

**Target:** the first real, end-to-end local production slice

## Outcome

v0.2 turns the v0.1 inspection foundation into a real, source-preserving Blender workflow:

1. Load a Blender scene with scripts and network access disabled.
2. Evaluate one explicit camera, frame, and optional rig/action pose.
3. Produce a conditioning bundle containing beauty, depth, normal, mask, and camera data.
4. Accept a user-supplied image aligned to that conditioning view.
5. Reproject the image into an existing target UV atlas, bake a base-color texture, and retain
   uncovered texels as explicitly unknown.
6. Save a derived `.blend`, export `.glb` and `.fbx`, and validate each artifact in a fresh
   Blender process.

This local round trip is provider-independent. A later v0.2 milestone may obtain the aligned
image through GPT Image 2, but the same view contract is used whether the image came from a user,
a fake test provider, or an approved remote provider.

## Binding Decisions

- `manifest-v1.schema.json` remains byte-compatible and inspect-only.
- Execution uses new, closed schemas: manifest v2, workflow plan v1, conditioning bundle v1,
  view v1, worker response v1, provider-evidence v1, provider-plan v1, and approval-receipt v1.
- The first verified Blender profile is exact Blender `5.2.0 LTS`. Every run records the full
  Blender version, build hash, platform, and executable fingerprint. Other versions fail
  preflight until an explicit compatibility profile and E2E evidence are added.
- Conditioning and baking use Cycles on CPU, one thread, fixed samples and seed, with adaptive
  sampling, denoise, motion blur, depth of field, and animated seed disabled.
- Production inputs must already contain a non-overlapping UV map in the 0..1 range. v0.2 never
  silently unwraps or repacks a user's mesh.
- The first acceptance path supports one scene, one explicit camera, one frame, one target mesh,
  and zero or one armature/action. Static props and posed characters both remain valid.
- The first view path requires exact resolution/aspect/camera alignment. Resizing, cropping,
  homography estimation, camera estimation, topology changes, or generative hole filling are not
  implicit fallbacks.
- Source `.blend` and image files are opened read-only and remain byte-identical. Every output is
  created below a new atomic run directory. Existing runs and sources are never overwritten.
- `.blend` is the editable authoring artifact. GLB is the primary runtime artifact. FBX is a
  compatibility export with an explicitly narrower validation claim.
- Face reconstruction remains outside this milestone. Generic local texturing of a user-supplied
  face/head mesh is allowed only as a non-inferential workflow with a user-declared asset kind and
  subject category. `real_person` requires a rights/consent receipt bound to the workflow plan;
  `unknown` is blocked until the user supplies a valid declaration. No face inference, identity
  embedding, landmark model, or real-person fixture is bundled or executed.

## Command Surface

The existing command remains unchanged:

```text
asset-mania inspect <input> ...
```

New local commands are stage-oriented:

```text
asset-mania scene preflight SOURCE.blend [--out RUNS]

asset-mania scene plan PREFLIGHT_MANIFEST
    --camera CAMERA_NAME
    --frame FRAME
    --target TARGET_NAME
    --asset-kind object|character|face-head
    --subject non-person|synthetic-person|real-person|unknown
    [--armature ARMATURE_NAME]
    [--action ACTION_NAME]
    [--resolution WIDTHxHEIGHT]
    [--out RUNS]

asset-mania scene condition SOURCE.blend
    --plan WORKFLOW_PLAN
    [--rights-receipt RIGHTS_RECEIPT]
    [--blender BLENDER_EXECUTABLE]
    [--out RUNS]

asset-mania view ingest VIEW_IMAGE
    --condition-manifest CONDITION_MANIFEST
    --origin observed|generated|unknown
    [--alignment-acknowledgement CONDITION_SHA256:VIEW_SHA256]
    [--out RUNS]

asset-mania texture bake
    --condition-manifest CONDITION_MANIFEST
    --view-manifest VIEW_MANIFEST
    [--blender BLENDER_EXECUTABLE]
    [--out RUNS]

asset-mania export BAKE_MANIFEST
    --format blend
    [--format glb]
    [--format fbx]
    [--blender BLENDER_EXECUTABLE]
    [--out RUNS]

asset-mania approval issue PLAN
    --gate face-rights|external-egress|paid-compute
    [--expires-in 30m]
    [--acknowledgement GATE:FULL_PLAN_SHA256]

asset-mania provider evidence refresh openai --out EVIDENCE

asset-mania view provider-plan CONDITION_MANIFEST
    --provider openai
    --model gpt-image-2-2026-04-21
    --prompt-file PRIVATE_PROMPT_FILE
    --evidence PROVIDER_EVIDENCE
    [provider output controls]
    [--out RUNS]

asset-mania view generate PROVIDER_PLAN
    --prompt-file PRIVATE_PROMPT_FILE
    --approval EXTERNAL_EGRESS_RECEIPT
    --approval PAID_COMPUTE_RECEIPT
    [--approval FACE_RIGHTS_RECEIPT]
    [--out RUNS]
```

Actual Blender datablock names are private execution inputs. Portable outputs replace them with
deterministic labels such as `camera-1`, `mesh-1`, `armature-1`, and `action-1`.

No command accepts a global `--yes`. Later provider approvals bind to one immutable plan digest.
`approval issue` displays the exact gate disclosure and, by default, requires the user to type the
full plan-bound acknowledgement interactively. Non-interactive automation must provide the exact
`GATE:FULL_PLAN_SHA256` string; a boolean flag cannot issue a receipt. Rights issuance additionally
shows the assertion that the user owns the material or has the depicted person's permission. The
receipt records that scoped user assertion, not the person's identity or a legal-consent signature.

CLI spellings use kebab case and normalize exactly at the parser boundary: `face-head` maps to
`face_head`, `non-person` to `non_person`, `synthetic-person` to `synthetic_person`, `real-person` to
`real_person`, `face-rights` to `face_rights`, `external-egress` to `external_egress`, and
`paid-compute` to `paid_compute`. Portable JSON accepts only the underscore forms.

A `scene-plan` run for `real_person` publishes the immutable plan with status `needs_approval` and
exit `5`. `scene condition` validates and atomically consumes the matching `face_rights` receipt
before it opens the source file. `unknown` is rejected before a plan is executable.

The successful condition manifest becomes the downstream local rights basis. View ingest, bake, and
export verify and inherit that immutable approval lineage instead of reusing a single-use receipt.
A later external provider plan is a new scope and requires a new `face_rights` receipt bound to the
provider-plan digest.

`view ingest` hashes the fully decoded source before accepting alignment. By default it displays the
condition and view digests and requires the user to type `CONDITION_SHA256:VIEW_SHA256` to attest
that the view was produced for that camera/framing. Non-interactive use must pass that exact string
through `--alignment-acknowledgement`; there is no boolean shortcut. The canonical declaration
object includes both digests, issuer `user`, statement `declared_aligned`, and issuance timestamp;
its hash is `alignment_attestation_sha256`. Provider generation creates the equivalent declaration
with issuer `provider` from the approved provider plan and response.

`view provider-plan` hashes the private prompt but never persists its text, binds the fixed edit
endpoint/attachments/model/evidence/controls, then exits `5` with exact missing gates. `view
generate` rereads the private prompt file and requires its digest to equal the approved plan before
it consumes every required receipt, accesses credentials, or reaches transport. A successful
provider response passes through the same internal view normalization/validation service and emits
`view-v1` directly with origin `generated` and provider-issued declared-alignment attestation; it
does not require a second public ingest run.

## Immutable Stage Graph

```text
inspect-v1 (unchanged)

scene-preflight
      |
      v
scene-plan --> [approval-issue when required] --> condition
                                                  |       \
                                                  |        -> view-ingest
provider-evidence --> provider-plan --> approval-issue(s) -> provider-generate
                                                                   |
                                                     view-v1 ------+--> bake --> export
```

Each stage creates a new run directory. A child references a parent by:

- parent run ID;
- parent manifest SHA-256;
- relationship;
- exact consumed artifact SHA-256 values.

A stage never mutates a parent run. Any mismatch between an expected parent digest and the file on
disk fails before Blender or any provider is invoked.

## Repository Architecture

```text
packages/contracts        Apache-2.0 schemas, diagnostics, and canonical builders
packages/pipeline         Apache-2.0 plans, lineage, stage store, view contract, validation
packages/blender-client   Apache-2.0 discovery, private envelopes, launch, response validation
packages/cli              Apache-2.0 argparse and stream mapping
packages/provider-openai  Apache-2.0 optional BYOK adapter in a later milestone

blender-addon/            GPL-3.0-or-later worker containing every bpy/mathutils import
```

Dependency direction:

```text
contracts <- pipeline <- blender-client <- cli
                 ^
                 +---- provider ports

Apache process -- JSON/files/subprocess --> GPL Blender process
```

The GPL worker never imports an `asset_mania_*` Apache package. Apache packages never import the
GPL worker. Root uv packaging does not include `blender-addon/` in Apache wheels. Release checks
verify the boundary and build the GPL worker as a distinct archive with its own license.

Production reprojection and baking have one owner: the GPL Blender worker. It has access to the
evaluated mesh, loop triangles, UVs, pose, camera, render passes, ray casting, and Blender image
data without transporting private geometry into an Apache process. Apache projection/reprojection
code is a small reference oracle for analytic test vectors and contract validation only; it is not
the production renderer and does not decode EXR. The worker reads EXR/image pixels through Blender,
creates the reprojection atlas and coverage, then performs a distinct emission bake into a separate
delivery image datablock. That second step is intentionally material consolidation for the derived
scene, not a second competing reprojection implementation.

## Execution Contracts

### Manifest v2

Manifest v2 is additive as a new schema, not a migration of v1. It contains:

- `schema_version: "2.0"`;
- `command` and closed `stage` enum;
- run/tool timestamps and versions;
- role-labeled, hashed inputs without basenames or absolute paths;
- parent manifest relationships;
- resolved parameters and plan digest;
- execution capabilities and exact Blender profile;
- artifact lineage, validation, sensitivity, upload eligibility, and content origin;
- approval receipt digests where applicable;
- terminal result and sorted diagnostic codes.

Artifacts add these fields to the v1 concepts:

- `role`;
- `parents` as artifact hashes;
- `operation`;
- `sensitivity`: `portable`, `user-content`, or `local-sensitive`;
- `upload_eligible` boolean;
- `content_origin`: `observed`, `derived`, `generated`, or `unknown`;
- optional semantic digest and validation profile.

Derived processing does not erase upstream provenance. Reprojecting a generated image produces a
generated texture; it does not become merely derived.

### Workflow plan v1

The immutable plan includes:

- source hash and preflight-manifest hash;
- private selections by digest plus their portable labels;
- frame, action evaluation, resolution, pixel aspect, units, and color settings;
- user-declared asset kind and subject category, never inferred from pixels or geometry;
- engine, device, threads, samples, seed, and disabled nondeterministic features;
- expected artifacts and validation profiles;
- source/output overwrite policy;
- exact compatible Blender profile;
- canonical plan digest.

Portable plan files contain no absolute paths, source basename, prompt text, credentials, signed
URLs, or unredacted datablock names.

### Normative v2 contract summary

Task 2 implements these exact closed enums and required identities before stage code may proceed:

| Field | Required values or shape |
| --- | --- |
| `stage` | `scene-preflight`, `scene-plan`, `provider-evidence`, `provider-plan`, `approval-issue`, `condition`, `view-ingest`, `provider-generate`, `bake`, `export` |
| `result.status` | `succeeded`, `failed`, `unsupported`, `needs_approval`, `canceled` |
| provider state | `planned` -> `approval_required` -> `approved` -> `submitted` -> `running` -> one terminal state |
| terminal provider state | `succeeded`, `failed`, or `canceled`; no transition leaves a terminal state |
| parent identity | `run_id`, `manifest_sha256`, `relationship`; all three required |
| artifact identity | `role`, `path`, `sha256`, `byte_size`, `media_type`, `parents`, `operation`, `content_origin`, `sensitivity`, `upload_eligible`, `validation` |
| content origin | `observed`, `derived`, `generated`, `unknown` |
| sensitivity | `portable`, `user-content`, `local-sensitive` |
| subject | `non_person`, `synthetic_person`, `real_person`, `unknown` |
| approval gate | `face_rights`, `external_egress`, `paid_compute` |
| approval scope | only `single_run` |

Each stage parameter object is a JSON Schema `oneOf` selected by `stage`; parameters for a different
stage are invalid rather than ignored. A minimal parent reference is:

```json
{
  "run_id": "run-example-1",
  "manifest_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "relationship": "conditioned_from"
}
```

Every canonical digest is SHA-256 over UTF-8 canonical JSON with sorted keys, no insignificant
whitespace, no omitted default, and one trailing newline. The digest preimage includes a schema
identifier and every field of the logical object except the digest field itself. Timestamps and
opaque IDs are part of run manifests, but immutable workflow/provider plan digests contain neither
implicit current time nor an automatically generated run ID.

Selection resolution uses a local-sensitive `selection-map.json`. It contains a random per-plan
salt plus the private Blender identifiers and is never upload-eligible. The portable plan contains
only stable labels and `selection_digest = HMAC-SHA256(salt, canonical selection identity)`. The
worker recomputes it after opening the exact source hash; a label, name, library identity, source
hash, or target-type change fails with `PLAN_TAMPERED`. Portable labels are deterministic within the
source inventory, while the salted selection digest intentionally is not usable to guess names.

An approval receipt requires `receipt_id`, `plan_sha256`, one gate, `issued_at`, `expires_at`,
`scope: "single_run"`, disclosure digest, and issuer type. It contains no person name or free-form
note. The stage store atomically writes an approval-consumption record before transport or protected
local face processing. A consumed, expired, wrong-gate, wrong-plan, or copied-within-the-same-store
receipt is rejected. This is accidental-scope protection, not a globally enforceable signature;
copying a run store is outside its guarantee.

`unknown` is rejected with `SUBJECT_DECLARATION_REQUIRED` before receipt issuance, local execution,
or provider planning. It is never a receipt-gated executable state. Only `real_person` can request
a `face_rights` receipt.

v2 CLI exit codes are fixed:

| Exit | Meaning |
| --- | --- |
| `0` | stage succeeded and a valid terminal manifest was published |
| `2` | CLI usage failed before a run was created |
| `3` | input/contract/unsupported failure after run creation |
| `4` | sanitized internal or worker failure after run creation |
| `5` | valid run stopped at `needs_approval` |
| `6` | valid run reached `canceled` |
| `73` | storage bootstrap/publication failed; no manifest is promised |

As in v1, exit 73 uses the stderr-only storage diagnostic. Normal stdout/stderr never contains raw
worker output, a private selection, or a local path.

### Closed schema field maps

These maps are normative. Task 2 translates them to JSON Schema and commits one valid example for
each row; it does not choose new public fields. Every object is closed. `sha256` means 64 lowercase
hex characters, `timestamp` means RFC 3339 UTC, and `portable_id` matches
`^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$`. `relative_path` forbids absolute paths, drive prefixes, `..`,
empty segments, and backslashes. Arrays reject duplicates and use the ordering named below.

#### `manifest-v2`

| Field | Type and rule |
| --- | --- |
| `schema_id` | constant `asset-mania/run-manifest` |
| `schema_version` | constant `2.0` |
| `run_id` | `portable_id` |
| `command` | `scene.preflight`, `scene.plan`, `provider.evidence.refresh`, `view.provider-plan`, `approval.issue`, `scene.condition`, `view.ingest`, `view.generate`, `texture.bake`, or `export` matching `stage` |
| `stage` | closed stage enum defined above |
| `tool_version`, `created_at` | semver string; timestamp |
| `inputs` | ordered array of input records |
| `parents` | ordered by relationship then run ID; parent records |
| `parameters` | exactly one stage parameter shape below |
| `plan_sha256` | sha256 or JSON `null` for `scene-preflight` and explicit `provider-evidence` only |
| `environment` | exact environment record below |
| `capabilities` | exact capability record below |
| `approvals` | ordered by gate; zero or more consumption records |
| `artifacts` | ordered by relative path; zero or more artifact records |
| `result` | exact result record below |
| `warnings` | sorted unique diagnostic-code strings |

Input record: `label` (`input-N`), `role` (`source_scene`, `source_view`, `parent_manifest`,
`workflow_plan`, `provider_evidence`, `provider_plan`, `approval_receipt`), `sha256`, `byte_size`
(nonnegative integer), `media_type`, `content_origin`, and `sensitivity`.

Parent record: `run_id`, `manifest_sha256`, and relationship from `planned_from`,
`evidenced_from`, `approved_by`, `conditioned_from`, `view_from`, `generated_from`, `baked_from`, or
`exported_from`.

Artifact record: `role`, `path`, `sha256`, `byte_size`, `media_type`, `parents` ordered by hash,
`operation`, `content_origin`, `sensitivity`, `upload_eligible`, and `validation`. Each artifact parent
is `{ "sha256": sha256, "relationship": "consumed" | "derived_from" | "generated_from" }`.
Validation is `{ "profile": nonempty ASCII, "status": "valid" | "invalid" | "incomplete" |
"unvalidated", "diagnostics": sorted unique codes, "semantic_digest": sha256 | null }`.

Environment is `{ "operating_system": "macos" | "linux", "architecture": "arm64" | "x86_64",
"python_version": x.y.z, "blender": null | { "profile": "blender-5.2.0-cpu-v1",
"version": "5.2.0", "build_hash": nonempty ASCII, "executable_sha256": sha256 } }`.

Capabilities is `{ "network": "denied" | "approval_gated" | "explicit_official_hosts", "filesystem":
"source_read_staging_write", "blender": "found" | "not_found" | "not_required", "provider":
"not_required" | "not_installed" | "approval_gated" | "available" }`.
Only the explicit `provider-evidence` command may use `explicit_official_hosts`, with the exact hosts
recorded in its parameters; every other preapproval stage is denied or approval-gated.

Approval consumption is `{ "gate": gate, "receipt_sha256": sha256, "consumption_id": portable_id,
"consumed_at": timestamp }`. Result is `{ "status": result status, "diagnostics": sorted unique
codes, "provider_state": provider state | null }`.

Stage parameter `oneOf` shapes are exact:

| Stage | Required parameters |
| --- | --- |
| `scene-preflight` | `source_input_label`, `profile` |
| `scene-plan` | `preflight_manifest_sha256`, `asset_kind`, `subject`, `camera_label`, `target_label`, `armature_label` nullable, `action_label` nullable, `frame`, `action_range` nullable `[start,end]`, `resolution` `[width,height]`, `profile` |
| `provider-evidence` | `provider`, sorted official `source_hosts`, `network_action: explicit` |
| `provider-plan` | `condition_manifest_sha256`, `evidence_sha256`, `prompt_sha256`, closed `controls` |
| `approval-issue` | `plan_sha256`, `gate`, `expires_at`, `acknowledgement_digest` |
| `condition` | `workflow_plan_sha256`, `rights_receipt_sha256` nullable |
| `view-ingest` | `condition_manifest_sha256`, `origin`, `alignment_attestation_sha256`, `rights_basis_manifest_sha256` nullable |
| `provider-generate` | `provider_plan_sha256`, `approval_receipt_sha256s` ordered by gate |
| `bake` | `condition_manifest_sha256`, `view_manifest_sha256`, `atlas_size`, `bake_margin`, `color_padding`, `minimum_coverage`, `profile` |
| `export` | `bake_manifest_sha256`, sorted unique `formats`, `profile` |

`asset_kind` is `object`, `character`, or `face_head`. Integer frames/action ranges are bounded to
Blender's supported range and the condition frame must lie within a non-null range. Width/height and
atlas dimensions are positive integers no greater than 4096 in this profile.

#### `workflow-plan-v1`

Required fields are: `schema_id: asset-mania/workflow-plan`, `schema_version: 1.0`,
`source_scene_sha256`, `preflight_manifest_sha256`, `selection` (portable camera/target/nullable
armature/nullable action labels plus `selection_digest`), `asset_kind`, `subject`, `frame`, nullable
`action_range`, `resolution`, `pixel_aspect: [1.0,1.0]`, `blender_profile`, `render_profile`,
`expected_artifact_roles` in stable order, `overwrite_policy: create_only`, and `plan_sha256`.
`render_profile` contains every binding table value rather than relying on Blender defaults.

`selection` is exactly `{ "camera_label": portable label, "target_label": portable label,
"armature_label": portable label | null, "action_label": portable label | null,
"selection_digest": sha256 }`. `expected_artifact_roles` is the ordered list
`conditioning_bundle`, `beauty_exr`, `beauty_preview`, `depth_exr`, `depth_preview`, `normal_exr`,
`normal_preview`, `object_index_exr`, `mask_png`, `scene_state_blend`.

`render_profile` is exactly:

```json
{
  "profile_id": "blender-5.2.0-cpu-v1",
  "blender_version": "5.2.0",
  "engine": "CYCLES",
  "device": "CPU",
  "threads": 1,
  "samples": 16,
  "seed": 0,
  "adaptive_sampling": false,
  "denoise": false,
  "animated_seed": false,
  "motion_blur": false,
  "depth_of_field": false,
  "render_border": false,
  "crop_to_border": false,
  "film_transparent": true,
  "pixel_aspect": [1.0, 1.0],
  "working_color_space": "scene_linear",
  "preview_color_space": "srgb",
  "target_object_index": 1,
  "pass_alpha_threshold": 0.5,
  "atlas_size": [1024, 1024],
  "bake_margin": 16,
  "color_padding": 8,
  "minimum_coverage": 0.15,
  "depth_absolute_tolerance_meters": 0.0001,
  "depth_relative_tolerance": 0.0002,
  "ray_epsilon_scale": 0.0000001,
  "ray_epsilon_min_meters": 0.0000001,
  "ray_epsilon_max_meters": 0.001,
  "matrix_decimal_places": 9,
  "worker_timeout_seconds": 300,
  "worker_response_max_bytes": 1048576,
  "dependency_policy": "packed_only",
  "unknown_texels": "transparent",
  "animation_profile": "selected_action_range_or_none"
}
```

The fixture workflow changes top-level resolution to `[64,64]`; its render profile changes only
`profile_id` to `blender-5.2.0-cpu-v1-fixture`, atlas size to `[64,64]`, bake margin to `2`, color
padding to `1`, and minimum coverage to `0.25`. All other keys and values remain identical.

#### `provider-evidence-v1`

Required fields are: `schema_id: asset-mania/provider-evidence`, `schema_version: 1.0`,
`provider: openai`, `model_snapshot: gpt-image-2-2026-04-21`, URL-sorted `sources`, aggregate
`retrieved_at`, `expires_at` exactly 24 hours later, aggregate `content_digest`, `data_policy`,
`pricing`, and `evidence_sha256`. This command's manifest records the explicit network action; the
evidence file contains no credential or account usage.
Allowed source hosts are exactly `developers.openai.com` and `platform.openai.com` in this profile.

`sources` is a URL-sorted array of `{ "url": exact HTTPS URL, "retrieved_at": timestamp,
"content_sha256": sha256, "parser_profile": "openai-gpt-image-2-v1" }`. `data_policy` is exactly
`{ "training_default": "not_used_unless_opted_in", "application_state": "none",
"abuse_monitoring_days": 30, "zdr": "eligible_with_approval", "csam_review_exception": true,
"effective_region": "unknown" | documented region }`.

`pricing` is exactly:

```json
{
  "currency": "USD",
  "rate_mode": "standard",
  "per_million_tokens": {
    "text_input": "5.000000",
    "cached_text_input": "1.250000",
    "image_input": "8.000000",
    "cached_image_input": "2.000000",
    "image_output": "30.000000"
  },
  "output_cost_rows": [
    {"quality": "low", "size": "1024x1024", "usd": "0.006000"},
    {"quality": "low", "size": "1024x1536", "usd": "0.005000"},
    {"quality": "low", "size": "1536x1024", "usd": "0.005000"},
    {"quality": "medium", "size": "1024x1024", "usd": "0.053000"},
    {"quality": "medium", "size": "1024x1536", "usd": "0.041000"},
    {"quality": "medium", "size": "1536x1024", "usd": "0.041000"},
    {"quality": "high", "size": "1024x1024", "usd": "0.211000"},
    {"quality": "high", "size": "1024x1536", "usd": "0.165000"},
    {"quality": "high", "size": "1536x1024", "usd": "0.165000"}
  ],
  "retrieved_at": "RFC3339 UTC timestamp",
  "content_sha256": "pricing-page sha256"
}
```

Rows are canonical in quality order `low`, `medium`, `high`, then size order `1024x1024`,
`1024x1536`, `1536x1024`; duplicate or reordered semantic rows are rejected by the builder. Decimal
strings match `^(0|[1-9][0-9]*)\.[0-9]{6}$`. The concrete numbers shown above are the design-date
example, not JSON Schema constants: only keys, currency/mode, row identities/order, and decimal
format are schema constants. Values must match the fresh official source parser; a later price
change creates a new evidence digest and invalidates old plans rather than changing the schema or an
existing artifact.

#### `provider-plan-v1`

Required fields are: `schema_id: asset-mania/provider-plan`, `schema_version: 1.0`,
`condition_manifest_sha256`, `provider: openai`, `endpoint: /v1/images/edits`,
`model: gpt-image-2-2026-04-21`, attachments ordered `beauty`, `depth_preview`, `normal_preview`,
`mask` (each with `multipart_field: image[]`, index `0..3`, sha256, byte size, media type, and upload
eligibility; optional API `mask` is absent), `prompt_sha256`, closed `controls`,
`subject`, `policy_evidence`, `cost_estimate`, `expected_view`, `required_gates` in gate order,
`overwrite_policy: create_only`, and `plan_sha256`.

Each attachment is `{ "role": fixed role, "multipart_field": "image[]", "index": 0..3,
"sha256": sha256, "byte_size": positive integer, "media_type": "image/png",
"upload_eligible": true }`. `expected_view` is `{ "count": 1, "width": positive integer,
"height": positive integer, "media_type": "image/png" | "image/jpeg" | "image/webp",
"origin": "generated", "alignment_issuer": "provider" }` and must match controls. Required gates
are `["external_egress","paid_compute"]` for non-person/synthetic input and
`["face_rights","external_egress","paid_compute"]` for real-person input; unknown is invalid.

`controls` is `{ "n": 1, "size": "WIDTHxHEIGHT", "quality": "low" | "medium" | "high",
"background": "auto" | "opaque", "output_format": "png" |
"jpeg" | "webp", "output_compression": integer 0..100 | null, "moderation": "auto" | "low" }`.
Size must equal the conditioning bundle resolution and be one of the official cost-table sizes in
this estimator profile: `1024x1024`, `1024x1536`, or `1536x1024`. Other API-valid custom sizes are
not executable because the official docs do not provide a stable arbitrary-dimension token/cost
formula; they fail before provider planning. Compression is null for PNG and required/allowed only
for JPEG/WebP.

`policy_evidence` is `{ "artifact_sha256": sha256, "source_urls": nonempty sorted HTTPS official
URLs, "retrieved_at": timestamp, "expires_at": timestamp, "content_digest": sha256,
"training_default": "not_used_unless_opted_in", "application_state": "none",
"abuse_monitoring_days": 30, "zdr": "eligible_with_approval", "csam_review_exception": true,
"effective_region": nonempty string | "unknown" }`.

`cost_estimate` is exactly `{ "currency": "USD", "rate_retrieved_at": timestamp,
"rate_digest": sha256, "text_input_tokens_assumed": nonnegative integer,
"image_input_tokens_assumed": nonnegative integer, "cached_text_input_tokens_assumed": 0,
"cached_image_input_tokens_assumed": 0, "n": 1, "size": control size,
"quality": control quality, "formula": "uncached_inputs_plus_published_output_row_v1",
"rounding": "ceiling_6_decimal_places", "estimate_uncertainty": "input_tokens_assumed",
"estimated_cost": six-decimal nonnegative string, "maximum_cost": six-decimal positive string }`.
`expected_view` fixes output count, dimensions, format, origin `generated`, and declared-alignment
issuer `provider`.

The preflight formula uses exact decimal arithmetic:

```text
input = text_input_tokens_assumed * text_input_rate / 1_000_000
      + image_input_tokens_assumed * image_input_rate / 1_000_000
output = the one published output_cost_rows USD value for (quality, size) * n
estimated_cost = ceil_to_0.000001(input + output)
```

Cached assumptions are fixed to zero, so cached rates are recorded for evidence completeness but
never used or double-counted in this profile. The `image_output` token rate is likewise recorded and
used only to audit returned provider usage; the published size/quality output-cost row is the sole
preflight output authority. `maximum_cost` must be at least `estimated_cost` and is an approval-bound
ceiling over stated assumptions, not a guarantee of the provider's eventual bill. Returned token
usage and actual cost are recorded separately after the one permitted call.

#### `approval-receipt-v1`

Required fields are: `schema_id: asset-mania/approval-receipt`, `schema_version: 1.0`, `receipt_id`,
`plan_sha256`, `gate`, `issued_at`, `expires_at`, `scope: single_run`, `disclosure_digest`,
`issuer_type: user`, `acknowledgement_digest`, and `receipt_sha256`. Every gate is a scoped user
decision; a maintainer or provider cannot issue approval on the user's behalf. `receipt_id` is a
`portable_id`. There is no free-form note or person identifier.

#### `conditioning-bundle-v1`

Required fields are: `schema_id: asset-mania/conditioning-bundle`, `schema_version: 1.0`, source and
evaluated geometry/UV/pose digests, portable target/camera/nullable armature/nullable action labels,
frame, resolution, pixel aspect, pixel origin, world/camera axis records, four row-major 4x4 matrices,
camera projection/lens/sensor/shift/clip fields, scene unit scale, depth semantics/unit/range, normal
space/channel convention, mask/index semantics, complete binding render profile, Blender fingerprint,
and pass artifacts ordered beauty/depth/normal/object-index/mask with relative paths and hashes.

The closed nested shapes are:

- `digests`: `{ "source_scene": sha256, "evaluated_geometry": sha256, "uv": sha256,
  "pose": sha256 }`.
- `selection`: the same portable-label object as workflow-plan v1, without private identifiers.
- `axes`: `{ "world": { "handedness": "right", "up": "+Z", "forward": "-Y" },
  "camera": { "right": "+X", "up": "+Y", "view": "-Z" } }`.
- `matrices`: `{ "layout": "row_major", "camera_to_world": 16 finite numbers,
  "world_to_camera": 16 finite numbers, "projection": 16 finite numbers,
  "world_to_clip": 16 finite numbers }`, quantized to the profile decimals.
- `camera`: `{ "projection_type": "perspective" | "orthographic", "lens_mm": positive number |
  null, "sensor_fit": "AUTO" | "HORIZONTAL" | "VERTICAL", "sensor_width_mm": positive number,
  "sensor_height_mm": positive number, "shift_x": finite number, "shift_y": finite number,
  "clip_start_meters": positive number, "clip_end_meters": greater number, "ortho_scale": positive
  number | null }`; lens is required only for perspective and ortho scale only for orthographic.
- `depth`: `{ "space": "camera_euclidean_distance", "unit": "meters",
  "background": "invalid_by_mask", "valid_min_meters": nonnegative finite number,
  "valid_max_meters": greater finite number }`.
- `normal`: `{ "space": "world", "channels": ["x","y","z"],
  "encoding": "float32_linear", "foreground_unit_expected": true }`.
- `mask`: `{ "target_object_index": 1, "foreground": 255, "background": 0,
  "pass_alpha_threshold": 0.5, "antialiasing": "none" }`.
- `blender`: `{ "profile": profile ID, "version": "5.2.0", "build_hash": nonempty ASCII,
  "executable_sha256": sha256 }`.

Top-level `frame` is integer, `resolution` is `[width,height]`, `pixel_aspect` is `[1.0,1.0]`,
`pixel_origin` is `top_left`, `scene_unit_scale_meters` is positive, and `render_profile` is the
exact resolved object above. `passes` is ordered by roles `beauty_exr`, `beauty_preview`,
`depth_exr`, `depth_preview`, `normal_exr`, `normal_preview`, `object_index_exr`, `mask_png`. Each is
`{ "role": role, "path": relative_path, "sha256": sha256, "byte_size": nonnegative integer,
"media_type": exact MIME, "color_space": "scene_linear" | "srgb" | "data",
"upload_eligible": true }`. The final top-level field is `bundle_sha256`.

Role mappings are fixed: every `_exr` role uses `image/x-exr`; every preview and mask uses
`image/png`. `beauty_exr` is `scene_linear`; beauty/normal/depth previews are `srgb`; depth/normal/
object-index EXR and mask PNG are `data`. No other role/MIME/color-space tuple is valid.

#### `view-v1`

Required fields are: `schema_id: asset-mania/view`, `schema_version: 1.0`, `image_sha256`, dimensions,
`media_type: image/png`, `color_space: srgb`, `alpha: straight | none`,
`condition_manifest_sha256`, `conditioning_bundle_sha256`, `camera_digest`, `origin`, `subject`,
`alignment: { "transform": "identity", "attestation_sha256": sha256, "issuer": "user" |
"provider", "status": "declared_unverified" | "verified_fixture" }`, nullable
`rights_basis_manifest_sha256`, sensitivity, upload eligibility, validation, and `view_sha256`.

The dimension fields are `width` and `height` positive integers. `origin` is `observed`, `generated`,
or `unknown`; `subject` uses the closed subject enum. `sensitivity` is `user-content` for user input
and `portable` only for the synthetic fixture; provider output remains `user-content`.
`upload_eligible` is `false` in v0.2 because this artifact is consumed locally. `validation` is the
common validation object with profile `view-v1`, status `valid` only for mechanically valid declared
input, and `semantic_digest` equal to the normalized decoded-pixel digest. Alignment status remains
`declared_unverified` unless it is the synthetic fiducial fixture.

#### `blender-response-v1`

Required fields are: `schema_id: asset-mania/blender-response`, `schema_version: 1.0`, `request_id`
as `portable_id`,
`operation: preflight | condition | bake | export | validate`, `status: succeeded | failed`, sorted
diagnostics, portable labels, output records ordered by relative path, closed numeric metrics, and
`response_sha256`. An output record has only role, relative staging path, sha256, byte size, media
type, and validation. No timestamps, absolute paths, source names, exception strings, or raw logs are
allowed. Each operation has its own metrics `oneOf`; unknown metrics fail validation.

`portable_labels` is sorted and contains only values matching
`^(camera|mesh|armature|action|bone)-[1-9][0-9]*$`.
The metrics `oneOf` shapes are exact, all counts nonnegative integers and all digests sha256:

- preflight: `{ "kind": "preflight", "object_count", "mesh_count", "camera_count",
  "armature_count", "action_count", "target_vertex_count", "target_triangle_count",
  "target_uv_layer_count", "target_bone_count", "external_dependency_count",
  "scene_semantic_digest" }`;
- condition: `{ "kind": "condition", "width", "height", "foreground_pixel_count",
  "finite_foreground_depth_count", "interior_unit_normal_count", "projection_max_error_pixels",
  "geometry_digest", "uv_digest", "pose_digest" }`;
- bake: `{ "kind": "bake", "atlas_width", "atlas_height", "observed_texel_count",
  "padded_texel_count", "finite_texel_count", "coverage_ratio", "texture_semantic_digest" }`;
- export: `{ "kind": "export", "format_count", "mesh_count", "bone_count", "action_count",
  "camera_count", "material_count", "texture_count", "scene_semantic_digest" }`;
- validate: `{ "kind": "validate", "profile": nonempty ASCII, "checked_artifact_count",
  "error_count", "warning_count", "semantic_digest" }`.

Ratios/errors are finite JSON numbers with `coverage_ratio` in `0..1`; projection error is
nonnegative or null when the non-fixture input has no fiducial oracle. An output record is
`{ "role": nonempty ASCII, "path": relative_path, "sha256": sha256,
"byte_size": nonnegative integer, "media_type": valid MIME, "validation": common validation
object }`. Failed responses have no outputs that claim valid publication; partial outputs are marked
incomplete.

### Private worker envelope

The Apache Blender client may create a private, ephemeral envelope containing absolute local paths
and datablock names. Its directory is mode `0700`, files are mode `0600`, paths must resolve below
the private staging root where applicable, and the directory is deleted in `finally`.

Raw Blender stdout/stderr is never forwarded or persisted because Blender may print private paths
and names. The GPL worker writes a closed response containing stable codes, portable labels,
relative staging paths, hashes, and numeric metrics. The client validates it before publication.

The Blender process starts from an empty environment. It receives only a fixed `PATH`, locale and
timezone, plus controller-created `HOME`, temporary, XDG, and Blender user resource/config/data/
script/extension directories below the private staging root. `BLENDER_USER_RESOURCES` points to an
empty mode-`0700` directory and `PYTHONNOUSERSITE=1` is forced. It never inherits `PYTHONPATH`,
`PYTHONHOME`, `BLENDER_USER_SCRIPTS`, `BLENDER_USER_EXTENSIONS`, `OCIO`, proxy variables, API/cloud
credentials, user-site settings, or the caller's home/config paths. Tests install malicious user
startup/add-on, environment-capture, network, and write sentinels and prove none execute or escape.

## Blender Launch Profile

The launcher uses the equivalent of:

```text
blender --background --factory-startup --disable-autoexec --offline-mode \
  --threads 1 --python-exit-code 86 \
  --python blender-addon/src/asset_mania_blender/entrypoint.py \
  -- --request PRIVATE_REQUEST.json --response PRIVATE_RESPONSE.json
```

Important properties:

- no source path or basename appears in process arguments or environment;
- the trusted worker reads the private envelope and calls `bpy.ops.wm.open_mainfile()` with UI and
  script execution disabled;
- scripts and Python drivers do not execute;
- Blender's offline flag is defense in depth, not an OS network sandbox guarantee;
- the release E2E also runs under an OS/container network deny boundary;
- a scene that requires trusted autoexec to produce its requested pose fails with
  `UNTRUSTED_AUTOEXEC_REQUIRED` rather than weakening the profile;
- all objects, bones, cameras, actions, materials, and views are processed in deterministic order;
- non-finite transforms, singular or negative-determinant transforms, external/unpacked
  dependencies, ambiguous selections, and
  topology-changing evaluation fail before artifact publication.

The first profile requires a self-contained `.blend`: images must be packed and linked libraries,
movie clips, fonts, volumes, simulation caches, and other external dependencies are rejected. This
keeps later stages bound only to hashed run artifacts. A future import stage may inventory, copy,
hash, and remap external dependencies, but v0.2 never follows a mutable external reference.

### Scene write-surface sanitization

Immediately after opening the file and before frame evaluation, dependency-graph evaluation,
rendering, or baking, the worker creates and operates on a fresh derived scene containing only the
allowlisted target mesh, optional armature/action, camera, packed material/image data, and explicit
light/world settings. It then:

- disables the source compositor and removes/ignores every pre-existing File Output node;
- disables sequencer evaluation, Freestyle including Python mode, OSL/custom shader execution, and
  uncontrolled render caches;
- sets Cycles texture-cache and auto-generate-texture-cache flags false when exposed by the pinned
  RNA API;
- disables render border/crop and redirects render, cache, temporary, and output paths below the
  private staging root;
- rejects unsupported node groups, drivers, handlers, constraints, modifiers, or data types that
  can execute code, change topology, or write outside the closed profile;
- uses separate source and destination image datablocks for every bake.

Release E2E runs with the source and its containing directory mounted/read-only, staging as the only
writable mount, and network denied. Malicious fixtures prove that an external compositor File Output
path and Blender 5.2 texture-cache settings cannot create or modify files outside staging.

### Binding `blender-5.2.0-cpu-v1` profile

| Setting | Value |
| --- | --- |
| Blender | exact `5.2.0 LTS`; build/platform fingerprint recorded |
| render engine/device | Cycles / CPU |
| threads | `1` |
| samples/seed | `16` / `0` |
| adaptive sampling/denoise/animated seed | disabled |
| motion blur/DOF/render border/crop | disabled |
| default condition resolution | `1024x1024`; E2E override `64x64` |
| pixel aspect | `1.0:1.0` |
| color management | scene-linear EXR plus explicitly recorded sRGB PNG transform |
| object index / alpha threshold | target `1`, every other rendered object not `1`, threshold `0.5` |
| default atlas | `1024x1024`; E2E override `64x64` |
| bake margin / color padding | `16` / `8` texels; E2E `2` / `1` |
| minimum observed UV coverage | `0.15`; E2E fixture `0.25` |
| depth tolerance | `max(0.0001 m, expected_depth * 0.0002)` |
| ray self-hit epsilon | bounding-box diagonal times `1e-7`, clamped to `1e-7..1e-3 m` |
| semantic matrix quantization | round finite values to 9 decimal places |
| worker timeout | `300` seconds per stage; plan range `1..1800` |
| worker response limit | `1 MiB`; artifacts are separately bounded and hashed |
| dependency policy | packed/self-contained only; external dependency rejected |
| unknown texels | transparent delivery alpha; separate observed and padded coverage |
| animation | selected Action only, explicit integer range and step `1`; static target exports none |

The conditioning frame must lie within the selected Action range. The first profile rejects drivers,
constraints, NLA composition, topology-changing modifiers, negative determinant transforms, and
multiple armatures/actions. These can receive later versioned profiles; they never trigger a silent
fallback.

## Conditioning Bundle

One conditioning run creates:

```text
artifacts/
  conditioning/
    bundle.json
    beauty.exr
    beauty.png
    depth.exr
    depth-preview.png
    normal.exr
    normal-preview.png
    object-index.exr
    mask.png
  local/
    scene-state.blend
```

Only the conditioning directory can be considered for later upload. `scene-state.blend` is
`local-sensitive` and `upload_eligible: false`.

`bundle.json` records:

- source and evaluated geometry digests;
- target, camera, armature, and action labels;
- frame and pose digest;
- width, height, pixel aspect, and top-left pixel origin;
- Blender world axes and camera local axes;
- camera-to-world, world-to-camera, projection, and world-to-clip matrices in row-major order;
- perspective/orthographic mode, lens, sensor fit/size, shift, and clipping range;
- scene unit scale;
- depth units and semantics;
- normal coordinate space and channel convention;
- object-index and mask semantics;
- engine, device, threads, samples, seed, Blender build, and color management;
- geometry and UV semantic digests;
- relative artifact paths and hashes.

The projection matrix is obtained from Blender's evaluated camera API rather than reimplemented.

### Render profile

- Cycles CPU;
- one thread;
- fixed samples and seed;
- transparent film;
- no adaptive sampling, denoise, animated seed, motion blur, or DOF;
- combined, Z, normal, and stable object-index passes enabled;
- scene-linear float OpenEXR is canonical;
- PNG files are previews or provider attachments and record their color transform;
- background depth validity is determined by the mask;
- normals record their exact space;
- the raw object-index pass is non-antialiased; mask limitations are documented;
- the target receives reserved `pass_index = 1` in the fresh derived scene, no other rendered
  object may use `1`, and `ViewLayer.pass_alpha_threshold` is fixed at `0.5`;
- `mask.png` is exact binary `255` where Object Index equals `1` and `0` elsewhere, and validation
  requires every mask-foreground pixel to have finite foreground depth.

## View Contract and Ingest

`view ingest` accepts fully decoded 8-bit sRGB PNG, JPEG, or WebP in RGB or straight-alpha RGBA and
requires exact conditioning width/height. EXIF orientation must be absent or `1`; any other value is
rejected rather than silently rotating an aligned view. A recognized sRGB ICC profile may be
removed after decoding; non-sRGB/unknown ICC, CMYK, grayscale, palette/indexed, 16-bit, float, and
premultiplied-alpha inputs are rejected in this profile. Transparent pixels have hidden RGB zeroed
in the normalized copy. All EXIF/IPTC/XMP/GPS and nonessential chunks are removed. The view manifest
records:

- decoded pixel dimensions, color space, alpha convention, and image hash;
- conditioning bundle and camera digest;
- user-declared origin: observed, generated, or unknown;
- user-declared subject category inherited from the immutable workflow plan;
- the condition manifest's inherited rights-basis digest for `real_person`, while `unknown` remains
  blocked;
- alignment transform, fixed to identity in the first milestone;
- an alignment attestation bound to the conditioning digest and issued by `user` or `provider`;
- source sensitivity and upload eligibility;
- validation status.

The original image is not rewritten. A normalized PNG copy may be placed in the new run and linked
to its source hash. Width, height, aspect, digest, alpha, and optional fixture fiducials are
mechanically validated, but an arbitrary same-sized image cannot be proven camera-aligned from
pixels alone. Normal user/provider input is therefore `declared_alignment` with an explicitly
unverified alignment status. `VIEW_ALIGNMENT_MISMATCH` is reserved for detectable contract
mismatches; the synthetic E2E alone can use `verified_fixture_alignment` through known fiducials.

## UV Reprojection

### Preconditions

- target mesh and evaluated topology match the conditioning plan;
- one explicit UV layer exists;
- UV triangles are finite, non-degenerate, non-overlapping, and contained in 0..1;
- source view resolution and camera digest exactly match the conditioning bundle;
- source pose/topology corresponds to the evaluated target;
- image color and alpha conventions are known.

Missing camera calibration, unknown source pose, topology drift, or invalid UVs fails. v0.2 does
not estimate them.

### Deterministic algorithm

Render border and crop are disabled, pixel aspect is 1:1, and source pixels use a top-left origin.
For UV texel `(x, y)`, the center is `((x + 0.5) / atlas_width,
1 - (y + 0.5) / atlas_height)`. Triangles use a top-left fill rule so a shared edge belongs to
exactly one triangle; triangles and pixels are visited by stable polygon index then row-major order.

For each UV triangle and texel center in stable scan order:

1. Compute barycentric coordinates in the UV triangle.
2. Interpolate the evaluated world position and normal using the corresponding mesh triangle.
3. Compute `clip = projection @ world_to_camera @ [x, y, z, 1]`; require `clip.w > 0` and all NDC
   components within `[-1, 1]`. Convert to pixel-center coordinates with
   `u = (ndc.x * 0.5 + 0.5) * width - 0.5` and
   `v = (1 - (ndc.y * 0.5 + 0.5)) * height - 0.5`.
4. Reject samples outside `[-0.5, width - 0.5)` / `[-0.5, height - 0.5)`, facing away from the
   camera, or outside the target mask. Mask and depth use nearest/conservative sampling; only color
   is bilinear.
5. Compare the Euclidean distance from camera origin to the interpolated world point with Blender's
   Z-pass nearest-surface distance using the binding absolute-plus-relative tolerance. A ray cast
   from camera to point must also first hit the same evaluated target face, using the binding
   self-hit epsilon; otherwise the texel is occluded.
6. Decode source RGB to linear light and sample bilinearly.
7. Store color, observed coverage, confidence, and `source_view_label`.

The first milestone uses one view. The schema already permits stable multi-view extension, where
weights combine facing, edge falloff, depth confidence, and declared confidence. Ties resolve by
stable view label. Uncovered texels remain alpha zero. A bounded, deterministic same-island seam
dilation may be planned; generative inpainting is forbidden in this stage.

Canonical texture intermediates are scene-linear float data. Delivery base color is straight-alpha
sRGB PNG, matching glTF base-color transfer semantics.

## Bake and Material Consolidation

The worker creates a derived scene only. Production reprojection writes a source atlas image
datablock. A distinct empty target image datablock is active for the Cycles emission bake; using one
image as both source and target is invalid. This consolidation bake copies the reprojection result
into the material-owned target UV without lighting contamination. The first profile uses the fixed
atlas, margin, and color-padding values from the binding plan.

Outputs:

```text
textures/albedo-linear.exr
textures/albedo.png
textures/coverage.png
textures/padded-coverage.png
textures/preview.png
local/scene-baked.blend
```

Validation rejects empty coverage, NaN/Inf values, unexpected dimensions, out-of-range values,
missing active bake targets, or coverage below the plan threshold. Low coverage may retain marked
incomplete artifacts but cannot produce a successful export run. `coverage.png` is authoritative
observed coverage; `padded-coverage.png` is kept separate. Margin/dilation may extend RGB for seam
filtering but never promotes observed coverage. Final alpha is written explicitly after bake: `255`
for observed texels and `0` for unknown texels.

## Export and Round-Trip Validation

Export order is derived `.blend`, GLB, then FBX. Each output is written to staging, validated, and
only then published atomically.

### BLEND

- Never save over the source.
- Make texture references run-relative or explicitly pack them.
- Reopen with scripts disabled in a fresh Blender process.
- Verify mesh/topology/UV, armature/bones, pose/action, camera, material, texture, and no absolute
  external references.

### GLB

- Export the selected mesh, optional armature, material, texture, and camera.
- Use glTF 2.0, meters, +Y-up conversion, skins, and the selected Action profile below.
- Connect base color and alpha to Principled BSDF, export unknown texels with glTF
  `alphaMode: "MASK"` and `alphaCutoff: 0.5`, and validate those JSON properties after export.
- Validate container structure and resources with Khronos glTF Validator.
- Import in a fresh Blender process and compare a semantic fingerprint.

GLB is the runtime delivery contract. It is not treated as an authoring-format substitute.

### FBX

- Export binary FBX with explicit axes, unit scale, no leaf bones, and baked visible animation.
- Treat textures as an artifact group if embedding is not round-trip reliable.
- Re-import in a fresh Blender process and verify the declared subset.

An FBX pass means compatibility with Asset Mania's pinned Blender subset, not universal FBX
interoperability. Unsupported constraints, shape-key behavior, or material conversions are reported
rather than silently dropped.

### Binding exporter profile

For a selected rig/action, outputs preserve the rest rig and exactly one selected Action over its
explicit integer frame range. The condition frame is not baked as a replacement rest pose. A static
target exports no animation. The derived scene contains no other Actions or NLA strips.

The GLB call binds `export_format="GLB"`, selected objects only, `export_yup=True`,
`export_skins=True`, `export_cameras=True`, `export_animations` according to the selected Action,
`export_animation_mode="ACTIONS"`, `export_frame_range=True`, `export_frame_step=1`,
`export_force_sampling=True`, `export_bake_animation=True`, `export_nla_strips=False`,
`export_current_frame=False`, `export_rest_position_armature=True`, `export_apply=False`, and no
geometry compression in the first profile. It also binds `export_image_format="AUTO"` for the PNG
source, `export_image_add_webp=False`, `export_image_webp_fallback=False`,
`export_keep_originals=False`, `export_materials="EXPORT"`, `export_texcoords=True`,
`export_normals=True`, `export_tangents=False`, `export_unused_images=False`, and
`export_unused_textures=False`. Runtime RNA preflight rejects a missing or changed operator property
rather than using a default.

The FBX call binds selected objects only, object types `ARMATURE`, `MESH`, and `CAMERA`, unit scale,
`global_scale=1.0`, `apply_unit_scale=True`, `apply_scale_options="FBX_SCALE_NONE"`,
`use_space_transform=True`, `bake_space_transform=False`, `axis_forward="-Z"`, `axis_up="Y"`,
`add_leaf_bones=False`, `primary_bone_axis="Y"`, `secondary_bone_axis="X"`, deform bones only,
`bake_anim` according to the selected Action, `bake_anim_use_all_actions=False`,
`bake_anim_use_nla_strips=False`, `bake_anim_force_startend_keying=True`, `bake_anim_step=1.0`, and
`bake_anim_simplify_factor=0.0`. It uses `path_mode="COPY"`, `embed_textures=False`,
`use_metadata=True`, and includes the PNG as a separate relative artifact in the same export group.
Fresh imports compare format-aware deformed vertex samples and bone matrices at the action start,
condition frame, and action end.

## Diagnostics

New stable diagnostics include:

- `BLENDER_VERSION_MISMATCH`
- `BLENDER_EXECUTION_FAILED`
- `BLENDER_RESPONSE_INVALID`
- `UNTRUSTED_AUTOEXEC_REQUIRED`
- `SOURCE_CHANGED_DURING_RUN`
- `PARENT_MANIFEST_MISMATCH`
- `PLAN_TAMPERED`
- `SELECTION_AMBIGUOUS`
- `TARGET_MESH_NOT_FOUND`
- `CAMERA_NOT_FOUND`
- `RIG_NOT_FOUND`
- `POSE_NONFINITE`
- `MISSING_LINKED_ASSET`
- `DEPSGRAPH_TOPOLOGY_CHANGED`
- `PASS_INVALID`
- `VIEW_ALIGNMENT_MISMATCH`
- `CAMERA_CALIBRATION_MISSING`
- `SOURCE_POSE_UNKNOWN`
- `UV_MISSING_OR_INVALID`
- `REPROJECTION_LOW_COVERAGE`
- `BAKE_CONTEXT_INVALID`
- `EXPORT_OPERATOR_UNAVAILABLE`
- `GLTF_VALIDATION_FAILED`
- `ROUNDTRIP_MISMATCH`
- `OUTPUT_COLLISION`
- `SUBJECT_DECLARATION_REQUIRED`
- `FACE_RIGHTS_CONFIRMATION_REQUIRED`
- `PROVIDER_EVIDENCE_STALE`

Expected failures produce a closed response and no traceback on normal stdout/stderr. Debug logs are
local-only, opt-in, redacted, and never upload-eligible.

## Privacy, Approval, and Provider Boundary

Local Blender stages perform no network access, model download, or paid action. They need no upload
approval but still preserve source integrity and sensitivity labels. Generic local face/head
texturing is not face reconstruction: `real_person` requires a plan-bound rights/consent receipt,
`unknown` is blocked, and the tool never infers either category from pixels or geometry. The
receipt records a user assertion and scope; it is not a legal-consent signature or legal clearance.

The full v0.2 release includes the GPT Image 2 adapter, while Tasks 1-9 form a publishable local
milestone and do not wait on provider credentials or a live call. The adapter is a separate
`asset-mania-provider-openai` wheel discovered through a provider entry point; the CLI wheel does
not depend on it. The root development workspace installs it only to run its fake-transport tests.

The scene-guided workflow binds `POST /v1/images/edits` and model snapshot
`gpt-image-2-2026-04-21`. Attachments are ordered and labeled as beauty, depth preview, normal
preview, and mask; all four are multipart `image[]` reference parts at indices `0..3`. This profile
does not send the optional API `mask` field—the binary mask is a visual reference image, not an
inpainting mask. The multipart field name, index, role, media type, byte size, and exact hash are
approval-bound. A future profile that uses the API `mask` field or generations endpoint requires a
different provider-plan profile; this release never silently changes the mapping or endpoint.

The GPT Image 2 adapter must implement a separate provider plan containing exact model
snapshot, attachment hashes, prompt hash, closed output controls, policy/pricing evidence, cost
ceiling, and expected outputs. GPT Image 2 output controls are exactly:

- `n`, fixed to `1` in this single-view profile;
- `size` as an explicit `1024x1024`, `1024x1536`, or `1536x1024` value equal to the conditioning
  bundle; `auto` and other custom sizes are rejected in this aligned-view/cost profile;
- `quality`: `low`, `medium`, or `high`; `auto` is rejected because it has no approval-bound
  published cost row;
- `background`: `auto` or `opaque`; `transparent` is rejected for GPT Image 2;
- `output_format`: `png`, `jpeg`, or `webp`;
- `output_compression` as an integer `0..100`, only when the format is JPEG or WebP;
- `moderation`: `auto` or `low`.

The plan rejects `input_fidelity` because GPT Image 2 applies high input fidelity automatically
and does not expose that control. Unknown provider fields and invalid control combinations fail
before approval.

The disclosure snapshot records official source URLs, retrieval timestamp, and source-version or
content digest. Its data-policy record states the documented default no-training-unless-opt-in,
Images API no-application-state behavior, default 30-day abuse-monitoring retention, Zero Data
Retention eligibility and approval requirement, and the potential-CSAM manual-review exception.
The effective processing region is recorded when documented, otherwise explicitly `unknown`.

The estimate record includes currency, timestamped rate basis, text/image input assumptions,
output-token estimate, `n`, size, and quality. Returned token usage and actual/billed cost, when the
provider exposes them, are recorded separately from the preflight estimate and ceiling. A stale or
changed policy/pricing evidence digest invalidates approval.

The estimator uses only the size/quality output-cost rows explicitly published in the evidence
artifact plus documented token rates for declared text/reference-image input assumptions. It never
scrapes the interactive calculator or invents an arbitrary-size formula. The exact estimator table
and source digest are approval-bound.

Provider policy/pricing evidence has a 24-hour TTL for an executable paid plan. It is never refreshed
implicitly. A maintainer or user must run the explicit, networked
`asset-mania provider evidence refresh openai --out EVIDENCE` command, restricted to inventoried
official OpenAI documentation/pricing hosts, to write a new hashed evidence artifact. Without a
fresh artifact, planning fails closed with `PROVIDER_EVIDENCE_STALE` before approval or credential
access.

Before transport, the runner requires fresh receipts bound to that exact plan for:

- real-person subject rights when applicable;
- external egress;
- paid compute.

`unknown` never reaches provider planning or receipt evaluation: it fails first with
`SUBJECT_DECLARATION_REQUIRED` until the user declares `non_person`, `synthetic_person`, or
`real_person`. Only `real_person` can satisfy the face-rights gate with a plan-bound receipt.

Any input, model, prompt, attachment, price ceiling, retention disclosure, or output change
invalidates the receipt. Paid retries require a new receipt. No source `.blend`, local scene copy,
credential, prompt text, or unlisted file can leave the machine.

The adapter is optional and never blocks the user-supplied view path. Tests use an injected fake
transport and deny sockets. A live provider canary is manual, disposable, price-capped, and not a
prerequisite for proving the local Blender pipeline.

Fake-transport evidence supports the public label `experimental, contract-verified`, not
`live-verified`. Any README/Skill claim that GPT Image 2 generation is available and working requires
a fresh, explicitly approved, capped live canary against the pinned snapshot and a recorded
sanitized result at the publication SHA.

## Licensing and Fixture Policy

- Apache components remain Apache-2.0.
- The Blender worker is GPL-3.0-or-later, separately packaged and inventoried.
- No Blender binary, model weight, provider credential, dataset, private face sample, or generated
  real-person image is redistributed.
- The supplied private face archive is exploratory input only. It is a 2D mirrored
  spin package without 3D geometry, provenance, or redistribution clearance and is not a fixture,
  golden, training source, or 3D acceptance artifact.
- One composite runtime fixture is procedurally generated: an asymmetric calibration/UV-checker
  scene with a rigged non-human robot. Malicious variants exercise external writes and invalid
  contracts. The generator, seed/profile, and generated outputs are dedicated CC0 for fixture use;
  binaries are generated at runtime and are not uploaded as CI artifacts.
- Any unavoidable tracked binary has an exact generator, seed, command, digest, license, and
  provenance entry. Prefer no tracked binary fixture.

## Determinism Claims

- Byte-exact: canonical serialization of the same logical value, immutable plan digests, binary
  masks, observed/padded coverage, and decoded 8-bit reprojection pixels in one pinned environment.
- Repeat-run equivalent: manifests and worker responses only after normalizing run IDs, timestamps,
  private staging paths, and parent run identities. Those run-specific fields remain integrity-bound
  in each real manifest; they are not omitted from hashes.
- Semantic-exact: sorted scene fingerprint, labels, topology, UV, rig, camera, material, and
  quantized transforms.
- Visual-tolerance: decoded EXR/render/bake float arrays under the pinned Blender/CPU profile; EXR
  container bytes are never compared across runs.
- Not claimed: byte-identical `.blend`, GLB, FBX, or beauty output across different Blender builds,
  operating systems, CPUs, GPUs, or remote providers.

The authoritative byte-exact environment is the pinned Linux x86_64 Blender 5.2.0 CPU E2E. macOS
is a required semantic/tolerance compatibility profile, not a cross-platform byte oracle.

## Testing and CI

### Fast CI

Runs on every push and pull request:

- v1 backward compatibility;
- every new schema and unsafe-field rejection;
- canonical plan/manifest/lineage hashing;
- approval tamper tests;
- private envelope permissions, path containment, deletion, and redaction;
- projection and reprojection math;
- view ingest and metadata stripping;
- malformed BLEND/GLB/FBX fast validators;
- GPL/Apache import and distribution boundary;
- source-integrity, no-overwrite, no-socket, and deterministic-order tests;
- Skill/schema/release inventory checks.

### Pinned Blender CPU E2E

A separate required workflow for 3D changes and releases:

1. Acquire an inventoried Blender 5.2.0 LTS build by exact URL and SHA-256.
2. Run in an OS/container network-deny boundary.
3. Generate the synthetic fixture at runtime.
4. Preflight and condition frame 2 through the real GPL worker.
5. Ingest a deterministic fake-provider checker view.
6. Reproject and bake the texture.
7. Save BLEND and export GLB/FBX.
8. Reopen or re-import every artifact in fresh Blender processes.
9. Run glTF Validator and compare semantic fingerprints.
10. Verify source hashes are unchanged and no absolute path or secret appears in portable output.

The tiny fixture has a two-bone non-human rig, asymmetric UV/color pattern, frame-2 deformation,
one calibrated camera, and a 64x64 render. Its numeric oracle is closed:

- named fiducials `root-left`, `joint-center`, and `tip-right` have expected top-left pixel-center
  coordinates in fixture metadata and maximum projection error `0.25` pixel;
- mask foreground contains at least `64` pixels and every foreground depth is finite and positive;
- every foreground normal is finite and within the encoded component range; on the one-pixel-eroded
  interior mask, pixels with magnitude above `1e-6` have maximum length error `1e-4` from unity;
- observed UV coverage is at least `0.25` of atlas texels and matches the analytic oracle outside a
  one-texel shared-edge/seam exclusion band;
- over the one-texel-eroded observed mask, encoded 8-bit sRGB channel error is maximum `1` code
  value and RMS at most `0.25` code value; linear-light `1/255` is not used as an oracle;
- observed alpha is exactly `255`, unknown alpha exactly `0`, and padded coverage never changes
  observed coverage;
- floating comparisons otherwise use numeric epsilon `1e-6` unless a tighter named tolerance above
  applies;
- Khronos glTF Validator reports zero errors.

GPU model workflows and live providers are manual/nightly canaries. They become release-blocking
only when a public capability claim depends on them.

## Acceptance Criteria

- v1 CLI, schema, fixtures, and behavior remain backward compatible.
- Every new portable document validates against its closed schema and contains no absolute path,
  basename, datablock name, credential, prompt text, image bytes, or identity feature.
- Apache wheels contain no GPL worker file; GPL archive contains its license and imports no Apache
  package.
- The source scene and source view are byte-identical after all success and failure paths.
- The real Blender 5.2.0 E2E creates and validates beauty/depth/normal/mask, baked texture, derived
  BLEND, GLB, and FBX artifacts.
- A mismatched view, invalid UV, topology drift, missing camera, autoexec-dependent scene, tampered
  plan/parent, output collision, or worker failure cannot publish a successful run.
- Unseen texels remain explicitly uncovered; no hidden geometry or texture is represented as
  observed fact.
- The Skill and README accurately distinguish the working local round trip from optional GPT
  generation, generic image-to-3D, face/head research, and future cloud execution.
- GitHub CI, release checks, license/provenance checks, and independent forward evaluations pass at
  the final public head.

## Primary References

- [Blender 5.2 Python API](https://docs.blender.org/api/5.2/)
- [Blender 5.2 command-line arguments](https://docs.blender.org/manual/en/5.2/advanced/command_line/arguments.html)
- [Blender 5.2 render passes](https://docs.blender.org/manual/en/5.2/render/layers/passes.html)
- [Blender 5.2 Cycles baking](https://docs.blender.org/manual/en/5.2/render/cycles/baking.html)
- [Blender 5.2 glTF exporter](https://docs.blender.org/manual/en/5.2/addons/scene_gltf2.html)
- [Blender 5.2 FBX exporter](https://docs.blender.org/manual/en/5.2/addons/import_export/scene_fbx.html)
- [glTF 2.0 specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)
- [Khronos glTF Validator](https://github.com/KhronosGroup/glTF-Validator)
- [GPT Image 2 model reference](https://developers.openai.com/api/docs/models/gpt-image-2)
- [OpenAI image generation guide](https://developers.openai.com/api/docs/guides/image-generation)
- [OpenAI API data controls](https://developers.openai.com/api/docs/guides/your-data)
