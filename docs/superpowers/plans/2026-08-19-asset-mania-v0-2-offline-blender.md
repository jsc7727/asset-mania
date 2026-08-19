# Asset Mania v0.2 Offline Blender Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use
> `superpowers:subagent-driven-development` task-by-task. Behavior changes follow
> red-green-refactor and receive a separate spec-compliance review before the next dependent task.

**Goal:** Ship a real local Blender round trip from a posed/camera-framed scene to conditioning
passes, a supplied aligned image, a baked UV texture, and validated BLEND/GLB/FBX outputs while
preserving v0.1 and the Apache/GPL boundary.

**Architecture:** New Apache packages own immutable execution contracts, stage orchestration, and
the sanitized Blender subprocess client. Every `bpy`/`mathutils` import lives in a separately
packaged GPL worker. The worker communicates only through closed JSON and relative files. Each
stage publishes a new atomic run referencing immutable parent digests.

**Tech stack:** CPython 3.11-3.13, uv workspace, pytest 9, JSON Schema 2020-12, Pillow, Blender
5.2.0 LTS/Cycles CPU, OpenEXR/PNG, glTF 2.0, Khronos glTF Validator, GitHub Actions.

**Spec:**
`docs/superpowers/specs/2026-08-19-asset-mania-v0-2-offline-blender-design.md`

## Global Constraints

- Keep `manifest-v1.schema.json`, v1 examples, and `asset-mania inspect` behavior unchanged.
- The source `.blend` and image are always byte-identical before and after success or failure.
- Never overwrite a source, parent run, existing run directory, or output artifact.
- Portable output contains no absolute path, basename, raw Blender datablock name, credential,
  prompt text, signed URL, image bytes, raw EXIF value, or face/identity feature.
- Every new schema is closed with `additionalProperties: false`; breaking changes use a new major.
- Root packages/Skill/docs remain Apache-2.0. Every `bpy`/`mathutils` import is confined to the
  separately distributed GPL-3.0-or-later `blender-addon/` tree.
- No network, model download, paid call, GPU allocation, or external upload occurs in Tasks 1-9.
- Task 10 tests providers with injected transports and denied sockets. A live GPT Image request is
  outside automated acceptance and requires its separate credential, egress, and cost authority.
- Do not bundle Blender, glTF Validator, weights, datasets, user assets, the private face ZIP, or
  real-person images.
- Runtime binary fixtures are procedurally generated. A tracked binary requires exact provenance.
- Use Conventional Commits and repository-local `jsc7727` identity. Preserve global Git/GitLab
  identity.

## Execution Setup

The isolated implementation worktree already exists below the primary checkout:

```text
.worktrees/v0-2-blender-pipeline
branch: codex/v0-2-blender-pipeline
base: 7b2d2c4737f3efa8ad397b86de5cfd0d231febf9
```

Before Task 1, verify:

```bash
git branch --show-current
git status --short
git rev-parse HEAD
git config --local --get user.name
git config --local --get user.email
git config --global --get user.name
git config --global --get user.email
```

Expected: the implementation branch, only the two planned docs modified, local `jsc7727` GitHub
identity, and unchanged company global identity.

Canonical local Blender executable:

```text
/Applications/Blender.app/Contents/MacOS/Blender
```

No task may rely on a shell alias or a `blender` PATH entry.

## Planned File Map

### Contracts

```text
packages/contracts/src/asset_mania_contracts/
  diagnostics.py
  execution.py
  models.py
  schema/
    manifest-v1.schema.json                  unchanged
    manifest-v2.schema.json
    workflow-plan-v1.schema.json
    provider-evidence-v1.schema.json
    provider-plan-v1.schema.json
    approval-receipt-v1.schema.json
    conditioning-bundle-v1.schema.json
    view-v1.schema.json
    blender-response-v1.schema.json
packages/contracts/tests/
  test_contracts.py                           unchanged v1 assertions
  test_manifest_v2.py
  test_workflow_plan.py
  test_provider_evidence.py
  test_provider_plan.py
  test_approval_receipt.py
  test_conditioning_bundle.py
  test_view_contract.py
  test_blender_response.py
```

### Pipeline

```text
packages/pipeline/
  LICENSE
  pyproject.toml
  src/asset_mania_pipeline/
    __init__.py
    artifacts.py
    hashing.py
    lineage.py
    plans.py
    approvals.py
    stage_store.py
    projection.py                 # reference oracle/test vectors only
    reprojection.py               # reference oracle/test vectors only
    views.py
    validators.py
  tests/
    test_stage_store.py
    test_lineage.py
    test_plans.py
    test_approvals.py
    test_projection.py
    test_reprojection.py
    test_views.py
    test_source_integrity.py
```

### Blender client and GPL worker

```text
packages/blender-client/
  LICENSE
  pyproject.toml
  src/asset_mania_blender_client/
    __init__.py
    discover.py
    envelope.py
    launcher.py
    response.py
    redaction.py
  tests/
    test_discover.py
    test_envelope.py
    test_launcher.py
    test_response.py
    test_redaction.py

blender-addon/
  LICENSE
  README.md
  pyproject.toml
  src/asset_mania_blender/
    __init__.py
    entrypoint.py
    protocol.py
    labels.py
    scene_inventory.py
    fixture_factory.py
    conditioning.py
    baking.py
    exporting.py
    validation.py
  tests/
    run_e2e.py
    validate_import.py
```

### CLI, provider, Skill, docs, and release

```text
packages/provider-openai/
  LICENSE
  pyproject.toml
  src/asset_mania_provider_openai/
    __init__.py
    client.py
    errors.py
    normalization.py
    transport.py
  tests/
    test_normalization.py
    test_transport_boundary.py
    test_response_validation.py
    test_retry_policy.py

packages/cli/src/asset_mania/
  cli.py
  execution_cli.py
packages/cli/tests/
  test_scene_cli.py
  test_view_cli.py
  test_bake_cli.py
  test_export_cli.py
  test_provider_cli.py

scripts/
  check_license_boundary.py
  check_schema_distribution.py
  run_blender_sandboxed.py
  run_blender_e2e.py
tools/
  blender-5.2.0.json
  gltf-validator.json
tests/
  test_license_boundary.py
  test_schema_distribution.py
  test_v2_privacy.py
  test_v2_skill_distribution.py
.github/workflows/
  ci.yml
  blender-e2e.yml
```

---

## Task 1: Freeze the v0.2 Design and Execution Ledger

**Files:**

- Add the v0.2 design and this plan.
- Add `.superpowers/sdd/2026-08-19-asset-mania-v0-2/progress.md` as an ignored local ledger.

**Acceptance:** all binding decisions are explicit; no implementation code changes.

- [ ] Verify the two docs contain no private local input basename or secret.
- [ ] Stage only the two docs, then run `git diff --check` and `make release-check` so the
  tracked-file-based release checker actually inspects the new files; unstage only if a fix is
  required.
- [ ] Record base SHA, Blender version/build hash, source worktree, and current test count in the
  ignored ledger.
- [ ] Commit:

```bash
git add docs/superpowers/specs/2026-08-19-asset-mania-v0-2-offline-blender-design.md \
  docs/superpowers/plans/2026-08-19-asset-mania-v0-2-offline-blender.md
git commit -m "docs: design offline Blender pipeline"
```

## Task 2: Add Execution Contract Schemas Without Moving v1

**Files:** contracts schemas/builders/tests listed above; copied Skill schemas; schema registry.

**Interfaces produced:**

- `canonical_digest(value) -> sha256`
- `build_manifest_v2(...)`
- `build_workflow_plan(...)`
- `build_provider_plan(...)`
- `build_approval_receipt(...)`
- schema loader by stable schema name/version

### TDD sequence

- [ ] Create failing tests proving:
  - current v1 fixture and schema bytes are unchanged;
  - valid examples for every new schema pass;
  - absolute paths, basenames, datablock names, prompt text, secret-like fields, image bytes, unknown
    properties, mutable parent references, and unclosed stage parameters fail;
  - artifact lineage preserves generated origin transitively;
  - asset kind and subject category are closed user declarations, not inferred fields;
  - canonical digest ignores no field and changes for every approval-sensitive field;
  - diagnostic ordering is stable;
  - run/receipt/consumption/request IDs reject separators, traversal, control characters, and values
    outside `^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$`;
  - only scene-preflight/provider-evidence permit null `plan_sha256`, and only provider-evidence may
    declare `network=explicit_official_hosts` with the exact official host parameters.
- [ ] Commit schema-valid normative examples covering every stage `oneOf`, parent/artifact identity,
  provider state transition, v2 exit mapping, selection HMAC preimage, and single-use approval
  issuance/expiry/consumption. Task code may not invent a field absent from these examples.
- [ ] Run focused tests and record RED.
- [ ] Implement only the closed schemas, builders, registry, and new diagnostic enum members.
- [ ] Copy distributed schemas mechanically to Skill references and validate byte parity.
- [ ] Run focused tests, all contract tests, and full `make test`; record GREEN.
- [ ] Run `make check`, `make skill-check`, `make release-check`, `git diff --check`.
- [ ] Commit `feat: define v2 execution contracts`.
- [ ] Independent reviewer checks v1 byte compatibility, privacy allowlists, and approval binding.

## Task 3: Implement Immutable Stage Storage, Plans, Lineage, and Approvals

**Files:** `packages/pipeline` metadata/core/tests; root workspace/lock/notices.

**Interfaces produced:**

- `StageStore.begin(...)`, `publish(...)`, and `fail(...)`
- parent manifest/artifact verification
- canonical plan hashing and tamper verification
- monotonic stage state machine
- approval validation by exact plan digest, gate, scope, and expiry
- receipt issuance from an exact interactive/non-interactive plan acknowledgement and atomic
  consumption journal

### TDD sequence

- [ ] Scaffold the workspace member and lock it without behavior.
- [ ] Write failing tests for atomic no-replace publication, storage loss, output collision, tampered
  parents, tampered plans, expired/wrong receipts, state rollback, canceled/failed terminal states,
  partial artifacts, source hash changes during execution, and portable path containment.
- [ ] Prove local `real_person` view processing requires a rights/consent receipt bound to the exact
  workflow-plan digest, `unknown` is blocked, and non-person/synthetic paths never invoke a pixel or
  geometry classifier.
- [ ] Reject maintainer/provider-issued approval for every gate; `face_rights`, `external_egress`,
  and `paid_compute` receipts require the scoped user acknowledgement.
- [ ] Write issuance tests for the exact CLI-to-JSON kebab/underscore mapping, full
  `GATE:PLAN_SHA256` acknowledgement, disclosure display, expiry, issuer enum, no free text/person
  identifier, and atomic single-use consumption. Boolean/global approval must fail.
- [ ] Add no-socket and no-subprocess fixtures for pure pipeline tests.
- [ ] Record RED, implement the minimum stage store/lineage/approval logic, and refactor green.
- [ ] Validate modes and cleanup on macOS and Linux-compatible code paths.
- [ ] Run focused tests, full suite, canonical gates, and external package builds.
- [ ] Update `THIRD_PARTY_NOTICES.md` for any new runtime/dev dependency.
- [ ] Commit `feat: add immutable pipeline stages`.
- [ ] Independent reviewer checks race/collision behavior and approval scope.

## Task 4: Establish the Apache/GPL Boundary and Sanitized Blender Client

**Files:** `packages/blender-client`, `blender-addon` scaffold, license checker/tests, pinned tool
inventories, sandbox runner, notices/docs.

**Interfaces produced:**

- exact Blender discovery/profile check;
- private request directory/envelope lifecycle;
- sanitized process launcher;
- closed worker response parsing and path containment;
- GPL archive packaging separate from uv wheels;
- source-read/staging-write/network-deny process isolation;
- pinned Blender and Khronos glTF Validator acquisition metadata used by later tasks.

### TDD sequence

- [ ] First write failing license-boundary tests that detect:
  - `bpy` or `mathutils` outside `blender-addon/`;
  - Apache package imports in GPL code;
  - GPL files in any Apache wheel/sdist;
  - missing GPL headers/license/archive inventory.
- [ ] Write failing client tests for wrong Blender version, inaccessible executable, malicious
  relative paths, symlink escape, missing/oversized/invalid response, timeout, signal, nonzero exit,
  stdout/stderr secrets, mode `0700/0600`, and guaranteed cleanup. Inspect the fake process argv and
  environment to prove the source path/basename never appears; only the private request path may be
  passed.
- [ ] Start the fake worker from an empty environment and capture it. Require isolated
  `BLENDER_USER_RESOURCES`, HOME/temp/XDG/Blender directories below staging,
  `PYTHONNOUSERSITE=1`, and a fixed locale/PATH; reject inherited Python/Blender/OCIO overrides,
  proxies, API/cloud credentials, user-site config, and caller home paths. Plant a malicious startup
  script/add-on sentinel and prove it never executes.
- [ ] Write failing sandbox-runner tests proving source/source-directory are read-only, staging is the
  only writable tree, and network is denied. Support the pinned macOS sandbox profile and Linux
  bubblewrap/container profile; fail closed when the requested isolation backend is unavailable.
- [ ] Pin and inventory official Blender 5.2.0 LTS builds and Khronos glTF Validator by exact
  version/revision, URL, SHA-256, license URL, and notice requirements before Tasks 5 and 9 consume
  them. Acquisition happens before the network-deny boundary; neither tool is redistributed.
- [ ] Use a fake executable in unit tests; do not invoke Blender in RED/GREEN unit tests.
- [ ] Implement the minimal client/scaffold/checker.
- [ ] Run focused tests, all packages, release checks, and inspect built archives.
- [ ] Commit `feat: isolate Blender GPL worker`.
- [ ] Independent reviewer verifies no license-linkage or privacy leak.

## Task 5: Generate a Synthetic Fixture and Deep Scene Preflight in Blender

**Files:** GPL labels/inventory/fixture factory/entrypoint, Apache response validation, Blender E2E
driver.

**Fixture:** a runtime-generated asymmetric, rigged non-human strip/robot with:

- six or more vertices and known non-overlapping UVs;
- two bones named privately but labeled `bone-1`, `bone-2` portably;
- rest frame and a frame-2 30-degree deformation;
- asymmetric checker/quadrant texture;
- one camera and one fixed area light;
- no external files, scripts, drivers, or identity content.

### TDD sequence

- [ ] Add worker protocol tests executable inside Blender, failing before implementation.
- [ ] Add a fixture-generation command that writes only to test staging.
- [ ] Record the composite fixture generator digest, fixed seed/profile, and CC0 disposition in the
  provenance report. Generate every binary at runtime and do not upload it as a CI artifact.
- [ ] Add failing deep-preflight cases: duplicate names, ambiguous camera/mesh, non-finite transform,
  singular/negative-determinant scale, every external/unpacked dependency, invalid UV,
  topology-changing modifier, and an autoexec marker that must never execute. Missing/zero rig
  weights fail only when an armature is selected; a static prop remains valid.
- [ ] Add malicious fixtures with a compositor File Output path outside staging and Blender 5.2
  texture-cache/auto-generate settings targeting a source-adjacent `.tx`. Neither may write.
- [ ] Start Blender without the source on argv; the trusted worker opens it from the envelope with UI
  and scripts disabled, sanitizes write surfaces, and only then inventories/evaluates.
- [ ] Implement deterministic portable labels, local-sensitive salted selection map/HMAC, and
  semantic scene fingerprint.
- [ ] Invoke real Blender 5.2.0 locally and record RED/GREEN transcripts in the ignored ledger.
- [ ] Verify fixture source hash before/after and private-path absence in response.
- [ ] Commit `feat: preflight Blender scenes`.
- [ ] Independent reviewer runs the worker from a fresh process.

## Task 6: Render and Validate the Conditioning Bundle

**Files:** GPL conditioning renderer; contracts/pipeline bundle validator; CLI service adapter tests.

**Required artifacts:** canonical EXR plus PNG previews for beauty, depth, normal, object index, mask,
camera/bundle metadata, and local derived scene state.

### TDD sequence

- [ ] Write failing bundle-schema and validator tests for dimensions, hashes, missing passes, NaN/Inf,
  empty mask, invalid depth/background semantics, non-finite/out-of-range foreground normals,
  non-unit eroded-interior normals, wrong camera matrix, upload-ineligible local scene, and unstable
  order.
- [ ] Write a real Blender RED E2E expecting missing pass artifacts.
- [ ] For a `real_person` plan, fail with exit `5` before launching Blender unless the exact
  `face_rights` receipt is supplied; consume it atomically before source open. `unknown` fails before
  receipt evaluation. Prove the worker is never invoked on either blocked path.
- [ ] Before dependency-graph evaluation, build a fresh allowlisted derived scene; disable compositor,
  sequencer, Freestyle/Python, OSL/custom execution, drivers/handlers, output nodes, texture caches,
  render caches and border/crop; route every output/temp path below staging.
- [ ] Implement the binding Cycles CPU profile, camera matrices, pose/topology digests, reserved
  target `pass_index=1`, alpha threshold `0.5`, exact binary mask, pass output, and bundle JSON.
- [ ] Keep raw EXR canonical; mark PNGs as transformed previews/attachments.
- [ ] Run twice and compare normalized run-document semantics, exact decoded masks/coverage, and
  tolerance-based decoded EXR/render arrays; never compare distinct-run manifest or EXR bytes.
- [ ] Execute under source-read/staging-write/network-deny isolation and recheck the full source tree
  inventory to prove no external File Output/cache artifact was created.
- [ ] Verify analytic fiducial projection error <= 0.25 pixel and source hash unchanged.
- [ ] Run full checks and commit `feat: render Blender conditioning bundles`.
- [ ] Independent reviewer inspects pass semantics and upload allowlist.

## Task 7: Ingest and Normalize a User-Supplied Aligned View

**Files:** pipeline view service, contract tests, CLI view command/tests.

### TDD sequence

- [ ] Write failing tests for PNG/JPEG/WebP success, corrupt/oversized inputs, decompression bombs,
  sensitive metadata, mismatched resolution/aspect, wrong condition hash, unsupported origin,
  non-identity hidden transform, alpha conventions, missing/tampered real-person approval lineage in
  the condition parent, blocked unknown subject, output collision, and source mutation.
- [ ] Add EXIF orientation, recognized/unknown ICC, non-sRGB, CMYK, grayscale, palette, 16-bit/float,
  premultiplied-alpha declaration, and transparent hidden-RGB cases. Accept only the binding 8-bit
  sRGB RGB/straight-RGBA profile, reject unsupported modes/orientation, and zero hidden RGB.
- [ ] Implement full decode, safe metadata stripping, mechanical dimension/aspect/digest checks,
  declared-alignment attestation, normalized PNG artifact, view manifest, and
  provenance/sensitivity propagation.
- [ ] Prove `observed`, `generated`, and `unknown` survive through the view contract without pixel
  classification.
- [ ] Require `user` or `provider` alignment attestation bound to the condition digest. Mark normal
  same-size input `declared_alignment`/unverified; reserve verified alignment for fixture fiducials
  and never claim dimensions prove correspondence.
- [ ] Default to an interactive exact `CONDITION_SHA256:VIEW_SHA256` acknowledgement after full
  decode/hash; non-interactive use must pass the same exact digest pair. Test wrong/boolean/missing
  acknowledgements and canonical user/provider attestation digest generation.
- [ ] Run focused/full tests and canonical gates.
- [ ] Commit `feat: ingest aligned texture views`.
- [ ] Independent reviewer checks privacy and no implicit resize/crop.

## Task 8: Reproject and Bake the View Into Existing UVs

**Files:** Apache reference projection/reprojection oracle, GPL production reprojection/bake worker,
validators/tests, CLI bake command.

### TDD sequence

- [ ] Start with Apache reference-oracle tests for UV `(x+0.5, y+0.5)` centers, top-left triangle
  ownership, `clip.w`/NDC/top-left source-pixel conversion, bilinear linear-light color sampling,
  nearest/conservative mask/depth sampling, backface/clip/bounds rejection, Euclidean Blender Z
  depth tolerance, ray first-hit/self-epsilon, stable scan order, seam padding, and uncovered alpha.
  This oracle uses synthetic arrays only and is not the production reprojection implementation.
- [ ] Add failing Blender cases for missing/degenerate/overlapping/out-of-range UV, topology drift,
  source pose mismatch, camera mismatch, empty/low coverage, missing active image target, NaN/Inf,
  output collision, and storage loss.
- [ ] Implement one-view production reprojection inside the GPL worker using evaluated mesh/loop/UV
  data, Blender image/EXR access, stored camera matrices, Z pass and ray casting. No private geometry
  snapshot crosses into Apache code. Compare it to the reference oracle and known fixture.
- [ ] Consume only the hashed `scene-state.blend` parent artifact and normalized view artifact; reject
  any mutable external dependency or digest mismatch.
- [ ] Use distinct source-atlas and empty target image datablocks; implement Cycles emission bake into
  the existing UV with the fixed atlas/margin and no lighting.
- [ ] Produce albedo-linear EXR, straight-alpha sRGB PNG, authoritative observed coverage, separate
  padded coverage, preview, and baked scene. Write alpha explicitly after bake: observed `255`,
  unknown `0`; padding never upgrades origin/coverage.
- [ ] Prove generated source origin remains generated and no uncovered texel is hallucinated.
- [ ] Run real Blender twice; validate exact/tolerance classes and source hashes.
- [ ] Commit `feat: reproject and bake texture views`.
- [ ] Independent reviewer checks geometry correspondence, occlusion, colorspace, and provenance.

## Task 9: Export and Fresh-Process Validate BLEND, GLB, and FBX

**Files:** GPL exporter/validators/import validator; Apache fast validators; CLI export tests; glTF
validator integration.

### TDD sequence

- [ ] Write failing fast validators for malformed BLEND, GLB header/length/chunks, and unsupported
  FBX header/version.
- [ ] Write failing deep tests for missing operator, absolute texture path, dropped mesh/UV/material,
  changed armature hierarchy, pose/action mismatch, invalid camera FOV, GLB validator error, FBX
  subset mismatch, and partial output publication.
- [ ] Save a new derived BLEND with relative/packed texture references and reopen it.
- [ ] Export selected GLB with the exact binding operator profile, camera enabled, rest rig preserved,
  only the selected Action/range sampled, no NLA/other Actions, and base-color alpha exported as
  `MASK` cutoff `0.5`. Validate JSON properties, pinned Khronos glTF Validator, and fresh import.
- [ ] Export binary FBX with the exact binding axes/units/no-leaf-bones/single-Action/no-NLA/no-
  simplification profile and validate via fresh Blender import.
- [ ] For a static target, prove both formats contain no animation. For a rigged target, compare
  format-aware semantic fingerprints, bone matrices, and deformed vertices at Action start,
  condition frame, and Action end rather than raw archive bytes.
- [ ] Publish successful requested formats atomically as one export run.
- [ ] Commit `feat: export validated 3D assets`.
- [ ] Independent reviewer runs full offline round trip from a clean fixture.

## Task 10: Add Provider Plans, Approval Gates, Fake Transport, and GPT Image 2 Adapter

**Files:** provider plan/approval logic, provider port, `packages/provider-openai`, provider tests.

**No live request is part of this task.** Tests use a fake transport with sockets denied. This task
is required for the full v0.2 release, but Tasks 1-9 may land as a truthful local-only milestone.
`asset-mania-provider-openai` is a separate optional wheel discovered through an entry point; the
CLI wheel has no runtime dependency on it even though the root development workspace tests it.

### TDD sequence

- [ ] Write failing gate tests proving transport is unreachable before exact egress, paid-compute,
  and applicable rights receipts are valid for the same plan digest.
- [ ] Add subject declarations `non_person`, `synthetic_person`, `real_person`, `unknown`; never infer
  them from pixels. `unknown` fails with `SUBJECT_DECLARATION_REQUIRED` before provider planning or
  receipt evaluation; only `real_person` can satisfy `face_rights` with a plan-bound receipt.
- [ ] Test every approval-sensitive mutation, expiry, single-run scope, paid retry, provider/model
  substitution, attachment substitution, retention disclosure change, price ceiling change, and
  cancellation.
- [ ] Write request-normalization tests for `POST /v1/images/edits`, exact snapshot
  `gpt-image-2-2026-04-21`, ordered beauty/depth-preview/normal-preview/mask multipart attachment
  inventory, prompt hash, and no prompt in portable files. Bind all four to `image[]` indices `0..3`
  and assert the optional API `mask` part is absent; any field/index/role change invalidates approval.
  Close the
  controls to `n`, `size`, `quality`, `background`, `output_format`, `output_compression`, and
  `moderation`; reject transparent background, compression with PNG, `input_fidelity`, invalid
  custom dimensions, `n != 1`, `size=auto`, `quality=auto`, size unequal to conditioning resolution,
  conditioning resolution outside `1024x1024`/`1024x1536`/`1536x1024`, compression outside integer
  `0..100`, and every unknown field before approval. Do not estimate arbitrary custom sizes.
- [ ] Test the approval disclosure snapshot: official source URLs, retrieval timestamp and
  source-version/digest; default no-training-unless-opt-in; no application state; default 30-day
  abuse-monitoring retention; Zero Data Retention eligibility/approval; potential-CSAM review
  exception; effective region or explicit `unknown`; currency/rate timestamp and text/image input,
  output-token, `n`, size, and quality estimate assumptions. A stale or changed evidence digest must
  invalidate approval.
- [ ] Parse only the official standard token rates and published size/quality output-cost rows into a
  versioned estimator table. Reject missing/duplicate/unit-changed rows and never depend on an
  interactive calculator or an invented arbitrary-dimension formula.
- [ ] Fix executable evidence TTL at 24 hours. Test fail-closed `PROVIDER_EVIDENCE_STALE` before
  credential access and prove there is no implicit refresh. Implement the explicit
  `provider evidence refresh openai` command separately with an official-host allowlist and a new
  hashed artifact.
- [ ] Write response tests for base64 PNG/JPEG/WebP, size/MIME mismatch, oversized payload,
  moderation errors, user errors, 429/5xx retry classification, timeout, cancellation, request ID,
  returned usage, actual cost separate from the preflight estimate, and quarantine-before-publish.
- [ ] Implement injected transport and the minimum adapter. Secrets are obtained only at runtime
  through the secret interface and never argv/manifests/logs.
- [ ] Implement `view provider-plan` and `view generate` services. The plan command binds the private
  prompt hash, exact attachments/evidence/model/endpoint and exits `5` with missing gates. Successful
  generation passes through shared normalization and emits `view-v1` directly as `generated` with
  provider-issued declared alignment.
- [ ] Require `view generate --prompt-file` to reread the private prompt and match the plan digest
  before receipt consumption, credential access, or transport. Never persist prompt text.
- [ ] Do not automatically retry a paid request. Retry returns a new approval requirement.
- [ ] Run all tests with socket creation denied and inspect built distributions.
- [ ] Commit `feat: add approval-gated image provider`.
- [ ] Independent reviewer checks official API fields, cost/retention claims, and zero preapproval
  network reachability.

## Task 11: Integrate CLI, Agent Skill, Docs, Rules, and Versioned Distribution

**Files:** CLI execution commands/tests, Skill instructions/references/evals, README/docs/rules,
package versions/notices.

### TDD sequence

- [ ] Write CLI tests for every command's success, usage error, unavailable Blender, needs-approval,
  worker failure, storage failure, sanitized stdout/stderr, and no traceback.
- [ ] Cover the complete executable surface: scene preflight/plan/condition, approval issue, view
  ingest/provider-plan/generate, provider evidence refresh, texture bake, and export. Verify CLI
  kebab values normalize to the closed JSON enums and receipt/provider outputs feed the next command
  without an undocumented conversion.
- [ ] Keep CLI parsing thin; services remain independently testable.
- [ ] Extend the Skill from inspect-only to local stage routing while keeping external generation
  behind exact approval. Do not put implementation logic into prompts.
- [ ] Add independent forward evals for:
  - posed rig to conditioning bundle;
  - supplied generated view to baked asset;
  - invalid/misaligned view;
  - missing/unsupported Blender;
  - output collision;
  - external generation without approvals;
  - unknown subject always blocked pending declaration;
  - real-person subject without the exact plan-bound rights receipt;
  - no-silent-provider fallback.
- [ ] Update README capability table with separately verified local and optional-provider claims.
- [ ] Without a freshly approved capped live canary, label the OpenAI adapter only
  `experimental, contract-verified`; a working/live-verified claim is forbidden.
- [ ] Update architecture, getting started, manifest concepts, security/privacy, research, roadmap,
  rules, contribution, notices, and package version `0.2.0`.
- [ ] Correct the v0.1 research wording that treated stock TRELLIS.2/InstantMesh runtimes as
  permissively licensed: document their non-commercial/custom dependency closure and keep them
  research-only unless those dependencies are replaced and independently cleared.
- [ ] Validate Skill metadata, schema copies, cold/prepared checkout launch behavior, and docs links.
- [ ] Commit `docs: publish Blender workflow guide` or split code/docs commits if reviewable scope
  requires it.
- [ ] Independent skill evaluator runs natural-language scenarios without implementation context.

## Task 12: Add Pinned Blender CI, Release Gates, Whole-Branch Review, and Publication

**Files:** Blender E2E workflow/runner, release/license/schema checkers, provenance/notices, reports.

### Verification sequence

- [ ] Re-verify the Task 4 Blender 5.2.0 LTS and Khronos glTF Validator inventories, acquire them by
  the pinned URL/digest before isolation, and read back their exact versions. Do not redistribute
  either tool.
- [ ] Add a required Blender CPU E2E job for relevant paths, main, and release candidates.
- [ ] Treat pinned Linux x86_64 as the authoritative byte-exact oracle. Run macOS as required
  semantic/tolerance compatibility and never compare cross-platform binary/container hashes.
- [ ] Add an always-present stable aggregator check (for example `required / Blender E2E`) that
  succeeds only when the path decision says legitimately not applicable or the real Blender job
  passed. Branch protection requires the stable aggregator name, never a conditionally absent job.
- [ ] Run the E2E inside a network-deny boundary after dependencies are present.
- [ ] Add release failures for missing schema copy, GPL leakage, model/archive/weight additions,
  unprovenanced binary, unlisted dependency/tool, stale capability claims, or private sample names.
- [ ] Extend opaque-binary provenance to every tracked file and built archive member, not only test
  fixtures. Always reject caches/bytecode and unapproved model formats including `.pyc` and
  `.tflite`; any exceptional binary needs an exact public provenance/license inventory entry.
- [ ] Build an ignored local deny inventory containing the private archive/member content hashes.
  Scan every reachable Git object, working tree file, built distribution, generated archive, and
  downloaded GitHub Actions artifact against it without printing the private hashes/names. Record a
  sanitized zero-match result; `gitleaks` remains a separate secret scan, not media evidence.
- [ ] Enforce a GitHub Actions artifact-upload allowlist. Default Blender/face/provider tests upload
  no binary fixture, render, model, view, or source; any allowed sanitized report has a named schema,
  provenance, and retention period. Read back the final run's artifact list.
- [ ] Pin/inventory the exact gitleaks version and acquisition digest before the final scan.
- [ ] Build every Apache wheel/sdist and the separate GPL archive outside the repository; inspect
  file lists, metadata, and license contents.
- [ ] Run:

```bash
uv lock --check
make check
make test
make skill-check
make release-check
python scripts/check_license_boundary.py
python scripts/check_schema_distribution.py
python scripts/run_blender_e2e.py \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
gitleaks git --redact --no-banner
git diff --check
```

- [ ] Obtain a whole-branch spec/compliance/security/license review. Fix every Critical/Important
  finding, commit the fixes with scoped Conventional Commit messages, rerun the entire acceptance
  matrix, and obtain a clean re-review.
- [ ] Before fast-forward/push, resolve and read back the exact `jsc7727/asset-mania` remote and the
  recorded user authorization for this goal's publication scope. The current goal already grants
  that repository push authority; it does not authorize any paid provider call.
- [ ] Fast-forward `main`, push, wait for every GitHub Actions job, and read back final head,
  default branch, workflow conclusions, dependency alerts, security settings, topics, and public
  README/capability claims.
- [ ] Record workflow run IDs and every check name/conclusion for the exact publication SHA. Read
  back branch protection (including the stable Blender aggregator), private vulnerability reporting,
  dependency alerts/updates, and Actions artifact inventory rather than inferring settings.
- [ ] Do not create a release tag until all acceptance criteria and final public readback pass.
- [ ] Record the already-reviewed publication SHA in the execution ledger. No fix may be committed
  after push without a new local verification, re-review, push, and final readback cycle.

## Final Acceptance Matrix

| Requirement | Evidence |
| --- | --- |
| v1 compatibility | committed v1 byte/snapshot tests and existing 126-test baseline preserved |
| Closed v2 contracts | schema validation, unsafe-field negatives, canonical digest tests |
| Source read-only | before/after SHA-256 in every real E2E and failure path |
| GPL separation | source scan plus built wheel/sdist/GPL-archive inspection |
| Real conditioning | Blender-created EXR/PNG passes, matrices, pose/geometry digests |
| Real reprojection/bake | analytic fixture color/coverage/occlusion thresholds |
| Editable output | reopened derived BLEND with mesh, rig, UV, material, camera |
| Runtime output | glTF Validator zero errors and fresh GLB import fingerprint |
| Compatibility output | declared FBX subset fresh import fingerprint |
| Privacy | portable allowlists, redaction tests, no raw Blender output |
| No hidden external action | sockets denied for local/fake-provider suites |
| Approval binding | mutation/expiry/retry/subject-state gate tests |
| Skill usability | independent forward evals and real CLI invocation |
| Public delivery | final GitHub Actions and repository state readback |

## Subsequent Goal Phases

Completing this plan closes only the real 3D-guided-image/local-bake foundation. The active product
goal continues through:

1. `triposr-local` generic single-image-to-mesh provider with an explicit model-download/license
   gate covering code, model, preprocessing, and every runtime dependency revision, hash, license,
   and download receipt; require pre-masked input or an exact audited background-removal model and
   forbid an unpinned `rembg` default;
2. optional AliceVision true multi-image photogrammetry provider;
3. FLAME 2023 Open geometry-only face/head boundary with user-supplied assets, recorded license
   URL/version/digest, attribution and restricted-use acknowledgement, no automatic download, no
   FLAME texture model, and an independently cleared fitter; academic FLAME, DECA, MICA,
   Pixel3DMM, and their weights remain research-only, and FLAME itself is never described as an
   image fitter, identity-preservation system, or exact-likeness guarantee;
4. research-only adapters clearly separated from commercial capability claims;
5. complete generic/face Blender export E2Es, Skill/docs/CI, and public readback;
6. only then, the optional cloud execution and collaboration service.

The goal must not be marked complete after v0.2.
