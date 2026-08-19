# Asset Mania Open-Source Skill Design

**Date:** 2026-08-19

**Status:** Approved for v0.1 implementation

**Repository:** `jsc7727/asset-mania`

**Initial release:** pre-alpha, source-only

## Purpose

Asset Mania will begin as an open-source Agent Skill backed by a deterministic CLI. It will establish a reproducible project format and safety contract for two eventual workflows:

1. **Image to editable 3D** for objects, stylized characters, and face/head assets.
2. **3D-guided image generation** where a Blender scene, rig, pose, and camera constrain an image-generation pass that can later be reprojected or baked onto the asset.

The open-source skill comes before the hosted product. A future cloud service must implement the same CLI/provider contract and produce the same manifests and artifacts, so local users are not locked into the hosted runtime.

## Product Sequence

The project will ship in four increments:

1. **v0.1 — Inspect and plan:** deterministic, offline, source-read-only inspection of supported inputs and the local runtime.
2. **v0.2 — 3D-guided image:** Blender render-pass planning and a BYOK GPT Image provider, with an explicit upload and cost gate.
3. **v0.3 — Generic image to 3D:** provider adapters for permissively licensed local engines such as TRELLIS.2 or InstantMesh.
4. **v0.4 — Face/head:** a separately gated face pipeline with explicit rights confirmation, provenance labels, and conservative likeness claims.

Cloud execution is considered only after the local contract, manifests, and validation fixtures are stable.

## Design Principles

- **CLI core, thin skill:** `SKILL.md` decides when and how to use deterministic commands; it does not hide pipeline logic in prompts.
- **Honest capability reporting:** unavailable or unfinished operations return stable diagnostics instead of silently selecting another model or pretending a 2D spin is true 3D.
- **Local-first by default:** v0.1 performs no network request, upload, model download, rendering, or GPU job.
- **Source-read-only:** inspection may create a new run directory but never modifies, moves, overwrites, or embeds data in the source asset.
- **Explicit provenance:** observed, derived, and generated information remain distinguishable in every future artifact.
- **Provider independence:** local, BYOK, custom remote, and hosted providers will share one interface without changing the user-facing project format.
- **Progressive disclosure:** shared routing stays in `SKILL.md`; workflow, manifest, safety, and provider details live in focused references.

## v0.1 Scope

### Included

The first public version exposes one command:

```text
asset-mania inspect <input> [--workflow image-to-3d|scene-to-image]
                           [--kind object|character|face-head]
                           [--out <directory>]
                           [--format json|text]
```

Option behavior is fixed for v0.1:

- `--format` controls stdout and defaults to `json`; persisted `manifest.json` and `report.json` are always JSON.
- `--out` selects the parent runs directory. The CLI always creates exactly one new timestamped child directory beneath it.
- For an image input, `--workflow` defaults to `image-to-3d`; for a `.blend` input, it defaults to `scene-to-image`.
- `image-to-3d` requires an image. `scene-to-image` requires a `.blend` file.
- `--kind` is valid only with `image-to-3d` and defaults to `object`. The value is user-declared; v0.1 performs no face recognition or identity classification.
- Declaring `--kind face-head` during local inspection does not request rights confirmation. It records an advisory that future external or generative processing will require confirmation.

It accepts:

- PNG, JPEG, and WebP images.
- Blender `.blend` files for header-level inspection.

It reports:

- Input existence, media type, byte size, and SHA-256.
- A strict image metadata allowlist: pixel width/height, bit depth, channel count or color model, alpha presence, numeric EXIF orientation, and presence-only flags for ICC, EXIF, IPTC, XMP, and GPS blocks.
- No camera model, capture time, creator, copyright, caption, contact, location value, free text, or unrecognized metadata value is emitted. Non-allowlisted blocks are represented only by type/presence diagnostics.
- Blender file header version, pointer size, endianness, syntactic validity, and whether a Blender executable is discoverable. v0.1 does not claim that the discovered executable can open the file.
- Host OS, architecture, Python version, Blender availability, supported provider capability, and missing prerequisites.
- Eligibility and diagnostics for the requested future workflow.
- A deterministic report plus a versioned run manifest.

### Excluded

v0.1 does **not**:

- Generate images, multiview sheets, geometry, textures, rigs, or animations.
- Call GPT Image or any other remote API.
- Download or execute model weights or installer scripts.
- Invoke Blender for rendering, scene mutation, export, or bake operations.
- Allocate a GPU or submit a remote job.
- Claim likeness reconstruction, photogrammetric accuracy, commercial clearance, or output ownership.
- Include the supplied face sample, its generated frames, real-person imagery, model weights, caches, datasets, or opaque binaries.

The README must call v0.1 an inspection and planning foundation, not a working image-to-3D generator.

## Run Directory and Manifest Contract

The default runs parent is `.asset-mania/runs/`, and a successful bootstrap creates:

```text
.asset-mania/runs/<UTC timestamp>-<short run id>/
├── manifest.json
├── report.json
└── logs/
```

The tool creates the run directory atomically and refuses to overwrite an existing directory. Portable outputs label inputs as `input-1`, `input-2`, and so on; they never contain source basenames or absolute paths. Source files are identified by content hash and are not copied by default.

`manifest.json` contains run-specific state:

```json
{
  "schema_version": "1.0",
  "run_id": "opaque-id",
  "command": "inspect",
  "tool_version": "0.1.0",
  "created_at": "UTC timestamp",
  "inputs": [],
  "environment": {},
  "parameters": {},
  "capabilities": {},
  "artifacts": [],
  "result": {},
  "warnings": []
}
```

Contract rules:

- Breaking changes require a new schema major version and migration documentation.
- Additive optional fields are allowed within schema v1.
- Portable manifests contain no API keys, tokens, cookies, signed URLs, private prompts, raw EXIF values, identity embeddings, or absolute source paths.
- Human-readable messages may change; machine-readable diagnostic codes are stable.
- After a run directory exists, failed, unsupported, and approval-blocked runs still finish with a valid manifest when the filesystem remains writable.
- Failure to create or continue writing the run directory is a bootstrap/storage exception: the CLI emits one sanitized diagnostic to stderr, exits nonzero, and does not claim that a manifest exists.
- `report.json` uses canonical key ordering and stable diagnostic ordering so identical inputs and parameters can be compared after masking run ID and timestamps.
- Every future artifact records a relative path, SHA-256, media type, provenance class (`observed`, `derived`, or `generated`), and validation status.

CLI stream and exit behavior:

- Exit `0`: inspection completed. A requested future workflow may still be reported as unavailable or planned inside the successful report.
- Exit `2`: invalid CLI usage or invalid input/workflow/`kind` combination; no run is created.
- Exit `3`: the run was created but the input was missing, unreadable, corrupt, or unsupported; manifest and report are written when possible.
- Exit `4`: sanitized internal failure after run creation; manifest and diagnostic are written when possible.
- Exit `73`: the output parent/run directory could not be created or written; stderr contains a sanitized bootstrap diagnostic and no manifest is promised.
- Exit `0`: stdout contains the report in the selected format and stderr is empty.
- Exit `2`: stdout is empty and stderr contains the usage diagnostic.
- Exit `3` or `4`: when `report.json` was written, stdout contains that report in the selected format and stderr contains the primary diagnostic; if storage is lost, behavior escalates to exit `73`.
- Exit `73`: stdout is empty and stderr contains only the sanitized storage diagnostic.

Diagnostics emitted by v0.1 include:

- `INPUT_NOT_FOUND`
- `UNSUPPORTED_MEDIA_TYPE`
- `INPUT_UNREADABLE`
- `EXIF_SENSITIVE_METADATA_PRESENT`
- `BLEND_HEADER_INVALID`
- `BLENDER_NOT_FOUND`
- `INTERNAL_ERROR`
- `OUTPUT_STORAGE_UNAVAILABLE` (stderr-only for exit 73; never persisted in a manifest)
- `WORKFLOW_NOT_IMPLEMENTED`

The following codes are reserved for later gated modes and are not emitted merely by inspecting a local file:

- `FACE_RIGHTS_CONFIRMATION_REQUIRED`
- `EXTERNAL_UPLOAD_APPROVAL_REQUIRED`
- `MODEL_DOWNLOAD_APPROVAL_REQUIRED`
- `PAID_COMPUTE_APPROVAL_REQUIRED`

## Repository Structure

```text
asset-mania/
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── AGENTS.md
├── pyproject.toml
├── uv.lock
├── scripts/
│   └── check_release.py
├── packages/
│   ├── contracts/
│   │   ├── pyproject.toml
│   │   └── src/asset_mania_contracts/
│   └── cli/
│       ├── pyproject.toml
│       └── src/asset_mania/
├── skills/asset-mania/
│   ├── SKILL.md
│   ├── agents/openai.yaml
│   ├── scripts/
│   └── references/
│       ├── cli-contract.md
│       ├── manifest-v1.schema.json
│       ├── safety-and-licenses.md
│       └── evals.md
├── docs/
│   ├── README.md
│   ├── getting-started.md
│   ├── architecture.md
│   ├── development.md
│   ├── concepts/
│   │   └── run-manifest.md
│   ├── security-and-privacy.md
│   ├── research.md
│   └── roadmap.md
├── rules/
│   ├── README.md
│   ├── index.md
│   ├── project/
│   ├── development/
│   ├── testing/
│   └── agent/
├── tests/
│   ├── fixtures/
│   │   └── PROVENANCE.md
│   └── test_inspect.py
└── .github/
    ├── ISSUE_TEMPLATE/
    └── pull_request_template.md
```

There is no skill-local README. The root README owns installation, status, examples, claims, and contribution guidance.

The root `pyproject.toml` defines a `uv` workspace and shared development tooling. `packages/contracts` owns the versioned schema, diagnostic codes, and portable contract types. `packages/cli` owns inspection and command behavior and depends on `asset-mania-contracts` through a workspace source. The skill invokes the installed CLI rather than duplicating its implementation. Future local or cloud providers become additional workspace packages without coupling the v0.1 CLI to their dependencies.

v0.1 supports CPython 3.11 through 3.13 on macOS 14 or newer and Ubuntu 22.04/24.04. Blender is optional because v0.1 reads only the portable file header; an absent executable is reported as a future capability limitation, not an inspection failure. Windows support and deep `bpy` inspection are outside v0.1.

A valid `.blend` header is exactly 12 bytes: the ASCII magic `BLENDER`, pointer marker `_` or `-`, endian marker `v` or `V`, and three ASCII version digits. The inspector accepts every syntactically valid version from `100` through `999` and reports the value without asserting load compatibility. Executable discovery checks configured paths and `PATH` without invoking Blender; it reports only `found` or `not_found`.

## Skill Behavior

The skill name is `asset-mania`. Automatic discovery remains enabled.

It should trigger for requests involving:

- Preflighting an image-to-3D or Blender-to-image workflow.
- Inspecting whether an image or `.blend` file is eligible for a supported pipeline.
- Planning Asset Mania generation without yet spending money or uploading data.

It should not trigger for ordinary image editing, unrelated Blender modeling, generic rendering questions, or existing commercial service support.

The v0.1 skill may run the offline `inspect` command directly. It must state that generation is not implemented and must not fabricate a provider result. Future operations that cross a permission gate must stop and request fresh approval.

For an otherwise valid input, `WORKFLOW_NOT_IMPLEMENTED` is a capability diagnostic inside a successful inspection rather than a command failure. Invalid input/workflow combinations fail as CLI usage before a run directory is created.

## Future Provider Boundary

Future providers implement these conceptual operations:

- `preflight(request) -> capabilities and diagnostics`
- `plan(request) -> immutable execution plan`
- `run(approved_plan) -> run handle`
- `status(run_handle) -> state and progress`
- `cancel(run_handle) -> terminal state`
- `fetch(run_handle) -> verified artifacts`
- `validate(artifacts) -> structured checks`

v0.1 does not freeze a cloud job schema. It records provider capability and planning diagnostics only. Cloud-specific fields may be added later without changing the core artifact/provenance contract.

## Permission and Stopping Rules

The default policy denies all actions beyond local inspection.

Before any future external or costly operation, the skill must show:

- Exact action and provider/model/revision.
- Files or derived data that will leave the machine.
- Known retention and region information, or an explicit statement that it is unknown.
- Estimated price, credits, model download size, GPU/runtime requirement, and output paths when known.
- Overwrite behavior and license/privacy implications.

It then requires fresh, explicit approval for that run. A global `--yes` flag cannot bypass upload, paid API, model-download, or paid-compute approval gates. Paid operations are not automatically retried. A retry requires an explained failure and renewed approval when it can incur cost or data egress.

The tool never silently changes provider, model, snapshot, quality, or workflow when a requested capability is unavailable.

## Face, Privacy, and Provenance

Face/head support is a distinct future workflow, not a subtype that silently reuses generic object behavior.

Required rules:

- Local processing is the default when a compatible face provider exists.
- External upload requires confirmation that the user has the right and consent to process the depicted person.
- No face input is added to examples, fixtures, telemetry, training data, galleries, or bug reports by default.
- Portable manifests omit identity embeddings, raw EXIF, absolute paths, and image bytes.
- Hidden or unobserved geometry is marked as generated or prior-filled, never reconstructed fact.
- Product copy avoids promises of exact likeness, anonymity, biometric safety, or legal clearance.
- Temporary and cached data locations and deletion commands must be documented before face execution ships.

Public tests use only tiny self-made or clearly redistributable synthetic fixtures.

## Licensing

- The standalone CLI/core, skill instructions, schemas, and documentation use Apache-2.0.
- Any future published Python code using Blender's `bpy` API lives under top-level `blender-addon/`, carries its own full GPL-3.0-or-later license and file headers, and is packaged separately from the Apache CLI wheel.
- The Blender component communicates with the core through the documented CLI/JSON/file boundary. It does not import Apache core modules into the `bpy` process, and root packaging must not present the optional add-on as Apache-licensed.
- The repository does not redistribute model weights. A future model registry records exact source URL, immutable revision, SHA-256, license/terms URL, attribution, access gate, and permitted-use restrictions.
- Runtime download does not imply that a model is open source or commercially usable.
- Third-party source, packages, assets, icons, HDRIs, textures, and fixtures are inventoried separately from model licenses in `THIRD_PARTY_NOTICES.md`. Each entry records source, version/revision, license, required notice, and redistribution evidence.
- Every tracked binary fixture must be listed in `tests/fixtures/PROVENANCE.md`. v0.1 fixtures are generated in-repository or explicitly dedicated under CC0; a release check fails when a binary fixture lacks an inventory entry.
- A root `NOTICE` file is added only when an included Apache-licensed dependency or asset requires preservation of an upstream notice.
- Generated outputs and user inputs are not relicensed by the repository license.

## README Contract

The initial README contains:

1. A concise statement of the long-term two-way workflow.
2. A prominent pre-alpha status banner explaining that v0.1 only inspects and plans.
3. A truthful capabilities table separating available, planned, and research-only behavior.
4. Installation and invocation for the working `inspect` command.
5. Example report output made from redistributable synthetic fixtures.
6. Architecture and run-manifest overview.
7. Local/offline, privacy, and approval guarantees scoped specifically to v0.1.
8. Model and component licensing boundaries.
9. Research sources and a roadmap.
10. Contribution and security-reporting links.

The README does not claim that Asset Mania currently creates 3D assets, preserves identity, is fully offline in future modes, or guarantees commercial-use rights.

## Error Handling

- User-correctable input errors produce a stable diagnostic and nonzero exit code without a traceback by default.
- Unexpected internal errors produce a sanitized diagnostic; verbose logs require an explicit debug flag.
- Manifests, reports, stdout, stderr, and logs use fixed input labels rather than basenames and redact credentials, query strings, absolute paths, EXIF values, and face-related metadata.
- Partial future artifacts are retained only inside the run directory and marked incomplete; they are never reported as successful output.
- A corrupt or unsupported input never triggers a network fallback.

## Validation and Evals

The first release includes:

- Skill structure validation with the bundled `quick_validate.py`.
- Unit tests for PNG, JPEG, WebP, valid `.blend` header, corrupt file, unsupported type, missing file, and unwritable output path.
- JSON Schema validation for every emitted manifest.
- Determinism tests that compare canonical reports after masking run ID and timestamp.
- Source-integrity tests that hash inputs before and after inspection.
- Collision tests proving that existing run directories are never overwritten.
- Secret tests proving that environment tokens, absolute home paths, and sensitive EXIF values do not appear in output.
- Backward-compatibility tests that current code can read committed schema-v1 fixtures.
- A CI matrix on CPython 3.11, 3.12, and 3.13 for Ubuntu 22.04, plus CPython 3.12 on Ubuntu 24.04 and macOS 14.
- `.blend` fixtures for valid versions `280` and `400`, plus invalid magic, pointer marker, endian marker, and version digits. No fixture is treated as proof that Blender can open the file.
- Independent forward evaluations using five fixed natural-language requests: inspect a generic image, preflight a declared face image, inspect a `.blend`, handle a missing Blender executable, and request paid cloud generation. For the paid-cloud request, the v0.1 skill may run only local inspection, must report `WORKFLOW_NOT_IMPLEMENTED`, and must perform no upload, approval prompt, network request, or paid action because no gated execution command exists yet. A pass requires truthful capability disclosure, no network activity, no source mutation, and the expected manifest/diagnostic behavior.

`python scripts/check_release.py` enforces the tracked-file denylist, fixture provenance inventory, third-party inventory, and common secret-pattern checks. `gitleaks git --redact --no-banner` performs the separate Git-history scan before the first push. Passing these checks is evidence for the documented release gate, not a guarantee that every possible licensing or secret issue has been detected.

## GitHub Publication

The repository will be public at `https://github.com/jsc7727/asset-mania` with `main` as the default branch.

The first public commit includes the working inspection foundation and community/security files. It does not create a release tag. The first source release is created only after v0.1 acceptance criteria pass and includes no weights, datasets, private samples, or opaque binaries.

Post-publication operations are owned by the `jsc7727` repository administrator and require the relevant GitHub permissions. They are not implementation acceptance criteria:

- Enable branch protection for `main` after the initial push.
- Enable Dependabot alerts when dependency manifests exist.
- Enable private vulnerability reporting.
- Add repository topics for agent skills, Blender, image-to-3D, 3D generation, and reproducible pipelines without claiming completed capabilities.

## v0.1 Acceptance Criteria

- The root README accurately distinguishes working, planned, and research-only capabilities.
- `asset-mania inspect` works offline on every supported fixture.
- The source input remains byte-identical after each test.
- Every run that successfully creates and retains a writable run directory emits a schema-valid manifest and deterministic report; bootstrap/storage failures follow the documented exit-73 exception.
- The skill passes structural validation and independent forward evaluation.
- The release checker, Git-history secret scan, third-party inventory review, and fixture provenance review all pass; no private face sample or downloaded model is intentionally tracked.
- Root Apache-2.0 scope and future Blender GPL boundary are explicit.
- Installation and tests succeed from fresh checkouts on the documented Python/OS CI matrix.
- GitHub authentication and repository-local commit identity use `jsc7727`; global company Git/GitLab identity remains unchanged.

## Cloud Evolution

The hosted service arrives only after v0.x proves that users value the workflow and the manifest contract is stable. It adds remote providers, GPU queues, artifact storage, collaboration, and organization controls behind the same `plan/run/status/cancel/fetch/validate` boundary. Local and BYOK providers remain first-class, and cloud artifacts remain portable to the open-source CLI.

The open-source project format, provenance model, validation checks, and provider interface are the durable product. The cloud service is an execution and collaboration option, not a separate incompatible product.
