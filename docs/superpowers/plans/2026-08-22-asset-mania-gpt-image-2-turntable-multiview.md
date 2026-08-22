# Asset Mania GPT Image 2 Turntable and Multi-view Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify an approval-gated GPT Image 2 eight-view face turntable whose eight local TripoSR meshes are fused by yaw-aware voxel consensus into a validated neutral GLB.

**Architecture:** Add three closed contracts, keep provider generation behind a new turntable-specific extension of the existing OpenAI adapter, audit the eight images locally without biometric claims, and implement fusion inside the optional TripoSR engine package. A maintainer E2E script composes planning, seven paid provider calls, local reconstruction, fusion, validation, and Blender preview without expanding the public CLI prematurely.

**Tech Stack:** Python 3.12, uv workspace, pytest, jsonschema 2020-12, Pillow, NumPy, Trimesh, optional local PyTorch/torchmcubes TripoSR runtime, GPT Image 2 Images API, Blender 5.2.

**Spec:** `docs/superpowers/specs/2026-08-22-asset-mania-gpt-image-2-turntable-multiview-design.md`

## Global Constraints

- Provider model is exactly `gpt-image-2-2026-04-21`; no model, provider, size, quality, or workflow substitution.
- Yaw schedule is exactly `[0, 45, 90, 135, 180, 225, 270, 315]`; pitch and roll are `0`.
- Yaw `0` is the observed source; the other seven views are generated and stay labeled `generated`.
- A real-person run requires exact `face_rights`, `external_egress`, and `paid_compute` receipts bound to one immutable turntable plan.
- Seven paid calls form one approved run, execute sequentially, and are never retried automatically.
- Source image, prompts, masks, generated views, meshes, and credentials never enter fixtures, logs, commits, telemetry, or galleries.
- Identity consistency remains `unmeasured`; no face detector, embedding, biometric score, or likeness guarantee is introduced.
- Generated views must pass the fixed structural audit before any TripoSR process is reachable.
- Fusion runs locally and offline, requires at least six closed winding-consistent meshes, and publishes only a watertight positive-volume neutral GLB.
- Every output is create-only beneath `.asset-mania/`; source bytes must remain identical.
- Public fixtures stay tiny, synthetic, redistributable, and non-human.

---

### Task 1: Closed turntable and multi-view contracts

**Files:**
- Create: `packages/contracts/src/asset_mania_contracts/turntable.py`
- Create: `packages/contracts/src/asset_mania_contracts/schema/turntable-plan-v1.schema.json`
- Create: `packages/contracts/src/asset_mania_contracts/schema/turntable-viewset-v1.schema.json`
- Create: `packages/contracts/src/asset_mania_contracts/schema/multiview-reconstruction-v1.schema.json`
- Create: `packages/contracts/tests/test_turntable_plan.py`
- Create: `packages/contracts/tests/test_turntable_viewset.py`
- Create: `packages/contracts/tests/test_multiview_reconstruction.py`
- Modify: `packages/contracts/src/asset_mania_contracts/execution.py`
- Modify: `packages/contracts/src/asset_mania_contracts/__init__.py`

**Interfaces:**
- Consumes: `canonical_digest`, `required_gates_for`, and `build_likeness_disclosure` from `asset_mania_contracts`.
- Produces:
  - `TURNTABLE_YAWS: tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 315)`
  - `build_turntable_plan(*, source_image_sha256: str, source_width: int, source_height: int, source_mask_sha256: str, source_cutout_sha256: str, prompt_sha256: str, provider_evidence_sha256: str, controls: Mapping[str, Any], subject: str, estimated_cost: str, maximum_cost: str) -> dict[str, Any]`
  - `build_turntable_viewset(*, plan_sha256: str, views: Sequence[Mapping[str, Any]], audit: Mapping[str, Any], reported_usage: Mapping[str, int | float], actual_cost: str | None) -> dict[str, Any]`
  - `build_multiview_reconstruction_record(*, turntable_plan_sha256: str, viewset_sha256: str, observed_source_image_sha256: str, meshes: Sequence[Mapping[str, Any]], fusion: Mapping[str, Any], fused_mesh: Mapping[str, Any], subject: str, rights_receipt_sha256: str | None) -> dict[str, Any]`
  - schema registry entries for `turntable-plan`, `turntable-viewset`, and `multiview-reconstruction`, all version `1.0`.

- [x] **Step 1: Write the failing plan contract tests**

```python
def test_real_person_plan_is_fixed_to_the_full_profile(validator_for):
    plan = build_turntable_plan(
        source_image_sha256="a1" * 32,
        source_width=1024,
        source_height=1024,
        source_mask_sha256="a2" * 32,
        source_cutout_sha256="a5" * 32,
        prompt_sha256="a3" * 32,
        provider_evidence_sha256="a4" * 32,
        controls={
            "size": "1024x1024",
            "quality": "medium",
            "background": "opaque",
            "output_format": "png",
            "moderation": "auto",
        },
        subject="real_person",
        estimated_cost="0.371000",
        maximum_cost="0.700000",
    )
    assert plan["yaws"] == list(TURNTABLE_YAWS)
    assert plan["model"] == "gpt-image-2-2026-04-21"
    assert plan["required_gates"] == ["face_rights", "external_egress", "paid_compute"]
    assert list(validator_for("turntable-plan", "1.0").iter_errors(plan)) == []
```

- [x] **Step 2: Run the plan test and verify RED**

Run: `uv run pytest packages/contracts/tests/test_turntable_plan.py -q`

Expected: collection fails because `build_turntable_plan` and the schema do not exist.

- [x] **Step 3: Implement the plan schema, builder, registry, and exports**

The builder fixes provider, endpoint, model snapshot, yaw schedule, pitch, roll, call count, prompt-template revision, overwrite policy, and required gates. It accepts only the source digests, evidence digest, closed controls, subject, and aggregate cost strings, then seals `plan_sha256` with `canonical_digest`.

- [x] **Step 4: Add mutation tests for every approval-bound field**

```python
@pytest.mark.parametrize("field", ["model", "yaws", "prompt_sha256", "controls", "maximum_cost"])
def test_editing_an_approval_bound_field_changes_the_digest(plan, field):
    mutated = copy.deepcopy(plan)
    mutated[field] = {"edited": True} if isinstance(mutated[field], dict) else "edited"
    preimage = {key: value for key, value in mutated.items() if key != "plan_sha256"}
    assert canonical_digest(preimage) != plan["plan_sha256"]
```

- [x] **Step 5: Write failing viewset and reconstruction-record tests**

The viewset test must require eight sorted records, observed yaw `0`, generated other yaws, no paths, `identity_consistency: unmeasured`, aggregate usage/cost, audit metrics, and a self-seal. The reconstruction test must require eight mesh records, fusion parameters, a neutral GLB record, and an eight-view likeness disclosure.

- [x] **Step 6: Run both new suites and verify RED**

Run: `uv run pytest packages/contracts/tests/test_turntable_viewset.py packages/contracts/tests/test_multiview_reconstruction.py -q`

Expected: failure because the builders and schemas are absent.

- [x] **Step 7: Implement the viewset and reconstruction builders**

Reject missing, duplicate, or reordered yaw records before sealing. Call `build_likeness_disclosure` with `views=8` for `face_head` output.

- [x] **Step 8: Run contract tests and verify GREEN**

Run: `uv run pytest packages/contracts/tests/test_turntable_plan.py packages/contracts/tests/test_turntable_viewset.py packages/contracts/tests/test_multiview_reconstruction.py -q`

Expected: all new contract tests pass.

- [x] **Step 9: Commit Task 1**

```powershell
git add packages/contracts
git commit -m "feat: add turntable multiview contracts"
```

---

### Task 2: Local source preparation and structural viewset audit

**Files:**
- Create: `packages/pipeline/src/asset_mania_pipeline/turntable.py`
- Create: `packages/pipeline/tests/test_turntable.py`
- Modify: `packages/pipeline/src/asset_mania_pipeline/__init__.py`
- Modify: `packages/pipeline/pyproject.toml`

**Interfaces:**
- Consumes: `prepare_input`, `sha256_file`, and the Task 1 builders.
- Produces:
  - `TurntableCandidate` dataclass with yaw, origin, image path, mask path, request ID, usage, and cost.
  - `prepare_turntable_source(image_path: Path, mask_path: Path, staging_root: Path) -> dict[str, Any]`
  - `derive_white_background_mask(image_path: Path, destination: Path) -> Path`
  - `audit_turntable(candidates: Sequence[TurntableCandidate]) -> dict[str, Any]`
  - `write_contact_sheet(candidates: Sequence[TurntableCandidate], destination: Path) -> Path`
  - `publish_turntable_viewset(*, plan: Mapping[str, Any], candidates: Sequence[TurntableCandidate], audit: Mapping[str, Any], actual_cost: str | None) -> dict[str, Any]`

- [x] **Step 1: Write failing source-preparation tests**

```python
def test_source_preparation_writes_an_rgba_cutout_without_changing_source(tmp_path):
    source, mask = synthetic_portrait_and_mask(tmp_path)
    before = sha256_file(source)
    prepared = prepare_turntable_source(source, mask, tmp_path / "run")
    assert prepared["cutout"].is_file()
    assert prepared["source_image_sha256"] == before
    assert sha256_file(source) == before
```

- [x] **Step 2: Run the focused test and verify RED**

Run: `uv run pytest packages/pipeline/tests/test_turntable.py::test_source_preparation_writes_an_rgba_cutout_without_changing_source -q`

Expected: import failure because `asset_mania_pipeline.turntable` is absent.

- [x] **Step 3: Implement bounded RGBA cutout preparation**

Reuse the existing bounded decoders, require equal image/mask dimensions, write create-only normalized PNGs below staging, zero hidden RGB, and return only paths plus digests to the caller.

- [x] **Step 4: Write audit boundary tests**

```python
@pytest.mark.parametrize(
    ("mutation", "diagnostic"),
    [
        ("missing_yaw", "VIEWSET_INCONSISTENT"),
        ("duplicate_pixels", "VIEWSET_INCONSISTENT"),
        ("off_center", "VIEWSET_INCONSISTENT"),
        ("border_contact", "VIEWSET_INCONSISTENT"),
        ("area_jump", "VIEWSET_INCONSISTENT"),
    ],
)
def test_structural_audit_fails_closed(turntable_candidates, mutation, diagnostic):
    candidates = mutate(turntable_candidates, mutation)
    result = audit_turntable(candidates)
    assert result["status"] == "failed"
    assert result["diagnostics"] == [diagnostic]
```

- [x] **Step 5: Run audit tests and verify RED**

Run: `uv run pytest packages/pipeline/tests/test_turntable.py -q`

Expected: failures because audit and mask functions are absent.

- [x] **Step 6: Implement white-background masks and the fixed audit**

Use Pillow plus NumPy. Background is the edge-connected near-white region; keep the largest foreground component. Compute coverage, normalized centroid, border-contact ratio, adjacent area ratios, byte digest, and decoded-pixel digest. Apply the exact thresholds from the spec and always emit `identity_consistency: unmeasured`.

- [x] **Step 7: Implement contact sheet and viewset publication**

The contact sheet is four columns by two rows, labeled with portable yaw text only, and written inside the private run. Publication calls `build_turntable_viewset`; it never embeds source paths or bytes.

- [x] **Step 8: Run pipeline tests and verify GREEN**

Run: `uv run pytest packages/pipeline/tests/test_turntable.py packages/pipeline/tests/test_reconstruction.py packages/pipeline/tests/test_views.py -q`

Expected: all selected tests pass.

- [x] **Step 9: Commit Task 2**

```powershell
git add packages/pipeline
git commit -m "feat: prepare and audit turntable views"
```

---

### Task 3: Approval-gated seven-call GPT Image 2 turntable provider

**Files:**
- Create: `packages/provider-openai/src/asset_mania_provider_openai/turntable.py`
- Create: `packages/provider-openai/src/asset_mania_provider_openai/live_transport.py`
- Create: `packages/provider-openai/tests/test_turntable_generation.py`
- Create: `packages/provider-openai/tests/test_live_transport.py`
- Modify: `packages/provider-openai/src/asset_mania_provider_openai/__init__.py`
- Modify: `packages/provider-openai/src/asset_mania_provider_openai/normalization.py`

**Interfaces:**
- Consumes: `consume_receipts`, `ProviderRequest`, `ProviderResponse`, `Transport`, Task 1 turntable plan, and Task 2 prepared cutout.
- Produces:
  - `build_turntable_prompt(base_prompt: str, yaw: int) -> str`
  - `build_turntable_request(plan: Mapping[str, Any], yaw: int, prompt: str, cutout: bytes) -> ProviderRequest`
  - `generate_turntable(*, plan: Mapping[str, Any], evidence: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]], base_prompt: str, cutout_path: Path, journal: ConsumptionJournal, now: datetime, consumed_at: str, transport: Transport, secret_resolver: SecretResolver) -> list[TurntableCallResult]`
  - `HTTPSMultipartTransport(connection_factory: Callable[[str, int], Any] | None = None)` implementing `Transport`.

- [x] **Step 1: Write the failing seven-call fake-transport test**

```python
def test_one_approved_run_sends_seven_ordered_calls(turntable_plan, evidence, receipts, cutout):
    transport = FakeTransport([png_response(yaw) for yaw in TURNTABLE_YAWS[1:]])
    results = generate_turntable(
        plan=turntable_plan,
        evidence=evidence,
        receipts=receipts,
        base_prompt="preserve the approved person",
        cutout_path=cutout,
        journal=ConsumptionJournal(cutout.parent / "journal"),
        now=NOW,
        consumed_at=NOW_TEXT,
        transport=transport,
        secret_resolver=lambda: "secret-not-recorded",
    )
    assert [result.yaw for result in results] == list(TURNTABLE_YAWS[1:])
    assert [sent["target_yaw"] for sent in transport.sent] == list(TURNTABLE_YAWS[1:])
```

- [x] **Step 2: Run the generation test and verify RED**

Run: `uv run pytest packages/provider-openai/tests/test_turntable_generation.py -q`

Expected: collection fails because the turntable provider does not exist.

- [x] **Step 3: Implement fixed prompt and request normalization**

The request uses `/v1/images/edits`, model `gpt-image-2-2026-04-21`, one `image[]` part named `source-cutout.png`, `n=1`, `1024x1024`, `medium`, opaque background, PNG, and the exact yaw-specific prompt. The redacted request record includes yaw and prompt digest, never prompt or bytes.

- [x] **Step 4: Implement approval ordering and sequential generation**

Verify evidence, plan seal, prompt-template digest, cutout digest, and all controls. Consume all three receipts once, resolve the secret once, then loop over yaws `45..315`. Validate each response before continuing. On the first exception, return no viewset and make no later call.

- [x] **Step 5: Add no-retry and quarantine tests**

```python
@pytest.mark.parametrize("failure_index", range(7))
def test_a_failed_paid_call_stops_without_retry(failure_index, prepared_run):
    transport = transport_failing_at(failure_index)
    with pytest.raises(ProviderTimeout):
        generate_turntable(
            plan=prepared_run.plan,
            evidence=prepared_run.evidence,
            receipts=prepared_run.receipts,
            base_prompt=prepared_run.base_prompt,
            cutout_path=prepared_run.cutout_path,
            journal=prepared_run.journal,
            now=prepared_run.now,
            consumed_at=prepared_run.consumed_at,
            transport=transport,
            secret_resolver=lambda: "secret-not-recorded",
        )
    assert len(transport.sent) == failure_index + 1
    assert not (prepared_run / "turntable-viewset.json").exists()
```

- [x] **Step 6: Write live transport tests with an injected fake HTTPS connection**

Assert exact host `api.openai.com`, POST endpoint, authorization header presence without logging, multipart field ordering, bounded response reads, status/request-id extraction, JSON rejection, timeout mapping, and refusal of redirects or another host.

- [x] **Step 7: Run transport tests and verify RED**

Run: `uv run pytest packages/provider-openai/tests/test_live_transport.py -q`

Expected: failure because `HTTPSMultipartTransport` is absent.

- [x] **Step 8: Implement the standard-library HTTPS transport**

Use `http.client.HTTPSConnection` with an injected factory, a random multipart boundary, exact-host construction, no proxy inheritance, an `Authorization` header containing the resolved bearer credential, and a `MAX_RESPONSE_BYTES + 1` bounded read. Return `ProviderResponse`; never print or persist the credential, multipart body, prompt, or image.

- [x] **Step 9: Run provider tests and verify GREEN**

Run: `uv run pytest packages/provider-openai/tests/test_turntable_generation.py packages/provider-openai/tests/test_live_transport.py packages/provider-openai/tests/test_transport_boundary.py packages/provider-openai/tests/test_response_validation.py -q`

Expected: all selected tests pass with sockets denied outside the injected live transport test.

- [x] **Step 10: Commit Task 3**

```powershell
git add packages/provider-openai
git commit -m "feat: generate approved GPT Image turntables"
```

---

### Task 4: Yaw-aware TripoSR voxel consensus

**Files:**
- Create: `packages/engine-triposr/src/asset_mania_engine_triposr/multiview.py`
- Create: `packages/engine-triposr/tests/test_multiview.py`
- Modify: `packages/engine-triposr/src/asset_mania_engine_triposr/__init__.py`

**Interfaces:**
- Consumes: local per-view GLB/PLY meshes, `TURNTABLE_YAWS`, existing bounded mesh repair, NumPy, Trimesh, and optional torchmcubes.
- Produces:
  - `YawMesh(yaw: int, path: Path, sha256: str)` dataclass.
  - `FusionSettings(grid_resolution: int = 192, minimum_votes: int | None = None)`.
  - `normalize_and_rotate(vertices: ndarray, yaw: int) -> ndarray`.
  - `vote_occupancy(grids: Sequence[ndarray], minimum_votes: int | None = None) -> ndarray`.
  - `fuse_turntable_meshes(inputs: Sequence[YawMesh], output_path: Path, settings: FusionSettings) -> FusionResult`.

- [x] **Step 1: Write failing pure normalization and voting tests**

```python
def test_known_yaw_is_removed_before_consensus():
    vertices = asymmetric_vertices()
    rotated = rotate_about_z(vertices, 90)
    restored = normalize_and_rotate(rotated, yaw=90)
    assert np.allclose(restored, normalize_and_rotate(vertices, yaw=0), atol=1e-6)


def test_four_of_eight_votes_survive_one_outlier():
    grids = [base_grid.copy() for _ in range(7)] + [outlier_grid]
    fused = vote_occupancy(grids)
    assert np.array_equal(fused, base_grid)
```

- [x] **Step 2: Run focused tests and verify RED**

Run: `uv run pytest packages/engine-triposr/tests/test_multiview.py -q`

Expected: import failure because `multiview.py` is absent.

- [x] **Step 3: Implement normalization, yaw removal, and occupancy voting**

Use `numpy.ptp(array, axis=0)`, never `ndarray.ptp`, so NumPy 1.x and 2.x both work. Centre on bounds, divide by longest extent, rotate `-yaw` around TripoSR-native +Z, and translate to the median centroid. Default votes are `ceil(count / 2)`.

- [x] **Step 4: Add failing mesh-validation tests**

Test refusal for fewer than six inputs, duplicate/missing yaw, open mesh, inconsistent winding, empty occupancy, existing output, and non-positive fused volume.

- [x] **Step 5: Run validation tests and verify RED**

Run: `uv run pytest packages/engine-triposr/tests/test_multiview.py -q`

Expected: failures because the end-to-end fusion function is incomplete.

- [x] **Step 6: Implement local mesh voxelization and extraction**

Load with `process=False`, normalize, rotate, voxelize and fill each mesh in the shared cube, vote, invoke CPU torchmcubes at level `0.5`, transform extracted vertices to world coordinates, orient faces, use the existing bounded hole policy, require watertight positive volume, and export create-only GLB through Trimesh.

Keep `torch` and `torchmcubes` imports inside the execution function so the Apache workspace can import and test the optional package without those runtime dependencies.

- [x] **Step 7: Add a runtime-gated synthetic fusion E2E**

```python
@pytest.mark.skipif(importlib.util.find_spec("torchmcubes") is None, reason="optional runtime")
def test_eight_noisy_ellipsoids_fuse_to_a_closed_glb(tmp_path):
    inputs = write_noisy_yaw_ellipsoids(tmp_path)
    result = fuse_turntable_meshes(inputs, tmp_path / "fused.glb", FusionSettings(48))
    assert result.manifold == "closed"
    assert result.signed_volume > 0
```

- [x] **Step 8: Run engine tests and verify GREEN**

Root workspace: `uv run pytest packages/engine-triposr/tests/test_multiview.py packages/engine-triposr/tests/test_port_triposr.py -q`

Installed optional runtime: `.asset-mania\triposr-venv\Scripts\python.exe -m pytest packages/engine-triposr/tests/test_multiview.py -q`

Expected: pure tests pass in both environments and the synthetic GLB E2E passes in the optional runtime.

- [x] **Step 9: Commit Task 4**

```powershell
git add packages/engine-triposr
git commit -m "feat: fuse yaw-aware TripoSR meshes"
```

---

### Task 5: Maintainer turntable multi-view E2E runner

**Files:**
- Create: `scripts/run_turntable_multiview_e2e.py`
- Create: `tests/test_turntable_multiview_e2e.py`
- Create: `tests/fixtures/v2/turntable-plan-v1.json`
- Create: `tests/fixtures/v2/turntable-viewset-v1.json`
- Create: `tests/fixtures/v2/multiview-reconstruction-v1.json`

**Interfaces:**
- Consumes: Tasks 1-4, existing engine clearance, approval receipts, provider evidence, OpenAI secret resolver, local TripoSR settings, and Blender preview launcher.
- Produces command groups:
  - `plan --image --mask --clearance --evidence --prompt-file --out`
  - `generate --plan PLAN --receipt FACE_RECEIPT --receipt EGRESS_RECEIPT --receipt PAID_RECEIPT --out RUNS_PARENT`
  - `reconstruct --viewset --clearance --engine-root --weights --hub-cache --out`
  - `verify --run --blender`

- [x] **Step 1: Write failing parser and offline-plan tests**

```python
def test_plan_is_offline_and_creates_no_provider_request(tmp_path, deny_sockets):
    image, mask, clearance, evidence, prompt = write_plan_inputs(tmp_path)
    code = main(
        [
            "plan",
            "--image",
            str(image),
            "--mask",
            str(mask),
            "--clearance",
            str(clearance),
            "--evidence",
            str(evidence),
            "--prompt-file",
            str(prompt),
            "--out",
            str(tmp_path / "runs"),
        ]
    )
    assert code == 0
    plan = load_only_child_json(tmp_path / "runs", "turntable-plan.json")
    assert plan["yaws"] == list(TURNTABLE_YAWS)
```

- [x] **Step 2: Run the E2E test and verify RED**

Run: `uv run pytest tests/test_turntable_multiview_e2e.py -q`

Expected: failure because the runner does not exist.

- [x] **Step 3: Implement create-only run layout and `plan`**

Run layout is `plan/`, `provider-quarantine/`, `viewset/`, `per-view-meshes/`, `fusion/`, and `verification/`. The source stays outside. Portable JSON contains no private path. Plan emits the exact acknowledgement strings needed for each gate but does not issue receipts.

- [x] **Step 4: Implement `generate` with injected provider dependencies**

The test entry accepts injected fake transport and secret resolver. The executable entry uses `HTTPSMultipartTransport` and resolves `OPENAI_API_KEY` only after approvals. On success, derive masks, audit, create the contact sheet, and publish `turntable-viewset.json`.

- [x] **Step 5: Implement `reconstruct` and fusion**

Run eight local TripoSR jobs sequentially, record each mesh, require six valid closed meshes, fuse with the selected profile, and publish `multiview-reconstruction.json` plus `fused.glb`.

- [x] **Step 6: Implement `verify`**

Validate all three schemas and seals, recompute every artifact hash, validate GLB structure, load it with Trimesh, require watertight positive volume, verify source hash, confirm all content remains under the run root, and call Blender to render a four-view preview.

- [x] **Step 7: Complete the deterministic fake-provider E2E test**

Use eight synthetic non-human ellipsoid portraits, fake provider responses, resolution `32`, fusion grid `48`, and a fake/optional extraction path. Assert ordered stages, no network, exact yaws, eight meshes, closed fused GLB, disclosure `views: 8`, and unchanged source.

- [x] **Step 8: Run deterministic E2E and verify GREEN**

Run: `uv run pytest tests/test_turntable_multiview_e2e.py -q`

Expected: all deterministic E2E tests pass without network, model downloads, paid calls, or real-person fixtures.

- [x] **Step 9: Commit Task 5**

```powershell
git add scripts/run_turntable_multiview_e2e.py tests/test_turntable_multiview_e2e.py tests/fixtures/v2
git commit -m "feat: add turntable multiview E2E runner"
```

---

### Task 6: Distribution, rules, and honest capability documentation

**Files:**
- Modify: `scripts/check_schema_distribution.py`
- Modify: `scripts/check_release.py`
- Modify: `skills/asset-mania/SKILL.md`
- Create: `skills/asset-mania/references/turntable-plan-v1.schema.json`
- Create: `skills/asset-mania/references/turntable-viewset-v1.schema.json`
- Create: `skills/asset-mania/references/multiview-reconstruction-v1.schema.json`
- Modify: `README.md`
- Modify: `docs/getting-started.md`
- Modify: `docs/research.md`
- Modify: `docs/security-and-privacy.md`
- Modify: `rules/agent/behavior-rules.md`
- Modify: `rules/testing/README.md`
- Modify: `THIRD_PARTY_NOTICES.md`
- Test: `tests/test_schema_distribution.py`
- Test: `tests/test_skill_distribution.py`
- Test: `tests/test_check_release.py`

**Interfaces:**
- Consumes: final command and artifact names from Task 5.
- Produces: distributed schema parity, maintainer usage instructions, approval language, live-E2E opt-in boundary, and measured capability statements.

- [x] **Step 1: Write failing distribution and publication tests**

Require all three schemas in contracts and Skill references byte-for-byte, reject private turntable assets and generated views from tracked/public content, and require README capability wording to distinguish fake E2E from live E2E.

- [x] **Step 2: Run release-focused tests and verify RED**

Run: `uv run pytest tests/test_schema_distribution.py tests/test_skill_distribution.py tests/test_check_release.py -q`

Expected: failures naming the three undistributed schemas and missing documentation clauses.

- [x] **Step 3: Distribute schemas and update rules**

Copy canonical schema bytes to Skill references. Document that GPT views are generated evidence, real-person viewsets need all three approvals, paid calls never retry, actual live E2E is opt-in, and identity remains unmeasured.

- [x] **Step 4: Update README and guides from measured results only**

Before the live run, describe the feature as deterministic/fake-transport verified and live-unverified. After Task 7, replace only the rows supported by live evidence and record actual counts, runtime, cost, viewset audit status, and mesh state without a likeness claim.

- [x] **Step 5: Run distribution, Skill, and release checks**

Run:

```powershell
uv run pytest tests/test_schema_distribution.py tests/test_skill_distribution.py tests/test_check_release.py -q
uv run python scripts/validate_skill.py skills/asset-mania
uv run python scripts/check_schema_distribution.py
uv run python scripts/check_release.py
```

Expected: all commands exit `0`.

- [x] **Step 6: Commit Task 6**

```powershell
git add README.md docs rules skills scripts/check_schema_distribution.py scripts/check_release.py THIRD_PARTY_NOTICES.md tests
git commit -m "docs: publish turntable multiview workflow"
```

---

### Task 7: Full verification and opt-in live E2E

**Files:**
- Modify after evidence: `README.md`
- Create only under ignored private storage: `.asset-mania/private-face-run/turntable-*/`
- Do not track: observed image, masks, prompts, provider responses, generated views, receipts, meshes, logs, or previews.

**Interfaces:**
- Consumes: completed Tasks 1-6, pinned policy/pricing evidence, user-authored approvals, `OPENAI_API_KEY`, local TripoSR runtime/assets, and Blender 5.2.
- Produces: one private eight-view viewset, eight per-view meshes, one fused GLB, contact sheet, preview, complete manifests, and verification evidence.

- [ ] **Step 1: Run focused suites from a clean process**

```powershell
uv run pytest packages/contracts/tests/test_turntable_plan.py packages/contracts/tests/test_turntable_viewset.py packages/contracts/tests/test_multiview_reconstruction.py -q
uv run pytest packages/pipeline/tests/test_turntable.py -q
uv run pytest packages/provider-openai/tests/test_turntable_generation.py packages/provider-openai/tests/test_live_transport.py -q
uv run pytest packages/engine-triposr/tests/test_multiview.py -q
uv run pytest tests/test_turntable_multiview_e2e.py -q
```

Expected: every focused suite passes with no warnings introduced by the feature.

- [ ] **Step 2: Run canonical repository checks**

On a supported environment with `make`:

```text
make check
make test
make skill-check
make release-check
```

On this Windows host, run the exact Makefile bodies because `make` is unavailable, and report existing POSIX-only failures separately:

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python scripts/validate_skill.py skills/asset-mania
uv run python scripts/check_release.py
```

- [ ] **Step 3: Verify the optional TripoSR runtime**

```powershell
.asset-mania\triposr-venv\Scripts\python.exe -m pytest packages/engine-triposr/tests/test_multiview.py -q
```

Expected: synthetic voxel fusion writes a watertight positive-volume GLB.

- [ ] **Step 4: Refresh provider evidence and inspect credentials safely**

Use the `openai-platform-api-key` skill. Never print the key. Refresh official model, policy, retention, and pricing evidence into a private artifact with a 24-hour TTL. If the key or verified organization access is absent, stop before issuing receipts and report the exact blocker.

- [ ] **Step 5: Build the real immutable plan and obtain exact acknowledgements**

Run `plan` against the approved face photo, mask, engine clearance, prompt file, and fresh provider evidence. Present the three exact plan-bound acknowledgement strings. Do not issue or consume a receipt until the user types each exact value.

- [ ] **Step 6: Execute the seven-call GPT Image 2 run**

Run `generate` once. Confirm model snapshot, request IDs, reported usage, actual cost, seven generated image hashes, eight-view order, structural audit, and contact sheet. A failure ends the run; do not retry.

- [ ] **Step 7: Execute eight local reconstructions and fusion**

First run TripoSR at resolution `128` for feasibility. If the audited set yields at least six closed meshes and fusion succeeds, create a new final local run at TripoSR resolution `256` and fusion grid `192`. Do not reuse consumed provider approvals; reuse the immutable generated viewset because local reconstruction is not a provider call.

- [ ] **Step 8: Verify final artifacts and render**

Run `verify`, then independently check:

```powershell
git diff --check
git status --short
```

Inspect the contact sheet and four-view Blender preview. Record actual mesh counts, volume, manifold, view audit metrics, elapsed time, provider usage/cost, and disclosure digest. Confirm source hash matches its pre-run value.

- [ ] **Step 9: Update measured documentation and commit**

Update README only with claims proved by Steps 6-8. If the live viewset or fusion fails, document the honest failed stage and keep the capability live-unverified.

```powershell
git add README.md
git commit -m "docs: record turntable multiview E2E evidence"
```

- [ ] **Step 10: Final review and completion audit**

Compare every acceptance criterion in the spec with a current file, test, or runtime artifact. Check commit diffs for private inputs, generated images, credentials, weights, opaque binaries, and unrelated changes. The goal is complete only if the live eight-view run and final fused GLB both have direct evidence.

#### 2026-08-22 OAuth execution evidence

The user explicitly replaced the direct API-key execution path with the Codex built-in
`imagegen` OAuth path. That tool disclosed neither a model snapshot nor request-level cost, so
the private provenance record says `codex-imagegen-oauth`, `model=unreported`, and
`cost=unreported`; it does not claim the pinned GPT Image 2 API snapshot.

- Eight yaw images were generated and the structural viewset audit passed. Identity
  consistency remains unmeasured.
- Head-only masks were required because shoulder and ponytail pixels dominated the first
  reconstruction attempt.
- Final TripoSR resolution `256` produced six closed and two open per-view meshes.
- Voxel resampling initially produced 47,352 disconnected closed fragments. One-cell lattice
  splatting, closing, hole filling, and largest-component retention reduced the output to one
  closed, positive-volume component.
- The resulting experimental fused GLB passed container, watertightness, winding, and volume
  checks, but failed Blender visual review because consensus removed recognizable facial
  detail. The capability therefore remains visually unverified and must not be described as a
  successful likeness reconstruction.
- Private evidence and deliverables remain under
  `.asset-mania/private-face-run/oauth-8view-20260822/`; no face image, generated view, mesh,
  receipt, or prompt is tracked.
