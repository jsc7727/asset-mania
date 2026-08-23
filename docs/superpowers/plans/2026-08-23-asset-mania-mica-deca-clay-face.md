# MICA + DECA Clay Face Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, rights-gated MICA identity plus DECA detail pipeline that must prove face
geometry in untextured Blender clay renders before any head, hair, or generated texture work.

**Architecture:** Add a new closed face-geometry plugin v1 protocol beside the DAD-specific v0
protocol. Isolated MICA and DECA adapters emit only numeric FLAME geometry; original pipeline code
validates, fuses, exports neutral GLBs, and produces a sealed private Blender comparison. MICA owns
identity proportions, DECA contributes bounded face-region normal displacement, and failure ends
the increment without fallback.

**Tech Stack:** Python 3.11+ workspace, isolated user-supplied MICA/DECA Python runtimes, NumPy,
trimesh 4.0.5, Pillow, Blender 5.2 LTS, pytest, Ruff, canonical JSON, local CUDA.

**Spec:** `docs/superpowers/specs/2026-08-23-asset-mania-local-face-geometry-design.md`

## Global Constraints

- Do not change production code until the linked spec is explicitly approved.
- Real-person input requires a plan-bound `face_rights` receipt consumed before source open.
- MICA and DECA run locally with network denied; source images are never uploaded.
- Identity features, detector crops, landmarks, FLAME parameter vectors, and source pixels are
  transient only and must never be persisted.
- Public code does not bundle or download MICA, DECA, FLAME, InsightFace, detector code, datasets,
  runtimes, or weights.
- Live model acquisition requires a fresh explicit user approval and user-supplied licensed FLAME
  assets; implementation tests use synthetic non-human fixtures only.
- MICA positions are the identity authority. DECA may contribute only bounded normal displacement
  inside the tapered face region.
- No DAD substitution, generated-view geometry, Stable Diffusion, head assembly, hair, or texture
  is in scope.
- Plugin outputs are create-only, numeric-only, pickle-disabled, and closed to one `geometry.npz`.
- glTF uses metres, Y-up, -Z forward, outward CCW winding, neutral embedded material, and no image.
- Publication remains conservative: no biometric, identification-grade, scan-accuracy, or exact-
  likeness claim.
- Every clay GLB has a sealed `likeness-disclosure-v1` with ground truth unavailable and no face
  benchmark; the artifact and disclosure cannot be published independently.

---

### Task 1: Add the closed face-geometry plugin v1 protocol

**Files:**
- Create: `packages/pipeline/src/asset_mania_pipeline/face_geometry_plugins.py`
- Modify: `packages/pipeline/src/asset_mania_pipeline/__init__.py`
- Create: `packages/pipeline/tests/test_face_geometry_plugins.py`

**Interfaces:**
- Consumes: existing `canonical_json`, `sha256_file`, absolute private paths, and subprocess pattern
  from `face_plugins.py`.
- Produces: `MICA_PLUGIN`, `DECA_PLUGIN`, `FaceGeometryPluginRequest`,
  `FaceGeometryPluginResult`, `build_face_geometry_plugin_request`,
  `write_face_geometry_plugin_request`, `load_face_geometry_plugin_result`, and
  `run_face_geometry_plugin`.

- [ ] **Step 1: Write failing request and result inventory tests**

```python
def test_geometry_request_binds_rights_topology_and_network(tmp_path: Path) -> None:
    request = build_face_geometry_plugin_request(
        plugin="mica-local",
        profile="identity-neutral-v1",
        plugin_revision="a" * 40,
        source_image=(tmp_path / "source.png"),
        output_directory=(tmp_path / "output"),
        device="cuda",
        checkpoint_sha256="b" * 64,
        topology="flame-2020-5023",
        face_rights_receipt_sha256="c" * 64,
    )
    assert request.network == "denied-during-inference"
    assert request.topology == "flame-2020-5023"
    assert request.face_rights_receipt_sha256 == "c" * 64


def test_success_inventory_is_exactly_numeric_geometry(tmp_path: Path) -> None:
    request = geometry_request(tmp_path)
    request.output_directory.mkdir()
    write_valid_geometry_npz(request.output_directory / "geometry.npz")
    write_result_json(tmp_path / "result.json", request)
    result = load_face_geometry_plugin_result(tmp_path / "result.json", request)
    assert result.persisted_identity_feature_count == 0
    (request.output_directory / "crop.png").write_bytes(b"private")
    with pytest.raises(ValueError, match="inventory is unexpected"):
        load_face_geometry_plugin_result(tmp_path / "result.json", request)
```

Also cover wrong plugin/profile pairs, relative paths, non-CUDA device, malformed revisions or
digests, failed results exposing paths, extra JSON fields, nonzero persisted-feature count, and a
result checkpoint differing from its request.

- [ ] **Step 2: Run tests and verify the protocol is absent**

Run:

```powershell
uv run pytest packages/pipeline/tests/test_face_geometry_plugins.py -q
```

Expected: collection failure because `asset_mania_pipeline.face_geometry_plugins` does not exist.

- [ ] **Step 3: Implement the dataclasses and closed serializers**

```python
MICA_PLUGIN = "mica-local"
DECA_PLUGIN = "deca-local"
REQUEST_SCHEMA = "asset-mania.face-geometry-plugin-request.v1"
RESULT_SCHEMA = "asset-mania.face-geometry-plugin-result.v1"


@dataclass(frozen=True, slots=True)
class FaceGeometryPluginRequest:
    schema: str
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


@dataclass(frozen=True, slots=True)
class FaceGeometryPluginResult:
    schema: str
    plugin: Literal["mica-local", "deca-local"]
    profile: Literal["identity-neutral-v1", "detail-displacement-v1"]
    status: Literal[
        "succeeded", "incompatible_runtime", "invalid_output", "execution_failed"
    ]
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

Enforce the exact plugin/profile mapping, a 40-character lowercase Git SHA, 64-character lowercase
SHA-256 fields, absolute paths, source outside output, and the successful output inventory
`{"geometry.npz"}`. `run_face_geometry_plugin` must use `shell=False`, a fixed timeout, captured
output, sanitized environment, and reject source path or basename disclosure.

- [ ] **Step 4: Run focused tests and format checks**

```powershell
uv run pytest packages/pipeline/tests/test_face_geometry_plugins.py -q
uv run ruff check packages/pipeline/src/asset_mania_pipeline/face_geometry_plugins.py packages/pipeline/tests/test_face_geometry_plugins.py
uv run ruff format --check packages/pipeline/src/asset_mania_pipeline/face_geometry_plugins.py packages/pipeline/tests/test_face_geometry_plugins.py
```

Expected: all pass.

- [ ] **Step 5: Commit the protocol**

```powershell
git add packages/pipeline/src/asset_mania_pipeline/face_geometry_plugins.py packages/pipeline/src/asset_mania_pipeline/__init__.py packages/pipeline/tests/test_face_geometry_plugins.py
git commit -m "feat: add closed face geometry plugin protocol"
```

---

### Task 2: Validate numeric FLAME geometry and fuse bounded detail

**Files:**
- Create: `packages/pipeline/src/asset_mania_pipeline/face_geometry.py`
- Modify: `packages/pipeline/src/asset_mania_pipeline/__init__.py`
- Create: `packages/pipeline/tests/test_face_geometry.py`

**Interfaces:**
- Consumes: successful plugin `geometry.npz` files and sealed FLAME face-index arrays.
- Produces: `FaceGeometryData`, `FaceGeometryMeasurements`, `load_face_geometry`,
  `fit_similarity_transform`, `build_face_taper`, and `fuse_mica_deca_geometry`.

- [ ] **Step 1: Write failing geometry validation tests**

```python
def test_numeric_geometry_requires_exact_flame_topology(tmp_path: Path) -> None:
    path = tmp_path / "geometry.npz"
    write_geometry(path, vertices=np.zeros((5023, 3)), faces=FLAME_FACES)
    data = load_face_geometry(path, expected_topology=FLAME_FACES)
    assert data.vertices.shape == (5023, 3)
    assert data.faces.shape == (9976, 3)


@pytest.mark.parametrize("bad_key", ["embedding", "landmarks", "crop", "shape_parameters"])
def test_geometry_archive_rejects_private_feature_fields(tmp_path: Path, bad_key: str) -> None:
    path = write_geometry(tmp_path / "geometry.npz", extra={bad_key: np.zeros(1)})
    with pytest.raises(ValueError, match="inventory"):
        load_face_geometry(path, expected_topology=FLAME_FACES)
```

Cover object arrays, pickle payloads, wrong shapes, changed topology, non-finite values, invalid
projection, nonzero MICA displacement, degenerate triangles, and unexpected keys.

- [ ] **Step 2: Write failing similarity, taper, and displacement tests**

```python
def test_similarity_fit_recovers_known_metric_transform() -> None:
    source = ANALYTIC_VERTICES
    target = source @ ROTATION.T * 1.07 + TRANSLATION
    transform = fit_similarity_transform(source, target)
    assert np.max(np.abs(transform.apply(source) - target)) < 1e-9


def test_fusion_keeps_identity_positions_and_tapers_detail() -> None:
    result, measured = fuse_mica_deca_geometry(
        mica=MICA_FIXTURE,
        deca=DECA_FIXTURE,
        face_indices=FACE_INDICES,
        inner_face_indices=INNER_FACE_INDICES,
    )
    assert np.array_equal(result.faces, MICA_FIXTURE.faces)
    assert np.allclose(result.vertices[OUTSIDE_FACE], MICA_FIXTURE.vertices[OUTSIDE_FACE])
    assert measured.maximum_displacement_metres <= 0.003
    assert measured.rms_displacement_metres <= 0.0015
```

Add explicit failures for a `0.0031` metre maximum, `0.0016` metre RMS, less than `0.90` face-region
coverage, displacement outside the taper, or a non-similarity DECA alignment.

- [ ] **Step 3: Run tests and verify the implementation is absent**

```powershell
uv run pytest packages/pipeline/tests/test_face_geometry.py -q
```

Expected: import failure for `asset_mania_pipeline.face_geometry`.

- [ ] **Step 4: Implement numeric loading and measurements**

```python
@dataclass(frozen=True, slots=True)
class FaceGeometryData:
    vertices: np.ndarray
    faces: np.ndarray
    source_projection: np.ndarray
    detail_displacement: np.ndarray


@dataclass(frozen=True, slots=True)
class FaceGeometryMeasurements:
    vertex_count: int
    triangle_count: int
    non_manifold_edge_count: int
    winding_consistent: bool
    longest_extent_metres: float
    maximum_displacement_metres: float
    rms_displacement_metres: float
    face_displacement_coverage: float
    outside_face_displacement_count: int
```

Use `np.load(path, allow_pickle=False)`, require the exact four-key archive, copy arrays before the
archive closes, and compare face bytes to the expected topology. Compute adjacency from edges and
build the two-ring cosine taper as weights `1.0`, `0.75`, `0.25`, then `0.0` outside the face.

- [ ] **Step 5: Implement similarity alignment and fusion**

Use Umeyama similarity fitting over `inner_face_indices`. Transform DECA coarse positions into MICA
space, transform displacement magnitudes by the fitted scale, validate thresholds before applying,
and compute MICA area-weighted vertex normals. Apply `normal * displacement * taper`; never clamp,
fill, smooth, or change MICA base positions outside that expression.

- [ ] **Step 6: Run focused tests and commit**

```powershell
uv run pytest packages/pipeline/tests/test_face_geometry.py -q
uv run ruff check packages/pipeline/src/asset_mania_pipeline/face_geometry.py packages/pipeline/tests/test_face_geometry.py
git add packages/pipeline/src/asset_mania_pipeline/face_geometry.py packages/pipeline/src/asset_mania_pipeline/__init__.py packages/pipeline/tests/test_face_geometry.py
git commit -m "feat: validate and fuse local face geometry"
```

Expected: all pass.

---

### Task 3: Add the isolated MICA identity adapter

**Files:**
- Create: `packages/engine-mica/pyproject.toml`
- Create: `packages/engine-mica/LICENSE`
- Create: `packages/engine-mica/src/asset_mania_engine_mica/__init__.py`
- Create: `packages/engine-mica/src/asset_mania_engine_mica/plugin.py`
- Create: `packages/engine-mica/tests/test_plugin.py`

**Interfaces:**
- Consumes: `FaceGeometryPluginRequest(profile="identity-neutral-v1")`, user-supplied pinned MICA
  checkout, checkpoint, FLAME2020 asset, InsightFace model files, and isolated home.
- Produces: one `geometry.npz` with canonical metric FLAME geometry and zero detail displacement.

- [ ] **Step 1: Write failing runtime and privacy tests**

Test exact revision/checkpoint/FLAME hashes, missing external files, CPU-only runtime, wrong profile,
existing output, network denial, provider credential removal, and output inventory. Inject a fake
backend whose in-memory result includes `vertices`, `faces`, detected 2D points, and camera; assert
the written archive contains only the four permitted arrays and no input basename.

```python
def test_mica_worker_persists_geometry_but_no_identity_feature(tmp_path: Path) -> None:
    result = execute_mica_request(
        request_path=REQUEST,
        result_path=RESULT,
        settings=SETTINGS,
        backend=FakeMicaBackend(),
    )
    assert result == 0
    with np.load(OUTPUT / "geometry.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {
            "vertices", "faces", "source_projection", "detail_displacement"
        }
        assert np.count_nonzero(archive["detail_displacement"]) == 0
    assert not any(OUTPUT.parent.rglob("identity*.npy"))
```

- [ ] **Step 2: Run tests and verify the package is absent**

```powershell
uv run pytest packages/engine-mica/tests -q
```

Expected: collection failure because `asset_mania_engine_mica` does not exist.

- [ ] **Step 3: Implement the guarded launcher and runtime validation**

Follow the DAD launcher pattern without importing external MICA modules at package import time.
Require these environment variables: `ASSET_MANIA_MICA_SOURCE_ROOT`,
`ASSET_MANIA_MICA_ISOLATED_HOME`, `ASSET_MANIA_MICA_FLAME_PATH`, and
`ASSET_MANIA_MICA_FLAME_SHA256`. Verify Git revision and every digest before source open. Replace
`HOME`, `USERPROFILE`, `XDG_CACHE_HOME`, `TORCH_HOME`, and `HF_HOME` with private isolated paths;
remove provider credentials and disable sockets and Python HTTP clients.

- [ ] **Step 4: Implement the in-memory MICA bridge**

Use the official API directly rather than `demo.py`, because the demo writes ArcFace blobs,
aligned crops, identity codes, and landmarks. Inside the network-denied worker:

```python
cfg = get_cfg_defaults()
model_class = util.find_model_using_name("micalib.models", cfg.model.name)
mica = model_class(cfg, "cuda:0")
load_checkpoint(settings.checkpoint, mica)
detector = LandmarksDetector(model=detectors.RETINAFACE)
bboxes, keypoints = detector.detect(image_bgr)
face = choose_face_nearest_image_center(bboxes, keypoints)
arcface_blob, aligned_image = get_arcface_input(face, image_bgr)
with torch.no_grad():
    codedict = mica.encode(to_image_tensor(aligned_image), to_tensor(arcface_blob))
    decoded = mica.decode(codedict)
    vertices = decoded["pred_canonical_shape_vertices"][0]
```

Read faces from `mica.flameModel.generator.faces_tensor`. Fit a weak-perspective camera in memory
from MICA landmarks to detected points, project all 5,023 vertices, convert the canonical result to
metres and the contract's +X right, +Y up, -Z forward coordinates, restore outward winding, then
write only the numeric archive. Delete tensor references and call `torch.cuda.empty_cache()` before
writing the result JSON. Mark `ephemeral_identity_feature_used=true` and
`persisted_identity_feature_count=0`.

- [ ] **Step 5: Run adapter tests and commit**

```powershell
uv run pytest packages/engine-mica/tests packages/pipeline/tests/test_face_geometry_plugins.py -q
uv run ruff check packages/engine-mica
uv run ruff format --check packages/engine-mica
git add packages/engine-mica
git commit -m "feat: add isolated MICA geometry adapter"
```

Expected: all pass without installing MICA or torch in the workspace environment.

---

### Task 4: Add the isolated DECA detail adapter

**Files:**
- Create: `packages/engine-deca/pyproject.toml`
- Create: `packages/engine-deca/LICENSE`
- Create: `packages/engine-deca/src/asset_mania_engine_deca/__init__.py`
- Create: `packages/engine-deca/src/asset_mania_engine_deca/plugin.py`
- Create: `packages/engine-deca/tests/test_plugin.py`

**Interfaces:**
- Consumes: `FaceGeometryPluginRequest(profile="detail-displacement-v1")`, user-supplied pinned
  DECA checkout/checkpoint/FLAME assets, and isolated home.
- Produces: one `geometry.npz` containing DECA coarse FLAME geometry, source projection, and sampled
  per-vertex signed detail displacement in metres.

- [ ] **Step 1: Write failing runtime, sampling, and inventory tests**

Use an analytic 4x4 UV displacement gradient and known vertex UV coordinates to prove sampling
orientation and units. Cover wrong revision/checkpoint/FLAME hashes, absent CUDA, texture-model
requests, external URI access, non-finite displacement, and any persisted crop, landmark, albedo,
normal map, visualization, OBJ, MAT, or parameter vector.

- [ ] **Step 2: Run tests and verify the package is absent**

```powershell
uv run pytest packages/engine-deca/tests -q
```

Expected: collection failure because `asset_mania_engine_deca` does not exist.

- [ ] **Step 3: Implement the guarded launcher and DECA bridge**

Mirror the MICA process isolation. Set `deca_cfg.model.use_tex=False` and
`deca_cfg.model.extract_tex=False`. Use `datasets.TestData` only in memory and never call
`save_obj`, `savemat`, `cv2.imwrite`, or visualization helpers.

```python
testdata = datasets.TestData([source_path], iscrop=True, face_detector="fan")
image = testdata[0]["image"].to("cuda")[None, ...]
deca = DECA(config=deca_cfg, device="cuda")
with torch.no_grad():
    codedict = deca.encode(image, use_detail=True)
    opdict = deca.decode(codedict, return_vis=False, use_detail=True)
vertices = opdict["verts"][0]
projection = opdict["trans_verts"][0]
uv_displacement = opdict["displacement_map"][0, 0]
```

Sample `uv_displacement` at `deca.render.raw_uvcoords[0]` with bilinear `grid_sample`, convert model
units to metres, and write the exact numeric archive. Do not write DECA's code dictionary,
landmarks, UV texture, normals, images, albedo, or dense mesh.

- [ ] **Step 4: Run adapter tests and commit**

```powershell
uv run pytest packages/engine-deca/tests packages/pipeline/tests/test_face_geometry.py -q
uv run ruff check packages/engine-deca
uv run ruff format --check packages/engine-deca
git add packages/engine-deca
git commit -m "feat: add isolated DECA detail adapter"
```

Expected: all pass without external DECA dependencies in the workspace environment.

---

### Task 5: Export neutral clay GLBs and deterministic Blender comparisons

**Files:**
- Create: `packages/pipeline/src/asset_mania_pipeline/face_geometry_glb.py`
- Modify: `packages/pipeline/src/asset_mania_pipeline/__init__.py`
- Create: `packages/pipeline/tests/test_face_geometry_glb.py`
- Modify: `blender-addon/src/asset_mania_blender/preview/render_mesh_preview.py`
- Modify: `tests/test_blender_preview_selection.py`

**Interfaces:**
- Consumes: validated MICA, DECA, and fused `FaceGeometryData` in canonical glTF coordinates.
- Produces: `export_clay_glb(data, output_path) -> FaceGeometryMeasurements` and identical Blender
  front/eight-view renders with neutral material.

- [ ] **Step 1: Write failing GLB material and axis tests**

```python
def test_clay_glb_has_neutral_material_and_no_texture(tmp_path: Path) -> None:
    output = tmp_path / "clay.glb"
    export_clay_glb(ANALYTIC_FACE, output)
    document = validate_glb(output).json_chunk
    assert document["materials"][0]["name"] == "Asset Mania neutral clay"
    assert "images" not in document
    assert "textures" not in document
    assert document["asset"]["generator"] == "Asset Mania face geometry"
```

Add a Blender fixture whose nose points toward the declared front camera and image-up remains
world-up after glTF import. Test create-only output and external URI rejection.

- [ ] **Step 2: Implement neutral export**

Use one PBR material with base color `(0.62, 0.62, 0.64, 1.0)`, metallic `0.0`, roughness `0.55`,
no vertex colors, no texture, no extras containing paths, and metres as asset units. Preserve
validated normals and topology; do not run trimesh repair or processing.

- [ ] **Step 3: Add explicit front-start rendering**

Extend the Blender preview with `--start-angle-degrees` defaulting to `-90.0`. For face geometry,
pass `90.0` only if the canonical contract's front is +Y after Blender glTF import; prove the value
with the analytic nose fixture rather than choosing by visual trial. Keep `--orbit-axis Z`, eight
views, identical lighting, and `0.15` elevation.

- [ ] **Step 4: Run tests and commit**

```powershell
uv run pytest packages/pipeline/tests/test_face_geometry_glb.py tests/test_blender_preview_selection.py -q
uv run ruff check packages/pipeline/src/asset_mania_pipeline/face_geometry_glb.py blender-addon/src/asset_mania_blender/preview/render_mesh_preview.py
git add packages/pipeline/src/asset_mania_pipeline/face_geometry_glb.py packages/pipeline/src/asset_mania_pipeline/__init__.py packages/pipeline/tests/test_face_geometry_glb.py blender-addon/src/asset_mania_blender/preview/render_mesh_preview.py tests/test_blender_preview_selection.py
git commit -m "feat: export and render neutral face geometry"
```

Expected: synthetic GLB tests pass; real Blender tests run when pinned Blender is discoverable.

---

### Task 6: Add the create-only private geometry E2E runner

**Files:**
- Create: `scripts/run_face_geometry_e2e.py`
- Create: `tests/test_face_geometry_e2e.py`

**Interfaces:**
- Consumes: authorized source, rights store, two explicit plugin commands/Python executables,
  sealed external topology, corrected DAD clay baseline, and Blender 5.2.
- Produces: six create-only stages, canonical records, three clay GLBs, an eight-view comparison,
  existing `likeness-disclosure-v1` records, and a separate sealed manual review.

- [ ] **Step 1: Write failing plan and stage-order tests**

Use fake plugins and synthetic geometry. Assert the plan seals source digest, consumed receipt,
plugin/profile/revision/checkpoint, topology digest, runtime probe, thresholds, and overwrite policy.
Assert exact stage order and that no generated yaw path is accepted.

```python
def test_geometry_plan_binds_two_plugins_and_rights(tmp_path: Path) -> None:
    run = invoke_plan_with_synthetic_inputs(tmp_path)
    plan = load_json(run / "geometry/plan.json")
    assert plan["plugins"] == [
        {"plugin": "mica-local", "profile": "identity-neutral-v1"},
        {"plugin": "deca-local", "profile": "detail-displacement-v1"},
    ]
    assert len(plan["face_rights_receipt_sha256"]) == 64
    assert "yaw" not in json.dumps(plan).lower()
```

- [ ] **Step 2: Write failing resume, privacy, and verdict tests**

Cover existing outputs, changed source/model/topology/runtime, plugin failure, extra files, source
mutation, unreviewed verification, duplicate review, and failed manual verdict. Assert portable
records contain no absolute path, basename, face-derived feature, or private output hash beyond
the approved artifact digests.

- [ ] **Step 3: Implement the commands**

```text
geometry-plan
mica-run
deca-run
geometry-fuse
geometry-verify
geometry-review
```

`geometry-plan` consumes the rights receipt before any plugin source open. `mica-run` and
`deca-run` reuse the sealed source copy and deny network. `geometry-fuse` validates both archives,
exports the three clay GLBs, seals metrics, and writes one matching `likeness-disclosure-v1` per
mesh with `ground_truth_available=false` and `face_benchmark=null`. `geometry-verify` renders rows in fixed order:
MICA, DECA coarse, MICA+DECA fusion, corrected-axis DAD. `geometry-review` writes
`manual-review.json` without changing the immutable verification report.

- [ ] **Step 4: Run runner tests and commit**

```powershell
uv run pytest tests/test_face_geometry_e2e.py packages/pipeline/tests/test_face_geometry.py -q
uv run ruff check scripts/run_face_geometry_e2e.py tests/test_face_geometry_e2e.py
uv run ruff format --check scripts/run_face_geometry_e2e.py tests/test_face_geometry_e2e.py
git add scripts/run_face_geometry_e2e.py tests/test_face_geometry_e2e.py
git commit -m "feat: add private clay face geometry E2E"
```

Expected: all synthetic stages and privacy failures pass.

---

### Task 7: Harden distribution, licensing, and documentation boundaries

**Files:**
- Modify: `scripts/check_release.py`
- Modify: `scripts/check_license_boundary.py`
- Modify: `scripts/check_publication.py`
- Modify: `tests/test_check_release.py`
- Modify: `tests/test_license_boundary.py`
- Modify: `tests/test_provider_distribution.py`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `rules/agent/behavior-rules.md`
- Modify: `docs/research.md`

**Interfaces:**
- Consumes: new package names, private output names, and the spec's transient-biometric boundary.
- Produces: release failures for tracked MICA/DECA/FLAME/InsightFace assets and documented local,
  non-commercial, no-redistribution behavior.

- [ ] **Step 1: Write failing forbidden-content tests**

Add parametrized fixtures for `mica.tar`, `deca_model.tar`, `generic_model.pkl`, FLAME files,
InsightFace ONNX files, identity arrays, aligned crops, landmark arrays, face geometry NPZ/GLB,
real-person previews, and external source directories. Require sanitized diagnostics that never echo
private basenames or paths.

- [ ] **Step 2: Extend checkers and notices**

Reject model and private geometry artifacts regardless of directory. Permit only original adapter
source under `packages/engine-mica` and `packages/engine-deca`. Document that public adapter code is
Apache-2.0 but external MICA/DECA/FLAME dependencies remain under their own non-commercial terms
and are neither bundled nor redistributed.

- [ ] **Step 3: Document the narrow privacy amendment**

In `rules/agent/behavior-rules.md`, state that a future approved face-geometry v1 plan may use a
detector, landmarks, and an identity feature only transiently inside a local network-denied plugin,
after a rights receipt, with zero persisted feature count. Keep identity comparison, recognition,
telemetry, and claims forbidden.

- [ ] **Step 4: Run all distribution gates and commit**

```powershell
uv run pytest tests/test_check_release.py tests/test_license_boundary.py tests/test_provider_distribution.py -q
uv run python scripts/check_release.py
uv run python scripts/check_license_boundary.py
uv run python scripts/check_schema_distribution.py
uv run python scripts/check_publication.py
uv run python scripts/validate_skill.py skills/asset-mania
git diff --check
git ls-files .asset-mania
```

Expected: all commands exit `0`; `git ls-files .asset-mania` prints nothing.

```powershell
git add scripts/check_release.py scripts/check_license_boundary.py scripts/check_publication.py tests/test_check_release.py tests/test_license_boundary.py tests/test_provider_distribution.py THIRD_PARTY_NOTICES.md rules/agent/behavior-rules.md docs/research.md
git commit -m "docs: bound local face geometry research"
```

---

### Task 8: Run approved private clay evaluation and stop at the gate

**Files:**
- Create only: ignored private run under `.asset-mania/private-face-geometry-runs/`
- Modify public docs only after measured evidence: `docs/research.md`

**Interfaces:**
- Consumes: fresh explicit approvals for each external source/weight set, user-supplied licensed
  FLAME2020 files, authorized portrait, rights receipt, isolated runtimes, RTX 4070, and Blender.
- Produces: sealed private plan, plugin records, fusion measurements, clay comparison, and pass/fail
  manual verdict.

- [ ] **Step 1: Verify authority and external inputs without downloading**

Require the user to supply or explicitly approve acquisition of each MICA, DECA, and InsightFace
source/weight. Record exact Git revisions and SHA-256 values. Verify FLAME license acceptance and
local file digests without reading credentials. Stop if any approval, license, source, or hash is
missing.

- [ ] **Step 2: Build isolated runtimes and run synthetic smoke**

Create separate private environments outside the workspace dependency graph. Probe Python, torch,
CUDA, device, model load, and one synthetic non-human fixture with network denied. Do not open the
real portrait until both smoke records pass.

- [ ] **Step 3: Execute the six create-only stages**

```powershell
uv run python scripts/run_face_geometry_e2e.py geometry-plan --out .asset-mania/private-face-geometry-runs --source $env:ASSET_MANIA_FACE_SOURCE --rights-store $env:ASSET_MANIA_RIGHTS_STORE --mica-python $env:ASSET_MANIA_MICA_PYTHON --mica-plugin $env:ASSET_MANIA_MICA_PLUGIN --mica-revision $env:ASSET_MANIA_MICA_REVISION --mica-checkpoint-sha256 $env:ASSET_MANIA_MICA_CHECKPOINT_SHA256 --deca-python $env:ASSET_MANIA_DECA_PYTHON --deca-plugin $env:ASSET_MANIA_DECA_PLUGIN --deca-revision $env:ASSET_MANIA_DECA_REVISION --deca-checkpoint-sha256 $env:ASSET_MANIA_DECA_CHECKPOINT_SHA256 --flame-topology $env:ASSET_MANIA_FLAME_TOPOLOGY --flame-sha256 $env:ASSET_MANIA_FLAME_SHA256
uv run python scripts/run_face_geometry_e2e.py mica-run --run $env:ASSET_MANIA_FACE_GEOMETRY_RUN
uv run python scripts/run_face_geometry_e2e.py deca-run --run $env:ASSET_MANIA_FACE_GEOMETRY_RUN
uv run python scripts/run_face_geometry_e2e.py geometry-fuse --run $env:ASSET_MANIA_FACE_GEOMETRY_RUN
uv run python scripts/run_face_geometry_e2e.py geometry-verify --run $env:ASSET_MANIA_FACE_GEOMETRY_RUN --blender $env:ASSET_MANIA_BLENDER --dad-baseline $env:ASSET_MANIA_DAD_CLAY
```

Expected before review: every automated gate passes and verification records
`visual_quality="unreviewed"`.

- [ ] **Step 4: Inspect front and three-quarter clay renders**

Judge only geometry. Compare MICA, DECA, fusion, and corrected DAD under identical cameras. Record
`passed` only if MICA is visibly closer to the authorized source's visible proportions than DAD and
DECA detail improves relief without changing the MICA proportions.

```powershell
uv run python scripts/run_face_geometry_e2e.py geometry-review --run $env:ASSET_MANIA_FACE_GEOMETRY_RUN --verdict passed --reason "MICA clay improves visible face proportions over corrected DAD; DECA detail adds bounded relief without changing MICA identity proportions."
```

Use `--verdict failed` with the exact observed failure if any manual criterion is not met.

- [ ] **Step 5: Run completion verification**

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python scripts/validate_skill.py skills/asset-mania
uv run python scripts/check_release.py
uv run python scripts/check_license_boundary.py
uv run python scripts/check_schema_distribution.py
uv run python scripts/check_publication.py
git diff --check
git ls-files .asset-mania
git status --short
```

Expected: focused face-geometry tests and every repository gate pass; full-suite platform-specific
failures, if unchanged from the Windows baseline, are reported separately; no private file is
tracked.

- [ ] **Step 6: Apply the decision gate**

If the sealed manual verdict is `passed`, open a new spec and plan for fixed head/ear/neck template
fitting. If it is `failed`, stop and report that one authorized frontal image did not support the
required geometry quality. Do not begin head, hair, Stable Diffusion UV, texture, or fallback work
in either case without its own approved design.
