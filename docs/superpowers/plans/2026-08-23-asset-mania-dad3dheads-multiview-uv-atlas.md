# DAD-3DHeads Multi-View UV Atlas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a recognizable private DAD head by replacing sparse vertex colors with one
embedded visibility-aware UV texture atlas built from the observed front and seven existing local
generated yaw views.

**Architecture:** Extend the guarded DAD projection artifact with camera-space vertices, then add a
pure texture module that audits per-view z-buffer visibility, selects one eligible tile per
triangle, creates seam-specific UV vertices, and exports a one-material embedded-texture GLB. Add
four create-only maintainer E2E stages and verify the live result in Blender against all prior
outputs.

**Tech Stack:** CPython 3.11-3.12, NumPy, Pillow, trimesh 4.0.5, glTF 2.0/GLB, PyTorch
2.13.0+cu130, pinned DAD-3DHeads, Blender 5.2, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-23-asset-mania-dad3dheads-multiview-uv-atlas-design.md`

## Global Constraints

- Reuse only the existing pinned DAD checkout/checkpoint/runtime and private local viewset.
- Make no model download, provider call, upload, paid call, view generation, or fallback.
- Yaw 0 uses the successful run's normalized observed source; generated yaw 0 is never consumed.
- Yaws 45-315 are inferred/generated content and must remain labeled as such.
- Keep all face images, masks, projections, atlases, meshes, renders, and paths private and ignored.
- Every stage and artifact is create-only; immutable matching inputs may resume a failed stage only
  before a success output exists.
- Keep identity consistency `unmeasured`; numeric gates never prove likeness.
- Public tests use generated non-human shapes and colors only.
- Work inline on `v0-2-blender-pipeline`; do not create a subagent or new worktree.

---

### Task 1: Extend DAD projection evidence with camera-space vertices

**Files:**
- Modify: `packages/engine-dad3dheads/src/asset_mania_engine_dad3dheads/plugin.py`
- Modify: `packages/engine-dad3dheads/tests/test_plugin.py`
- Modify: `tests/test_face_plugin_e2e.py`

**Interfaces:**
- Consumes: one fixed-topology DAD inference result.
- Produces: `projection.npz` with `projected_vertices`, `camera_vertices`, and `image_shape`.

- [ ] **Step 1: Write a failing projection payload test**

Inject fake predictions and assert the saved archive contains finite `(N, 2)` projected vertices,
finite `(N, 3)` camera vertices, and a two-element image shape. Add a malformed-shape rejection.

```python
with np.load(projection, allow_pickle=False) as archive:
    assert archive["projected_vertices"].shape == (4, 2)
    assert archive["camera_vertices"].shape == (4, 3)
    assert archive["image_shape"].tolist() == [64, 64]
```

- [ ] **Step 2: Run the test and confirm RED**

```powershell
uv run pytest packages/engine-dad3dheads/tests/test_plugin.py -k camera_vertices -q
```

Expected: `camera_vertices` is absent.

- [ ] **Step 3: Save the exact predicted vertices**

Write the same validated `vertices` array used for `head.obj` into `projection.npz` under
`camera_vertices`. Do not transform or normalize it. Keep the closed result/output inventory
unchanged.

- [ ] **Step 4: Update fake-plugin projection fixtures**

Every fake result used by staged E2E writes camera vertices matching its OBJ topology. Keep
real-person data absent.

- [ ] **Step 5: Verify and commit**

```powershell
uv run pytest packages/engine-dad3dheads/tests/test_plugin.py tests/test_face_plugin_e2e.py -q
uv run ruff check packages/engine-dad3dheads/src/asset_mania_engine_dad3dheads/plugin.py packages/engine-dad3dheads/tests/test_plugin.py tests/test_face_plugin_e2e.py
git add packages/engine-dad3dheads/src/asset_mania_engine_dad3dheads/plugin.py packages/engine-dad3dheads/tests/test_plugin.py tests/test_face_plugin_e2e.py
git commit -m "feat: preserve DAD camera-space projections"
```

---

### Task 2: View visibility and deterministic triangle selection

**Files:**
- Create: `packages/engine-dad3dheads/src/asset_mania_engine_dad3dheads/texture.py`
- Modify: `packages/engine-dad3dheads/src/asset_mania_engine_dad3dheads/__init__.py`
- Create: `packages/engine-dad3dheads/tests/test_texture.py`

**Interfaces:**
- Consumes: eight `DADTextureView` records, fixed faces, and face-region indices.
- Produces: `ViewVisibility`, `select_triangle_views`, and per-triangle tile assignments.

- [ ] **Step 1: Write failing closed-input tests**

Define these exact types:

```python
@dataclass(frozen=True, slots=True)
class DADTextureView:
    yaw: int
    origin: Literal["observed", "generated"]
    image_path: Path
    mask_path: Path
    projection_path: Path


@dataclass(frozen=True, slots=True)
class ViewVisibility:
    yaw: int
    eligible: np.ndarray
    score: np.ndarray
    visible_pixels: np.ndarray
```

Reject missing/reordered yaws, wrong origin labels, non-1024 images/masks, mismatched vertex counts,
non-finite arrays, wrong projection dimensions, or topology mismatch.

- [ ] **Step 2: Confirm RED**

```powershell
uv run pytest packages/engine-dad3dheads/tests/test_texture.py -k "input or order" -q
```

- [ ] **Step 3: Implement camera-facing and mask eligibility**

Scale projections from 1024 to 512. Compute triangle normals in each view's camera-space geometry.
Use negative raw Z as camera-facing. Require all projected vertices in bounds and either all three
mask pixels or centroid plus two mask pixels.

- [ ] **Step 4: Write and run a failing z-buffer occlusion test**

Create front and rear triangles with identical projected pixels and depths `-0.4` and `0.2`.
Assert only the front triangle owns pixels and only it is eligible. Require at least four owned
pixels within `1e-6` depth tolerance.

- [ ] **Step 5: Implement deterministic raster depth ownership**

Rasterize triangle bounding boxes at 512 square with barycentric coordinates. Interpolate raw Z,
retain the minimum depth per pixel, then count pixels owned by each face.

- [ ] **Step 6: Write selection-priority tests**

Prove yaw 0 receives the `4.0` multiplier for indexed face triangles only after eligibility, the
`1.5` front-head multiplier elsewhere, generated views have no provenance bonus, and ties resolve
by visible pixels, wrapped yaw magnitude, then yaw number. Unseen faces select neutral tile `8`.

- [ ] **Step 7: Implement and commit**

```powershell
uv run pytest packages/engine-dad3dheads/tests/test_texture.py -q
uv run ruff check packages/engine-dad3dheads/src/asset_mania_engine_dad3dheads/texture.py packages/engine-dad3dheads/tests/test_texture.py
git add packages/engine-dad3dheads/src/asset_mania_engine_dad3dheads/texture.py packages/engine-dad3dheads/src/asset_mania_engine_dad3dheads/__init__.py packages/engine-dad3dheads/tests/test_texture.py
git commit -m "feat: select visible DAD texture views"
```

---

### Task 3: Atlas, seam UVs, embedded material, and metrics

**Files:**
- Modify: `packages/engine-dad3dheads/src/asset_mania_engine_dad3dheads/texture.py`
- Modify: `packages/engine-dad3dheads/src/asset_mania_engine_dad3dheads/__init__.py`
- Modify: `packages/engine-dad3dheads/tests/test_texture.py`

**Interfaces:**
- Consumes: observed geometry, eight views, triangle tile assignments, and output paths.
- Produces: atlas PNG, embedded-texture GLB, and `DADTextureMeasurements`.

- [ ] **Step 1: Write failing atlas layout and hash tests**

Use eight solid-color 1024 images and masks. Assert the 1536-square atlas has the fixed 3-by-3 tile
layout, Lanczos resizing, neutral tile `(160, 145, 140)`, deterministic bytes, and create-only
behavior.

- [ ] **Step 2: Implement `build_texture_atlas`**

```python
def build_texture_atlas(views: Sequence[DADTextureView], output_path: Path) -> Image.Image: ...
```

Use two-pixel tile insets for UV generation; atlas pixels themselves retain the complete resized
tile.

- [ ] **Step 3: Write failing seam-vertex tests**

Assign adjacent faces sharing one original vertex to different tiles. Assert output vertices are
keyed by `(original_vertex, tile)`, faces retain their count/order, UVs differ at the seam, and
copied smooth normals remain equal.

- [ ] **Step 4: Implement seam remapping and UV conversion**

Convert image-down projected pixel centres into the selected tile and glTF UV-up coordinates.
Neutral faces use the neutral tile centre. Copy positions from the fixed observed geometry after
the existing DAD-to-Blender transform.

- [ ] **Step 5: Write failing textured GLB readback tests**

Require one embedded image, texture, and PBR material, one `TEXCOORD_0` accessor, no external URI,
9,976 preserved faces in the full fixture, under 40,184 vertices, fixed components, zero
non-manifold edges, and consistent winding.

- [ ] **Step 6: Implement export and exact measurements**

Define:

```python
@dataclass(frozen=True, slots=True)
class DADTextureMeasurements:
    triangle_count: int
    vertex_count: int
    textured_triangle_fraction: float
    textured_surface_area_fraction: float
    observed_face_area_fraction: float
    generated_surface_area_fraction: float
    neutral_surface_area_fraction: float
    yaw_triangle_counts: dict[int, int]
    back_projection_violation_count: int
    non_manifold_edge_count: int
    winding_consistent: bool
```

Implement `build_textured_dad_glb(...) -> DADTextureMeasurements`, enforce every spec threshold,
and independently parse the GLB JSON/material/image chunks after export.

- [ ] **Step 7: Verify and commit**

```powershell
uv run pytest packages/engine-dad3dheads/tests/test_texture.py packages/engine-dad3dheads/tests/test_mesh.py -q
uv run ruff check packages/engine-dad3dheads
git add packages/engine-dad3dheads/src/asset_mania_engine_dad3dheads packages/engine-dad3dheads/tests/test_texture.py
git commit -m "feat: export textured DAD UV atlases"
```

---

### Task 4: Texture planning and seven-view inference stages

**Files:**
- Modify: `scripts/run_face_plugin_e2e.py`
- Modify: `tests/test_face_plugin_e2e.py`

**Interfaces:**
- Consumes: successful DAD run, ordered private viewset, explicit CUDA Python/plugin, masks, and
  existing acquisition/smoke records.
- Produces: sealed `texture/plan.json`, seven view projection outputs, and `texture/infer.json`.

- [ ] **Step 1: Write failing `texture-plan` tests**

Build a synthetic ordered viewset. Assert the plan records yaws, origin labels, image/mask hashes,
profile `dad-multiview-uv-atlas-v1`, tile/atlas sizes, fixed gates, checkpoint/revision, no paths,
and a canonical digest. Reject generated yaw-0 consumption.

- [ ] **Step 2: Implement create-only `texture-plan`**

Add `--run` and `--views`. Use the existing observed `inference/source.png` and front projection,
then yaws 45-315 from `--views`.

- [ ] **Step 3: Write failing `texture-infer` tests**

Inject a recording fake plugin. Assert exactly seven calls in yaw order, no call for yaw 0, same
checkpoint/runtime/environment each time, immutable request resume, source integrity, and no
fallback after failure.

- [ ] **Step 4: Implement sequential generated-view inference**

Write each result under `texture/views/yaw-###/`. Verify the returned topology/counts and extended
projection payload immediately. Seal image, mask, OBJ, projection, and result hashes into private
`texture/infer.json` without serializing paths.

- [ ] **Step 5: Verify and commit**

```powershell
uv run pytest tests/test_face_plugin_e2e.py -k "texture_plan or texture_infer" -q
uv run ruff check scripts/run_face_plugin_e2e.py tests/test_face_plugin_e2e.py
git add scripts/run_face_plugin_e2e.py tests/test_face_plugin_e2e.py
git commit -m "feat: infer DAD multi-view texture projections"
```

---

### Task 5: Texture build, verification, and Blender comparison stages

**Files:**
- Modify: `scripts/run_face_plugin_e2e.py`
- Modify: `tests/test_face_plugin_e2e.py`

**Interfaces:**
- Consumes: sealed texture plan/inference, external face indices, observed geometry, Blender, and
  three prior GLBs.
- Produces: private atlas, textured GLB, metrics, four-row comparison, and manual-verdict report.

- [ ] **Step 1: Write failing `texture-build` E2E test**

Inject analytic view projections and a private synthetic face-index file. Assert source hashes and
topology are rechecked, atlas/GLB are create-only, face-index digest is recorded, metrics are
sealed, paths are absent, and identity remains unmeasured.

- [ ] **Step 2: Implement `texture-build`**

Load the external pinned `face.npy`, call `build_textured_dad_glb`, validate the GLB container and
embedded material, and write `texture/build.json`.

- [ ] **Step 3: Write failing `texture-verify` comparison tests**

Inject a preview runner and require four identical render calls in fixed row order: textured DAD,
sparse-color DAD, TripoSR anchor, TripoSR hybrid. Recheck original/viewset hashes and output GLB
hash. Report `visual_quality="unreviewed"` and `identity_consistency="unmeasured"`.

- [ ] **Step 4: Implement comparison and interactive-output metadata**

Use 16 samples, 500 resolution, four views, and real texture material. Write a four-row contact
sheet and private report. Include the DAD face-facing axis so Computer Use can open the correct
Blender view later.

- [ ] **Step 5: Verify and commit**

```powershell
uv run pytest tests/test_face_plugin_e2e.py packages/engine-dad3dheads/tests/test_texture.py -q
uv run ruff check scripts/run_face_plugin_e2e.py tests/test_face_plugin_e2e.py
git add scripts/run_face_plugin_e2e.py tests/test_face_plugin_e2e.py
git commit -m "feat: verify textured DAD faces in Blender"
```

---

### Task 6: Publication and research boundaries

**Files:**
- Modify: `tests/test_check_release.py`
- Modify: `scripts/check_release.py`
- Modify: `scripts/check_publication.py`
- Modify after deterministic evidence: `README.md`
- Modify after deterministic evidence: `docs/getting-started.md`
- Modify after deterministic evidence: `docs/research.md`

- [ ] **Step 1: Write failing distribution tests for the new artifact classes**

Prove tracked projection `.npz`, copied FLAME `.npy`, and private texture records are rejected.
Retain the existing PNG/GLB and `.asset-mania` coverage as regression assertions.

- [ ] **Step 2: Reject NumPy projection/index binaries explicitly**

Add `.npz` and `.npy` to the release/publication binary suffix sets. Preserve public synthetic
source fixtures and adapter code; tests create NumPy arrays at runtime instead of tracking them.

- [ ] **Step 3: Document deterministic status only**

Before live review, state fake-plugin/synthetic atlas verification and `live texture quality
unverified`. Preserve DAD's non-commercial restriction and inferred-view provenance.

- [ ] **Step 4: Run gates and commit**

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest packages/engine-dad3dheads/tests tests/test_face_plugin_e2e.py tests/test_check_release.py -q
uv run python scripts/validate_skill.py skills/asset-mania
uv run python scripts/check_license_boundary.py
uv run python scripts/check_schema_distribution.py
uv run python scripts/check_publication.py
uv run python scripts/check_release.py
git diff --check
git add README.md docs/getting-started.md docs/research.md scripts/check_release.py scripts/check_publication.py tests/test_check_release.py
git commit -m "docs: bound DAD multi-view texture research"
```

---

### Task 7: Actual private atlas E2E and completion audit

**Live correction:** Task 7 uses the spec's `dad-multiview-fixed-uv-blend-v2` correction after the
v1 per-triangle atlas failed manual seam review. The model, checkpoint, views, and plugin remain
unchanged. A fresh create-only texture attempt is required; v1/v2 evidence is retained.

**Files:**
- Create only under the ignored successful DAD run's `texture/` and `verification/` directories.
- Modify `README.md` and `docs/research.md` only from measured evidence.

- [x] **Step 1: Prove the current run and viewset inputs**

Verify the pinned checkpoint/revision, compatibility patch, successful observed inference, original
source hash, ordered view/mask hashes, CUDA Python, plugin executable, Blender 5.2, and absence of
existing texture outputs.

- [x] **Step 2: Run the four live texture stages with fixed-UV profile v2**

```powershell
uv run python scripts/run_face_plugin_e2e.py texture-plan --run $env:ASSET_MANIA_DAD_RUN --views $env:ASSET_MANIA_DAD_VIEWS
uv run python scripts/run_face_plugin_e2e.py texture-infer --run $env:ASSET_MANIA_DAD_RUN --python $env:ASSET_MANIA_DAD_PYTHON --plugin-command $env:ASSET_MANIA_DAD_PLUGIN
uv run python scripts/run_face_plugin_e2e.py texture-build --run $env:ASSET_MANIA_DAD_RUN
uv run python scripts/run_face_plugin_e2e.py texture-verify --run $env:ASSET_MANIA_DAD_RUN --source $env:ASSET_MANIA_FACE_SOURCE --views $env:ASSET_MANIA_DAD_VIEWS --blender $env:ASSET_MANIA_BLENDER --sparse-dad $env:ASSET_MANIA_DAD_SPARSE --triposr-anchor $env:ASSET_MANIA_TRIPOSR_ANCHOR --triposr-hybrid $env:ASSET_MANIA_TRIPOSR_HYBRID
```

- [x] **Step 3: Inspect the atlas, eight-view renders, and Blender material**

Open the atlas and comparison. Import the textured GLB into an empty Blender scene, frame the mesh,
switch to material preview, choose the face-facing axis, and confirm the texture is visible without
node edits.

- [x] **Step 4: Record the manual verdict honestly**

Set `visual_quality=passed` only when the original person is recognizable from front and one
three-quarter view, central facial features align, no rear face leakage exists, no central seam is
obvious, and the result improves on sparse DAD. Otherwise set `failed` with the exact reason.

- [x] **Step 5: Run full verification and private-data audit**

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

Require focused tests and every repository gate to pass. Report the known Windows/POSIX failures
separately from new regression evidence. Require no tracked private file.

- [x] **Step 6: Commit measured documentation and finish**

Update public wording only with measured atlas coverage and manual verdict, commit documentation,
leave the non-commercial private artifacts ignored, and open the final GLB in Blender.

### Task 7 measured outcome

- Profile: `dad-multiview-fixed-uv-blend-v2`
- Fixed-UV valid-pixel coverage: `0.9370746580`
- Observed-face area retained by yaw 0: `0.8899959429`
- Textured surface area: `0.8534889984`
- Back-projection violations: `0`
- Output topology: `5,118` seam-aware vertices, `9,976` triangles, no non-manifold edges,
  consistent winding
- Manual review: passed for observed-front recognizability, both three-quarter views, central
  feature alignment, absence of central-face seams, absence of rear face leakage, and improvement
  over the sparse DAD and TripoSR comparisons
- Repository gates: check, skill, release, license, schema, and publication passed
- Full Windows suite: `1,193 passed`, `115 skipped`, `83 failed`; all 83 are the existing
  POSIX/fake-executable, chmod, symlink, or platform-boundary failures, while all changed DAD,
  runner, and Blender-preview tests passed
