# Face-Anchor Visual Hull Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a private face reconstruction profile that preserves observed-front
TripoSR detail while completing the side and rear head from eight silhouettes.

**Architecture:** Add an isolated `face_hybrid.py` module beside the existing majority-fusion
module. It canonicalizes eight private views, carves a robust visual hull, aligns one observed
front TripoSR anchor, blends the two occupancies, projects vertex colors, and exports a verified
GLB. A maintainer-only runner performs prepare, CUDA anchor, fuse, and Blender verification stages.

**Tech Stack:** Python 3.12, NumPy, SciPy ndimage, Pillow, trimesh 4.0.5, PyTorch 2.13,
torchmcubes, local TripoSR, CUDA 13/PyTorch cu130, Blender 5.2.

**Spec:** `docs/superpowers/specs/2026-08-23-asset-mania-face-anchor-visual-hull-design.md`

## Global Constraints

- Work on the existing scoped `v0-2-blender-pipeline` branch and preserve unrelated changes.
- Never modify, move, overwrite, upload, or track the private source image, masks, generated
  views, meshes, previews, prompts, weights, or GPU logs.
- Do not download a model, contact a provider, spend money, or change the approved workflow.
- Keep `voxel-consensus-v1` behavior unchanged; this is a separate experimental profile.
- Use the complete ordered yaws `(0, 45, 90, 135, 180, 225, 270, 315)`.
- Real execution uses the existing local TripoSR revision, weights, clearance, and CUDA runtime.
- All output writes are create-only and portable evidence contains hashes, not private paths.
- Follow red-green-refactor for every production behavior.

---

### Task 1: Canonical view frames and projection math

**Files:**
- Create: `packages/engine-triposr/src/asset_mania_engine_triposr/face_hybrid.py`
- Modify: `packages/engine-triposr/src/asset_mania_engine_triposr/__init__.py`
- Create: `packages/engine-triposr/tests/test_face_hybrid.py`

**Interfaces:**
- Consumes: ordered 1024-square private image/mask paths.
- Produces: `CanonicalView`, `FaceHybridSettings`, `canonicalize_views`, and
  `project_points(points, yaw, resolution)`.

- [ ] **Step 1: Write failing projection-axis tests**

```python
def test_projection_axes_follow_triposr_yaw_convention() -> None:
    points = np.array([[1.0, 0.0, 0.30], [0.0, 1.0, -0.30]])

    yaw0 = project_points(points, yaw=0, resolution=101)
    yaw90 = project_points(points, yaw=90, resolution=101)

    assert yaw0[0] == pytest.approx([50.0, 25.0])
    assert yaw90[1] == pytest.approx([50.0, 75.0])
```

Use an explicit synthetic point set whose expected horizontal and vertical image coordinates are
derived from `u=-sin(theta)*x+cos(theta)*y` and `v=z`.

- [ ] **Step 2: Run the projection test and confirm RED**

Run:

```powershell
uv run pytest packages/engine-triposr/tests/test_face_hybrid.py::test_projection_axes_follow_triposr_yaw_convention -q
```

Expected: import failure because `face_hybrid.py` and `project_points` do not exist.

- [ ] **Step 3: Implement dataclasses and projection math**

Implement these exact public types:

```python
@dataclass(frozen=True, slots=True)
class CanonicalView:
    yaw: int
    image_path: Path
    mask_path: Path


@dataclass(frozen=True, slots=True)
class FaceHybridSettings:
    grid_resolution: int = 192
    minimum_silhouette_votes: int = 7
    front_seam: float = 0.08
```

Validate yaw as an integer in the fixed schedule. Map projected coordinates from the common cube
`[-0.6, 0.6]` to pixel centres in `[0, resolution-1]` and invert vertical image Y.

- [ ] **Step 4: Write failing canonicalization tests**

```python
def test_canonicalization_centres_and_scales_without_touching_inputs(tmp_path: Path) -> None:
    views = synthetic_shifted_views(tmp_path)
    before = {view.image_path: sha256_file(view.image_path) for view in views}

    result = canonicalize_views(views, tmp_path / "canonical")

    assert [view.yaw for view in result] == list(TURNTABLE_YAWS)
    assert all(mask_bbox_ratio(view.mask_path) == pytest.approx(0.82, abs=0.01) for view in result)
    assert before == {path: sha256_file(path) for path in before}
```

Add separate tests for wrong order, wrong dimensions, empty masks, coverage below 0.15, coverage
above 0.65, duplicate decoded pixels, and an existing output directory.

- [ ] **Step 5: Run canonicalization tests and confirm RED**

Run:

```powershell
uv run pytest packages/engine-triposr/tests/test_face_hybrid.py -q
```

Expected: projection passes and canonicalization tests fail because `canonicalize_views` is absent.

- [ ] **Step 6: Implement create-only canonicalization**

Decode via Pillow, threshold masks at 128, compute coverage/bounds/centroid, and write normalized
1024 PNGs using the existing normalized-PNG helper. Apply the same affine transform to image and
mask. Refuse the entire output directory when it already exists; never partially replace files.

- [ ] **Step 7: Run focused tests and commit**

Run:

```powershell
uv run pytest packages/engine-triposr/tests/test_face_hybrid.py -q
uv run ruff check packages/engine-triposr/src/asset_mania_engine_triposr/face_hybrid.py packages/engine-triposr/tests/test_face_hybrid.py
```

Expected: all Task 1 tests pass.

Commit:

```powershell
git add packages/engine-triposr/src/asset_mania_engine_triposr/face_hybrid.py packages/engine-triposr/src/asset_mania_engine_triposr/__init__.py packages/engine-triposr/tests/test_face_hybrid.py
git commit -m "feat: canonicalize face hybrid views"
```

---

### Task 2: Robust visual-hull carving and reprojection gates

**Files:**
- Modify: `packages/engine-triposr/src/asset_mania_engine_triposr/face_hybrid.py`
- Modify: `packages/engine-triposr/tests/test_face_hybrid.py`

**Interfaces:**
- Consumes: canonical ordered views and `FaceHybridSettings`.
- Produces: `build_visual_hull(views, settings) -> tuple[np.ndarray, dict[str, float]]`.

- [ ] **Step 1: Write a failing synthetic visual-hull test**

Generate eight 128-square silhouettes by analytically projecting an ellipsoid with a positive-X
front bump. Use test settings `grid_resolution=48` and assert:

```python
occupancy, metrics = build_visual_hull(views, FaceHybridSettings(48, 7, 0.08))
assert occupancy.dtype == bool
assert occupancy.shape == (48, 48, 48)
assert occupancy.any()
assert metrics["minimum_reprojection_iou"] >= 0.72
assert metrics["mean_reprojection_iou"] >= 0.82
assert connected_component_count(occupancy) == 1
```

Add one test where a single yaw mask is replaced by a smaller silhouette and the seven-of-eight
hull still passes. Add another where two opposite masks are inconsistent and the numeric gate
raises `ValueError("visual hull reprojection gate failed")`.

- [ ] **Step 2: Run the new tests and confirm RED**

Run:

```powershell
uv run pytest packages/engine-triposr/tests/test_face_hybrid.py -k "visual_hull" -q
```

Expected: failure because `build_visual_hull` is absent.

- [ ] **Step 3: Implement slabbed voxel carving**

Allocate one boolean `(R,R,R)` output, but project coordinates in Z slabs of at most 16 layers to
bound temporary memory. For every yaw, sample its thresholded canonical mask using nearest pixel
centres and accumulate `uint8` votes. Keep votes greater than or equal to
`minimum_silhouette_votes`.

Call a private `_clean_volume` that performs one 26-neighbour closing, fills holes, retains the
largest 26-connected component, and rejects grid-boundary contact. Reproject occupied voxels to
each mask and compute exact binary IoU.

- [ ] **Step 4: Run tests and commit**

Run:

```powershell
uv run pytest packages/engine-triposr/tests/test_face_hybrid.py -q
```

Expected: all canonicalization and visual-hull tests pass.

Commit:

```powershell
git add packages/engine-triposr/src/asset_mania_engine_triposr/face_hybrid.py packages/engine-triposr/tests/test_face_hybrid.py
git commit -m "feat: carve robust face visual hulls"
```

---

### Task 3: Anchor alignment and bounded occupancy blend

**Files:**
- Modify: `packages/engine-triposr/src/asset_mania_engine_triposr/face_hybrid.py`
- Modify: `packages/engine-triposr/tests/test_face_hybrid.py`

**Interfaces:**
- Consumes: a closed TripoSR anchor mesh, visual-hull occupancy, and yaw-0 canonical mask.
- Produces: private `_align_anchor`, `_voxelize_aligned_anchor`, and `_blend_face_anchor` helpers.

- [ ] **Step 1: Write failing deterministic alignment tests**

Create a closed ellipsoid anchor with a positive-X bump, then apply scale `1.08` and Y/Z
translation `(0.04, -0.03)`. Assert the bounded search returns the inverse placement within one
search step and reaches projection IoU at least 0.90. Add an anchor shifted outside the allowed
range and assert `ValueError("anchor alignment gate failed")`.

- [ ] **Step 2: Run alignment tests and confirm RED**

Run:

```powershell
uv run pytest packages/engine-triposr/tests/test_face_hybrid.py -k "anchor_alignment" -q
```

Expected: failure because `_align_anchor` is absent.

- [ ] **Step 3: Implement bounded alignment**

Normalize anchor vertices on bounds centre and longest extent. Search scale values
`np.linspace(0.88, 1.12, 13)` and Y/Z translations `np.linspace(-0.08, 0.08, 17)`. Project the
transformed vertices into yaw 0, rasterize the projected convex surface through a temporary
orthographic voxel silhouette, and maximize IoU. Break ties lexicographically by smallest absolute
translation and scale distance from 1.0.

- [ ] **Step 4: Write failing blend tests**

```python
hybrid, retention = _blend_face_anchor(anchor_grid, hull_grid, settings)
assert retention >= 0.85
assert hybrid[front_bump_index]
assert hybrid[rear_completion_index]
assert connected_component_count(hybrid) == 1
```

Add a case where hull clipping removes too much positive-X anchor and assert
`ValueError("front anchor retention gate failed")`.

- [ ] **Step 5: Implement the fixed seam profile**

Use world-X coordinates for the three fixed bands from the spec. Intersect anchor voxels with a
one-cell-dilated hull in the face region, use hull voxels in the rear, and use their bounded union
in the transition. Clean the result and measure positive-X anchor retention.

- [ ] **Step 6: Run focused tests and commit**

Run:

```powershell
uv run pytest packages/engine-triposr/tests/test_face_hybrid.py -q
```

Expected: all Task 1-3 tests pass.

Commit:

```powershell
git add packages/engine-triposr/src/asset_mania_engine_triposr/face_hybrid.py packages/engine-triposr/tests/test_face_hybrid.py
git commit -m "feat: blend observed face anchors with visual hulls"
```

---

### Task 4: Surface extraction, vertex colors, and measured result

**Files:**
- Modify: `packages/engine-triposr/src/asset_mania_engine_triposr/face_hybrid.py`
- Modify: `packages/engine-triposr/src/asset_mania_engine_triposr/__init__.py`
- Modify: `packages/engine-triposr/tests/test_face_hybrid.py`

**Interfaces:**
- Consumes: anchor path, canonical views, create-only GLB destination, settings.
- Produces: `FaceHybridResult` and `fuse_face_anchor(...) -> FaceHybridResult`.

The result type is fixed before implementation:

```python
@dataclass(frozen=True, slots=True)
class FaceHybridResult:
    triangle_count: int
    vertex_count: int
    manifold: str
    signed_volume: float
    component_count: int
    minimum_reprojection_iou: float
    mean_reprojection_iou: float
    front_anchor_retention: float
    color_coverage: float
```

- [ ] **Step 1: Write failing color-projection tests**

Use eight solid-color images and matching masks. Give yaw 0 red, yaw 90 green, yaw 180 blue, and
the remaining yaws distinct colors. For synthetic final vertices with outward normals toward each
camera, assert their sampled colors come from the geometrically facing view. At a yaw-0/yaw-45
boundary, assert the observed yaw-0 multiplier changes the blend toward red without selecting an
otherwise invalid view.

- [ ] **Step 2: Run color tests and confirm RED**

Run:

```powershell
uv run pytest packages/engine-triposr/tests/test_face_hybrid.py -k "vertex_color" -q
```

Expected: failure because `_project_vertex_colors` is absent.

- [ ] **Step 3: Implement color projection**

Estimate mesh vertex normals after marching cubes. For each vertex, compute camera-direction
cosines, select the two highest positive valid views whose projected mask pixel is foreground,
apply the yaw-0 multiplier only after selection, and blend RGB in float before rounding to uint8.
Return RGBA with alpha 255 and measured non-neutral coverage.

- [ ] **Step 4: Write the optional-runtime GLB test**

Under the existing torchmcubes skip guard, export the asymmetric synthetic hybrid and assert:

```python
result = fuse_face_anchor(
    anchor_mesh=anchor,
    views=views,
    output_path=output,
    settings=FaceHybridSettings(48, 7, 0.08),
)
mesh = trimesh.load(output, process=False, force="mesh")
assert result.manifold == "closed"
assert result.component_count == 1
assert result.signed_volume > 0
assert mesh.is_watertight and mesh.is_winding_consistent
assert len(mesh.visual.vertex_colors) == len(mesh.vertices)
```

Also assert create-only refusal and malformed/open anchor refusal.

- [ ] **Step 5: Implement surface export and result measurement**

Reuse the existing package-local marching-cubes extraction and bounded `_normalise`. Attach vertex
colors before GLB export. Reload the written file to recompute component count, winding, volume,
counts, and color coverage. Fail rather than repair a profile-gate violation.

- [ ] **Step 6: Run root and optional-runtime tests and commit**

Run:

```powershell
uv run pytest packages/engine-triposr/tests/test_face_hybrid.py -q
.asset-mania\triposr-venv\Scripts\python.exe -m pytest packages/engine-triposr/tests/test_face_hybrid.py -q
```

Expected: root tests pass with only the declared optional skip; optional runtime runs every test
and writes a watertight GLB.

Commit:

```powershell
git add packages/engine-triposr/src/asset_mania_engine_triposr/face_hybrid.py packages/engine-triposr/src/asset_mania_engine_triposr/__init__.py packages/engine-triposr/tests/test_face_hybrid.py
git commit -m "feat: export colored face hybrid meshes"
```

---

### Task 5: Maintainer E2E runner

**Files:**
- Create: `scripts/run_face_hybrid_e2e.py`
- Create: `tests/test_face_hybrid_e2e.py`

**Interfaces:**
- Consumes: a private source view directory, existing engine clearance/assets, fixed CUDA device,
  and an output parent.
- Produces: staged private run directories, anchor GLB, hybrid GLB, metrics JSON, and preview.

- [ ] **Step 1: Write failing prepare-stage E2E test**

Create synthetic ordered `yaw-###.png` and `yaw-###-mask.png` inputs, record their hashes, invoke
`main(["prepare", ...])`, and assert a create-only run with canonical frames and no source mutation.
Assert serialized JSON contains no input path or basename.

- [ ] **Step 2: Write failing anchor/fuse/verify E2E tests**

Inject:

```python
def fake_anchor_runner(*, output_path: Path, **kwargs):
    synthetic_anchor().export(output_path, file_type="glb")
    return {"device": "cuda", "manifold": "closed"}


def fake_preview_runner(mesh: Path, preview: Path, blender: Path | None):
    Image.new("RGB", (64, 64), (20, 30, 40)).save(preview)
```

Assert `anchor` refuses device values other than `cuda`, `fuse` seals exact hashes, and `verify`
recomputes source integrity and writes the preview/report without overwriting.

- [ ] **Step 3: Run E2E tests and confirm RED**

Run:

```powershell
uv run pytest tests/test_face_hybrid_e2e.py -q
```

Expected: import failure because the runner does not exist.

- [ ] **Step 4: Implement four create-only stages**

Use `prepare`, `anchor`, `fuse`, and `verify` subparsers. The real anchor path constructs one
`TripoSRPort` with `device="cuda"`, resolution 256, the existing hub cache, and verified engine
clearance. Before and after execution, require `torch.cuda.is_available()` and record only device
type, elapsed seconds, peak allocated bytes, counts, and hashes. Never serialize a local source
path or GPU name.

The verify stage invokes the existing safe Blender preview launcher and records
`visual_quality="unreviewed"`; manual inspection changes only a private report.

- [ ] **Step 5: Run deterministic E2E and regression tests**

Run:

```powershell
uv run pytest tests/test_face_hybrid_e2e.py packages/engine-triposr/tests/test_face_hybrid.py tests/test_turntable_multiview_e2e.py -q
```

Expected: all tests pass and the old runner remains unchanged.

- [ ] **Step 6: Commit the runner**

```powershell
git add scripts/run_face_hybrid_e2e.py tests/test_face_hybrid_e2e.py
git commit -m "feat: add private face hybrid E2E runner"
```

---

### Task 6: Research documentation and release boundaries

**Files:**
- Modify: `docs/research.md`
- Modify: `docs/getting-started.md`
- Modify: `README.md`
- Modify if required by checks: `scripts/check_release.py`
- Test: `tests/test_check_release.py`

**Interfaces:**
- Consumes: implemented private research profile and measured deterministic evidence.
- Produces: honest experimental documentation with no live-quality claim before Task 7.

- [ ] **Step 1: Add documentation assertions before prose**

Extend the relevant release/check test so README/research documentation must contain
`face-anchor-visual-hull-v1`, `identity consistency remains unmeasured`, and
`live face quality unverified`, while rejecting a sentence that calls the profile accurate or
identity-preserving.

- [ ] **Step 2: Run the release test and confirm RED**

Run:

```powershell
uv run pytest tests/test_check_release.py -q
```

Expected: failure because the required documentation text is absent.

- [ ] **Step 3: Document only deterministic evidence**

Describe the new profile as private, experimental, no-new-model, and deterministic/synthetic
verified. Explain that GPU acceleration changes runtime, not likeness. Keep the old majority
profile and its failed visual result visible.

- [ ] **Step 4: Run canonical focused checks and commit**

Run:

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest packages/engine-triposr/tests/test_face_hybrid.py tests/test_face_hybrid_e2e.py tests/test_check_release.py -q
uv run python scripts/validate_skill.py skills/asset-mania
uv run python scripts/check_license_boundary.py
uv run python scripts/check_schema_distribution.py
uv run python scripts/check_release.py
```

Expected: all commands exit 0, apart from no optional-runtime skips in the dedicated runtime.

Commit:

```powershell
git add README.md docs/research.md docs/getting-started.md scripts/check_release.py tests/test_check_release.py
git commit -m "docs: describe face hybrid research profile"
```

---

### Task 7: Actual private CUDA face run and Blender comparison

**Files:**
- Create only under ignored storage:
  `.asset-mania/private-face-run/face-hybrid-*/`
- Do not track: private images, masks, canonical frames, meshes, reports, logs, or previews.

**Interfaces:**
- Consumes: existing approved source/view bytes, head-only masks, engine clearance/assets, CUDA
  runtime, and completed Tasks 1-6.
- Produces: one private anchor GLB, one hybrid GLB, measured report, comparison previews, and an
  honest manual visual verdict.

- [ ] **Step 1: Prove source and viewset integrity**

Hash the original source image, original mask, eight selected view images, and eight masks before
the run. Compare them to the prior private evidence where available. Stop on any mismatch.

- [ ] **Step 2: Run create-only prepare**

Use `oauth-viewset-v5-head` as the read-only input and a new timestamped output. Confirm the
canonical audit and write hashes only to the private run.

- [ ] **Step 3: Run one observed-front CUDA anchor**

Use `.asset-mania\triposr-cuda-venv\Scripts\python.exe`, `device=cuda`, TripoSR resolution 256,
the pinned local engine/weights/cache, and the existing valid clearance. Record elapsed time and
peak allocated VRAM without printing face paths.

- [ ] **Step 4: Fuse and verify the real hybrid**

Run visual hull and anchor fusion at grid 192. Require every numeric gate from the spec and
validate the final GLB independently with trimesh and the GLB container validator.

- [ ] **Step 5: Render comparable Blender previews**

Using Blender 5.2 and identical camera/material settings, render:

- observed-front anchor;
- prior `fused-360-experimental.glb` majority result;
- new face-hybrid GLB.

Write a side-by-side contact image under the private run and inspect front, right, rear, and left.

- [ ] **Step 6: Record manual verdict**

Set private `visual_quality` to `passed` only when the new mesh preserves a recognizable front
surface and removes the prior plate/blob silhouette. Otherwise set `failed` with the exact visual
reason. Keep `identity_consistency="unmeasured"` in both cases.

- [ ] **Step 7: Run final repository verification**

Run the valid Makefile bodies on Windows:

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
git status --short
```

Report focused success separately from known POSIX-only Windows failures. Inspect the staged diff
and confirm no private data, model assets, binaries, or unrelated files are tracked.

- [ ] **Step 8: Commit measured public documentation only if supported**

If the live hybrid passes visual review, update only the measured research row without claiming
identity accuracy. If it fails, document the failed stage and next recommended face-model family.

```powershell
git add README.md docs/research.md
git commit -m "docs: record face hybrid E2E evidence"
```
