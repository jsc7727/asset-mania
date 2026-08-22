# Asset Mania Face-Anchor Visual Hull Design

**Status:** approved for implementation on 2026-08-23

## Purpose

Improve the private photo-to-3D face experiment without downloading another model or sending
additional face data to an external provider. The new profile combines one observed-front
TripoSR mesh with an eight-silhouette visual hull instead of averaging eight independently
hallucinated TripoSR meshes.

The profile is research-only. It must not claim biometric identity, exact likeness, or measured
face accuracy.

## Evidence behind the change

The previous eight-mesh occupancy-majority experiment completed but failed Blender visual
review:

- the generated viewset passed its structural audit, while identity consistency remained
  unmeasured;
- TripoSR resolution 256 produced six closed and two open meshes;
- lattice cleanup reduced 47,352 fragments to one watertight positive-volume component;
- majority consensus removed recognizable facial detail even after resolution, vote threshold,
  yaw-axis, no-yaw, ICP, and smooth-shading comparisons.

The CUDA smoke test proves runtime acceleration only. It used `synthetic.png`, not a face, and
therefore supplies no face-quality evidence. The verified CUDA environment may run the observed
front anchor, while torchmcubes and visual-hull post-processing may remain on CPU.

## Goals

- Preserve the observed-front TripoSR mesh's nose, mouth, chin, and face-side surface.
- Derive a clean side and rear head envelope from all eight head-only masks.
- Produce one create-only GLB that is a single connected, watertight, winding-consistent,
  positive-volume mesh.
- Add view-projected vertex colors without treating generated views as observed evidence.
- Reproject the final mesh into all eight yaws and publish measured silhouette metrics.
- Run the real private face input locally on the verified RTX 4070 path and inspect it in
  Blender against the previous outputs.
- Preserve source image bytes and keep every real-person artifact under ignored private storage.

## Non-goals

- No new model, weight, provider, API call, remote generation, upload, or paid compute.
- No DECA, EMOCA, FLAME, NeRF, 3D Gaussian Splatting, or photogrammetry dependency.
- No public CLI or stable portable schema in this iteration.
- No rigging, animation, expression controls, production topology, UV atlas, or texture bake.
- No claim that generated side and rear colors reproduce the real person.
- No replacement or silent fallback for the existing `voxel-consensus-v1` profile.

## Considered approaches

### Selected: observed-front anchor plus robust visual hull

Run TripoSR once on the observed front image and its head-only mask. Build the full head envelope
by carving a common voxel grid against the eight canonicalized silhouettes. Retain the anchor in
the camera-facing half and the visual hull elsewhere, with a bounded transition band.

This keeps the one surface that contains observed-front detail and uses generated views only for
coarse envelope support.

### Rejected: higher-resolution majority fusion

Resolution 256, alternative vote thresholds, ICP, and multiple yaw assumptions were already
measured. They changed topology and smoothness but did not restore the face. Resolution 512 would
refine the same disagreement rather than repair it.

### Deferred: a face-specific model

DECA/EMOCA/FLAME is the likely next family if this hybrid fails, but it introduces new model
assets, licences, clearance, installation, and a weak hair/back-head story. It requires a separate
approval and design.

## Architecture

The feature lives in the optional TripoSR engine package and a maintainer-only runner:

```text
8 private images + 8 private head masks
                 |
                 v
canonical silhouette frames -----> robust 7-of-8 visual hull
                 |                             |
observed yaw 0 image/mask                      |
                 |                             |
                 v                             |
GPU TripoSR anchor -> projection alignment ----+
                                               |
                                               v
                                  bounded front/back occupancy blend
                                               |
                                               v
                                   cleanup + marching cubes + GLB
                                               |
                                               v
                             view-projected vertex colors + verification
```

The existing `multiview.py` and its majority profile remain unchanged. The new implementation is
isolated in `face_hybrid.py` so later rejection of the experiment does not alter object fusion.

## Coordinate system

TripoSR declares a right-handed world with `x back, y right, z up`; rendered azimuth starts at
camera position `+X` and advances toward `+Y`. The observed front camera is yaw 0 at `+X`.

For yaw `theta`, orthographic projection uses:

```text
u = -sin(theta) * x + cos(theta) * y
v = z
```

`u` maps to image horizontal and `v` maps to image vertical with the image Y direction inverted.
Every projection helper is tested with asymmetric synthetic geometry so a sign or axis reversal
cannot pass unnoticed.

## Canonical silhouette frames

Inputs are the complete ordered yaws `(0, 45, 90, 135, 180, 225, 270, 315)`. Each image and
mask must be 1024 by 1024. The runner never modifies those inputs.

For each mask:

1. Threshold foreground at 128.
2. Require one non-empty foreground and coverage in `[0.15, 0.65]`.
3. Compute the foreground bounding box and centroid.
4. Centre the foreground in a 1024-square canonical frame.
5. Uniformly scale the longest foreground-box edge to 82 percent of the frame.
6. Apply the identical affine transform to its color image.

The affine parameters and normalized image/mask hashes are private run evidence. Independent
centering deliberately removes ImageGen framing drift; it does not assert identity consistency.

## Robust visual hull

The final profile uses a `192^3` grid spanning `[-0.6, 0.6]^3`. Each voxel projects into all
eight canonical masks. A voxel survives when at least seven masks contain its projection.

Seven-of-eight support tolerates one inconsistent generated silhouette while remaining stricter
than the old six-input mesh majority. After voting:

- apply one 26-neighbour binary closing;
- fill enclosed holes;
- retain only the largest 26-connected component;
- fail if occupancy is empty or touches the grid boundary.

The visual hull must reproject with foreground IoU at least 0.72 in every view and mean IoU at
least 0.82. These thresholds are geometric quality gates, not likeness measurements.

## Observed-front anchor

The maintainer runner executes TripoSR once at resolution 256 using the observed yaw-0 image and
head-only mask. It reuses the verified local engine checkout, weights, architecture cache, and
engine clearance. `device=cuda` is required for the live private run; no network access occurs.

The anchor must be closed, winding-consistent, and positive-volume. Bounds-centre and
longest-extent normalization place it in the common cube. A deterministic grid search optimizes
only uniform scale in `[0.88, 1.12]` and Y/Z translation in `[-0.08, 0.08]` against the yaw-0
canonical silhouette. Rotation and non-uniform scaling are forbidden. The aligned anchor must
reach yaw-0 projection IoU at least 0.60.

## Hybrid occupancy

The aligned anchor is voxelized into the common grid and conservatively splatted by one
6-neighbour cell. Positive X is the observed-front side.

The fixed profile is:

- `x >= 0.08`: anchor occupancy intersected with a one-cell-dilated visual hull;
- `x <= -0.08`: visual-hull occupancy;
- `-0.08 < x < 0.08`: union of anchor and hull, intersected with the dilated hull.

The combined volume receives the same closing, hole fill, and largest-component cleanup. It must
retain at least 85 percent of anchor voxels in the positive-X front region. Marching cubes extracts
the surface at level 0.5, and the existing bounded geometry normalization verifies the GLB.

## Vertex colors

The geometry remains the primary acceptance surface. Vertex colors are produced by deterministic
orthographic projection into the canonical color images:

1. Estimate each final vertex normal.
2. Rank yaw cameras by positive normal-to-camera cosine.
3. Sample the two best valid views whose masks contain the projected point.
4. Blend samples by normalized cosine weight.
5. Give the observed yaw-0 view a 1.5 weight multiplier only when it is already one of those two
   geometrically valid views.
6. Use neutral gray when no view is valid.

Portable/private metadata labels colors sampled from yaw 0 as observed-derived and every other
yaw as generated-derived. Color coverage is measured, but no color or identity accuracy claim is
made.

## Interfaces

`packages/engine-triposr/src/asset_mania_engine_triposr/face_hybrid.py` provides:

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


def canonicalize_views(
    views: Sequence[CanonicalView], output_directory: Path
) -> list[CanonicalView]: ...


def build_visual_hull(
    views: Sequence[CanonicalView], settings: FaceHybridSettings
) -> tuple[np.ndarray, dict[str, float]]: ...


def fuse_face_anchor(
    *,
    anchor_mesh: Path,
    views: Sequence[CanonicalView],
    output_path: Path,
    settings: FaceHybridSettings,
) -> FaceHybridResult: ...
```

All writes are create-only. Public or portable records contain hashes and measurements, never
private paths, basenames, prompts, or face pixels.

## Maintainer runner

`scripts/run_face_hybrid_e2e.py` exposes:

```text
prepare      canonicalize and audit eight private image/mask pairs
anchor       run observed-front TripoSR with an existing clearance
fuse         create the hybrid GLB and private evidence record
verify       recompute GLB metrics and run the Blender four-view preview
```

The runner supports injected anchor/fusion/preview functions for deterministic E2E tests. The
real `anchor` command requires the local CUDA runtime and never falls back to CPU silently.

## Failure handling

The profile fails closed on:

- missing, duplicated, unordered, wrong-sized, or empty views;
- mask coverage outside the declared range;
- visual-hull boundary contact or failed reprojection thresholds;
- an open or non-positive anchor;
- insufficient yaw-0 anchor alignment;
- insufficient front-anchor retention;
- empty or multi-component final occupancy;
- an existing output path;
- a non-watertight, inconsistent, or non-positive final GLB;
- Blender preview failure.

A geometrically valid mesh can still fail manual likeness review. The private report records
`visual_quality=failed` rather than weakening a gate or presenting the output as successful.

## Testing

Unit tests use only synthetic redistributable shapes:

- projection-axis tests use an asymmetric nose-like wedge;
- visual-hull tests reconstruct an ellipsoid with a front bump from eight generated silhouettes;
- one deliberately inconsistent silhouette confirms seven-of-eight tolerance;
- two inconsistent silhouettes fail the reprojection gate;
- anchor blending proves positive-X bump retention and rear-hull completion;
- create-only and malformed-input tests fail closed;
- color projection uses distinct synthetic colors per yaw and verifies observed-front weighting;
- optional torchmcubes tests verify one watertight positive-volume GLB.

The deterministic maintainer E2E injects fake anchor and preview functions. The real private E2E
then runs one CUDA TripoSR anchor, CPU visual-hull fusion, GLB validation, and Blender rendering.

## Acceptance criteria

Implementation is complete only when:

1. focused unit and deterministic E2E tests pass;
2. the optional runtime writes a single-component watertight positive-volume synthetic GLB;
3. the actual observed-front anchor runs on CUDA without network access or fallback;
4. the real eight-view visual hull and hybrid pass all numeric gates;
5. Blender produces comparable front/side/back previews for the old anchor, old majority fusion,
   and new hybrid;
6. a human visual review labels the new hybrid either `passed` or `failed` without an identity
   claim;
7. original image and mask hashes remain unchanged;
8. Git contains no private images, masks, meshes, previews, model weights, or GPU logs.

