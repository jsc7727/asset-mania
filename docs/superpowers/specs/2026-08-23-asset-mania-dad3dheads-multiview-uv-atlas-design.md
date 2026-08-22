# Asset Mania DAD-3DHeads Multi-View UV Atlas Design

**Status:** approved for implementation on 2026-08-23

## Purpose

Make the private DAD-3DHeads result visually recognizable from the authorized portrait instead of
showing a mostly neutral FLAME head. Replace sparse vertex colors with a real embedded UV texture
atlas built from the observed front image and the seven existing locally generated yaw views.

The geometry remains the pinned DAD result. No new model, checkpoint, provider, upload, paid call,
or generated view is introduced. The experiment remains CC BY-NC-SA 4.0 non-commercial research,
and identity consistency remains unmeasured.

## Evidence behind the change

The first DAD experiment improved geometry but failed the user's actual visual objective:

- the fixed FLAME output was coherent at 5,023 vertices and 9,976 triangles;
- it preserved nose, lips, chin, cheeks, eye sockets, ears, and side profiles;
- it was structurally better than both TripoSR voxel results;
- observed-front vertex colors covered only 30.6 percent of vertices;
- most vertices therefore received neutral color, and Blender showed an almost white silhouette;
- the result did not make the supplied person readily recognizable.

The local private viewset already contains ordered yaw images and head masks for
`(0, 45, 90, 135, 180, 225, 270, 315)`. DAD emits the same fixed topology and per-vertex projected
coordinates for every inference. That allows view-specific visibility and UV assignment without
fusing independent geometry.

## Goals

- Keep the observed-front DAD geometry and its fixed head plus two eye-shell topology.
- Run the already acquired pinned DAD checkpoint on the seven existing generated yaw images.
- Use the observed original image, not the generated yaw-0 frame, for the atlas's front tile.
- Build a 3-by-3 atlas containing eight 512-square view tiles and one neutral fallback tile.
- Assign every triangle to one view only after camera-facing, mask, bounds, and z-buffer checks.
- Prioritize the observed yaw-0 tile for the indexed face region whenever it is truly visible.
- Duplicate vertices only across view seams so one vertex can have different UVs on adjacent faces.
- Preserve smooth normals across duplicated seam vertices.
- Export one create-only GLB with an embedded base-color texture and a real glTF material.
- Open the final textured GLB in Blender with its face-facing view selected and material preview
  visible without manual node editing.
- Compare textured DAD, sparse-color DAD, TripoSR anchor, and TripoSR hybrid under identical Blender
  render settings.
- Record an honest manual verdict against recognizability, not geometry alone.

## Non-goals

- No new model, checkpoint, Python runtime, provider, API call, upload, or paid compute.
- No regeneration or editing of any of the eight source views.
- No DAD geometry averaging, shape fusion, retraining, or fine-tuning.
- No biometric embedding, face recognition, identity score, or exact-likeness claim.
- No claim that generated side/rear views reveal the person's real unseen appearance.
- No new hair geometry, eyelashes, teeth, tongue, or realistic eye shader.
- No automatic fallback to vertex colors, TripoSR, MICA, SMIRK, DECA, NeRF, or 3DGS.
- No silent UV repair when a projection or visibility gate fails.
- No public real-person image, atlas, mesh, render, projection, or local path.

## Considered approaches

### Selected: fixed DAD geometry plus visibility-aware multi-view UV atlas

Run the same DAD model on each existing view, keep only its projected vertices and camera-space
depth, and texture the original observed-front mesh. Each triangle chooses the best eligible view,
while the indexed face region strongly prefers the observed yaw-0 image.

This addresses the measured failure directly: sparse low-resolution vertex colors become a real
texture without introducing another model or pretending generated views are observations.

### Deferred: MICA or SMIRK geometry replacement

These models may improve identity shape but require separate checkpoints, FLAME access and terms,
dependency clearance, and compatibility work. They do not solve texture delivery by themselves.
They remain a separate experiment if the atlas still fails recognizability review.

### Rejected: NeRF or 3D Gaussian Splatting from generated views

The seven side/rear images are model inferences, not synchronized camera observations. A radiance
field could encode their inconsistencies while presenting them as measured geometry. That is not a
valid identity reconstruction path.

## Architecture

```text
observed original image + projection ────────────────┐
                                                     │
7 existing generated yaw images + masks             │
                    │                                │
                    v                                │
        pinned local DAD projection inference        │
                    │                                │
                    v                                v
      8 fixed-topology view projections ----> visibility audit
                    │                       (facing + mask + z-buffer)
                    v                                │
           per-triangle view scores <────────────────┘
                    │
          observed-face priority
                    │
                    v
        3x3 texture atlas + seam UVs
                    │
                    v
      embedded-texture GLB + Blender comparison
```

The existing sparse-color converter stays available as historical evidence. The new implementation
lives in `packages/engine-dad3dheads/src/asset_mania_engine_dad3dheads/texture.py` and is invoked by
new create-only stages in `scripts/run_face_plugin_e2e.py`.

## Input set and provenance

The ordered views are:

| Yaw | Image origin | Geometry use | Texture use |
| --- | --- | --- | --- |
| 0 | observed original normalized PNG | front DAD geometry | preferred observed face/front tile |
| 45–315 | existing generated private viewset | projection and visibility only | side/rear inferred tiles |

Every source image and mask is fingerprinted before processing and verified unchanged afterward.
The private records label yaw 0 `observed` and all other yaws `generated`. Public records contain
only aggregate measurements and model identifiers.

The generated yaw-0 image is not consumed. The original normalized source from the successful DAD
run is the only yaw-0 texture and projection source.

## Multi-view DAD projection stage

The plugin result gains no portable public face data. A private per-view output contains:

```text
head.obj
projection.npz
result.json
```

`projection.npz` already stores `projected_vertices` and `image_shape`. It is extended with
camera-space `vertices` so the atlas stage can compute normals, depth, and occlusion for the exact
view-specific DAD prediction. The raw fixed face index and triangle files remain in the external
private checkout and are never copied into the public repository.

The runner executes generated views sequentially in yaw order and reuses the same checkpoint,
revision, CUDA Python, plugin executable, and network-denied environment as the successful front
run. A failure stops the stage. It never skips a yaw or substitutes another image.

For every view, require:

- exactly 5,023 vertices and 9,976 triangles;
- the same face index array and triangle topology as yaw 0;
- finite projected coordinates and camera-space vertices;
- image dimensions matching the projection record;
- a matching source image and mask digest;
- the pinned checkpoint digest and CUDA device.

## View-specific visibility

DAD camera coordinates use image `X` as horizontal, image `Y` as vertical, and raw `Z` as depth.
The face is toward negative `Z`; smaller depth is closer to the camera.

For each view:

1. Compute area-weighted face normals from that view's predicted vertices.
2. Reject back-facing triangles whose camera-facing cosine is not positive.
3. Project triangle vertices into a 512-square working frame using the DAD projection coordinates.
4. Require projected vertices to stay inside image bounds and the triangle centroid to lie inside
   the supplied head mask.
5. Rasterize eligible triangles into a depth buffer with perspective-correctness unnecessary for
   DAD's orthographic projection.
6. A triangle is visible only if it owns at least four pixels within a `1e-6` depth tolerance.

This visibility set is the hard eligibility boundary. A scoring bonus cannot select an occluded,
back-facing, out-of-mask, or out-of-bounds triangle.

Synthetic tests fix the depth sign and camera-facing sign with an asymmetric nose wedge and a rear
triangle that projects to the same pixels.

## Triangle view selection

For every eligible `(triangle, view)` pair, compute:

```text
score = camera_facing_cosine * visible_pixel_fraction * mask_confidence
```

`mask_confidence` is `1.0` when all three projected vertices are inside the mask and `0.8` when the
centroid plus two vertices are inside. Other cases are ineligible.

The observed face-region rule is applied after eligibility:

- load the external pinned `flame_indices/face.npy` and hash it into private evidence;
- a triangle is an indexed face triangle when at least two vertices occur in that array;
- eligible yaw 0 receives a fixed `4.0` score multiplier for indexed face triangles;
- eligible yaw 0 receives a `1.5` multiplier for other front-facing head triangles;
- generated views receive no provenance multiplier;
- ties resolve by higher visible pixel count, then smaller absolute wrapped yaw, then yaw number.

No triangle receives yaw 0 solely because it belongs to the face region; yaw-0 visibility remains
mandatory. Unassigned triangles use the neutral fallback tile.

## Atlas layout

The atlas is a deterministic 3-by-3 grid of 512-square tiles, producing one 1536-square sRGB PNG:

```text
0 observed | 45 generated | 90 generated
135 gen.   | 180 gen.     | 225 gen.
270 gen.   | 315 gen.     | neutral
```

Each source is resized from 1024 to 512 with Lanczos. The same transform is applied to its mask and
projection coordinates. The neutral tile is `(160, 145, 140)`.

UV coordinates use pixel centres with a two-pixel inset from tile edges. Vertical coordinates are
converted from image-down to glTF UV-up. Atlas PNG encoding is deterministic and create-only.

## UV seam topology and normals

glTF stores one UV per vertex, while one FLAME vertex may touch triangles assigned to different
views. The exporter therefore creates one output vertex per `(original_vertex_index, assigned_tile)`
pair.

For each duplicate:

- position comes from the observed-front DAD geometry after the fixed DAD-to-Blender transform;
- UV comes from that tile's projected vertex coordinate;
- normal is copied from the smooth original observed-front vertex normal;
- provenance records the tile yaw, not the source path.

Faces assigned to the neutral tile use the tile centre UV. Eye shells follow the same visibility
rules and may use neutral fallback if no view is safe.

The output must preserve 9,976 triangles, stay below 40,184 vertices, retain the fixed three
components, have zero non-manifold edges, and retain consistent winding.

## Textured GLB material

The output uses one glTF PBR material:

- embedded atlas PNG as `baseColorTexture`;
- base-color factor `(1, 1, 1, 1)`;
- metallic factor `0.0`;
- roughness factor `0.65`;
- opaque alpha mode;
- smooth vertex normals from the observed geometry.

The GLB is reloaded independently. Verification requires one embedded image, one texture, one
material referencing that texture, a `TEXCOORD_0` accessor, and no absolute/external URI.

The public converter never writes the private atlas or mesh. Real outputs stay under ignored run
storage.

## Metrics and acceptance gates

The private atlas record contains:

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

Hard gates:

- textured triangle fraction at least `0.80`;
- textured surface-area fraction at least `0.85`;
- observed yaw-0 coverage of eligible indexed face area at least `0.75`;
- neutral surface-area fraction at most `0.15`;
- back-projection violation count exactly `0`;
- all eight non-neutral tiles own at least one triangle;
- no output topology or GLB material validation failure.

These are texture-delivery gates, not identity measurements.

## Maintainer E2E stages

Extend `scripts/run_face_plugin_e2e.py` with:

```text
texture-plan   seal the ordered inputs, masks, model/runtime, atlas profile, and output paths
texture-infer  run DAD projection inference for the seven generated views
texture-build  audit visibility, select triangle views, build atlas, and export textured GLB
texture-verify validate source integrity, GLB material, metrics, and Blender comparison
```

All stages are create-only and use a new `texture/` subtree inside the existing successful private
DAD run. A failed stage may resume only when its existing immutable request/input bytes match the
sealed plan and no success output exists.

The Blender comparison contains four rows rendered with identical cameras, lights, background,
samples, and resolution:

1. DAD multi-view textured GLB;
2. DAD sparse vertex-color GLB;
3. TripoSR front anchor;
4. TripoSR face hybrid.

The final Blender interactive presentation imports only the textured GLB into an empty scene,
selects the mesh, frames it, switches to material preview, and chooses the DAD face-facing axis.

## Manual visual verdict

`visual_quality=passed` requires all of the following:

- the supplied person is recognizable from the observed front and at least one three-quarter view;
- eyes, brows, nose, lips, jawline, and skin placement align with the geometry closely enough to
  read as a face rather than a projected photograph on a blank shell;
- yaw-0 face pixels do not appear on the rear scalp or neck;
- no obvious tile seam crosses the central face;
- side/rear inferred textures do not overwrite the observed central face;
- the result is visibly more recognizable than the sparse-color DAD result.

Hair and unseen rear identity are not required to match reality. If recognizability or seam review
fails, the experiment records `visual_quality=failed`; numeric coverage cannot override that.

## Failure handling

Fail closed on:

- missing, duplicated, reordered, modified, wrong-sized, or empty views/masks;
- a DAD checkpoint, revision, runtime, topology, face-index, or CUDA mismatch;
- any inference network attempt;
- projection/image dimension mismatch or non-finite data;
- wrong camera-depth sign in synthetic or live audit;
- an existing output, atlas, record, or comparison path;
- texture coverage, observed-face coverage, neutral-area, or back-projection gate failure;
- external texture URI, absent material binding, invalid GLB, or Blender import/render failure;
- a geometrically valid but visually unrecognizable result.

No failure triggers another model, new view generation, a retry that overwrites evidence, or a
weaker gate.

## Testing

Public tests use only synthetic non-human fixtures:

- fixed-topology ellipsoid with a nose wedge and two eye shells;
- eight analytic yaw projections with distinct solid-color tiles;
- overlapping front/rear triangles proving z-buffer occlusion;
- back-facing and out-of-mask rejection;
- observed face-region priority only when yaw 0 is eligible;
- deterministic tie resolution;
- `(vertex, tile)` seam duplication and smooth-normal preservation;
- embedded-texture GLB readback with no URI;
- neutral fallback and every numeric gate;
- fake-plugin staged E2E, create-only resume, redaction, and no fallback.

The real private run verifies all source hashes before and after, runs seven sequential CUDA
inferences, builds the atlas, reloads the GLB, renders the four-row comparison, opens the result in
Blender, and records the manual verdict.

## License, privacy, and publication

- DAD source, checkpoint, FLAME assets, indices, runtime, private patches, and outputs remain under
  the existing CC BY-NC-SA 4.0 non-commercial research boundary.
- The observed image and every generated view remain private and local.
- Generated yaw textures are explicitly inferred content, not observations.
- No private path, basename, image, mask, projection, atlas, mesh, render, or face-derived hash is
  committed.
- Public code contains only original adapter/atlas logic and synthetic fixtures under Apache-2.0.
- Release and publication gates continue rejecting `.trcd`, external DAD/FLAME assets, private
  output directories, textures, OBJ/GLB artifacts, and real-person fixtures.
- README/research claims change only after the private live comparison is manually reviewed.
