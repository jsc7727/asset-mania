# SPDX-License-Identifier: GPL-3.0-or-later
"""Production reprojection and consolidation bake.

This is the real implementation. It runs inside Blender because that is where the
evaluated mesh, loop triangles, UVs, pose, camera matrices, render passes, ray casting, and
image datablocks all live -- so no private geometry has to cross into an Apache process.

The Apache oracle in `asset_mania_pipeline.reprojection` states the same conventions over
synthetic arrays. The two are compared against the known fixture; where they disagree, the
oracle is the specification and this module is the bug.

Coverage is the part worth reading carefully. `observed` is set only by a texel that
actually sampled the view and survived every rejection. Padding extends RGB for seam
filtering and is tracked separately; it never promotes observed coverage, and final alpha
comes from `observed` alone. No uncovered texel is given an invented colour.
"""

import hashlib
import math
from pathlib import Path

import bpy
import numpy
import OpenImageIO as oiio
from mathutils import Vector

from . import scene_inventory
from .selection import canonical_json

ALBEDO_LINEAR = "textures/albedo-linear.exr"
ALBEDO_PNG = "textures/albedo.png"
COVERAGE_PNG = "textures/coverage.png"
PADDED_COVERAGE_PNG = "textures/padded-coverage.png"
PREVIEW_PNG = "textures/preview.png"
SCENE_BAKED = "local/scene-baked.blend"

SOURCE_ATLAS_IMAGE = "AssetManiaReprojectedAtlas"
BAKE_TARGET_IMAGE = "AssetManiaBakeTarget"

DEPTH_ABSOLUTE_TOLERANCE_METERS = 0.0001
DEPTH_RELATIVE_TOLERANCE = 0.0002
RAY_EPSILON_SCALE = 1e-7
RAY_EPSILON_MIN_METERS = 1e-7
RAY_EPSILON_MAX_METERS = 1e-3


class ReprojectionFailed(Exception):
    """A closed-profile violation found while reprojecting or baking."""

    def __init__(self, diagnostics: list[str]) -> None:
        super().__init__(", ".join(sorted(set(diagnostics))))
        self.diagnostics = sorted(set(diagnostics))


# --- Conventions, restated to match the Apache oracle --------------------------


def texel_center(x: int, y: int, atlas_width: int, atlas_height: int) -> tuple[float, float]:
    return ((x + 0.5) / atlas_width, 1.0 - (y + 0.5) / atlas_height)


def srgb_to_linear(value: numpy.ndarray) -> numpy.ndarray:
    clipped = numpy.clip(value, 0.0, 1.0)
    return numpy.where(clipped <= 0.04045, clipped / 12.92, ((clipped + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(value: numpy.ndarray) -> numpy.ndarray:
    clipped = numpy.clip(value, 0.0, 1.0)
    return numpy.where(
        clipped <= 0.0031308,
        clipped * 12.92,
        1.055 * numpy.power(numpy.maximum(clipped, 1e-8), 1.0 / 2.4) - 0.055,
    )


def depth_tolerance(expected: float) -> float:
    return max(DEPTH_ABSOLUTE_TOLERANCE_METERS, abs(expected) * DEPTH_RELATIVE_TOLERANCE)


def ray_epsilon(diagonal: float) -> float:
    return min(
        max(abs(diagonal) * RAY_EPSILON_SCALE, RAY_EPSILON_MIN_METERS), RAY_EPSILON_MAX_METERS
    )


def _edge(a, b, point) -> float:
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])


def _is_top_left_edge(a, b) -> bool:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    if dy == 0.0:
        return dx < 0.0
    return dy > 0.0


def owns_texel(point, triangle) -> bool:
    """The same top-left fill rule the oracle uses, so ownership is total and disjoint."""
    a, b, c = triangle
    area = _edge(a, b, c)
    if area == 0.0 or not math.isfinite(area):
        return False
    if area < 0.0:
        b, c = c, b

    for start, end in ((a, b), (b, c), (c, a)):
        value = _edge(start, end, point)
        if value > 0.0:
            continue
        if value == 0.0 and _is_top_left_edge(start, end):
            continue
        return False
    return True


def barycentric(point, triangle):
    a, b, c = triangle
    doubled = _edge(a, b, c)
    if doubled == 0.0 or not math.isfinite(doubled):
        return None
    return (
        _edge(b, c, point) / doubled,
        _edge(c, a, point) / doubled,
        _edge(a, b, point) / doubled,
    )


def clip_to_pixel(clip, width: int, height: int):
    if clip[3] <= 0.0 or not all(math.isfinite(component) for component in clip):
        return None
    ndc = (clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3])
    if not all(-1.0 <= component <= 1.0 for component in ndc):
        return None
    return ((ndc[0] * 0.5 + 0.5) * width - 0.5, (1.0 - (ndc[1] * 0.5 + 0.5)) * height - 0.5)


def within_pixel_bounds(u: float, v: float, width: int, height: int) -> bool:
    return -0.5 <= u < width - 0.5 and -0.5 <= v < height - 0.5


def depth_neighbourhood(array: numpy.ndarray, u: float, v: float) -> tuple[float, float]:
    """The nearest and farthest finite depths over the 2x2 neighbourhood of `(u, v)`."""
    height, width = array.shape[0], array.shape[1]
    x0 = min(max(math.floor(u), 0), width - 1)
    y0 = min(max(math.floor(v), 0), height - 1)
    nearest = math.inf
    farthest = -math.inf
    for y in (y0, min(y0 + 1, height - 1)):
        for x in (x0, min(x0 + 1, width - 1)):
            value = float(array[y, x][0])
            if math.isfinite(value):
                nearest = min(nearest, value)
                farthest = max(farthest, value)
    return nearest, farthest


def is_occluded(expected: float, array: numpy.ndarray, u: float, v: float) -> bool:
    """Whether a texel lies behind the surface the depth pass recorded.

    This test exists to catch occlusion -- something standing between the camera and the
    point -- so it must not fire on quantization. Two allowances make that precise:

    * the nearest finite neighbour is the reference, because a sample between pixel
      centres is legitimately nearer than any single one of them; and
    * the neighbourhood's own depth span is added to the tolerance, because a surface
      slanted away from the camera changes depth across one pixel by more than the
      binding tolerance at modest resolutions.

    A genuinely hidden texel sits behind its occluder by far more than one pixel of local
    gradient, so it is still rejected.
    """
    nearest, farthest = depth_neighbourhood(array, u, v)
    if not math.isfinite(nearest):
        return True
    allowance = depth_tolerance(expected) + max(farthest - nearest, 0.0)
    return (expected - nearest) > allowance


def _nearest(array: numpy.ndarray, u: float, v: float):
    height, width = array.shape[0], array.shape[1]
    x = min(max(math.floor(u + 0.5), 0), width - 1)
    y = min(max(math.floor(v + 0.5), 0), height - 1)
    return array[y, x]


def _bilinear(array: numpy.ndarray, u: float, v: float) -> numpy.ndarray:
    height, width = array.shape[0], array.shape[1]
    x0 = math.floor(u)
    y0 = math.floor(v)
    fx = u - x0
    fy = v - y0

    def at(x: int, y: int) -> numpy.ndarray:
        return array[min(max(y, 0), height - 1), min(max(x, 0), width - 1)]

    top = at(x0, y0) * (1.0 - fx) + at(x0 + 1, y0) * fx
    bottom = at(x0, y0 + 1) * (1.0 - fx) + at(x0 + 1, y0 + 1) * fx
    return top * (1.0 - fy) + bottom * fy


# --- Reading the conditioning passes -------------------------------------------


def read_single_layer(path: Path) -> numpy.ndarray:
    """Read one canonical single-layer artifact written by the conditioning stage."""
    source = oiio.ImageInput.open(str(path))
    if source is None:
        raise ReprojectionFailed(["PASS_INVALID"])
    try:
        spec = source.spec()
        pixels = source.read_image(0, 0, 0, spec.nchannels, oiio.FLOAT)
        if pixels is None:
            raise ReprojectionFailed(["PASS_INVALID"])
        return numpy.asarray(pixels, dtype=numpy.float32).reshape(
            spec.height, spec.width, spec.nchannels
        )
    finally:
        source.close()


def read_view_png(path: Path) -> numpy.ndarray:
    """Read the normalized straight-alpha sRGB view as 0..1 floats."""
    source = oiio.ImageInput.open(str(path))
    if source is None:
        raise ReprojectionFailed(["VIEW_ALIGNMENT_MISMATCH"])
    try:
        spec = source.spec()
        pixels = source.read_image(0, 0, 0, spec.nchannels, oiio.FLOAT)
        if pixels is None:
            raise ReprojectionFailed(["VIEW_ALIGNMENT_MISMATCH"])
        return numpy.asarray(pixels, dtype=numpy.float32).reshape(
            spec.height, spec.width, spec.nchannels
        )
    finally:
        source.close()


# --- Evaluated geometry --------------------------------------------------------


def evaluated_triangles(target: bpy.types.Object):
    """Loop triangles of the evaluated target with world position, normal, and UV.

    Ordered by polygon index then triangle index, so the scan order is stable.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = target.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        if not len(mesh.uv_layers):
            raise ReprojectionFailed(["UV_MISSING_OR_INVALID"])
        uv_layer = mesh.uv_layers[0]
        matrix = evaluated.matrix_world
        rotation = matrix.to_3x3().inverted_safe().transposed()

        triangles = []
        for triangle in sorted(
            mesh.loop_triangles, key=lambda item: (item.polygon_index, tuple(item.loops))
        ):
            uv = tuple(tuple(uv_layer.data[loop].uv) for loop in triangle.loops)
            world = tuple(tuple(matrix @ mesh.vertices[index].co) for index in triangle.vertices)
            normals = tuple(
                tuple((rotation @ mesh.vertices[index].normal).normalized())
                for index in triangle.vertices
            )
            triangles.append(
                {
                    "polygon_index": int(triangle.polygon_index),
                    "uv": uv,
                    "world": world,
                    "normal": normals,
                }
            )
        return triangles
    finally:
        evaluated.to_mesh_clear()


def bounding_box_diagonal(target: bpy.types.Object) -> float:
    corners = [target.matrix_world @ Vector(corner) for corner in target.bound_box]
    if not corners:
        return 0.0
    lows = [min(corner[axis] for corner in corners) for axis in range(3)]
    highs = [max(corner[axis] for corner in corners) for axis in range(3)]
    return math.sqrt(sum((highs[axis] - lows[axis]) ** 2 for axis in range(3)))


# --- The deterministic texel loop ----------------------------------------------


def reproject(
    *,
    target: bpy.types.Object,
    world_to_clip,
    camera_origin,
    view_rgba: numpy.ndarray,
    mask: numpy.ndarray,
    depth: numpy.ndarray,
    atlas_size: tuple[int, int],
    scene: bpy.types.Scene,
    view_label: str = "view-1",
) -> dict:
    """Project every owned atlas texel back into the one supplied view."""
    atlas_width, atlas_height = atlas_size
    view_height, view_width = view_rgba.shape[0], view_rgba.shape[1]
    if (mask.shape[0], mask.shape[1]) != (view_height, view_width):
        raise ReprojectionFailed(["VIEW_ALIGNMENT_MISMATCH"])
    if (depth.shape[0], depth.shape[1]) != (view_height, view_width):
        raise ReprojectionFailed(["PASS_INVALID"])

    colour = numpy.zeros((atlas_height, atlas_width, 3), dtype=numpy.float32)
    observed = numpy.zeros((atlas_height, atlas_width), dtype=bool)
    rejected = {
        "degenerate_uv": 0,
        "outside_clip": 0,
        "outside_bounds": 0,
        "backfacing": 0,
        "outside_mask": 0,
        "depth_mismatch": 0,
        "occluded": 0,
    }

    linear_view = srgb_to_linear(view_rgba[:, :, :3])
    depsgraph = bpy.context.evaluated_depsgraph_get()
    epsilon = ray_epsilon(bounding_box_diagonal(target))
    evaluated_target = target.evaluated_get(depsgraph)

    for triangle in evaluated_triangles(target):
        for y in range(atlas_height):
            for x in range(atlas_width):
                if observed[y, x]:
                    continue
                centre = texel_center(x, y, atlas_width, atlas_height)
                if not owns_texel(centre, triangle["uv"]):
                    continue

                weights = barycentric(centre, triangle["uv"])
                if weights is None:
                    rejected["degenerate_uv"] += 1
                    continue

                world = Vector(
                    tuple(
                        sum(weights[i] * triangle["world"][i][axis] for i in range(3))
                        for axis in range(3)
                    )
                )
                normal = Vector(
                    tuple(
                        sum(weights[i] * triangle["normal"][i][axis] for i in range(3))
                        for axis in range(3)
                    )
                )

                clip = world_to_clip @ world.to_4d()
                pixel = clip_to_pixel(tuple(clip), view_width, view_height)
                if pixel is None:
                    rejected["outside_clip"] += 1
                    continue
                u, v = pixel
                if not within_pixel_bounds(u, v, view_width, view_height):
                    rejected["outside_bounds"] += 1
                    continue

                to_camera = camera_origin - world
                if normal.length == 0.0 or normal.dot(to_camera) <= 0.0:
                    rejected["backfacing"] += 1
                    continue
                if float(_nearest(mask, u, v)[0]) <= 0.0:
                    rejected["outside_mask"] += 1
                    continue

                expected = to_camera.length
                if is_occluded(expected, depth, u, v):
                    rejected["depth_mismatch"] += 1
                    continue

                direction = (world - camera_origin).normalized()
                origin = camera_origin + direction * epsilon
                hit, _location, _normal, _index, hit_object, _matrix = scene.ray_cast(
                    depsgraph, origin, direction
                )
                if not hit or hit_object is None or hit_object.original != target:
                    rejected["occluded"] += 1
                    continue

                colour[y, x] = _bilinear(linear_view, u, v)[:3]
                observed[y, x] = True

    padded = observed.copy()
    return {
        "colour": colour,
        "observed": observed,
        "padded": padded,
        "rejected": rejected,
        "view_label": view_label,
        "evaluated_target": evaluated_target,
    }


def dilate(result: dict, *, margin: int) -> dict:
    """Bounded seam dilation. RGB spreads; observed coverage does not."""
    if margin <= 0:
        return result

    colour = result["colour"]
    padded = result["padded"]
    height, width = padded.shape
    for _ in range(margin):
        additions = []
        for y in range(height):
            for x in range(width):
                if padded[y, x]:
                    continue
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and padded[ny, nx]:
                        additions.append((x, y, colour[ny, nx].copy()))
                        break
        for x, y, value in additions:
            colour[y, x] = value
            padded[y, x] = True
    return result


# --- Consolidation bake --------------------------------------------------------


def _write_exr(path: Path, data: numpy.ndarray, channels: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, count = data.shape
    spec = oiio.ImageSpec(width, height, count, oiio.FLOAT)
    spec.channelnames = channels
    output = oiio.ImageOutput.create(str(path))
    if output is None or not output.open(str(path), spec):
        raise ReprojectionFailed(["BAKE_CONTEXT_INVALID"])
    try:
        if not output.write_image(numpy.ascontiguousarray(data, dtype=numpy.float32)):
            raise ReprojectionFailed(["BAKE_CONTEXT_INVALID"])
    finally:
        output.close()


def _write_png(path: Path, data: numpy.ndarray, channels: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, count = data.shape
    quantized = numpy.clip(numpy.rint(data * 255.0), 0, 255).astype(numpy.uint8)
    spec = oiio.ImageSpec(width, height, count, oiio.UINT8)
    spec.channelnames = channels
    output = oiio.ImageOutput.create(str(path))
    if output is None or not output.open(str(path), spec):
        raise ReprojectionFailed(["BAKE_CONTEXT_INVALID"])
    try:
        if not output.write_image(numpy.ascontiguousarray(quantized)):
            raise ReprojectionFailed(["BAKE_CONTEXT_INVALID"])
    finally:
        output.close()


def consolidate_bake(
    *,
    scene: bpy.types.Scene,
    target: bpy.types.Object,
    colour: numpy.ndarray,
    observed: numpy.ndarray,
    atlas_size: tuple[int, int],
    bake_margin: int,
) -> numpy.ndarray:
    """Copy the reprojected atlas into the material-owned UV with no lighting.

    Two distinct image datablocks are used on purpose: the reprojection result is the
    emission source and a separate empty image is the bake target. Using one image as both
    is invalid, and Blender will happily produce garbage if asked to.
    """
    atlas_width, atlas_height = atlas_size

    for name in (SOURCE_ATLAS_IMAGE, BAKE_TARGET_IMAGE):
        existing = bpy.data.images.get(name)
        if existing is not None:
            bpy.data.images.remove(existing)

    source = bpy.data.images.new(
        SOURCE_ATLAS_IMAGE, width=atlas_width, height=atlas_height, alpha=True, float_buffer=True
    )
    source.colorspace_settings.name = "Non-Color"
    flat = numpy.zeros((atlas_height, atlas_width, 4), dtype=numpy.float32)
    flat[:, :, :3] = colour
    flat[:, :, 3] = observed.astype(numpy.float32)
    # Blender addresses image pixels bottom-up; the atlas is stored top-down.
    source.pixels.foreach_set(numpy.flipud(flat).ravel())
    source.update()

    destination = bpy.data.images.new(
        BAKE_TARGET_IMAGE, width=atlas_width, height=atlas_height, alpha=True, float_buffer=True
    )
    destination.colorspace_settings.name = "Non-Color"
    if source.name == destination.name:
        raise ReprojectionFailed(["BAKE_CONTEXT_INVALID"])

    material = _emission_material(source)
    target.data.materials.clear()
    target.data.materials.append(material)
    _activate_bake_target(material, destination)

    scene.cycles.bake_type = "EMIT"
    scene.render.bake.margin = int(bake_margin)
    scene.render.bake.use_selected_to_active = False
    scene.render.bake.use_clear = True

    previous_active = bpy.context.view_layer.objects.active
    for item in bpy.context.view_layer.objects:
        item.select_set(False)
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    try:
        bpy.ops.object.bake(type="EMIT")
    except RuntimeError as error:
        raise ReprojectionFailed(["BAKE_CONTEXT_INVALID"]) from error
    finally:
        bpy.context.view_layer.objects.active = previous_active

    baked = numpy.zeros(atlas_width * atlas_height * 4, dtype=numpy.float32)
    destination.pixels.foreach_get(baked)
    return numpy.flipud(baked.reshape(atlas_height, atlas_width, 4))


def _emission_material(source: bpy.types.Image) -> bpy.types.Material:
    """An unlit emission material, so the bake copies colour rather than shading it."""
    material = bpy.data.materials.new("AssetManiaConsolidation")
    material.use_nodes = True
    tree = material.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)

    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = source
    texture.interpolation = "Closest"
    texture.extension = "EXTEND"
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    output = tree.nodes.new("ShaderNodeOutputMaterial")

    tree.links.new(texture.outputs["Color"], emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _activate_bake_target(material: bpy.types.Material, destination: bpy.types.Image) -> None:
    tree = material.node_tree
    node = tree.nodes.new("ShaderNodeTexImage")
    node.image = destination
    node.label = "AssetManiaBakeTarget"
    tree.nodes.active = node


def publish(
    *,
    staging_root: Path,
    colour: numpy.ndarray,
    observed: numpy.ndarray,
    padded: numpy.ndarray,
    baked: numpy.ndarray,
    scene: bpy.types.Scene,
) -> dict:
    """Write the texture artifacts with alpha set explicitly from observed coverage."""
    height, width = observed.shape
    alpha = observed.astype(numpy.float32)

    linear = numpy.zeros((height, width, 4), dtype=numpy.float32)
    linear[:, :, :3] = baked[:, :, :3]
    linear[:, :, 3] = alpha
    if not numpy.isfinite(linear).all():
        raise ReprojectionFailed(["BAKE_CONTEXT_INVALID"])
    _write_exr(staging_root / ALBEDO_LINEAR, linear, ("R", "G", "B", "A"))

    delivery = numpy.zeros((height, width, 4), dtype=numpy.float32)
    delivery[:, :, :3] = linear_to_srgb(linear[:, :, :3])
    delivery[:, :, 3] = alpha
    _write_png(staging_root / ALBEDO_PNG, delivery, ("R", "G", "B", "A"))

    _write_png(staging_root / COVERAGE_PNG, alpha[:, :, None], ("Y",))
    _write_png(staging_root / PADDED_COVERAGE_PNG, padded.astype(numpy.float32)[:, :, None], ("Y",))
    # The preview shows the delivered colour over the observed mask only, so an uncovered
    # texel is visibly uncovered rather than quietly black.
    preview = numpy.zeros((height, width, 3), dtype=numpy.float32)
    preview[observed] = delivery[:, :, :3][observed]
    _write_png(staging_root / PREVIEW_PNG, preview, ("R", "G", "B"))

    baked_scene = staging_root / SCENE_BAKED
    baked_scene.parent.mkdir(parents=True, exist_ok=True)
    bpy.data.libraries.write(str(baked_scene), {scene}, fake_user=True, compress=False)

    return {
        "observed_texel_count": int(observed.sum()),
        "padded_texel_count": int(padded.sum() - observed.sum()),
        "finite_texel_count": int(numpy.isfinite(linear).all(axis=-1).sum()),
        "coverage_ratio": float(observed.sum()) / float(width * height),
        "texture_semantic_digest": hashlib.sha256(
            numpy.ascontiguousarray(delivery).tobytes()
        ).hexdigest(),
    }


def bake(request: dict) -> dict:
    """Run the reprojection and consolidation bake stage."""
    staging_root = Path(str(request["staging_root"]))
    condition_directory = Path(str(request["condition_run_directory"]))
    atlas_size = tuple(int(value) for value in request["atlas_size"])
    minimum_coverage = float(request["minimum_coverage"])
    bake_margin = int(request["bake_margin"])
    color_padding = int(request["color_padding"])

    resolved, diagnostics = scene_inventory.resolve_selection(request)
    if diagnostics:
        raise ReprojectionFailed(diagnostics)
    target = resolved["target"]
    camera = resolved["camera"]
    rig = resolved["armature"]

    if not scene_inventory.selection_matches_request(
        request, target=target, camera=camera, rig=rig
    ):
        raise ReprojectionFailed([scene_inventory.PLAN_TAMPERED])

    scene = bpy.context.scene
    scene.frame_set(int(request["frame"]))

    view_rgba = read_view_png(Path(str(request["view_path"])))
    mask = read_single_layer(condition_directory / "artifacts/conditioning/mask.png")
    depth = read_single_layer(condition_directory / "artifacts/conditioning/depth.exr")

    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_camera = camera.evaluated_get(depsgraph)
    projection = evaluated_camera.calc_matrix_camera(
        depsgraph,
        x=scene.render.resolution_x,
        y=scene.render.resolution_y,
        scale_x=scene.render.pixel_aspect_x,
        scale_y=scene.render.pixel_aspect_y,
    )
    world_to_clip = projection @ evaluated_camera.matrix_world.inverted()

    result = reproject(
        target=target,
        world_to_clip=world_to_clip,
        camera_origin=evaluated_camera.matrix_world.translation.copy(),
        view_rgba=view_rgba,
        mask=mask,
        depth=depth,
        atlas_size=atlas_size,
        scene=scene,
    )
    if not result["observed"].any():
        raise ReprojectionFailed(["REPROJECTION_LOW_COVERAGE"])

    dilate(result, margin=color_padding)
    baked = consolidate_bake(
        scene=scene,
        target=target,
        colour=result["colour"],
        observed=result["observed"],
        atlas_size=atlas_size,
        bake_margin=bake_margin,
    )
    metrics = publish(
        staging_root=staging_root,
        colour=result["colour"],
        observed=result["observed"],
        padded=result["padded"],
        baked=baked,
        scene=scene,
    )

    diagnostics = []
    if metrics["coverage_ratio"] < minimum_coverage:
        # Low coverage keeps its artifacts, marked incomplete, but cannot feed an export.
        diagnostics.append("REPROJECTION_LOW_COVERAGE")

    atlas_width, atlas_height = atlas_size
    return {
        "metrics": {
            "kind": "bake",
            "atlas_width": atlas_width,
            "atlas_height": atlas_height,
            "observed_texel_count": metrics["observed_texel_count"],
            "padded_texel_count": metrics["padded_texel_count"],
            "finite_texel_count": metrics["finite_texel_count"],
            "coverage_ratio": scene_inventory.quantize(metrics["coverage_ratio"]),
            "texture_semantic_digest": metrics["texture_semantic_digest"],
        },
        "rejected": result["rejected"],
        "diagnostics": diagnostics,
        "artifacts": [
            ALBEDO_LINEAR,
            ALBEDO_PNG,
            COVERAGE_PNG,
            PADDED_COVERAGE_PNG,
            PREVIEW_PNG,
            SCENE_BAKED,
        ],
        "semantic": canonical_json(
            {"coverage": metrics["coverage_ratio"], "digest": metrics["texture_semantic_digest"]}
        ),
    }
