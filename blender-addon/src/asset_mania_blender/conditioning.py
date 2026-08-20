# SPDX-License-Identifier: GPL-3.0-or-later
"""Conditioning: render the closed Cycles CPU profile and publish the pass bundle.

The worker builds a fresh derived scene holding only allowlisted data, sanitizes every
write surface, renders the combined, Z, normal, and object-index passes, then converts
them into the canonical single-layer artifacts the bundle declares.

Two API facts of the pinned Blender 5.2.0 shape this module, both measured rather than
assumed:

* the compositor File Output node writes OpenEXR *multilayer* containers only, and
  Blender's Python API cannot read a multilayer EXR's pixels back; so
* the passes are re-read with the OpenImageIO module that ships inside Blender and
  rewritten as single-layer EXRs. The pixel data therefore still comes from Cycles -- no
  pass is substituted with an analytic approximation -- while every published artifact
  stays readable and verifiable.
"""

import hashlib
from pathlib import Path

import bpy
import numpy
import OpenImageIO as oiio
from mathutils import Vector

from . import scene_inventory
from .selection import canonical_json

BUNDLE_NAME = "bundle.json"
CONDITIONING_DIRECTORY = "artifacts/conditioning"
LOCAL_DIRECTORY = "artifacts/local"
SCENE_STATE_NAME = "scene-state.blend"
TARGET_PASS_INDEX = 1
PASS_ALPHA_THRESHOLD = 0.5

#: (compositor item name, render-layer socket, artifact role, relative path)
PASS_SPECS = (
    ("beauty", "Image", "beauty_exr", "beauty.exr"),
    ("depth", "Depth", "depth_exr", "depth.exr"),
    ("normal", "Normal", "normal_exr", "normal.exr"),
    ("object-index", "Object Index", "object_index_exr", "object-index.exr"),
)
PREVIEW_SPECS = (
    ("beauty_preview", "beauty.png", "beauty"),
    ("depth_preview", "depth-preview.png", "depth"),
    ("normal_preview", "normal-preview.png", "normal"),
)
MASK_ROLE = "mask_png"
MASK_PATH = "mask.png"


#: A silhouette pixel averages foreground and background samples, so its normal is not a
#: unit vector. The unit-length requirement therefore applies to the eroded interior of the
#: mask, which is what `interior_unit_normal_count` counts.
NORMAL_UNIT_TOLERANCE = 1e-2


def erode(mask: "numpy.ndarray") -> "numpy.ndarray":
    """The pixels whose eight neighbours are all inside the mask."""
    padded = numpy.pad(mask, 1, mode="constant", constant_values=False)
    interior = numpy.ones_like(mask, dtype=bool)
    for row in (0, 1, 2):
        for column in (0, 1, 2):
            height, width = mask.shape
            interior &= padded[row : row + height, column : column + width]
    return interior


class ConditioningFailed(Exception):
    """A closed-profile violation found while conditioning."""

    def __init__(self, diagnostics: list[str]) -> None:
        super().__init__(", ".join(sorted(set(diagnostics))))
        self.diagnostics = sorted(set(diagnostics))


# --- The derived scene ---------------------------------------------------------


def build_derived_scene(
    *,
    target: bpy.types.Object,
    camera: bpy.types.Object,
    rig: bpy.types.Object | None,
    action: bpy.types.Action | None,
    frame: int,
    resolution: tuple[int, int],
    render_profile: dict,
) -> bpy.types.Scene:
    """A fresh scene holding only the allowlisted data, configured to the profile.

    Nothing from the source scene's render settings, compositor, sequencer, or object list
    carries over: the scene is new, and only the named objects plus one explicit light are
    linked into it.
    """
    scene = bpy.data.scenes.new("AssetManiaDerived")
    collection = scene.collection

    for item in (target, camera, rig):
        if item is not None:
            collection.objects.link(item)

    light_data = bpy.data.lights.new("AssetManiaDerivedKey", type="AREA")
    light_data.energy = 500.0
    light_data.size = 4.0
    light = bpy.data.objects.new("AssetManiaDerivedKey", light_data)
    light.location = Vector((1.5, -4.0, 5.0))
    collection.objects.link(light)

    scene.world = bpy.data.worlds.new("AssetManiaDerivedWorld")
    scene.world.use_nodes = False
    scene.camera = camera

    for item in collection.objects:
        item.pass_index = TARGET_PASS_INDEX if item is target else 0

    if rig is not None and action is not None:
        rig.animation_data_create()
        rig.animation_data.action = action

    width, height = resolution
    render = scene.render
    render.engine = "CYCLES"
    render.resolution_x = width
    render.resolution_y = height
    render.resolution_percentage = 100
    render.pixel_aspect_x, render.pixel_aspect_y = render_profile["pixel_aspect"]
    render.film_transparent = bool(render_profile["film_transparent"])
    render.use_motion_blur = bool(render_profile["motion_blur"])
    render.use_border = bool(render_profile["render_border"])
    render.use_crop_to_border = bool(render_profile["crop_to_border"])
    render.threads_mode = "FIXED"
    render.threads = int(render_profile["threads"])

    cycles = scene.cycles
    cycles.device = render_profile["device"]
    cycles.samples = int(render_profile["samples"])
    cycles.seed = int(render_profile["seed"])
    cycles.use_adaptive_sampling = bool(render_profile["adaptive_sampling"])
    cycles.use_denoising = bool(render_profile["denoise"])
    cycles.use_animated_seed = bool(render_profile["animated_seed"])

    camera.data.dof.use_dof = bool(render_profile["depth_of_field"])

    view_layer = scene.view_layers[0]
    view_layer.use_pass_combined = True
    view_layer.use_pass_z = True
    view_layer.use_pass_normal = True
    view_layer.use_pass_object_index = True
    view_layer.pass_alpha_threshold = float(render_profile["pass_alpha_threshold"])

    scene.frame_start = frame
    scene.frame_end = frame
    scene.frame_step = 1
    scene.frame_set(frame)
    return scene


def _reserved_index_is_unique(scene: bpy.types.Scene, target: bpy.types.Object) -> bool:
    return all(
        item.pass_index != TARGET_PASS_INDEX
        for item in scene.collection.objects
        if item is not target
    )


# --- Rendering ------------------------------------------------------------------


def attach_pass_writer(scene: bpy.types.Scene, render_directory: Path) -> None:
    """Wire one File Output node per pass into a worker-owned compositor graph.

    The source compositor was already detached by sanitization; this graph is created by
    the worker, writes only below staging, and holds no other node type.
    """
    group = bpy.data.node_groups.new("AssetManiaConditioning", "CompositorNodeTree")
    # The render-layers node only exposes Depth, Normal, and Object Index once it points at
    # a scene whose view layer has those passes enabled, so bind it before linking.
    scene.compositing_node_group = group
    scene.use_nodes = True
    layers = group.nodes.new("CompositorNodeRLayers")
    layers.scene = scene

    missing = [socket for _item, socket, _role, _path in PASS_SPECS if socket not in layers.outputs]
    if missing:
        raise ConditioningFailed(["PASS_INVALID"])

    for index, (item_name, socket, _role, _path) in enumerate(PASS_SPECS):
        node = group.nodes.new("CompositorNodeOutputFile")
        node.location = (320, index * -160)
        node.directory = str(render_directory)
        node.file_name = item_name
        node.use_file_extension = True
        node.file_output_items.clear()
        socket_type = "RGBA" if socket == "Image" else ("VECTOR" if socket == "Normal" else "FLOAT")
        node.file_output_items.new(socket_type, item_name)
        group.links.new(layers.outputs[socket], node.inputs[item_name])


def render_passes(scene: bpy.types.Scene, render_directory: Path) -> dict[str, Path]:
    """Render once and return the multilayer container written for each pass."""
    render_directory.mkdir(parents=True, exist_ok=True)
    attach_pass_writer(scene, render_directory)

    previous = bpy.context.window.scene if bpy.context.window else None
    if bpy.context.window is not None:
        bpy.context.window.scene = scene
    try:
        bpy.ops.render.render(scene=scene.name, write_still=False)
    finally:
        if previous is not None and bpy.context.window is not None:
            bpy.context.window.scene = previous

    written: dict[str, Path] = {}
    for item_name, _socket, _role, _path in PASS_SPECS:
        candidate = render_directory / f"{item_name}.exr"
        if not candidate.is_file():
            raise ConditioningFailed(["PASS_INVALID"])
        written[item_name] = candidate
    return written


# --- Pass conversion ------------------------------------------------------------


def read_pass(path: Path) -> numpy.ndarray:
    """Read one multilayer container written by the compositor as a float array."""
    source = oiio.ImageInput.open(str(path))
    if source is None:
        raise ConditioningFailed(["PASS_INVALID"])
    try:
        spec = source.spec()
        pixels = source.read_image(0, 0, 0, spec.nchannels, oiio.FLOAT)
        if pixels is None:
            raise ConditioningFailed(["PASS_INVALID"])
        return numpy.asarray(pixels, dtype=numpy.float32).reshape(
            spec.height, spec.width, spec.nchannels
        )
    finally:
        source.close()


def write_exr(path: Path, data: numpy.ndarray, channel_names: tuple[str, ...]) -> None:
    """Write a canonical single-layer 32-bit float EXR."""
    height, width, channels = data.shape
    spec = oiio.ImageSpec(width, height, channels, oiio.FLOAT)
    spec.channelnames = channel_names
    output = oiio.ImageOutput.create(str(path))
    if output is None or not output.open(str(path), spec):
        raise ConditioningFailed(["PASS_INVALID"])
    try:
        if not output.write_image(numpy.ascontiguousarray(data, dtype=numpy.float32)):
            raise ConditioningFailed(["PASS_INVALID"])
    finally:
        output.close()


def write_png(path: Path, data: numpy.ndarray, channel_names: tuple[str, ...]) -> None:
    """Write an 8-bit PNG from values already in 0..1."""
    height, width, channels = data.shape
    quantized = numpy.clip(numpy.rint(data * 255.0), 0, 255).astype(numpy.uint8)
    spec = oiio.ImageSpec(width, height, channels, oiio.UINT8)
    spec.channelnames = channel_names
    output = oiio.ImageOutput.create(str(path))
    if output is None or not output.open(str(path), spec):
        raise ConditioningFailed(["PASS_INVALID"])
    try:
        if not output.write_image(numpy.ascontiguousarray(quantized)):
            raise ConditioningFailed(["PASS_INVALID"])
    finally:
        output.close()


def encode_srgb(linear: numpy.ndarray) -> numpy.ndarray:
    """The exact sRGB transfer function, applied explicitly and recorded in the bundle."""
    clipped = numpy.clip(linear, 0.0, 1.0)
    low = clipped * 12.92
    high = 1.055 * numpy.power(numpy.maximum(clipped, 1e-8), 1.0 / 2.4) - 0.055
    return numpy.where(clipped <= 0.0031308, low, high)


def euclidean_depth_scale(
    scene: bpy.types.Scene, camera: bpy.types.Object, width: int, height: int
) -> numpy.ndarray:
    """Per-pixel factor converting Cycles' planar Z pass to Euclidean distance.

    Measured, not assumed: Cycles' `Depth` pass records distance along the camera's view
    axis, not radial distance from the camera origin. The bundle declares
    `camera_euclidean_distance`, so the pass has to be converted rather than the
    declaration weakened -- a downstream reprojection compares |camera - point| against
    this pass, and a planar value is short by 1/cos(angle) toward the frame edges.

    The factor is `|d| / |d.z|` for the camera-space ray `d` through each pixel centre,
    taken from Blender's own view frame so the lens, sensor fit, and shift are all
    included.
    """
    top_right, bottom_right, bottom_left, top_left = camera.data.view_frame(scene=scene)
    scale = numpy.empty((height, width), dtype=numpy.float32)
    for y in range(height):
        v = (y + 0.5) / height
        left = top_left.lerp(bottom_left, v)
        right = top_right.lerp(bottom_right, v)
        for x in range(width):
            direction = left.lerp(right, (x + 0.5) / width)
            axis = abs(direction.z)
            scale[y, x] = (direction.length / axis) if axis > 0.0 else 1.0
    return scale


def convert_passes(
    *,
    containers: dict[str, Path],
    artifact_directory: Path,
    depth_range: tuple[float, float],
    depth_scale: numpy.ndarray,
) -> dict[str, dict]:
    """Rewrite every pass as a canonical single-layer artifact plus its preview.

    Returns a role-keyed description carrying the relative path and the measured
    statistics the bundle and its validation need.
    """
    artifact_directory.mkdir(parents=True, exist_ok=True)
    beauty = read_pass(containers["beauty"])
    planar_depth = read_pass(containers["depth"])[:, :, :1]
    if planar_depth.shape[:2] != depth_scale.shape:
        raise ConditioningFailed(["PASS_INVALID"])
    depth = planar_depth * depth_scale[:, :, None]
    normal = read_pass(containers["normal"])[:, :, :3]
    index = read_pass(containers["object-index"])[:, :, :1]

    height, width = beauty.shape[0], beauty.shape[1]
    for array in (depth, normal, index):
        if array.shape[0] != height or array.shape[1] != width:
            raise ConditioningFailed(["PASS_INVALID"])

    foreground = numpy.isclose(index[:, :, 0], float(TARGET_PASS_INDEX), atol=1e-6)
    if not foreground.any():
        raise ConditioningFailed(["PASS_INVALID"])

    foreground_depth = depth[:, :, 0][foreground]
    if not numpy.isfinite(foreground_depth).all():
        raise ConditioningFailed(["PASS_INVALID"])
    if not numpy.isfinite(beauty[foreground]).all():
        raise ConditioningFailed(["PASS_INVALID"])
    if not numpy.isfinite(normal[foreground]).all():
        raise ConditioningFailed(["PASS_INVALID"])

    interior = erode(foreground)
    lengths = numpy.linalg.norm(normal[interior], axis=-1)
    if lengths.size and float(numpy.abs(lengths - 1.0).max()) > NORMAL_UNIT_TOLERANCE:
        raise ConditioningFailed(["PASS_INVALID"])

    write_exr(artifact_directory / "beauty.exr", beauty, ("R", "G", "B", "A"))
    write_exr(artifact_directory / "depth.exr", depth, ("Y",))
    write_exr(artifact_directory / "normal.exr", normal, ("R", "G", "B"))
    write_exr(artifact_directory / "object-index.exr", index, ("Y",))

    write_png(
        artifact_directory / "beauty.png",
        numpy.concatenate(
            [encode_srgb(beauty[:, :, :3]), numpy.clip(beauty[:, :, 3:4], 0.0, 1.0)], axis=-1
        ),
        ("R", "G", "B", "A"),
    )

    # The depth preview is a normalized visualization over the declared valid range; the
    # canonical metric depth stays in the EXR.
    valid_min, valid_max = depth_range
    span = max(valid_max - valid_min, 1e-6)
    normalized = numpy.clip((depth - valid_min) / span, 0.0, 1.0)
    normalized = numpy.where(foreground[:, :, None], normalized, 0.0)
    write_png(artifact_directory / "depth-preview.png", normalized, ("Y",))

    # The normal preview maps the unit vector into 0..1 per channel and says so.
    write_png(
        artifact_directory / "normal-preview.png",
        numpy.where(foreground[:, :, None], (normal + 1.0) * 0.5, 0.0),
        ("R", "G", "B"),
    )

    mask = numpy.where(foreground, 1.0, 0.0).astype(numpy.float32)[:, :, None]
    write_png(artifact_directory / MASK_PATH, mask, ("Y",))

    return {
        "resolution": (int(width), int(height)),
        "foreground_pixel_count": int(foreground.sum()),
        "finite_foreground_depth_count": int(numpy.isfinite(foreground_depth).sum()),
        "interior_pixel_count": int(interior.sum()),
        "interior_unit_normal_count": int(
            numpy.logical_and(
                interior,
                numpy.abs(numpy.linalg.norm(normal, axis=-1) - 1.0) <= NORMAL_UNIT_TOLERANCE,
            ).sum()
        ),
        "depth_observed_min": float(foreground_depth.min()),
        "depth_observed_max": float(foreground_depth.max()),
    }


# --- Camera, digests, and the bundle -------------------------------------------


def _matrix(values) -> list[float]:
    return [scene_inventory.quantize(component) for row in values for component in row]


def camera_record(scene: bpy.types.Scene, camera: bpy.types.Object) -> tuple[dict, dict]:
    """The camera record and the four row-major matrices, from Blender's own API."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = camera.evaluated_get(depsgraph)
    projection = evaluated.calc_matrix_camera(
        depsgraph,
        x=scene.render.resolution_x,
        y=scene.render.resolution_y,
        scale_x=scene.render.pixel_aspect_x,
        scale_y=scene.render.pixel_aspect_y,
    )
    camera_to_world = evaluated.matrix_world
    world_to_camera = camera_to_world.inverted()
    data = camera.data

    record = {
        "projection_type": "perspective" if data.type == "PERSP" else "orthographic",
        "lens_mm": scene_inventory.quantize(data.lens) if data.type == "PERSP" else None,
        "sensor_fit": data.sensor_fit,
        "sensor_width_mm": scene_inventory.quantize(data.sensor_width),
        "sensor_height_mm": scene_inventory.quantize(data.sensor_height),
        "shift_x": scene_inventory.quantize(data.shift_x),
        "shift_y": scene_inventory.quantize(data.shift_y),
        "clip_start_meters": scene_inventory.quantize(data.clip_start),
        "clip_end_meters": scene_inventory.quantize(data.clip_end),
        "ortho_scale": (
            None if data.type == "PERSP" else scene_inventory.quantize(data.ortho_scale)
        ),
    }
    matrices = {
        "layout": "row_major",
        "camera_to_world": _matrix(camera_to_world),
        "world_to_camera": _matrix(world_to_camera),
        "projection": _matrix(projection),
        "world_to_clip": _matrix(projection @ world_to_camera),
    }
    return record, matrices


def projection_error_pixels(
    *,
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    target: bpy.types.Object,
    world_to_clip,
) -> float:
    """Max pixel disagreement between our matrix chain and Blender's own projection.

    The target's evaluated vertices are the fiducials. Their pixel positions are computed
    twice: once through the `world_to_clip` matrix this bundle publishes, and once through
    `bpy_extras.object_utils.world_to_camera_view`, which is an independent Blender code
    path. A downstream stage trusts the published matrices, so the two must agree; the
    binding profile allows at most a quarter pixel.
    """
    from bpy_extras.object_utils import world_to_camera_view

    width = scene.render.resolution_x
    height = scene.render.resolution_y
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = target.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    worst = 0.0
    try:
        for vertex in mesh.vertices:
            world = evaluated.matrix_world @ vertex.co
            clip = world_to_clip @ world.to_4d()
            if abs(clip.w) < 1e-9:
                continue
            ours_x = (clip.x / clip.w * 0.5 + 0.5) * width
            ours_y = (1.0 - (clip.y / clip.w * 0.5 + 0.5)) * height

            reference = world_to_camera_view(scene, camera, world)
            theirs_x = reference.x * width
            theirs_y = (1.0 - reference.y) * height

            worst = max(worst, abs(ours_x - theirs_x), abs(ours_y - theirs_y))
    finally:
        evaluated.to_mesh_clear()
    return worst


PROJECTION_ERROR_LIMIT_PIXELS = 0.25


def evaluated_digests(target: bpy.types.Object, rig: bpy.types.Object | None) -> dict[str, str]:
    """Geometry, UV, and pose digests taken from the evaluated dependency graph."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = target.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        geometry = {
            "vertices": sorted(
                [scene_inventory.quantize(component) for component in vertex.co]
                for vertex in mesh.vertices
            ),
            "polygons": sorted(
                sorted(int(index) for index in polygon.vertices) for polygon in mesh.polygons
            ),
        }
        uv_layer = mesh.uv_layers[0] if len(mesh.uv_layers) else None
        uv = (
            sorted(
                [
                    scene_inventory.quantize(uv_layer.data[index].uv[0]),
                    scene_inventory.quantize(uv_layer.data[index].uv[1]),
                ]
                for polygon in mesh.polygons
                for index in polygon.loop_indices
            )
            if uv_layer is not None
            else []
        )
    finally:
        evaluated.to_mesh_clear()

    pose = (
        []
        if rig is None
        else sorted(_matrix(bone.matrix) for bone in rig.evaluated_get(depsgraph).pose.bones)
    )
    return {
        "evaluated_geometry": _digest(geometry),
        "uv": _digest(uv),
        "pose": _digest(pose),
    }


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


PASS_MEDIA_TYPES = {
    "beauty_exr": "image/x-exr",
    "beauty_preview": "image/png",
    "depth_exr": "image/x-exr",
    "depth_preview": "image/png",
    "normal_exr": "image/x-exr",
    "normal_preview": "image/png",
    "object_index_exr": "image/x-exr",
    "mask_png": "image/png",
}
PASS_COLOR_SPACES = {
    "beauty_exr": "scene_linear",
    "beauty_preview": "srgb",
    "depth_exr": "data",
    "depth_preview": "srgb",
    "normal_exr": "data",
    "normal_preview": "srgb",
    "object_index_exr": "data",
    "mask_png": "data",
}
PASS_ORDER = (
    ("beauty_exr", "beauty.exr"),
    ("beauty_preview", "beauty.png"),
    ("depth_exr", "depth.exr"),
    ("depth_preview", "depth-preview.png"),
    ("normal_exr", "normal.exr"),
    ("normal_preview", "normal-preview.png"),
    ("object_index_exr", "object-index.exr"),
    ("mask_png", MASK_PATH),
)


def build_bundle(
    *,
    staging_root: Path,
    source_scene_sha256: str,
    digests: dict[str, str],
    portable_selection: dict,
    frame: int,
    statistics: dict,
    camera: dict,
    matrices: dict,
    render_profile: dict,
    blender: dict,
    depth_range: tuple[float, float],
) -> dict:
    """Assemble `conditioning-bundle-v1` from measured values only."""
    width, height = statistics["resolution"]
    artifact_directory = staging_root / CONDITIONING_DIRECTORY

    passes = []
    for role, relative in PASS_ORDER:
        path = artifact_directory / relative
        if not path.is_file():
            raise ConditioningFailed(["PASS_INVALID"])
        passes.append(
            {
                "role": role,
                "path": f"{CONDITIONING_DIRECTORY}/{relative}",
                "sha256": _sha256_file(path),
                "byte_size": path.stat().st_size,
                "media_type": PASS_MEDIA_TYPES[role],
                "color_space": PASS_COLOR_SPACES[role],
                "upload_eligible": True,
            }
        )

    bundle = {
        "schema_id": "asset-mania/conditioning-bundle",
        "schema_version": "1.0",
        "digests": {"source_scene": source_scene_sha256, **digests},
        "selection": dict(portable_selection),
        "frame": int(frame),
        "resolution": [int(width), int(height)],
        "pixel_aspect": [1.0, 1.0],
        "pixel_origin": "top_left",
        "axes": {
            "world": {"handedness": "right", "up": "+Z", "forward": "-Y"},
            "camera": {"right": "+X", "up": "+Y", "view": "-Z"},
        },
        "matrices": matrices,
        "camera": camera,
        "scene_unit_scale_meters": 1.0,
        "depth": {
            "space": "camera_euclidean_distance",
            "unit": "meters",
            "background": "invalid_by_mask",
            "valid_min_meters": scene_inventory.quantize(depth_range[0]),
            "valid_max_meters": scene_inventory.quantize(depth_range[1]),
        },
        "normal": {
            "space": "world",
            "channels": ["x", "y", "z"],
            "encoding": "float32_linear",
            "foreground_unit_expected": True,
        },
        "mask": {
            "target_object_index": TARGET_PASS_INDEX,
            "foreground": 255,
            "background": 0,
            "pass_alpha_threshold": PASS_ALPHA_THRESHOLD,
            "antialiasing": "none",
        },
        "render_profile": dict(render_profile),
        "blender": dict(blender),
        "passes": passes,
        "bundle_sha256": "",
    }
    preimage = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    return {**preimage, "bundle_sha256": _digest(preimage)}


def write_scene_state(scene: bpy.types.Scene, path: Path) -> None:
    """Save only the derived scene, so the local artifact carries no unrelated data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.data.libraries.write(str(path), {scene}, fake_user=True, compress=False)


def condition(request: dict) -> dict:
    """Run the conditioning stage and return the bundle plus its artifact inventory."""
    staging_root = Path(str(request["staging_root"]))
    frame = int(request["frame"])
    render_profile = dict(request["render_profile"])
    resolution = tuple(int(value) for value in request["resolution"])

    resolved, diagnostics = scene_inventory.resolve_selection(request)
    if diagnostics:
        raise ConditioningFailed(diagnostics)

    target = resolved["target"]
    camera_object = resolved["camera"]
    rig = resolved["armature"]
    action_name = request.get("action_name")
    action = bpy.data.actions.get(str(action_name)) if action_name else None
    if action_name and action is None:
        raise ConditioningFailed([scene_inventory.POSE_UNKNOWN])

    # The plan binding is re-verified here, against the file this stage actually opened.
    # Conditioning publishes artifacts, so a plan edited after approval must fail before
    # anything is rendered.
    if not scene_inventory.selection_matches_request(
        request, target=target, camera=camera_object, rig=rig
    ):
        raise ConditioningFailed([scene_inventory.PLAN_TAMPERED])

    scene = build_derived_scene(
        target=target,
        camera=camera_object,
        rig=rig,
        action=action,
        frame=frame,
        resolution=resolution,
        render_profile=render_profile,
    )
    if not _reserved_index_is_unique(scene, target):
        raise ConditioningFailed(["PASS_INVALID"])

    scene_inventory.sanitize_write_surfaces(str(staging_root))

    depth_range = (
        float(camera_object.data.clip_start),
        float(camera_object.data.clip_end),
    )
    containers = render_passes(scene, staging_root / "render")
    statistics = convert_passes(
        containers=containers,
        artifact_directory=staging_root / CONDITIONING_DIRECTORY,
        depth_range=depth_range,
        depth_scale=euclidean_depth_scale(
            scene, camera_object, scene.render.resolution_x, scene.render.resolution_y
        ),
    )

    camera_data, matrices = camera_record(scene, camera_object)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    world_to_clip = (
        camera_object.evaluated_get(depsgraph).calc_matrix_camera(
            depsgraph,
            x=scene.render.resolution_x,
            y=scene.render.resolution_y,
            scale_x=scene.render.pixel_aspect_x,
            scale_y=scene.render.pixel_aspect_y,
        )
        @ camera_object.evaluated_get(depsgraph).matrix_world.inverted()
    )
    projection_error = projection_error_pixels(
        scene=scene, camera=camera_object, target=target, world_to_clip=world_to_clip
    )
    if projection_error > PROJECTION_ERROR_LIMIT_PIXELS:
        raise ConditioningFailed(["CAMERA_CALIBRATION_MISSING"])

    scene_state = staging_root / LOCAL_DIRECTORY / SCENE_STATE_NAME
    write_scene_state(scene, scene_state)

    bundle = build_bundle(
        staging_root=staging_root,
        source_scene_sha256=str(request["source_scene_sha256"]),
        digests=evaluated_digests(target, rig),
        portable_selection=dict(request["portable_selection"]),
        frame=frame,
        statistics=statistics,
        camera=camera_data,
        matrices=matrices,
        render_profile=render_profile,
        blender=dict(request["blender"]),
        depth_range=depth_range,
    )

    bundle_path = staging_root / CONDITIONING_DIRECTORY / BUNDLE_NAME
    bundle_path.write_text(canonical_json(bundle), encoding="utf-8")

    outputs = [
        {
            "role": "conditioning_bundle",
            "path": f"{CONDITIONING_DIRECTORY}/{BUNDLE_NAME}",
            "sha256": _sha256_file(bundle_path),
            "byte_size": bundle_path.stat().st_size,
            "media_type": "application/json",
            "validation": {
                "profile": "conditioning-bundle-v1",
                "status": "valid",
                "diagnostics": [],
                "semantic_digest": bundle["bundle_sha256"],
            },
        },
        *(
            {
                "role": item["role"],
                "path": item["path"],
                "sha256": item["sha256"],
                "byte_size": item["byte_size"],
                "media_type": item["media_type"],
                "validation": {
                    "profile": "conditioning-pass-v1",
                    "status": "valid",
                    "diagnostics": [],
                    "semantic_digest": None,
                },
            }
            for item in bundle["passes"]
        ),
        {
            "role": "scene_state_blend",
            "path": f"{LOCAL_DIRECTORY}/{SCENE_STATE_NAME}",
            "sha256": _sha256_file(scene_state),
            "byte_size": scene_state.stat().st_size,
            "media_type": "application/x-blender",
            "validation": {
                "profile": "scene-state-v1",
                "status": "valid",
                "diagnostics": [],
                "semantic_digest": bundle["digests"]["evaluated_geometry"],
            },
        },
    ]

    metrics = {
        "kind": "condition",
        "width": bundle["resolution"][0],
        "height": bundle["resolution"][1],
        "foreground_pixel_count": statistics["foreground_pixel_count"],
        "finite_foreground_depth_count": statistics["finite_foreground_depth_count"],
        "interior_unit_normal_count": statistics["interior_unit_normal_count"],
        "projection_max_error_pixels": scene_inventory.quantize(projection_error),
        "geometry_digest": bundle["digests"]["evaluated_geometry"],
        "uv_digest": bundle["digests"]["uv"],
        "pose_digest": bundle["digests"]["pose"],
    }
    return {
        "bundle": bundle,
        "outputs": sorted(outputs, key=lambda item: item["path"]),
        "metrics": metrics,
    }
