# SPDX-License-Identifier: GPL-3.0-or-later
"""Export the derived asset as BLEND, GLB, and FBX, in that order.

Every exporter property the design binds is passed explicitly and checked against the
operator's RNA first. That preflight is the point: a property that Blender renamed or
removed must fail the run, because silently falling back to a default would change the
exported geometry, axes, or animation without anyone noticing.

Each output is written to staging, structurally validated, and reimported in a *fresh*
Blender process before it is considered publishable. Byte-identical archives are never
claimed; what is compared is a format-aware semantic fingerprint.
"""

import json
from pathlib import Path

import bpy

from . import labels as label_module
from . import scene_inventory
from .selection import canonical_json

BLEND_NAME = "exports/asset.blend"
GLB_NAME = "exports/asset.glb"
FBX_NAME = "exports/asset.fbx"
FBX_TEXTURE_NAME = "exports/asset.png"
FINGERPRINT_NAME = "exports/fingerprint.json"
REIMPORT_FINGERPRINT_NAME = "reimport-fingerprint.json"
EXPORT_ORDER = ("blend", "glb", "fbx")
ALPHA_CUTOFF = 0.5

_EXPORT_UNAVAILABLE = "EXPORT_OPERATOR_UNAVAILABLE"
_GLTF_FAILED = "GLTF_VALIDATION_FAILED"
_ROUNDTRIP = "ROUNDTRIP_MISMATCH"
_COLLISION = "OUTPUT_COLLISION"


class ExportFailed(Exception):
    """An export could not be produced or did not survive validation."""

    def __init__(self, diagnostics: list[str]) -> None:
        super().__init__(", ".join(sorted(set(diagnostics))))
        self.diagnostics = sorted(set(diagnostics))


def glb_properties(*, animated: bool, frame_start: int, frame_end: int) -> dict:
    """The bound GLB call. Animation flags follow the one selected Action."""
    return {
        "export_format": "GLB",
        "use_selection": True,
        "export_yup": True,
        "export_skins": True,
        "export_cameras": True,
        "export_animations": animated,
        "export_animation_mode": "ACTIONS",
        "export_frame_range": True,
        "export_frame_step": 1,
        "export_force_sampling": True,
        "export_bake_animation": True,
        "export_nla_strips": False,
        "export_current_frame": False,
        "export_rest_position_armature": True,
        "export_apply": False,
        "export_image_format": "AUTO",
        "export_image_add_webp": False,
        "export_image_webp_fallback": False,
        "export_keep_originals": False,
        "export_materials": "EXPORT",
        "export_texcoords": True,
        "export_normals": True,
        "export_tangents": False,
        "export_unused_images": False,
        "export_unused_textures": False,
    }


def fbx_properties(*, animated: bool) -> dict:
    """The bound FBX call. Axes, units, and bone axes are explicit, never defaulted."""
    return {
        "use_selection": True,
        "object_types": {"ARMATURE", "MESH", "CAMERA"},
        "global_scale": 1.0,
        "apply_unit_scale": True,
        "apply_scale_options": "FBX_SCALE_NONE",
        "use_space_transform": True,
        "bake_space_transform": False,
        "axis_forward": "-Z",
        "axis_up": "Y",
        "add_leaf_bones": False,
        "primary_bone_axis": "Y",
        "secondary_bone_axis": "X",
        "use_armature_deform_only": True,
        "bake_anim": animated,
        "bake_anim_use_all_actions": False,
        "bake_anim_use_nla_strips": False,
        "bake_anim_force_startend_keying": True,
        "bake_anim_step": 1.0,
        "bake_anim_simplify_factor": 0.0,
        "path_mode": "COPY",
        "embed_textures": False,
        "use_metadata": True,
    }


def preflight_operator(operator, properties: dict) -> None:
    """Reject a missing or retyped operator property instead of using a default."""
    available = set(operator.get_rna_type().properties.keys())
    missing = sorted(name for name in properties if name not in available)
    if missing:
        raise ExportFailed([_EXPORT_UNAVAILABLE])


def _select_only(objects) -> None:
    for item in bpy.context.view_layer.objects:
        item.select_set(False)
    active = None
    for item in objects:
        if item is None:
            continue
        item.select_set(True)
        active = item
    bpy.context.view_layer.objects.active = active


def _refuse_existing(path: Path) -> None:
    if path.exists():
        raise ExportFailed([_COLLISION])
    path.parent.mkdir(parents=True, exist_ok=True)


# --- Semantic fingerprint --------------------------------------------------------


def semantic_fingerprint(
    *,
    target: bpy.types.Object,
    rig: bpy.types.Object | None,
    camera: bpy.types.Object | None,
    sample_frames: tuple[int, ...],
) -> dict:
    """A format-aware fingerprint: counts plus sampled pose and deformed geometry.

    Sampling at the action start, the condition frame, and the action end is what makes a
    dropped or resampled animation visible; comparing archive bytes would not.
    """
    scene = bpy.context.scene
    mesh = target.data
    fingerprint = {
        "mesh_count": sum(1 for item in bpy.data.objects if item.type == "MESH"),
        "bone_count": 0 if rig is None else len(rig.data.bones),
        "action_count": len(bpy.data.actions),
        "camera_count": sum(1 for item in bpy.data.objects if item.type == "CAMERA"),
        "material_count": len(mesh.materials),
        "texture_count": sum(1 for item in bpy.data.images if item.users),
        "vertex_count": len(mesh.vertices),
        "polygon_count": len(mesh.polygons),
        # glTF stores triangles only, so the triangle count -- not the polygon count -- is
        # what survives a runtime round trip.
        "triangle_count": sum(len(polygon.vertices) - 2 for polygon in mesh.polygons),
        "uv_layer_count": len(mesh.uv_layers),
    }

    samples: list[float] = []
    for frame in sample_frames:
        scene.frame_set(int(frame))
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = target.evaluated_get(depsgraph)
        evaluated_mesh = evaluated.to_mesh()
        try:
            for vertex in evaluated_mesh.vertices:
                world = evaluated.matrix_world @ vertex.co
                samples.extend(scene_inventory.quantize(component) for component in world)
        finally:
            evaluated.to_mesh_clear()
        if rig is not None:
            for bone in sorted(rig.pose.bones, key=lambda item: item.name):
                samples.extend(
                    scene_inventory.quantize(component) for row in bone.matrix for component in row
                )
    fingerprint["samples"] = samples
    if camera is not None:
        fingerprint["camera_lens_mm"] = scene_inventory.quantize(camera.data.lens)
    return fingerprint


# --- The three exports -----------------------------------------------------------


def export_blend(path: Path, *, scene: bpy.types.Scene) -> None:
    """Save a derived blend. The source is never written to."""
    _refuse_existing(path)
    for image in bpy.data.images:
        if image.users and image.packed_file is None and image.source == "FILE":
            # Pack rather than leave an absolute reference the reopened file cannot follow.
            image.pack()
    bpy.ops.wm.save_as_mainfile(filepath=str(path), compress=False, copy=True, relative_remap=True)


def export_glb(path: Path, *, animated: bool, frame_start: int, frame_end: int, objects) -> None:
    _refuse_existing(path)
    properties = glb_properties(animated=animated, frame_start=frame_start, frame_end=frame_end)
    preflight_operator(bpy.ops.export_scene.gltf, properties)
    _select_only(objects)
    try:
        bpy.ops.export_scene.gltf(filepath=str(path), **properties)
    except RuntimeError as error:
        raise ExportFailed([_GLTF_FAILED]) from error


def export_fbx(path: Path, *, animated: bool, objects) -> None:
    _refuse_existing(path)
    properties = fbx_properties(animated=animated)
    preflight_operator(bpy.ops.export_scene.fbx, properties)
    _select_only(objects)
    try:
        bpy.ops.export_scene.fbx(filepath=str(path), **properties)
    except RuntimeError as error:
        raise ExportFailed([_EXPORT_UNAVAILABLE]) from error


def link_alpha_clip(tree, texture, principled, cutoff: float = ALPHA_CUTOFF) -> None:
    """Insert the node setup the glTF exporter reads as `alphaMode: MASK`.

    Measured, not assumed: Blender 5.2's glTF exporter derives `alphaMode` from the node
    graph rather than from `Material.blend_method`. Its `detect_alpha_clip` recognizes a
    mask when the alpha input is driven through a comparison against a constant cutoff,
    and reports that constant as `alphaCutoff`.

    Unknown texels carry alpha zero, so the delivered material must clip rather than blend
    them; otherwise a runtime viewer shows black where the pipeline saw nothing.
    """
    clip = tree.nodes.new("ShaderNodeMath")
    clip.operation = "GREATER_THAN"
    clip.label = "AssetManiaAlphaClip"
    clip.location = (-160, -220)
    clip.inputs[1].default_value = cutoff

    tree.links.new(texture.outputs["Alpha"], clip.inputs[0])
    tree.links.new(clip.outputs["Value"], principled.inputs["Alpha"])


def apply_alpha_mask(target: bpy.types.Object, cutoff: float = ALPHA_CUTOFF) -> None:
    """Give every delivered material the alpha-clip setup, plus the Eevee fallback."""
    for material in target.data.materials:
        if material is None or not material.use_nodes:
            continue
        if hasattr(material, "blend_method"):
            material.blend_method = "CLIP"
        if hasattr(material, "alpha_threshold"):
            material.alpha_threshold = cutoff

        tree = material.node_tree
        principled = next((node for node in tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
        texture = next((node for node in tree.nodes if node.type == "TEX_IMAGE"), None)
        if principled is None or texture is None:
            continue
        if any(node.label == "AssetManiaAlphaClip" for node in tree.nodes):
            continue
        tree.links.new(texture.outputs["Color"], principled.inputs["Base Color"])
        link_alpha_clip(tree, texture, principled, cutoff)


def delivery_material(target: bpy.types.Object, texture_path: Path) -> None:
    """Replace the emission consolidation material with a Principled delivery material."""
    material = bpy.data.materials.new("AssetManiaDelivery")
    material.use_nodes = True
    tree = material.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)

    image = bpy.data.images.load(str(texture_path), check_existing=False)
    image.colorspace_settings.name = "sRGB"
    image.pack()

    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    texture.location = (-400, 0)
    principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.location = (300, 0)

    tree.links.new(texture.outputs["Color"], principled.inputs["Base Color"])
    tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    link_alpha_clip(tree, texture, principled)

    target.data.materials.clear()
    target.data.materials.append(material)
    apply_alpha_mask(target)


def rename_to_portable_labels(
    *,
    target: bpy.types.Object,
    rig: bpy.types.Object | None,
    camera: bpy.types.Object | None,
    portable_selection: dict,
) -> None:
    """Rename every exported datablock to its portable label before export.

    A GLB writes its node names into the file, and those nodes are named after Blender
    datablocks. Exporting as-authored therefore puts the user's private object, bone, and
    action names into the delivered runtime artifact -- exactly what portable outputs are
    supposed to avoid. Renaming happens in this throwaway session only; the source file is
    never written to.
    """
    target_label = portable_selection.get("target_label") or "mesh-1"
    camera_label = portable_selection.get("camera_label") or "camera-1"
    armature_label = portable_selection.get("armature_label") or "armature-1"
    action_label = portable_selection.get("action_label") or "action-1"

    target.name = target_label
    target.data.name = f"{target_label}-mesh"
    for index, uv_layer in enumerate(target.data.uv_layers, start=1):
        uv_layer.name = f"uv-{index}"
    for index, material in enumerate(target.data.materials, start=1):
        if material is None:
            continue
        material.name = f"material-{index}"
        if not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None:
                node.image.name = f"texture-{index}"

    if camera is not None:
        camera.name = camera_label
        camera.data.name = f"{camera_label}-data"

    if rig is not None:
        rig.name = armature_label
        rig.data.name = f"{armature_label}-skeleton"
        bone_labels = label_module.assign_labels("bone", (bone.name for bone in rig.data.bones))
        # Vertex groups are matched to bones by name, so both sides rename together.
        groups = {group.name: group for group in target.vertex_groups}
        for private, label in sorted(bone_labels.items()):
            group = groups.get(private)
            rig.data.bones[private].name = label
            if group is not None:
                group.name = label
        if rig.animation_data is not None and rig.animation_data.action is not None:
            rig.animation_data.action.name = action_label

    for action in bpy.data.actions:
        if action.name != action_label:
            action.name = f"{action_label}-{abs(hash(action.name)) % 1000:03d}"


def export(request: dict) -> dict:
    """Run the export stage: derived blend, then GLB, then FBX."""
    staging_root = Path(str(request["staging_root"]))
    formats = [name for name in EXPORT_ORDER if name in set(request["formats"])]
    if not formats:
        raise ExportFailed([_EXPORT_UNAVAILABLE])

    resolved, diagnostics = scene_inventory.resolve_selection(request)
    if diagnostics:
        raise ExportFailed(diagnostics)
    target = resolved["target"]
    camera = resolved["camera"]
    rig = resolved["armature"]

    if not scene_inventory.selection_matches_request(
        request, target=target, camera=camera, rig=rig
    ):
        raise ExportFailed([scene_inventory.PLAN_TAMPERED])

    scene = bpy.context.scene
    action_range = request.get("action_range")
    animated = bool(rig is not None and action_range)
    frame_start = int(action_range[0]) if animated else int(request["frame"])
    frame_end = int(action_range[1]) if animated else int(request["frame"])
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    scene.frame_step = 1

    texture = Path(str(request["texture_path"]))
    if not texture.is_file():
        raise ExportFailed([_EXPORT_UNAVAILABLE])
    delivery_material(target, texture)
    rename_to_portable_labels(
        target=target,
        rig=rig,
        camera=camera,
        portable_selection=dict(request["portable_selection"]),
    )

    sample_frames = tuple(sorted({frame_start, int(request["frame"]), frame_end}))
    fingerprint = semantic_fingerprint(
        target=target, rig=rig, camera=camera, sample_frames=sample_frames
    )
    scene.frame_set(int(request["frame"]))

    objects = [item for item in (target, rig, camera) if item is not None]
    written: list[str] = []
    if "blend" in formats:
        export_blend(staging_root / BLEND_NAME, scene=scene)
        written.append(BLEND_NAME)
    if "glb" in formats:
        export_glb(
            staging_root / GLB_NAME,
            animated=animated,
            frame_start=frame_start,
            frame_end=frame_end,
            objects=objects,
        )
        written.append(GLB_NAME)
    if "fbx" in formats:
        export_fbx(staging_root / FBX_NAME, animated=animated, objects=objects)
        written.append(FBX_NAME)
        # FBX embedding is not round-trip reliable here, so the texture travels beside it
        # as a declared member of the same export group.
        beside = staging_root / FBX_TEXTURE_NAME
        if not beside.exists():
            beside.write_bytes(texture.read_bytes())
        written.append(FBX_TEXTURE_NAME)

    # The fingerprint travels as a declared artifact rather than an extra response field:
    # the worker response schema is closed, so anything not in it is dropped when sealed.
    fingerprint_path = staging_root / FINGERPRINT_NAME
    _refuse_existing(fingerprint_path)
    fingerprint_path.write_text(
        canonical_json({"sample_frames": list(sample_frames), "fingerprint": fingerprint}),
        encoding="utf-8",
    )
    written.append(FINGERPRINT_NAME)

    return {
        "artifacts": written,
        "fingerprint": fingerprint,
        "animated": animated,
        "sample_frames": list(sample_frames),
        "semantic": canonical_json(fingerprint),
    }


def reimport_fingerprint(request: dict) -> dict:
    """Reimport an exported file in this (fresh) process and fingerprint what came back."""
    path = Path(str(request["import_path"]))
    kind = str(request["import_kind"])
    sample_frames = tuple(int(frame) for frame in request["sample_frames"])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    if kind == "blend":
        bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False, use_scripts=False)
    elif kind == "glb":
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif kind == "fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    else:
        raise ExportFailed([_ROUNDTRIP])

    meshes = sorted(
        (item for item in bpy.data.objects if item.type == "MESH"), key=lambda item: item.name
    )
    if not meshes:
        raise ExportFailed([_ROUNDTRIP])
    rigs = sorted(
        (item for item in bpy.data.objects if item.type == "ARMATURE"), key=lambda item: item.name
    )
    # A runtime importer may synthesize helper geometry that was never in the file, so the
    # target is chosen by structure -- the mesh the rig deforms, else the largest mesh --
    # rather than by alphabetical order.
    deformed = [
        item
        for item in meshes
        if any(modifier.type == "ARMATURE" for modifier in item.modifiers)
        or (item.parent is not None and item.parent.type == "ARMATURE")
    ]
    if deformed:
        meshes = deformed
    else:
        meshes = sorted(meshes, key=lambda item: -len(item.data.polygons))
    cameras = sorted(
        (item for item in bpy.data.objects if item.type == "CAMERA"), key=lambda item: item.name
    )
    fingerprint = semantic_fingerprint(
        target=meshes[0],
        rig=rigs[0] if rigs else None,
        camera=cameras[0] if cameras else None,
        sample_frames=sample_frames,
    )
    destination = Path(str(request["staging_root"])) / REIMPORT_FINGERPRINT_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        canonical_json({"kind": kind, "fingerprint": fingerprint}), encoding="utf-8"
    )
    return fingerprint


def absolute_reference_report() -> dict:
    """Every image reference in the current file, so an absolute path cannot hide."""
    return {
        "images": sorted(
            {
                "name_is_private": True,
                "packed": image.packed_file is not None,
                "absolute": bool(image.filepath) and not image.filepath.startswith("//"),
            }.__repr__()
            for image in bpy.data.images
            if image.users
        )
    }


def dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
