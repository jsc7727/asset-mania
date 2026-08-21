# SPDX-License-Identifier: GPL-3.0-or-later
"""Write-surface sanitization, deterministic inventory, and deep scene preflight.

Order matters. The worker opens the file with the UI and script execution disabled,
immediately sanitizes every write surface, and only then inventories or evaluates
anything. A scene that would need trusted autoexec, an external dependency, a
topology-changing modifier, or a non-reproducible pose fails with a stable code instead of
being coerced into the closed profile.
"""

import hashlib
import math

import bpy

from . import labels as label_module
from .selection import canonical_json

QUANTIZE_PLACES = 9
ALLOWED_MODIFIER_TYPES = frozenset({"ARMATURE"})
_UV_EPSILON = 1e-6
_DETERMINANT_EPSILON = 1e-9

CODE_EXECUTION = "UNTRUSTED_AUTOEXEC_REQUIRED"
EXTERNAL_DEPENDENCY = "MISSING_LINKED_ASSET"
TOPOLOGY_CHANGED = "DEPSGRAPH_TOPOLOGY_CHANGED"
NONFINITE_POSE = "POSE_NONFINITE"
UV_INVALID = "UV_MISSING_OR_INVALID"
AMBIGUOUS = "SELECTION_AMBIGUOUS"
TARGET_MISSING = "TARGET_MESH_NOT_FOUND"
CAMERA_MISSING = "CAMERA_NOT_FOUND"
RIG_MISSING = "RIG_NOT_FOUND"
POSE_UNKNOWN = "SOURCE_POSE_UNKNOWN"
PLAN_TAMPERED = "PLAN_TAMPERED"


def quantize(value: float) -> float:
    """Round a finite value to the profile's semantic precision."""
    return round(float(value), QUANTIZE_PLACES)


def matrix_values(matrix) -> list[float]:
    """A row-major 4x4 matrix as sixteen quantized numbers."""
    return [quantize(component) for row in matrix for component in row]


def _finite(values) -> bool:
    return all(math.isfinite(float(value)) for value in values)


# --- Write-surface sanitization ------------------------------------------------


def sanitize_write_surfaces(staging_root: str) -> list[str]:
    """Neutralize every surface that could execute code or write outside staging.

    Returns the list of actions taken so a caller can record what it disabled. This runs
    before frame evaluation, dependency-graph evaluation, rendering, or baking.
    """
    actions: list[str] = []
    scene = bpy.context.scene

    # Blender 5 exposes the compositor as `compositing_node_group`, not `node_tree`, and
    # detaching that group is what actually disables it: `Scene.use_nodes` is writable
    # through RNA but assigning False to it has no effect on 5.2, so this code does not
    # pretend to flip it. A scene with no attached group runs no compositor.
    node_group = getattr(scene, "compositing_node_group", None)
    if node_group is not None:
        removed = 0
        for node in list(node_group.nodes):
            if node.type in ("OUTPUT_FILE", "R_LAYERS", "COMPOSITE"):
                node_group.nodes.remove(node)
                removed += 1
        if removed:
            actions.append(f"removed_output_nodes:{removed}")
        scene.compositing_node_group = None
        actions.append("detached_compositor_node_group")

    if getattr(scene, "sequence_editor", None) is not None:
        bpy.context.scene.sequence_editor_clear()
        actions.append("cleared_sequencer")

    render = scene.render
    render.use_border = False
    render.use_crop_to_border = False
    render.use_motion_blur = False
    render.filepath = f"{staging_root.rstrip('/')}/render/"
    actions.append("redirected_render_output")

    for attribute in ("use_freestyle",):
        if getattr(render, attribute, False):
            setattr(render, attribute, False)
            actions.append(f"disabled_{attribute}")

    cycles = getattr(scene, "cycles", None)
    if cycles is not None:
        # `shading_system` is OSL, which executes shader code. The texture-cache flags are
        # the ones the design names; 5.2 spells two of them `debug_*`, and any name the
        # pinned RNA does not expose is simply absent rather than assumed.
        for attribute in (
            "shading_system",
            "use_auto_tile",
            "use_texture_cache",
            "use_auto_generate_texture_cache",
            "debug_use_texture_cache_eviction",
            "debug_texture_cache_preserve_unused",
        ):
            if getattr(cycles, attribute, False):
                setattr(cycles, attribute, False)
                actions.append(f"disabled_cycles_{attribute}")

    preferences = bpy.context.preferences.filepaths
    preferences.temporary_directory = f"{staging_root.rstrip('/')}/blender-temp"
    actions.append("redirected_temporary_directory")

    return sorted(actions)


# --- Deterministic inventory ---------------------------------------------------


def _objects_of_type(object_type: str) -> list[bpy.types.Object]:
    return sorted(
        (item for item in bpy.data.objects if item.type == object_type),
        key=lambda item: item.name,
    )


def collect_label_maps() -> dict[str, dict[str, str]]:
    """Assign portable labels for every labelled datablock kind, in sorted name order."""
    meshes = _objects_of_type("MESH")
    return {
        "camera": label_module.assign_labels(
            "camera", (item.name for item in _objects_of_type("CAMERA"))
        ),
        "mesh": label_module.assign_labels("mesh", (item.name for item in meshes)),
        "armature": label_module.assign_labels(
            "armature", (item.name for item in _objects_of_type("ARMATURE"))
        ),
        "action": label_module.assign_labels("action", (item.name for item in bpy.data.actions)),
        "bone": label_module.assign_labels(
            "bone",
            (bone.name for armature in bpy.data.armatures for bone in armature.bones),
        ),
    }


def external_dependencies() -> list[str]:
    """Every reference this profile refuses to follow, as stable kind strings."""
    findings: set[str] = set()

    for image in bpy.data.images:
        if image.source in ("FILE", "SEQUENCE", "MOVIE", "TILED") and image.packed_file is None:
            findings.add("unpacked_image")
        if image.source == "MOVIE":
            findings.add("movie_clip")
    for library in bpy.data.libraries:
        findings.add("linked_library")
    for collection in (
        bpy.data.movieclips,
        bpy.data.fonts,
        bpy.data.volumes,
        bpy.data.sounds,
        bpy.data.cache_files,
    ):
        if len(collection):
            findings.add(f"external_{collection.rna_type.identifier.lower()}")

    return sorted(findings)


def code_execution_surfaces() -> list[str]:
    """Every surface that would need trusted autoexec to reproduce the requested result."""
    findings: set[str] = set()

    for block in bpy.data.objects:
        animation = block.animation_data
        if animation is not None and len(animation.drivers):
            findings.add("object_driver")
    for block in bpy.data.materials:
        animation = block.animation_data
        if animation is not None and len(animation.drivers):
            findings.add("material_driver")
    for scene in bpy.data.scenes:
        animation = scene.animation_data
        if animation is not None and len(animation.drivers):
            findings.add("scene_driver")
        if scene.render.use_freestyle:
            findings.add("freestyle")
        cycles = getattr(scene, "cycles", None)
        if cycles is not None and getattr(cycles, "shading_system", False):
            findings.add("open_shading_language")
    if len(bpy.data.texts):
        findings.add("embedded_text_block")

    return sorted(findings)


def _pose_composition(rig: bpy.types.Object | None) -> list[str]:
    """Pose sources this profile cannot reproduce deterministically."""
    if rig is None:
        return []

    findings: set[str] = set()
    animation = rig.animation_data
    if animation is not None and len(animation.nla_tracks):
        findings.add("nla_track")
    for bone in rig.pose.bones:
        if len(bone.constraints):
            findings.add("bone_constraint")
    if len(rig.constraints):
        findings.add("object_constraint")
    return sorted(findings)


def matrix_findings(values, determinant) -> list[str]:
    """Classify one transform from its sixteen components and its 3x3 determinant.

    Kept free of `bpy` so the in-Blender worker tests can exercise the non-finite branch
    directly. That branch is unreachable through the Python API -- Blender sanitizes a
    non-finite location or scale to zero at the RNA boundary -- but a hand-authored
    `.blend` is not bound by that, so the check stays.
    """
    findings: set[str] = set()
    if not _finite(values):
        findings.add("nonfinite_matrix")
        return sorted(findings)

    if not math.isfinite(determinant):
        findings.add("nonfinite_determinant")
    elif abs(determinant) < _DETERMINANT_EPSILON:
        findings.add("singular_transform")
    elif determinant < 0:
        findings.add("negative_determinant")
    return sorted(findings)


def _transform_findings(objects) -> list[str]:
    findings: set[str] = set()
    for item in objects:
        matrix = item.matrix_world
        findings.update(
            matrix_findings(
                [component for row in matrix for component in row],
                matrix.to_3x3().determinant(),
            )
        )
    return sorted(findings)


def _uv_findings(mesh: bpy.types.Mesh) -> list[str]:
    findings: set[str] = set()
    if len(mesh.uv_layers) == 0:
        return ["missing_uv_layer"]
    if len(mesh.uv_layers) > 1:
        findings.add("multiple_uv_layers")

    uv_layer = mesh.uv_layers[0]
    boxes: list[tuple[int, float, float, float, float]] = []
    for polygon in mesh.polygons:
        coordinates = [uv_layer.data[index].uv for index in polygon.loop_indices]
        if not _finite(component for uv in coordinates for component in uv):
            findings.add("nonfinite_uv")
            continue
        us = [uv[0] for uv in coordinates]
        vs = [uv[1] for uv in coordinates]
        if min(us) < -_UV_EPSILON or max(us) > 1 + _UV_EPSILON:
            findings.add("uv_outside_unit_range")
        if min(vs) < -_UV_EPSILON or max(vs) > 1 + _UV_EPSILON:
            findings.add("uv_outside_unit_range")
        boxes.append((polygon.index, min(us), max(us), min(vs), max(vs)))

    for index, (_, u_low, u_high, v_low, v_high) in enumerate(boxes):
        for _, other_u_low, other_u_high, other_v_low, other_v_high in boxes[index + 1 :]:
            overlaps_u = u_low < other_u_high - _UV_EPSILON and other_u_low < u_high - _UV_EPSILON
            overlaps_v = v_low < other_v_high - _UV_EPSILON and other_v_low < v_high - _UV_EPSILON
            if overlaps_u and overlaps_v:
                findings.add("overlapping_uv_islands")
                break

    return sorted(findings)


def _modifier_findings(target: bpy.types.Object) -> list[str]:
    return sorted(
        {
            f"unsupported_modifier:{modifier.type}"
            for modifier in target.modifiers
            if modifier.type not in ALLOWED_MODIFIER_TYPES
        }
    )


def _evaluated_topology(target: bpy.types.Object) -> tuple[int, int]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = target.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return len(mesh.vertices), len(mesh.polygons)
    finally:
        evaluated.to_mesh_clear()


def _rig_weight_findings(target: bpy.types.Object, rig: bpy.types.Object | None) -> list[str]:
    """Weight checks apply only when an armature is selected; a static prop is valid."""
    if rig is None:
        return []

    bone_names = {bone.name for bone in rig.data.bones}
    group_indices = {
        group.index: group.name for group in target.vertex_groups if group.name in bone_names
    }
    if not group_indices:
        return ["missing_rig_weights"]

    for vertex in target.data.vertices:
        total = sum(element.weight for element in vertex.groups if element.group in group_indices)
        if not math.isfinite(total) or total <= 0.0:
            return ["zero_rig_weights"]
    return []


# --- Selection resolution ------------------------------------------------------


def _unique_object(name: str, object_type: str) -> tuple[bpy.types.Object | None, str | None]:
    """Resolve one object by exact name, refusing an ambiguous or absent match."""
    exact = [item for item in bpy.data.objects if item.name == name and item.type == object_type]
    if len(exact) == 1:
        return exact[0], None
    if len(exact) > 1:
        return None, AMBIGUOUS

    # A name that differs only by case or surrounding whitespace is treated as ambiguous
    # rather than silently resolved, because the plan named an exact datablock.
    folded = name.strip().casefold()
    near = [
        item
        for item in bpy.data.objects
        if item.type == object_type and item.name.strip().casefold() == folded
    ]
    if near:
        return None, AMBIGUOUS
    return None, None


def resolve_selection(request: dict) -> tuple[dict, list[str]]:
    """Resolve the private selection the envelope names, or report why it failed."""
    diagnostics: list[str] = []
    resolved: dict[str, bpy.types.Object | None] = {
        "target": None,
        "camera": None,
        "armature": None,
    }

    target_name = request.get("target_name")
    if target_name:
        target, problem = _unique_object(str(target_name), "MESH")
        resolved["target"] = target
        if target is None:
            diagnostics.append(problem or TARGET_MISSING)
    else:
        meshes = _objects_of_type("MESH")
        if len(meshes) == 1:
            resolved["target"] = meshes[0]
        elif not meshes:
            diagnostics.append(TARGET_MISSING)
        else:
            diagnostics.append(AMBIGUOUS)

    camera_name = request.get("camera_name")
    if camera_name:
        camera, problem = _unique_object(str(camera_name), "CAMERA")
        resolved["camera"] = camera
        if camera is None:
            diagnostics.append(problem or CAMERA_MISSING)
    else:
        cameras = _objects_of_type("CAMERA")
        if len(cameras) == 1:
            resolved["camera"] = cameras[0]
        elif not cameras:
            diagnostics.append(CAMERA_MISSING)
        else:
            diagnostics.append(AMBIGUOUS)

    armature_name = request.get("armature_name")
    if armature_name:
        rig, problem = _unique_object(str(armature_name), "ARMATURE")
        resolved["armature"] = rig
        if rig is None:
            diagnostics.append(problem or RIG_MISSING)

    action_name = request.get("action_name")
    if action_name and action_name not in bpy.data.actions:
        diagnostics.append(POSE_UNKNOWN)

    if len(_objects_of_type("ARMATURE")) > 1 and not armature_name:
        diagnostics.append(AMBIGUOUS)
    if len(bpy.data.actions) > 1 and not action_name:
        diagnostics.append(POSE_UNKNOWN)

    return resolved, sorted(set(diagnostics))


# --- Semantic fingerprint ------------------------------------------------------


def scene_semantic_description(
    *,
    label_maps: dict[str, dict[str, str]],
    target: bpy.types.Object,
    camera: bpy.types.Object,
    rig: bpy.types.Object | None,
) -> dict:
    """A sorted, label-only, quantized description of the scene.

    It carries no datablock name, no path, and no basename, so its digest is a portable
    semantic identity rather than a fingerprint of the user's naming.
    """
    mesh = target.data
    uv_layer = mesh.uv_layers[0] if len(mesh.uv_layers) else None
    return {
        "labels": label_module.sorted_labels(*label_maps.values()),
        "target": {
            "label": label_maps["mesh"][target.name],
            "vertex_count": len(mesh.vertices),
            "polygon_count": len(mesh.polygons),
            "triangle_count": sum(len(polygon.vertices) - 2 for polygon in mesh.polygons),
            "uv_layer_count": len(mesh.uv_layers),
            "matrix_world": matrix_values(target.matrix_world),
            "vertices": sorted(
                [quantize(component) for component in vertex.co] for vertex in mesh.vertices
            ),
            "polygons": sorted(
                sorted(int(index) for index in polygon.vertices) for polygon in mesh.polygons
            ),
            "uv": sorted(
                [quantize(uv_layer.data[index].uv[0]), quantize(uv_layer.data[index].uv[1])]
                for polygon in mesh.polygons
                for index in polygon.loop_indices
            )
            if uv_layer is not None
            else [],
        },
        "camera": {
            "label": label_maps["camera"][camera.name],
            "projection_type": "perspective" if camera.data.type == "PERSP" else "orthographic",
            "lens_mm": quantize(camera.data.lens) if camera.data.type == "PERSP" else None,
            "sensor_fit": camera.data.sensor_fit,
            "sensor_width_mm": quantize(camera.data.sensor_width),
            "sensor_height_mm": quantize(camera.data.sensor_height),
            "shift_x": quantize(camera.data.shift_x),
            "shift_y": quantize(camera.data.shift_y),
            "clip_start_meters": quantize(camera.data.clip_start),
            "clip_end_meters": quantize(camera.data.clip_end),
            "matrix_world": matrix_values(camera.matrix_world),
        },
        "rig": None
        if rig is None
        else {
            "label": label_maps["armature"][rig.name],
            "bone_labels": sorted(label_maps["bone"][bone.name] for bone in rig.data.bones),
            "bone_rest_matrices": sorted(
                matrix_values(bone.matrix_local) for bone in rig.data.bones
            ),
            "matrix_world": matrix_values(rig.matrix_world),
        },
        "materials": sorted(
            {
                "slot_count": len(target.data.materials),
                "packed_image_count": sum(
                    1 for image in bpy.data.images if image.packed_file is not None
                ),
            }.items()
        ),
    }


def semantic_digest(description: dict) -> str:
    return hashlib.sha256(canonical_json(description).encode("utf-8")).hexdigest()


# --- Deep preflight ------------------------------------------------------------


def selection_matches_request(
    request: dict,
    *,
    target: bpy.types.Object,
    camera: bpy.types.Object,
    rig: bpy.types.Object | None,
) -> bool:
    """Recompute the salted selection digest against the file that was actually opened.

    Every stage that opens a source must call this, not just preflight: a plan whose labels
    or private identity were edited after approval must fail with `PLAN_TAMPERED` before any
    artifact is produced. A request that carries no salt or portable selection is not
    plan-bound and is accepted as-is.
    """
    salt_hex = request.get("selection_salt")
    portable_selection = request.get("portable_selection")
    if not salt_hex or not portable_selection:
        return True

    from .selection import verify_selection

    identity = {
        "source_scene_sha256": request.get("source_scene_sha256", ""),
        "camera": camera.name,
        "target": target.name,
        "target_type": target.type,
        "armature": rig.name if rig is not None else None,
        "action": request.get("action_name"),
    }
    return verify_selection(
        salt_hex=str(salt_hex), identity=identity, portable_selection=portable_selection
    )


def preflight(request: dict) -> tuple[dict | None, list[str], dict]:
    """Run every closed-profile check and return metrics, diagnostics, and labels.

    Metrics are `None` whenever a check failed, so a caller can never mistake a partial
    inventory for a usable one.
    """
    diagnostics: set[str] = set()

    dependencies = external_dependencies()
    if dependencies:
        diagnostics.add(EXTERNAL_DEPENDENCY)
    if code_execution_surfaces():
        diagnostics.add(CODE_EXECUTION)

    label_maps = collect_label_maps()
    resolved, selection_diagnostics = resolve_selection(request)
    diagnostics.update(selection_diagnostics)

    target = resolved["target"]
    camera = resolved["camera"]
    rig = resolved["armature"]

    transform_subjects = [item for item in (target, camera, rig) if item is not None]
    if _transform_findings(transform_subjects):
        diagnostics.add(NONFINITE_POSE)
    if _pose_composition(rig):
        diagnostics.add(POSE_UNKNOWN)

    if target is not None:
        if _uv_findings(target.data):
            diagnostics.add(UV_INVALID)
        if _modifier_findings(target):
            diagnostics.add(TOPOLOGY_CHANGED)
        else:
            vertex_count, polygon_count = _evaluated_topology(target)
            if (vertex_count, polygon_count) != (
                len(target.data.vertices),
                len(target.data.polygons),
            ):
                diagnostics.add(TOPOLOGY_CHANGED)
        if _rig_weight_findings(target, rig):
            diagnostics.add(RIG_MISSING)

    portable_labels = label_module.sorted_labels(*label_maps.values())
    if diagnostics or target is None or camera is None:
        return None, sorted(diagnostics), {"portable_labels": portable_labels}

    if not selection_matches_request(request, target=target, camera=camera, rig=rig):
        return None, [PLAN_TAMPERED], {"portable_labels": portable_labels}

    description = scene_semantic_description(
        label_maps=label_maps, target=target, camera=camera, rig=rig
    )
    metrics = {
        "kind": "preflight",
        "object_count": len(bpy.data.objects),
        "mesh_count": len(_objects_of_type("MESH")),
        "camera_count": len(_objects_of_type("CAMERA")),
        "armature_count": len(_objects_of_type("ARMATURE")),
        "action_count": len(bpy.data.actions),
        "target_vertex_count": len(target.data.vertices),
        "target_triangle_count": sum(len(polygon.vertices) - 2 for polygon in target.data.polygons),
        "target_uv_layer_count": len(target.data.uv_layers),
        "target_bone_count": 0 if rig is None else len(rig.data.bones),
        "external_dependency_count": len(dependencies),
        "scene_semantic_digest": semantic_digest(description),
    }
    return metrics, [], {"portable_labels": portable_labels}
