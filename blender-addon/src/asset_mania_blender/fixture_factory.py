# SPDX-License-Identifier: GPL-3.0-or-later
"""Procedural generation of the one composite runtime fixture.

The fixture is an asymmetric, rigged, non-human strip: a tapered bar with a known
non-overlapping UV layout, two bones, a rest frame plus a deformed frame, a packed
asymmetric quadrant texture, one camera, and one fixed area light. It contains no external
file, script, driver, or identity content, and every byte is generated at runtime into a
staging directory rather than tracked in the repository.

Datablock names here are deliberately private-looking. The portable label mapping in
`labels` is what a manifest records; nothing in this module is a portable name.
"""

import math

import bpy
from mathutils import Euler, Quaternion, Vector

FIXTURE_PROFILE = "blender-5.2.0-cpu-v1-fixture"
SEED = 0

TARGET_NAME = "Robot_Strip_Body"
ARMATURE_OBJECT_NAME = "Robot_Rig"
ARMATURE_DATA_NAME = "Robot_Rig_Skeleton"
BASE_BONE_NAME = "Base_Joint"
TIP_BONE_NAME = "Tip_Joint"
ACTION_NAME = "Robot_Flex"
CAMERA_NAME = "Shot_Camera"
LIGHT_NAME = "Key_Light"
MATERIAL_NAME = "Robot_Strip_Surface"
IMAGE_NAME = "Robot_Strip_Quadrants"

TARGET_PASS_INDEX = 1
REST_FRAME = 1
DEFORMED_FRAME = 2
DEFORMATION_DEGREES = 30.0

TEXTURE_SIZE = 64
_BONE_SPLIT_X = 1.5

# A tapered, slightly bent strip: asymmetric in Y and Z so the depth and normal passes
# carry real variation and a mirrored scene cannot be mistaken for the original.
_PROFILE = (
    (0.0, 0.30, 0.00),
    (1.0, 0.25, 0.05),
    (2.0, 0.18, 0.15),
    (3.0, 0.10, 0.30),
)
# Three quads, each in its own non-overlapping island well inside 0..1.
_UV_ISLANDS = (
    (0.02, 0.32, 0.02, 0.48),
    (0.35, 0.65, 0.02, 0.48),
    (0.68, 0.98, 0.02, 0.48),
)


def _reset_scene() -> None:
    """Start from an empty file so nothing from a previous run leaks into the fixture."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _build_mesh() -> bpy.types.Object:
    vertices: list[tuple[float, float, float]] = []
    for x, half_width, z in _PROFILE:
        vertices.append((x, -half_width, z))
        vertices.append((x, half_width, z))

    faces = [(index, index + 1, index + 3, index + 2) for index in range(0, 6, 2)]

    mesh = bpy.data.meshes.new(f"{TARGET_NAME}_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(verbose=False)
    mesh.update()

    uv_layer = mesh.uv_layers.new(name="Robot_Strip_UV")
    for face_index, polygon in enumerate(mesh.polygons):
        u_low, u_high, v_low, v_high = _UV_ISLANDS[face_index]
        corners = ((u_low, v_low), (u_high, v_low), (u_high, v_high), (u_low, v_high))
        for corner_index, loop_index in enumerate(polygon.loop_indices):
            uv_layer.data[loop_index].uv = corners[corner_index]

    target = bpy.data.objects.new(TARGET_NAME, mesh)
    target.pass_index = TARGET_PASS_INDEX
    bpy.context.collection.objects.link(target)
    return target


def _build_texture() -> bpy.types.Image:
    """An asymmetric quadrant checker, packed so the fixture needs no external file."""
    image = bpy.data.images.new(IMAGE_NAME, width=TEXTURE_SIZE, height=TEXTURE_SIZE, alpha=True)
    quadrant_colors = (
        (0.90, 0.15, 0.10),
        (0.10, 0.65, 0.90),
        (0.95, 0.80, 0.10),
        (0.20, 0.85, 0.35),
    )
    pixels: list[float] = []
    half = TEXTURE_SIZE // 2
    for row in range(TEXTURE_SIZE):
        for column in range(TEXTURE_SIZE):
            quadrant = (0 if row < half else 2) + (0 if column < half else 1)
            red, green, blue = quadrant_colors[quadrant]
            # An 8-pixel checker over the quadrant colour keeps the pattern locally
            # distinguishable, which is what the reprojection oracle needs.
            if ((row // 8) + (column // 8)) % 2:
                red, green, blue = red * 0.45, green * 0.45, blue * 0.45
            pixels.extend((red, green, blue, 1.0))

    image.pixels[:] = pixels
    image.pack()
    return image


def _build_material(target: bpy.types.Object, image: bpy.types.Image) -> None:
    material = bpy.data.materials.new(MATERIAL_NAME)
    material.use_nodes = True
    tree = material.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)

    texture_node = tree.nodes.new("ShaderNodeTexImage")
    texture_node.image = image
    texture_node.interpolation = "Closest"
    texture_node.extension = "CLIP"
    texture_node.location = (-400, 0)

    shader = tree.nodes.new("ShaderNodeBsdfDiffuse")
    shader.location = (-120, 0)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.location = (120, 0)

    tree.links.new(texture_node.outputs["Color"], shader.inputs["Color"])
    tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    target.data.materials.append(material)


def _build_armature(target: bpy.types.Object) -> bpy.types.Object:
    armature = bpy.data.armatures.new(ARMATURE_DATA_NAME)
    rig = bpy.data.objects.new(ARMATURE_OBJECT_NAME, armature)
    bpy.context.collection.objects.link(rig)

    previous = bpy.context.view_layer.objects.active
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    base = armature.edit_bones.new(BASE_BONE_NAME)
    base.head = Vector((0.0, 0.0, 0.0))
    base.tail = Vector((_BONE_SPLIT_X, 0.0, 0.0))
    tip = armature.edit_bones.new(TIP_BONE_NAME)
    tip.head = Vector((_BONE_SPLIT_X, 0.0, 0.0))
    tip.tail = Vector((3.0, 0.0, 0.0))
    tip.parent = base
    tip.use_connect = True
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.objects.active = previous

    # Every vertex is weighted, so a selected armature never sees a zero-weight target.
    base_group = target.vertex_groups.new(name=BASE_BONE_NAME)
    tip_group = target.vertex_groups.new(name=TIP_BONE_NAME)
    for vertex in target.data.vertices:
        group = base_group if vertex.co.x < _BONE_SPLIT_X else tip_group
        group.add([vertex.index], 1.0, "REPLACE")

    target.parent = rig
    modifier = target.modifiers.new(name="Robot_Strip_Skin", type="ARMATURE")
    modifier.object = rig
    return rig


def _build_action(rig: bpy.types.Object) -> bpy.types.Action:
    """Rest at frame 1, a 30-degree tip rotation at frame 2. No driver, no NLA."""
    action = bpy.data.actions.new(ACTION_NAME)
    rig.animation_data_create()
    rig.animation_data.action = action

    for bone in rig.pose.bones:
        bone.rotation_mode = "QUATERNION"

    tip = rig.pose.bones[TIP_BONE_NAME]
    tip.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
    tip.keyframe_insert(data_path="rotation_quaternion", frame=REST_FRAME)
    tip.rotation_quaternion = Quaternion(Vector((0.0, 0.0, 1.0)), math.radians(DEFORMATION_DEGREES))
    tip.keyframe_insert(data_path="rotation_quaternion", frame=DEFORMED_FRAME)
    tip.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))

    for curve in action_fcurves(action):
        for keyframe in curve.keyframe_points:
            keyframe.interpolation = "LINEAR"
    return action


def action_fcurves(action: bpy.types.Action) -> list[bpy.types.FCurve]:
    """Every F-curve of a Blender 5 slotted action.

    `Action.fcurves` was removed when actions became layered: curves now live in a
    channelbag per (strip, slot). Reading them through layers and strips keeps this
    module working on the pinned 5.2.0 API instead of a pre-slot one.
    """
    curves: list[bpy.types.FCurve] = []
    for layer in action.layers:
        for strip in layer.strips:
            if strip.type != "KEYFRAME":
                continue
            for slot in action.slots:
                channelbag = strip.channelbag(slot)
                if channelbag is not None:
                    curves.extend(channelbag.fcurves)
    return curves


def _build_camera() -> bpy.types.Object:
    camera_data = bpy.data.cameras.new(CAMERA_NAME)
    camera_data.type = "PERSP"
    camera_data.lens = 50.0
    camera_data.sensor_fit = "AUTO"
    camera_data.sensor_width = 36.0
    camera_data.sensor_height = 24.0
    camera_data.shift_x = 0.0
    camera_data.shift_y = 0.0
    camera_data.clip_start = 0.1
    camera_data.clip_end = 100.0
    camera_data.dof.use_dof = False

    camera = bpy.data.objects.new(CAMERA_NAME, camera_data)
    camera.location = Vector((1.5, -7.0, 1.4))
    camera.rotation_mode = "XYZ"
    camera.rotation_euler = Euler((math.radians(84.0), 0.0, 0.0), "XYZ")
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def _build_light() -> bpy.types.Object:
    light_data = bpy.data.lights.new(LIGHT_NAME, type="AREA")
    light_data.energy = 500.0
    light_data.size = 4.0
    light = bpy.data.objects.new(LIGHT_NAME, light_data)
    light.location = Vector((1.5, -4.0, 5.0))
    light.rotation_mode = "XYZ"
    light.rotation_euler = Euler((0.0, 0.0, 0.0), "XYZ")
    bpy.context.collection.objects.link(light)
    return light


def _configure_scene() -> None:
    scene = bpy.context.scene
    scene.frame_start = REST_FRAME
    scene.frame_end = DEFORMED_FRAME
    scene.frame_step = 1
    scene.frame_set(REST_FRAME)
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.seed = SEED
    scene.cycles.samples = 16
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.use_animated_seed = False
    scene.render.use_motion_blur = False
    scene.render.film_transparent = True
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 1
    scene.render.resolution_x = 64
    scene.render.resolution_y = 64
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.use_nodes = False


def build_fixture() -> dict[str, object]:
    """Build the fixture in the current Blender session and describe it."""
    _reset_scene()
    target = _build_mesh()
    image = _build_texture()
    _build_material(target, image)
    rig = _build_armature(target)
    action = _build_action(rig)
    camera = _build_camera()
    light = _build_light()
    _configure_scene()

    return {
        "profile": FIXTURE_PROFILE,
        "seed": SEED,
        "target_name": target.name,
        "armature_object_name": rig.name,
        "action_name": action.name,
        "camera_name": camera.name,
        "light_name": light.name,
        "bone_names": [BASE_BONE_NAME, TIP_BONE_NAME],
        "vertex_count": len(target.data.vertices),
        "triangle_count": sum(len(polygon.vertices) - 2 for polygon in target.data.polygons),
        "uv_layer_count": len(target.data.uv_layers),
        "rest_frame": REST_FRAME,
        "deformed_frame": DEFORMED_FRAME,
        "deformation_degrees": DEFORMATION_DEGREES,
        "texture_size": [TEXTURE_SIZE, TEXTURE_SIZE],
        "packed_image_count": sum(1 for item in bpy.data.images if item.packed_file is not None),
        "external_dependency_count": 0,
    }


def write_fixture(path: str) -> dict[str, object]:
    """Build the fixture and save it to `path`, which must live below staging."""
    description = build_fixture()
    bpy.ops.wm.save_as_mainfile(filepath=path, compress=False, copy=True, relative_remap=False)
    return description
