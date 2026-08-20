# SPDX-License-Identifier: GPL-3.0-or-later
"""Negative and malicious variants of the runtime fixture.

Each variant starts from the valid composite fixture and breaks exactly one closed-profile
invariant, so a preflight failure can be attributed to that one cause. The malicious
variants additionally try to write outside staging; the write must never happen.

Like the base fixture, every variant is generated at runtime into staging. Nothing here is
tracked, uploaded, or derived from a real person.
"""

import bpy
from mathutils import Vector

from . import fixture_factory

#: A write attempt that must never succeed. The path is supplied by the caller so the test
#: can assert the exact file does not exist afterwards.
SENTINEL_PLACEHOLDER = "__ASSET_MANIA_SENTINEL__"


def _target() -> bpy.types.Object:
    return bpy.data.objects[fixture_factory.TARGET_NAME]


def _rig() -> bpy.types.Object:
    return bpy.data.objects[fixture_factory.ARMATURE_OBJECT_NAME]


def _valid() -> None:
    """The unmodified fixture."""


def _static_prop() -> None:
    """A valid static prop: no armature, no action, no rig weights needed."""
    target = _target()
    for modifier in list(target.modifiers):
        target.modifiers.remove(modifier)
    target.parent = None
    for group in list(target.vertex_groups):
        target.vertex_groups.remove(group)

    rig = _rig()
    bpy.data.objects.remove(rig, do_unlink=True)
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)


def _ambiguous_mesh() -> None:
    """A second mesh, so an unqualified target cannot be resolved."""
    duplicate = _target().copy()
    duplicate.data = _target().data.copy()
    duplicate.name = "Robot_Strip_Body_Spare"
    duplicate.location = Vector((0.0, 3.0, 0.0))
    bpy.context.collection.objects.link(duplicate)


def _ambiguous_camera() -> None:
    camera = bpy.data.objects[fixture_factory.CAMERA_NAME]
    duplicate = camera.copy()
    duplicate.data = camera.data.copy()
    duplicate.name = "Shot_Camera_Spare"
    bpy.context.collection.objects.link(duplicate)


def _case_colliding_name() -> None:
    """A name differing only by case must not be silently resolved."""
    duplicate = _target().copy()
    duplicate.data = _target().data.copy()
    duplicate.name = fixture_factory.TARGET_NAME.lower()
    bpy.context.collection.objects.link(duplicate)


# There is deliberately no non-finite-transform variant. Blender sanitizes a non-finite
# location or scale to zero when the value crosses the RNA boundary, so such a scene cannot
# be authored through the Python API. `scene_inventory.matrix_findings` still rejects a
# non-finite matrix for a hand-authored file, and the in-Blender worker tests cover that
# branch directly.
def _singular_scale() -> None:
    _target().scale = Vector((1.0, 0.0, 1.0))


def _negative_determinant() -> None:
    _target().scale = Vector((-1.0, 1.0, 1.0))


def _unpacked_image() -> None:
    """An image that still points at an external file this profile refuses to follow."""
    for image in bpy.data.images:
        if image.packed_file is not None:
            image.unpack(method="REMOVE")
            image.source = "FILE"
            image.filepath = "//textures/robot_strip_quadrants.png"


def _overlapping_uv() -> None:
    mesh = _target().data
    uv_layer = mesh.uv_layers[0]
    for polygon in mesh.polygons:
        corners = ((0.10, 0.10), (0.90, 0.10), (0.90, 0.90), (0.10, 0.90))
        for index, loop_index in enumerate(polygon.loop_indices):
            uv_layer.data[loop_index].uv = corners[index]


def _uv_outside_unit_range() -> None:
    mesh = _target().data
    uv_layer = mesh.uv_layers[0]
    for loop_index in range(len(uv_layer.data)):
        uv = uv_layer.data[loop_index].uv
        uv_layer.data[loop_index].uv = (uv[0] + 1.5, uv[1])


def _missing_uv() -> None:
    mesh = _target().data
    while len(mesh.uv_layers):
        mesh.uv_layers.remove(mesh.uv_layers[0])


def _topology_modifier() -> None:
    modifier = _target().modifiers.new(name="Robot_Strip_Subdivide", type="SUBSURF")
    modifier.levels = 1


def _bone_constraint() -> None:
    """A constraint makes the pose non-reproducible under the closed profile."""
    bone = _rig().pose.bones[fixture_factory.TIP_BONE_NAME]
    bone.constraints.new(type="LIMIT_ROTATION")


def _zero_rig_weights() -> None:
    target = _target()
    for group in target.vertex_groups:
        group.remove([vertex.index for vertex in target.data.vertices])


def _autoexec_driver(sentinel_path: str) -> None:
    """A driver whose expression would write a sentinel if autoexec were trusted."""
    target = _target()
    target.animation_data_create()
    curve = target.driver_add("location", 2)
    curve.driver.type = "SCRIPTED"
    curve.driver.expression = (
        f"__import__('pathlib').Path({sentinel_path!r}).write_text('driver ran') or 0.0"
    )


def _compositor_file_output(sentinel_path: str) -> None:
    """A File Output node aimed outside staging. Sanitization must remove it."""
    scene = bpy.context.scene
    group = bpy.data.node_groups.new("Robot_Strip_Composite", "CompositorNodeTree")
    output = group.nodes.new("CompositorNodeOutputFile")
    # 5.2's file-output node spells its write surface `directory` + `file_name`; the
    # pre-5 `base_path` and `file_slots` attributes are gone.
    directory, _, file_name = str(sentinel_path).rpartition("/")
    output.directory = directory
    output.file_name = file_name
    scene.compositing_node_group = group
    scene.use_nodes = True


def _texture_cache(sentinel_path: str) -> None:
    """Cycles texture-cache settings aimed at a source-adjacent path."""
    cycles = getattr(bpy.context.scene, "cycles", None)
    if cycles is None:
        return
    for attribute in (
        "use_texture_cache",
        "use_auto_generate_texture_cache",
        "debug_use_texture_cache_eviction",
        "use_auto_tile",
    ):
        if hasattr(cycles, attribute):
            setattr(cycles, attribute, True)
    for attribute in ("texture_cache_path", "texture_cache_directory"):
        if hasattr(cycles, attribute):
            setattr(cycles, attribute, str(sentinel_path).rsplit("/", 1)[0])


_PLAIN_VARIANTS = {
    "valid": _valid,
    "static-prop": _static_prop,
    "ambiguous-mesh": _ambiguous_mesh,
    "ambiguous-camera": _ambiguous_camera,
    "case-colliding-name": _case_colliding_name,
    "singular-scale": _singular_scale,
    "negative-determinant": _negative_determinant,
    "unpacked-image": _unpacked_image,
    "overlapping-uv": _overlapping_uv,
    "uv-outside-unit-range": _uv_outside_unit_range,
    "missing-uv": _missing_uv,
    "topology-modifier": _topology_modifier,
    "bone-constraint": _bone_constraint,
    "zero-rig-weights": _zero_rig_weights,
}
_SENTINEL_VARIANTS = {
    "autoexec-driver": _autoexec_driver,
    "compositor-file-output": _compositor_file_output,
    "texture-cache": _texture_cache,
}
VARIANTS = tuple(sorted([*_PLAIN_VARIANTS, *_SENTINEL_VARIANTS]))


def write_variant(*, variant: str, path: str, sentinel_path: str | None = None) -> dict:
    """Build one variant on top of the base fixture and save it to `path`."""
    if variant not in VARIANTS:
        raise ValueError(f"{variant!r} is not a known fixture variant")

    description = fixture_factory.build_fixture()
    if variant in _SENTINEL_VARIANTS:
        if sentinel_path is None:
            raise ValueError(f"variant {variant!r} requires a sentinel path")
        _SENTINEL_VARIANTS[variant](sentinel_path)
    else:
        _PLAIN_VARIANTS[variant]()

    bpy.ops.wm.save_as_mainfile(filepath=path, compress=False, copy=True, relative_remap=False)
    return {**description, "variant": variant}
