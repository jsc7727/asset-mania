# SPDX-License-Identifier: GPL-3.0-or-later
"""Render a reconstructed mesh in Blender, so a result can be looked at rather than described.

Run through Blender, not through the project venv:

    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \\
        --disable-autoexec --offline-mode --threads 1 --python-exit-code 86 \\
        --python scripts/render_mesh_preview.py -- --mesh out.obj --out preview.png

Uses the same flags the rest of the pipeline does: `--factory-startup` so a user's preferences
cannot change the image, `--offline-mode` and `--disable-autoexec` because a mesh file is
untrusted input, and one thread so two runs of the same input agree.

Cycles at 32 samples on CPU. Not a determinism-class claim -- the reconstruction fixtures own
that -- just a picture that shows silhouette, shading, and which way the normals face, which a
triangle count cannot.
"""

from __future__ import annotations

import argparse
import math
import sys

import bpy


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, help="mesh to import (.obj, .ply, .glb)")
    parser.add_argument("--out", required=True, help="PNG to write")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=720)
    parser.add_argument("--elevation", type=float, default=0.42)
    parser.add_argument("--start-angle-degrees", type=float, default=-90.0)
    parser.add_argument(
        "--orbit-axis",
        choices=("X", "Y", "Z"),
        default="Z",
        help="world axis around which the camera orbits",
    )
    parser.add_argument(
        "--views",
        type=int,
        default=4,
        help="orbit positions rendered side by side into one strip",
    )
    parser.add_argument(
        "--vertex-colors",
        action="store_true",
        help="shade from the mesh's colour attribute instead of a neutral grey",
    )
    parser.add_argument(
        "--use-imported-material",
        action="store_true",
        help="preserve the material and embedded texture imported from the mesh",
    )
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def select_import_target(objects) -> bpy.types.Object:
    """Return the first mesh, ignoring GLB hierarchy objects such as root empties."""
    for candidate in objects:
        if candidate.type == "MESH":
            return candidate
    raise ValueError("the imported scene contains no mesh object")


def import_mesh(path: str) -> bpy.types.Object:
    lowered = path.lower()
    if lowered.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=path)
    elif lowered.endswith(".ply"):
        bpy.ops.wm.ply_import(filepath=path)
    elif lowered.endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=path)
    elif lowered.endswith(".stl"):
        bpy.ops.wm.stl_import(filepath=path)
    else:
        raise SystemExit(f"unsupported mesh extension: {path}")

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit(f"{path} imported no mesh")
    if len(meshes) > 1:
        for extra in meshes[1:]:
            meshes[0].data.materials.clear()
        bpy.ops.object.select_all(action="SELECT")
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
    return select_import_target(bpy.context.scene.objects)


def build_material(
    target: bpy.types.Object,
    use_vertex_colors: bool,
    use_imported_material: bool = False,
) -> None:
    """A plain dielectric. Colour comes from the mesh when it has any, grey when it does not."""
    for polygon in target.data.polygons:
        polygon.use_smooth = True
    if use_imported_material and target.data.materials:
        material = target.data.materials[0]
        material.use_nodes = True
        nodes, links = material.node_tree.nodes, material.node_tree.links
        principled = nodes.get("Principled BSDF")
        texture_nodes = [node for node in nodes if node.bl_idname == "ShaderNodeTexImage"]
        if principled is not None:
            imported_images = [
                image
                for image in bpy.data.images
                if image.source == "FILE" and image.size[0] > 0 and image.size[1] > 0
            ]
            if len(imported_images) != 1:
                raise ValueError("imported textured GLB must expose exactly one file image")
            texture = texture_nodes[0] if texture_nodes else nodes.new("ShaderNodeTexImage")
            texture.image = imported_images[0]
            if not texture.inputs["Vector"].is_linked:
                coordinates = nodes.new("ShaderNodeTexCoord")
                links.new(coordinates.outputs["UV"], texture.inputs["Vector"])
            if not principled.inputs["Base Color"].is_linked:
                links.new(texture.outputs["Color"], principled.inputs["Base Color"])
            principled.inputs["Roughness"].default_value = 0.65
            emission = principled.inputs.get("Emission Color")
            emission_strength = principled.inputs.get("Emission Strength")
            if emission is not None and emission_strength is not None:
                if not emission.is_linked:
                    links.new(texture.outputs["Color"], emission)
                emission_strength.default_value = 0.35
        return
    material = bpy.data.materials.new("preview")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    principled = nodes["Principled BSDF"]
    principled.inputs["Roughness"].default_value = 0.45

    attribute_names = [a.name for a in target.data.color_attributes]
    if use_vertex_colors and attribute_names:
        attribute = nodes.new("ShaderNodeVertexColor")
        attribute.layer_name = attribute_names[0]
        links.new(attribute.outputs["Color"], principled.inputs["Base Color"])
    else:
        principled.inputs["Base Color"].default_value = (0.62, 0.62, 0.64, 1.0)

    target.data.materials.clear()
    target.data.materials.append(material)


def frame_and_light(target: bpy.types.Object) -> float:
    """Centre the subject at the origin and return the orbit radius that fits it."""
    bpy.context.view_layer.update()
    corners = [target.matrix_world @ v.co for v in target.data.vertices]
    lo = [min(c[i] for c in corners) for i in range(3)]
    hi = [max(c[i] for c in corners) for i in range(3)]
    centre = [(lo[i] + hi[i]) / 2 for i in range(3)]
    target.location = [target.location[i] - centre[i] for i in range(3)]

    extent = max(hi[i] - lo[i] for i in range(3)) or 1.0

    key = bpy.data.objects.new("key", bpy.data.lights.new("key", type="AREA"))
    key.data.energy = 900 * extent**2
    key.data.size = extent * 3
    key.location = (extent * 2.5, -extent * 2.5, extent * 3)
    key.rotation_euler = (math.radians(50), 0, math.radians(40))
    bpy.context.scene.collection.objects.link(key)

    fill = bpy.data.objects.new("fill", bpy.data.lights.new("fill", type="AREA"))
    fill.data.energy = 250 * extent**2
    fill.data.size = extent * 4
    fill.location = (-extent * 3, -extent * 1.5, extent * 0.5)
    fill.rotation_euler = (math.radians(80), 0, math.radians(-60))
    bpy.context.scene.collection.objects.link(fill)

    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.05, 0.05, 0.06, 1)
    bpy.context.scene.world = world

    return extent * 2.6


def configure_render(samples: int, resolution: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.seed = 0
    scene.render.use_motion_blur = False
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 1
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"


def camera_location(
    radius: float,
    angle: float,
    elevation: float = 0.42,
    orbit_axis: str = "Z",
) -> tuple[float, float, float]:
    radial_a = radius * math.cos(angle)
    radial_b = radius * math.sin(angle)
    offset = radius * elevation
    values = {
        "X": (offset, radial_a, radial_b),
        "Y": (radial_a, offset, radial_b),
        "Z": (radial_a, radial_b, offset),
    }
    if orbit_axis not in values:
        raise ValueError(f"unsupported orbit axis: {orbit_axis}")
    return tuple(0.0 if abs(value) < 1e-12 else value for value in values[orbit_axis])


def add_camera(
    radius: float,
    angle: float,
    elevation: float = 0.42,
    orbit_axis: str = "Z",
) -> bpy.types.Object:
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    camera.location = camera_location(radius, angle, elevation, orbit_axis)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    track = camera.constraints.new("TRACK_TO")
    empty = bpy.data.objects.new("focus", None)
    bpy.context.scene.collection.objects.link(empty)
    track.target = empty
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    return camera


def orbit_angles(views: int, start_angle_degrees: float) -> list[float]:
    if views <= 0:
        raise ValueError("views must be positive")
    start = math.radians(start_angle_degrees)
    return [start + index * (2 * math.pi / views) for index in range(views)]


def main() -> int:
    args = parse_args()
    clear_scene()
    target = import_mesh(args.mesh)
    build_material(target, args.vertex_colors, args.use_imported_material)
    radius = frame_and_light(target)
    configure_render(args.samples, args.resolution)

    tiles: list[str] = []
    for index, angle in enumerate(orbit_angles(args.views, args.start_angle_degrees)):
        camera = add_camera(radius, angle, args.elevation, args.orbit_axis)
        tile = f"{args.out}.view{index}.png"
        bpy.context.scene.render.filepath = tile
        bpy.ops.render.render(write_still=True)
        tiles.append(tile)
        bpy.data.objects.remove(camera, do_unlink=True)

    # Stitch the orbit into one strip using Blender's own image API, so this script needs
    # nothing from the project venv -- it runs inside Blender, which has no access to it.
    loaded = [bpy.data.images.load(t) for t in tiles]
    width, height = loaded[0].size
    strip = bpy.data.images.new("strip", width=width * len(loaded), height=height)
    pixels = [0.0] * (width * len(loaded) * height * 4)
    for column, image in enumerate(loaded):
        source = list(image.pixels)
        for row in range(height):
            src = row * width * 4
            dst = (row * width * len(loaded) + column * width) * 4
            pixels[dst : dst + width * 4] = source[src : src + width * 4]
    strip.pixels = pixels
    strip.filepath_raw = args.out
    strip.file_format = "PNG"
    strip.save()

    print(f"wrote {args.out}  ({len(loaded)} views, {args.samples} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
