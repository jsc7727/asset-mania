# SPDX-License-Identifier: GPL-3.0-or-later
"""Render a known solid in Blender to make a reconstruction input whose answer is known.

Run through Blender:

    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \\
        --disable-autoexec --offline-mode --threads 1 --python-exit-code 86 \\
        --python scripts/make_reconstruction_fixture.py -- --shape monkey --out /tmp/in.png

The synthetic silhouettes used until now were a poor test and produced a misleading result: a
flat cutout with uniform shading gives a monocular model no depth cue at all, so it returned a
hollow shell, and nothing in the pipeline was wrong. Shading *is* the signal, which means a
fixture has to have some.

Rendering a solid also gives the one thing a photograph cannot: the ground truth. The mesh that
produced the image is written alongside it, so reconstruction error becomes a number -- volume
ratio, Hausdorff distance, silhouette agreement -- rather than an impression from a picture.

Writes three files next to `--out`:

  <out>              RGBA, subject on transparency, ready for the port
  <out>.truth.obj    the exact solid that was rendered, in the same units
  <out>.camera.json  focal length, sensor, and the camera-to-world matrix
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import bpy
import mathutils

#: Shapes with a closed analytic or generated form, so ground-truth volume is meaningful.
SHAPES = ("monkey", "sphere", "torus", "cone", "cylinder", "icosphere")


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=SHAPES, default="monkey")
    parser.add_argument("--out", required=True, help="RGBA PNG to write")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument(
        "--elevation",
        type=float,
        default=12.0,
        help="camera elevation in degrees; a monocular model sees more shape off-axis",
    )
    parser.add_argument("--azimuth", type=float, default=25.0)
    parser.add_argument("--subdivide", type=int, default=2, help="smoothing for the monkey")
    return parser.parse_args(argv)


def build_shape(name: str, subdivide: int) -> bpy.types.Object:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    factory = {
        "monkey": bpy.ops.mesh.primitive_monkey_add,
        "sphere": bpy.ops.mesh.primitive_uv_sphere_add,
        "torus": bpy.ops.mesh.primitive_torus_add,
        "cone": bpy.ops.mesh.primitive_cone_add,
        "cylinder": bpy.ops.mesh.primitive_cylinder_add,
        "icosphere": bpy.ops.mesh.primitive_ico_sphere_add,
    }[name]
    factory()
    target = bpy.context.active_object

    if name == "monkey" and subdivide:
        modifier = target.modifiers.new("subsurf", "SUBSURF")
        modifier.levels = subdivide
        modifier.render_levels = subdivide
        bpy.ops.object.modifier_apply(modifier=modifier.name)

    # Normalise to a unit-ish scale so ground-truth numbers are comparable across shapes.
    dimensions = max(target.dimensions)
    target.scale = [1.0 / dimensions] * 3
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    target.location = (0, 0, 0)
    return target


def build_material(target: bpy.types.Object) -> None:
    """A matte dielectric with a slight colour gradient.

    Deliberately not flat: uniform albedo makes every pixel's brightness a function of the
    normal alone, which is the cue the model needs, and a texture-free surface still leaves
    the silhouette doing most of the work.
    """
    material = bpy.data.materials.new("subject")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    principled = nodes["Principled BSDF"]
    principled.inputs["Roughness"].default_value = 0.55
    principled.inputs["Base Color"].default_value = (0.72, 0.58, 0.48, 1.0)
    principled.inputs["Specular IOR Level"].default_value = 0.35

    coords = nodes.new("ShaderNodeTexCoord")
    gradient = nodes.new("ShaderNodeTexGradient")
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.68, 0.52, 0.42, 1.0)
    ramp.color_ramp.elements[1].color = (0.80, 0.68, 0.58, 1.0)
    links.new(coords.outputs["Object"], gradient.inputs["Vector"])
    links.new(gradient.outputs["Color"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], principled.inputs["Base Color"])

    target.data.materials.clear()
    target.data.materials.append(material)


def build_lighting(extent: float) -> None:
    """Three-point light. Directional variation is what carries shape information."""
    # Energies tuned by measuring the result: the first attempt used values two orders of
    # magnitude higher and blew the subject out to near-white, which destroys the only cue a
    # monocular model has. Mean luminance of the foreground is the number to watch, not the
    # look of the numbers.
    for name, energy, size, location, rotation in (
        ("key", 60, 2.0, (2.4, -2.4, 2.8), (50, 0, 40)),
        ("fill", 18, 3.5, (-2.8, -1.6, 0.6), (80, 0, -60)),
        ("rim", 26, 1.5, (-1.2, 2.6, 2.0), (60, 0, 200)),
    ):
        light = bpy.data.lights.new(name, type="AREA")
        light.energy = energy * extent**2
        light.size = size * extent
        obj = bpy.data.objects.new(name, light)
        obj.location = tuple(c * extent for c in location)
        obj.rotation_euler = tuple(math.radians(a) for a in rotation)
        bpy.context.scene.collection.objects.link(obj)

    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    # Transparent film, so the alpha channel is the mask and no background remover is needed.
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.25
    bpy.context.scene.world = world


def place_camera(extent: float, elevation: float, azimuth: float) -> bpy.types.Object:
    radius = extent * 1.9  # fill the frame; the port crops to the alpha bbox anyway
    theta, phi = math.radians(azimuth), math.radians(elevation)
    camera_data = bpy.data.cameras.new("camera")
    camera_data.lens = 50.0
    camera_data.sensor_width = 36.0
    camera = bpy.data.objects.new("camera", camera_data)
    camera.location = (
        radius * math.cos(phi) * math.sin(theta),
        -radius * math.cos(phi) * math.cos(theta),
        radius * math.sin(phi),
    )
    direction = -mathutils.Vector(camera.location)
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def configure_render(resolution: int, samples: int) -> None:
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
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"


def write_truth(target: bpy.types.Object, path: str) -> dict:
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    bpy.ops.wm.obj_export(
        filepath=path,
        export_selected_objects=True,
        export_materials=False,
        export_uv=False,
        export_normals=True,
        apply_modifiers=True,
    )
    mesh = target.data
    return {
        "vertices": len(mesh.vertices),
        "polygons": len(mesh.polygons),
        "dimensions": [round(v, 6) for v in target.dimensions],
    }


def main() -> int:
    args = parse_args()
    target = build_shape(args.shape, args.subdivide)
    build_material(target)
    extent = max(target.dimensions)
    build_lighting(extent)
    camera = place_camera(extent, args.elevation, args.azimuth)
    configure_render(args.resolution, args.samples)

    bpy.context.scene.render.filepath = args.out
    bpy.ops.render.render(write_still=True)

    truth = write_truth(target, f"{args.out}.truth.obj")
    with open(f"{args.out}.camera.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "shape": args.shape,
                "lens_mm": camera.data.lens,
                "sensor_width_mm": camera.data.sensor_width,
                "resolution": [args.resolution, args.resolution],
                "elevation_deg": args.elevation,
                "azimuth_deg": args.azimuth,
                "camera_to_world": [list(row) for row in camera.matrix_world],
                "truth": truth,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    print(f"wrote {args.out}")
    print(f"wrote {args.out}.truth.obj  ({truth['polygons']} polygons)")
    print(f"wrote {args.out}.camera.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
