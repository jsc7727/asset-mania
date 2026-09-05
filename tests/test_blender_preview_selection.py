"""Regression coverage for GLB imports that include a root EMPTY object."""

import importlib.util
import math
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "blender-addon" / "src" / "asset_mania_blender" / "preview" / "render_mesh_preview.py"
)


def test_single_glb_mesh_is_selected_instead_of_its_root_empty(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "bpy", types.ModuleType("bpy"))
    spec = importlib.util.spec_from_file_location("render_mesh_preview_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root_empty = types.SimpleNamespace(type="EMPTY", name="Scene")
    mesh = types.SimpleNamespace(type="MESH", name="geometry_0")

    assert module.select_import_target([root_empty, mesh]) is mesh


def test_camera_location_orbits_selected_world_axis(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "bpy", types.ModuleType("bpy"))
    spec = importlib.util.spec_from_file_location("render_mesh_preview_axis_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.camera_location(2.0, 0.0, 0.5, "Z") == (2.0, 0.0, 1.0)
    assert module.camera_location(2.0, math.pi / 2, 0.5, "Y") == (0.0, 1.0, 2.0)
    assert module.camera_location(2.0, math.pi, 0.5, "X") == (1.0, -2.0, 0.0)


def test_orbit_angles_start_from_explicit_face_axis(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "bpy", types.ModuleType("bpy"))
    spec = importlib.util.spec_from_file_location("render_mesh_preview_angles_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.orbit_angles(4, 90.0) == [
        math.pi / 2,
        math.pi,
        3 * math.pi / 2,
        2 * math.pi,
    ]


def test_light_rig_rotates_with_front_camera_angle(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "bpy", types.ModuleType("bpy"))
    spec = importlib.util.spec_from_file_location("render_mesh_preview_lights_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    location = (2.5, -2.5, 3.0)
    orientation = (math.radians(50), 0.0, math.radians(40))

    unchanged = module.rotated_light_setup(location, orientation, -90.0)
    front_positive_y = module.rotated_light_setup(location, orientation, 90.0)

    assert unchanged == (location, orientation)
    assert all(
        math.isclose(actual, expected, abs_tol=1e-12)
        for actual, expected in zip(front_positive_y[0], (-2.5, 2.5, 3.0), strict=True)
    )
    assert math.isclose(front_positive_y[1][0], math.radians(50))
    assert front_positive_y[1][1] == 0.0
    assert math.isclose(front_positive_y[1][2], math.radians(220))


def test_scene_setup_passes_render_start_angle_to_light_rig(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "bpy", types.ModuleType("bpy"))
    spec = importlib.util.spec_from_file_location("render_mesh_preview_setup_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = object()
    captured = []
    monkeypatch.setattr(module, "clear_scene", lambda: None)
    monkeypatch.setattr(module, "import_mesh", lambda _path: target)
    monkeypatch.setattr(module, "build_material", lambda *_args: None)
    monkeypatch.setattr(
        module,
        "frame_and_light",
        lambda supplied, angle: captured.append((supplied, angle)) or 2.6,
    )
    monkeypatch.setattr(module, "configure_render", lambda *_args: None)
    args = types.SimpleNamespace(
        mesh="face.glb",
        vertex_colors=False,
        use_imported_material=False,
        start_angle_degrees=90.0,
        samples=16,
        resolution=500,
    )

    radius = module.prepare_scene(args)

    assert radius == 2.6
    assert captured == [(target, 90.0)]
