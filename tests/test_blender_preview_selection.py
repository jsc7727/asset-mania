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
