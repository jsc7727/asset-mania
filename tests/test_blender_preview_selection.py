"""Regression coverage for GLB imports that include a root EMPTY object."""

import importlib.util
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
