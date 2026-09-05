"""Optional external DAD-3DHeads research adapter."""

from .mesh import DADMeshMeasurements, convert_dad_mesh, inspect_dad_mesh
from .plugin import DADPluginSettings, run_face_plugin, validate_dad_runtime
from .texture import (
    ATLAS_SIZE,
    DAD_TEXTURE_YAWS,
    DADTextureMeasurements,
    DADTextureView,
    ViewVisibility,
    build_texture_atlas,
    build_textured_dad_glb,
    compute_view_visibility,
    select_triangle_views,
)

__all__ = [
    "ATLAS_SIZE",
    "DAD_TEXTURE_YAWS",
    "DADMeshMeasurements",
    "DADPluginSettings",
    "DADTextureMeasurements",
    "DADTextureView",
    "ViewVisibility",
    "build_texture_atlas",
    "build_textured_dad_glb",
    "compute_view_visibility",
    "convert_dad_mesh",
    "inspect_dad_mesh",
    "run_face_plugin",
    "select_triangle_views",
    "validate_dad_runtime",
]
