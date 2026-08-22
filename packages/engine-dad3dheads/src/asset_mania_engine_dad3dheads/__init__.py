"""Optional external DAD-3DHeads research adapter."""

from .mesh import DADMeshMeasurements, convert_dad_mesh, inspect_dad_mesh
from .plugin import DADPluginSettings, run_face_plugin, validate_dad_runtime
from .texture import (
    DAD_TEXTURE_YAWS,
    DADTextureView,
    ViewVisibility,
    compute_view_visibility,
    select_triangle_views,
)

__all__ = [
    "DAD_TEXTURE_YAWS",
    "DADMeshMeasurements",
    "DADPluginSettings",
    "DADTextureView",
    "ViewVisibility",
    "compute_view_visibility",
    "convert_dad_mesh",
    "inspect_dad_mesh",
    "run_face_plugin",
    "select_triangle_views",
    "validate_dad_runtime",
]
