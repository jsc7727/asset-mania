"""Optional clearance-gated single-image and multi-view mesh engine adapter."""

from .face_hybrid import (
    CanonicalView,
    FaceHybridSettings,
    build_visual_hull,
    canonicalize_views,
    project_points,
)
from .multiview import (
    FusionResult,
    FusionSettings,
    YawMesh,
    fuse_turntable_meshes,
    normalize_and_rotate,
    vote_occupancy,
)

__all__ = [
    "CanonicalView",
    "FaceHybridSettings",
    "FusionResult",
    "FusionSettings",
    "YawMesh",
    "build_visual_hull",
    "canonicalize_views",
    "fuse_turntable_meshes",
    "normalize_and_rotate",
    "project_points",
    "vote_occupancy",
]
