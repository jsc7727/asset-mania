"""Optional clearance-gated single-image and multi-view mesh engine adapter."""

from .face_hybrid import (
    CanonicalView,
    FaceHybridResult,
    FaceHybridSettings,
    build_visual_hull,
    canonicalize_views,
    fuse_face_anchor,
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
    "FaceHybridResult",
    "FaceHybridSettings",
    "FusionResult",
    "FusionSettings",
    "YawMesh",
    "build_visual_hull",
    "canonicalize_views",
    "fuse_face_anchor",
    "fuse_turntable_meshes",
    "normalize_and_rotate",
    "project_points",
    "vote_occupancy",
]
