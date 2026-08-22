"""Optional clearance-gated single-image and multi-view mesh engine adapter."""

from .multiview import (
    FusionResult,
    FusionSettings,
    YawMesh,
    fuse_turntable_meshes,
    normalize_and_rotate,
    vote_occupancy,
)

__all__ = [
    "FusionResult",
    "FusionSettings",
    "YawMesh",
    "fuse_turntable_meshes",
    "normalize_and_rotate",
    "vote_occupancy",
]
