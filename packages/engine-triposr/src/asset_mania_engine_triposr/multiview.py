"""Yaw-aware normalization and voxel consensus for local TripoSR meshes."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class YawMesh:
    yaw: int
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class FusionSettings:
    grid_resolution: int = 192
    minimum_votes: int | None = None


@dataclass(frozen=True, slots=True)
class FusionResult:
    triangle_count: int
    vertex_count: int
    manifold: str
    signed_volume: float
    eligible_mesh_count: int
    minimum_votes: int


def normalize_and_rotate(vertices: np.ndarray, yaw: int) -> np.ndarray:
    """Centre, unit-scale, and remove one declared yaw rotation about +Z."""
    array = np.asarray(vertices, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) < 3:
        raise ValueError("vertices must be an Nx3 array with at least three rows")
    if not np.isfinite(array).all():
        raise ValueError("vertices must be finite")
    if not isinstance(yaw, int) or isinstance(yaw, bool) or not 0 <= yaw < 360:
        raise ValueError("yaw must be an integer in 0..359")
    lower = array.min(axis=0)
    upper = array.max(axis=0)
    extent = np.ptp(array, axis=0)
    scale = float(extent.max())
    if scale <= 0:
        raise ValueError("vertices have no positive extent")
    normalized = (array - (lower + upper) * 0.5) / scale
    radians = np.deg2rad(-yaw)
    rotation = np.array(
        [
            [np.cos(radians), -np.sin(radians), 0.0],
            [np.sin(radians), np.cos(radians), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return normalized @ rotation.T


def vote_occupancy(grids: Sequence[np.ndarray], minimum_votes: int | None = None) -> np.ndarray:
    """Return voxels occupied by at least half of six to eight eligible meshes."""
    arrays = [np.asarray(grid, dtype=bool) for grid in grids]
    if not 6 <= len(arrays) <= 8:
        raise ValueError("voxel consensus requires six to eight grids")
    shape = arrays[0].shape
    if len(shape) != 3 or any(array.shape != shape for array in arrays):
        raise ValueError("voxel grids must have one identical 3D shape")
    votes = (len(arrays) + 1) // 2 if minimum_votes is None else minimum_votes
    if not isinstance(votes, int) or isinstance(votes, bool) or not 1 <= votes <= len(arrays):
        raise ValueError("minimum_votes must be an integer within the grid count")
    return np.sum(np.stack(arrays, axis=0), axis=0) >= votes


def _load_mesh(path: Path):
    import trimesh

    return trimesh.load(str(path), process=False, force="mesh")


def _voxelize(mesh, *, yaw: int, resolution: int) -> np.ndarray:
    import trimesh

    vertices = normalize_and_rotate(np.asarray(mesh.vertices), yaw)
    normalized = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(mesh.faces),
        process=False,
    )
    pitch = 1.2 / (resolution - 1)
    voxel = normalized.voxelized(pitch).fill()
    points = np.asarray(voxel.points, dtype=float)
    grid = np.zeros((resolution, resolution, resolution), dtype=bool)
    if not len(points):
        return grid
    indices = np.rint((points + 0.6) / pitch).astype(int)
    valid = np.logical_and(indices >= 0, indices < resolution).all(axis=1)
    indices = indices[valid]
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    return grid


def _extract_surface(occupancy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torchmcubes import marching_cubes

    volume = torch.from_numpy(occupancy.astype(np.float32, copy=False))
    vertices, faces = marching_cubes(volume.contiguous(), 0.5)
    return vertices.cpu().numpy(), faces.cpu().numpy()


def fuse_turntable_meshes(
    inputs: Sequence[YawMesh],
    output_path: Path,
    settings: FusionSettings,
) -> FusionResult:
    """Fuse six to eight closed yaw meshes into one create-only neutral GLB."""
    records = list(inputs)
    expected_yaws = [0, 45, 90, 135, 180, 225, 270, 315]
    if [item.yaw for item in records] != expected_yaws:
        raise ValueError(f"input yaws must be exactly {expected_yaws} in order")
    if output_path.exists():
        raise ValueError(f"refusing to overwrite {output_path}")
    if not isinstance(settings.grid_resolution, int) or not 16 <= settings.grid_resolution <= 512:
        raise ValueError("grid_resolution must be an integer in 16..512")

    eligible: list[tuple[YawMesh, object]] = []
    for record in records:
        if not record.path.is_file():
            raise ValueError(f"mesh for yaw {record.yaw} is missing")
        actual = hashlib.sha256(record.path.read_bytes()).hexdigest()
        if actual != record.sha256:
            raise ValueError(f"mesh for yaw {record.yaw} does not match its approved digest")
        mesh = _load_mesh(record.path)
        if bool(mesh.is_watertight) and bool(mesh.is_winding_consistent):
            eligible.append((record, mesh))
    if len(eligible) < 6:
        raise ValueError("multiview fusion requires at least six closed meshes")

    minimum_votes = (
        (len(eligible) + 1) // 2 if settings.minimum_votes is None else settings.minimum_votes
    )
    grids = [
        _voxelize(mesh, yaw=record.yaw, resolution=settings.grid_resolution)
        for record, mesh in eligible
    ]
    consensus = vote_occupancy(grids, minimum_votes)
    if not consensus.any():
        raise ValueError("voxel consensus is empty")
    vertices, faces = _extract_surface(consensus)
    if not len(vertices) or not len(faces):
        raise ValueError("voxel consensus produced no surface")

    pitch = 1.2 / (settings.grid_resolution - 1)
    vertices = vertices * pitch - 0.6
    import trimesh

    from .ports.triposr import _normalise

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh, manifold = _normalise(mesh)
    signed_volume = float(mesh.volume)
    if manifold != "closed" or not mesh.is_winding_consistent or signed_volume <= 0:
        raise ValueError("fused mesh must be closed, winding-consistent, and positive-volume")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path), file_type="glb")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise ValueError("fused GLB was not written")
    return FusionResult(
        triangle_count=len(mesh.faces),
        vertex_count=len(mesh.vertices),
        manifold=manifold,
        signed_volume=signed_volume,
        eligible_mesh_count=len(eligible),
        minimum_votes=minimum_votes,
    )


__all__ = [
    "FusionResult",
    "FusionSettings",
    "YawMesh",
    "fuse_turntable_meshes",
    "normalize_and_rotate",
    "vote_occupancy",
]
