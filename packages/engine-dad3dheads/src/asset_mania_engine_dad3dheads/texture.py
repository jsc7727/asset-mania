"""Visibility-aware multi-view texture selection for fixed-topology DAD heads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

DAD_TEXTURE_YAWS = (0, 45, 90, 135, 180, 225, 270, 315)


@dataclass(frozen=True, slots=True)
class DADTextureView:
    yaw: int
    origin: Literal["observed", "generated"]
    image_path: Path
    mask_path: Path
    projection_path: Path

    def __post_init__(self) -> None:
        if self.yaw not in DAD_TEXTURE_YAWS:
            raise ValueError("unsupported DAD texture yaw")
        expected = "observed" if self.yaw == 0 else "generated"
        if self.origin != expected:
            raise ValueError(f"yaw {self.yaw} texture origin must be {expected}")


@dataclass(frozen=True, slots=True)
class ViewVisibility:
    yaw: int
    eligible: np.ndarray
    score: np.ndarray
    visible_pixels: np.ndarray


def _load_view_arrays(view: DADTextureView) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with Image.open(view.image_path) as opened:
            opened.load()
            image_size = opened.size
        with Image.open(view.mask_path) as opened:
            opened.load()
            mask = np.asarray(opened.convert("L"), dtype=np.uint8) >= 128
        with np.load(view.projection_path, allow_pickle=False) as archive:
            projected = np.asarray(archive["projected_vertices"], dtype=np.float64)
            camera = np.asarray(archive["camera_vertices"], dtype=np.float64)
            declared_shape = np.asarray(archive["image_shape"], dtype=np.int64)
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(f"yaw {view.yaw} texture view is unreadable") from error
    if image_size != (1024, 1024) or mask.shape != (1024, 1024):
        raise ValueError(f"yaw {view.yaw} image and mask must be 1024 square")
    if declared_shape.tolist() != [1024, 1024]:
        raise ValueError(f"yaw {view.yaw} projection dimensions do not match the image")
    if projected.ndim != 2 or projected.shape[1] != 2:
        raise ValueError(f"yaw {view.yaw} projected vertices are invalid")
    if camera.shape != (len(projected), 3):
        raise ValueError(f"yaw {view.yaw} camera vertices are invalid")
    if not np.isfinite(projected).all() or not np.isfinite(camera).all():
        raise ValueError(f"yaw {view.yaw} projection contains non-finite values")
    return projected, camera, mask


def _barycentric_grid(points: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    x0, y0 = points[0]
    x1, y1 = points[1]
    x2, y2 = points[2]
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denominator) <= 1e-12:
        return None
    w0 = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / denominator
    w1 = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / denominator
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
    return w0, w1, w2, inside


def compute_view_visibility(
    view: DADTextureView,
    faces: np.ndarray,
    *,
    resolution: int = 512,
    minimum_visible_pixels: int = 4,
) -> ViewVisibility:
    projected, camera, mask = _load_view_arrays(view)
    triangles = np.asarray(faces, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.size == 0:
        raise ValueError("DAD texture faces must be non-empty triangles")
    if triangles.min() < 0 or triangles.max() >= len(camera):
        raise ValueError("DAD texture face index is out of range")
    points3 = camera[triangles]
    cross = np.cross(points3[:, 1] - points3[:, 0], points3[:, 2] - points3[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    cosine = np.zeros(len(triangles), dtype=np.float64)
    valid_normal = lengths > 1e-12
    cosine[valid_normal] = -cross[valid_normal, 2] / lengths[valid_normal]
    projected_work = projected * (resolution / 1024.0)
    points2 = projected_work[triangles]
    in_bounds = np.logical_and(points2 >= 0, points2 < resolution).all(axis=(1, 2))

    rounded = np.clip(np.rint(projected).astype(np.int64), 0, 1023)
    vertex_mask = mask[rounded[:, 1], rounded[:, 0]][triangles]
    centroid = projected[triangles].mean(axis=1)
    centroid_rounded = np.clip(np.rint(centroid).astype(np.int64), 0, 1023)
    centroid_mask = mask[centroid_rounded[:, 1], centroid_rounded[:, 0]]
    inside_count = vertex_mask.sum(axis=1)
    mask_confidence = np.where(
        inside_count == 3, 1.0, np.where((inside_count >= 2) & centroid_mask, 0.8, 0.0)
    )
    preliminary = (cosine > 0) & in_bounds & (mask_confidence > 0)

    depth = np.full((resolution, resolution), np.inf, dtype=np.float64)
    owner = np.full((resolution, resolution), -1, dtype=np.int64)
    candidate_pixels = np.zeros(len(triangles), dtype=np.int64)
    for face_index in np.flatnonzero(preliminary):
        triangle2 = points2[face_index]
        lower = np.maximum(np.floor(triangle2.min(axis=0)).astype(int), 0)
        upper = np.minimum(np.ceil(triangle2.max(axis=0)).astype(int), resolution - 1)
        if np.any(upper < lower):
            continue
        grid_x, grid_y = np.meshgrid(
            np.arange(lower[0], upper[0] + 1, dtype=np.float64) + 0.5,
            np.arange(lower[1], upper[1] + 1, dtype=np.float64) + 0.5,
        )
        barycentric = _barycentric_grid(triangle2, grid_x, grid_y)
        if barycentric is None:
            continue
        w0, w1, w2, inside = barycentric
        count = int(np.count_nonzero(inside))
        candidate_pixels[face_index] = count
        if count == 0:
            continue
        triangle_depth = points3[face_index, :, 2]
        interpolated = w0 * triangle_depth[0] + w1 * triangle_depth[1] + w2 * triangle_depth[2]
        ys = slice(lower[1], upper[1] + 1)
        xs = slice(lower[0], upper[0] + 1)
        current_depth = depth[ys, xs]
        update = inside & (interpolated < current_depth - 1e-6)
        current_depth[update] = interpolated[update]
        owner_slice = owner[ys, xs]
        owner_slice[update] = face_index

    owned = owner[owner >= 0]
    visible_pixels = np.bincount(owned, minlength=len(triangles)).astype(np.int64)
    eligible = preliminary & (visible_pixels >= minimum_visible_pixels)
    fraction = np.divide(
        visible_pixels,
        np.maximum(candidate_pixels, 1),
        out=np.zeros(len(triangles), dtype=np.float64),
        where=candidate_pixels > 0,
    )
    score = cosine * fraction * mask_confidence
    score[~eligible] = 0.0
    return ViewVisibility(view.yaw, eligible, score, visible_pixels)


def select_triangle_views(
    visibilities: list[ViewVisibility],
    faces: np.ndarray,
    face_indices: np.ndarray,
) -> np.ndarray:
    if [item.yaw for item in visibilities] != list(DAD_TEXTURE_YAWS):
        raise ValueError(f"texture visibility yaws must be {list(DAD_TEXTURE_YAWS)}")
    triangles = np.asarray(faces, dtype=np.int64)
    face_set = {int(index) for index in np.asarray(face_indices, dtype=np.int64)}
    indexed_face = np.asarray(
        [sum(int(vertex) in face_set for vertex in triangle) >= 2 for triangle in triangles],
        dtype=bool,
    )
    result = np.full(len(triangles), -1, dtype=np.int64)
    for face_index in range(len(triangles)):
        candidates = []
        for item in visibilities:
            if not bool(item.eligible[face_index]):
                continue
            multiplier = 1.0
            if item.yaw == 0:
                multiplier = 4.0 if indexed_face[face_index] else 1.5
            weighted = float(item.score[face_index]) * multiplier
            wrapped = min(item.yaw, 360 - item.yaw)
            candidates.append(
                (
                    weighted,
                    int(item.visible_pixels[face_index]),
                    -wrapped,
                    -item.yaw,
                    item.yaw,
                )
            )
        if candidates:
            result[face_index] = max(candidates)[-1]
    return result
