"""Observed-front face anchor and generated-view visual-hull research profile."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from asset_mania_contracts import TURNTABLE_YAWS
from PIL import Image

COMMON_RADIUS = 0.6
CANONICAL_RESOLUTION = 1024
CANONICAL_FOREGROUND_RATIO = 0.82


@dataclass(frozen=True, slots=True)
class CanonicalView:
    yaw: int
    image_path: Path
    mask_path: Path


@dataclass(frozen=True, slots=True)
class FaceHybridSettings:
    grid_resolution: int = 192
    minimum_silhouette_votes: int = 7
    front_seam: float = 0.08


def _require_yaw(yaw: int) -> None:
    if not isinstance(yaw, int) or isinstance(yaw, bool) or yaw not in TURNTABLE_YAWS:
        raise ValueError(f"yaw must be one of {list(TURNTABLE_YAWS)}")


def project_points(points: np.ndarray, *, yaw: int, resolution: int) -> np.ndarray:
    """Project common TripoSR-world points into one orthographic yaw image."""
    _require_yaw(yaw)
    if not isinstance(resolution, int) or isinstance(resolution, bool) or resolution < 2:
        raise ValueError("resolution must be an integer of at least 2")
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError("points must be a finite Nx3 array")
    radians = np.deg2rad(yaw)
    horizontal = -np.sin(radians) * array[:, 0] + np.cos(radians) * array[:, 1]
    vertical = array[:, 2]
    scale = (resolution - 1) / (2 * COMMON_RADIUS)
    pixel_x = (horizontal + COMMON_RADIUS) * scale
    pixel_y = (COMMON_RADIUS - vertical) * scale
    return np.column_stack((pixel_x, pixel_y))


def _load_pair(view: CanonicalView) -> tuple[Image.Image, Image.Image, np.ndarray]:
    _require_yaw(view.yaw)
    try:
        with Image.open(view.image_path) as opened:
            opened.load()
            image = opened.convert("RGBA")
        with Image.open(view.mask_path) as opened:
            opened.load()
            mask = opened.convert("L")
    except (OSError, ValueError) as error:
        raise ValueError(f"yaw {view.yaw} image or mask is unreadable") from error
    expected = (CANONICAL_RESOLUTION, CANONICAL_RESOLUTION)
    if image.size != expected or mask.size != expected:
        raise ValueError(f"yaw {view.yaw} image and mask must be 1024x1024")
    foreground = np.asarray(mask) >= 128
    coverage = float(foreground.mean())
    if not 0.15 <= coverage <= 0.65:
        raise ValueError(f"yaw {view.yaw} foreground coverage is outside [0.15, 0.65]")
    return image, mask, foreground


def _affine_for(foreground: np.ndarray) -> tuple[float, float, float, float, float, float]:
    ys, xs = np.nonzero(foreground)
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    scale = (CANONICAL_RESOLUTION * CANONICAL_FOREGROUND_RATIO) / max(width, height)
    centre = (CANONICAL_RESOLUTION - 1) / 2
    inverse = 1.0 / scale
    source_x = float(xs.mean())
    source_y = float(ys.mean())
    return (
        inverse,
        0.0,
        source_x - centre * inverse,
        0.0,
        inverse,
        source_y - centre * inverse,
    )


def canonicalize_views(
    views: Sequence[CanonicalView], output_directory: Path
) -> list[CanonicalView]:
    """Centre and uniformly scale eight private views without changing their source bytes."""
    records = list(views)
    if [record.yaw for record in records] != list(TURNTABLE_YAWS):
        raise ValueError(f"views must use the ordered yaws {list(TURNTABLE_YAWS)}")
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")

    decoded: list[tuple[CanonicalView, Image.Image, Image.Image, np.ndarray]] = []
    digests: list[str] = []
    for record in records:
        image, mask, foreground = _load_pair(record)
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        digests.append(digest)
        decoded.append((record, image, mask, foreground))
    if len(digests) != len(set(digests)):
        raise ValueError("turntable contains duplicate decoded images")

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}-staging-",
            dir=output_directory.parent,
        )
    )
    try:
        for record, image, mask, foreground in decoded:
            affine = _affine_for(foreground)
            normalized_image = image.transform(
                (CANONICAL_RESOLUTION, CANONICAL_RESOLUTION),
                Image.Transform.AFFINE,
                affine,
                resample=Image.Resampling.BICUBIC,
                fillcolor=(0, 0, 0, 0),
            )
            normalized_mask = mask.transform(
                (CANONICAL_RESOLUTION, CANONICAL_RESOLUTION),
                Image.Transform.AFFINE,
                affine,
                resample=Image.Resampling.NEAREST,
                fillcolor=0,
            )
            normalized_image.save(staging / f"yaw-{record.yaw:03d}.png", format="PNG")
            normalized_mask.save(staging / f"yaw-{record.yaw:03d}-mask.png", format="PNG")
        staging.replace(output_directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        for _record, image, mask, _foreground in decoded:
            image.close()
            mask.close()

    return [
        CanonicalView(
            yaw=yaw,
            image_path=output_directory / f"yaw-{yaw:03d}.png",
            mask_path=output_directory / f"yaw-{yaw:03d}-mask.png",
        )
        for yaw in TURNTABLE_YAWS
    ]


def _validate_settings(settings: FaceHybridSettings) -> None:
    if (
        not isinstance(settings.grid_resolution, int)
        or isinstance(settings.grid_resolution, bool)
        or not 16 <= settings.grid_resolution <= 512
    ):
        raise ValueError("grid_resolution must be an integer in 16..512")
    if (
        not isinstance(settings.minimum_silhouette_votes, int)
        or isinstance(settings.minimum_silhouette_votes, bool)
        or not 1 <= settings.minimum_silhouette_votes <= len(TURNTABLE_YAWS)
    ):
        raise ValueError("minimum_silhouette_votes must be an integer in 1..8")
    if not isinstance(settings.front_seam, (int, float)) or not 0 < settings.front_seam < 0.6:
        raise ValueError("front_seam must be within the common cube")


def _load_canonical_masks(views: Sequence[CanonicalView]) -> list[np.ndarray]:
    records = list(views)
    if [record.yaw for record in records] != list(TURNTABLE_YAWS):
        raise ValueError(f"views must use the ordered yaws {list(TURNTABLE_YAWS)}")
    masks = []
    for record in records:
        try:
            with Image.open(record.mask_path) as opened:
                opened.load()
                if opened.size != (CANONICAL_RESOLUTION, CANONICAL_RESOLUTION):
                    raise ValueError(f"yaw {record.yaw} mask must be 1024x1024")
                mask = np.asarray(opened.convert("L")) >= 128
        except OSError as error:
            raise ValueError(f"yaw {record.yaw} mask is unreadable") from error
        if not mask.any():
            raise ValueError(f"yaw {record.yaw} mask has no foreground")
        masks.append(mask)
    return masks


def _clean_volume(occupancy: np.ndarray) -> np.ndarray:
    from scipy.ndimage import (
        binary_closing,
        binary_fill_holes,
        generate_binary_structure,
        label,
    )

    structure = generate_binary_structure(3, 3)
    cleaned = binary_closing(np.asarray(occupancy, dtype=bool), structure=structure, iterations=1)
    cleaned = binary_fill_holes(cleaned)
    labels, count = label(cleaned, structure=structure)
    if count == 0:
        raise ValueError("visual hull is empty")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    cleaned = labels == int(np.argmax(sizes))
    boundary = np.concatenate(
        (
            cleaned[0].ravel(),
            cleaned[-1].ravel(),
            cleaned[:, 0].ravel(),
            cleaned[:, -1].ravel(),
            cleaned[:, :, 0].ravel(),
            cleaned[:, :, -1].ravel(),
        )
    )
    if boundary.any():
        raise ValueError("visual hull touches the common grid boundary")
    return cleaned


def _reprojection_metrics(
    occupancy: np.ndarray, views: Sequence[CanonicalView], masks: Sequence[np.ndarray]
) -> dict[str, float]:
    resolution = occupancy.shape[0]
    indices = np.argwhere(occupancy)
    coordinates = -COMMON_RADIUS + indices * (2 * COMMON_RADIUS / (resolution - 1))
    values = []
    for record, mask in zip(views, masks, strict=True):
        projected = project_points(coordinates, yaw=record.yaw, resolution=resolution)
        pixels = np.rint(projected).astype(int)
        valid = np.logical_and(pixels >= 0, pixels < resolution).all(axis=1)
        silhouette = np.zeros((resolution, resolution), dtype=bool)
        pixels = pixels[valid]
        silhouette[pixels[:, 1], pixels[:, 0]] = True
        target = np.asarray(
            Image.fromarray(mask).resize((resolution, resolution), Image.Resampling.NEAREST),
            dtype=bool,
        )
        union = np.logical_or(silhouette, target).sum()
        intersection = np.logical_and(silhouette, target).sum()
        values.append(float(intersection / union) if union else 0.0)
    return {
        "minimum_reprojection_iou": float(min(values)),
        "mean_reprojection_iou": float(np.mean(values)),
    }


def build_visual_hull(
    views: Sequence[CanonicalView], settings: FaceHybridSettings
) -> tuple[np.ndarray, dict[str, float]]:
    """Carve a robust seven-of-eight visual hull from canonical head silhouettes."""
    _validate_settings(settings)
    records = list(views)
    masks = _load_canonical_masks(records)
    resolution = settings.grid_resolution
    coordinates = np.linspace(-COMMON_RADIUS, COMMON_RADIUS, resolution)
    occupancy = np.zeros((resolution, resolution, resolution), dtype=bool)
    slab_depth = min(16, resolution)
    for start in range(0, resolution, slab_depth):
        stop = min(start + slab_depth, resolution)
        x, y, z = np.meshgrid(
            coordinates,
            coordinates,
            coordinates[start:stop],
            indexing="ij",
        )
        points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        votes = np.zeros(len(points), dtype=np.uint8)
        for record, mask in zip(records, masks, strict=True):
            projected = project_points(points, yaw=record.yaw, resolution=CANONICAL_RESOLUTION)
            pixels = np.rint(projected).astype(int)
            valid = np.logical_and(pixels >= 0, pixels < CANONICAL_RESOLUTION).all(axis=1)
            supported = np.zeros(len(points), dtype=bool)
            valid_pixels = pixels[valid]
            supported[valid] = mask[valid_pixels[:, 1], valid_pixels[:, 0]]
            votes += supported
        occupancy[:, :, start:stop] = (
            votes.reshape(resolution, resolution, stop - start) >= settings.minimum_silhouette_votes
        )
    occupancy = _clean_volume(occupancy)
    metrics = _reprojection_metrics(occupancy, records, masks)
    if metrics["minimum_reprojection_iou"] < 0.72 or metrics["mean_reprojection_iou"] < 0.82:
        raise ValueError("visual hull reprojection gate failed")
    return occupancy, metrics


__all__ = [
    "CanonicalView",
    "FaceHybridSettings",
    "build_visual_hull",
    "canonicalize_views",
    "project_points",
]
