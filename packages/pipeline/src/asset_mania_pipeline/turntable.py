"""Local preparation and structural validation for generated turntable views."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from asset_mania_contracts import TURNTABLE_YAWS, build_turntable_viewset
from PIL import Image, ImageDraw

from .hashing import sha256_bytes, sha256_file
from .reconstruction import decode_mask, prepare_input
from .views import decode_still, write_normalized_png

SOURCE_CUTOUT_NAME = "source-cutout.png"
VIEWSET_INCONSISTENT = "VIEWSET_INCONSISTENT"


@dataclass(frozen=True, slots=True)
class TurntableCandidate:
    yaw: int
    origin: str
    image_path: Path
    mask_path: Path
    provider_request_id: str | None
    reported_usage: Mapping[str, int | float]
    actual_cost: str | None


def prepare_turntable_source(
    *, image_path: Path, mask_path: Path, staging_root: Path
) -> dict[str, Any]:
    """Validate one observed source and write its private normalized RGBA cutout."""
    prepared = prepare_input(
        image_path=image_path,
        mask_path=mask_path,
        staging_root=staging_root,
    )
    with (
        Image.open(prepared["normalized_image"]) as image,
        Image.open(prepared["normalized_mask"]) as mask,
    ):
        rgba = image.convert("RGBA")
        alpha = mask.convert("L")
        pixels = bytearray(rgba.tobytes())
        alpha_bytes = alpha.tobytes()
        for index, alpha_value in enumerate(alpha_bytes):
            offset = index * 4
            pixels[offset + 3] = alpha_value
            if alpha_value == 0:
                pixels[offset] = pixels[offset + 1] = pixels[offset + 2] = 0

    cutout = write_normalized_png(
        pixels=bytes(pixels),
        width=prepared["width"],
        height=prepared["height"],
        destination=staging_root / SOURCE_CUTOUT_NAME,
    )
    return {
        **prepared,
        "source_image_sha256": prepared["image_sha256"],
        "source_mask_sha256": prepared["mask_sha256"],
        "cutout": cutout,
        "cutout_sha256": sha256_file(cutout),
    }


def derive_white_background_mask(image_path: Path, destination: Path) -> Path:
    """Remove only near-white pixels connected to the image edge."""
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    image = decode_still(image_path)
    try:
        rgba = image.convert("RGBA")
        width, height = rgba.size
        pixels = rgba.tobytes()
    finally:
        image.close()

    count = width * height
    candidate = bytearray(count)
    for index in range(count):
        offset = index * 4
        red, green, blue, alpha = pixels[offset : offset + 4]
        candidate[index] = int(
            alpha == 0
            or (
                red >= 245
                and green >= 245
                and blue >= 245
                and max(red, green, blue) - min(red, green, blue) <= 15
            )
        )

    connected = bytearray(count)
    pending: deque[int] = deque()
    for x in range(width):
        pending.append(x)
        pending.append((height - 1) * width + x)
    for y in range(1, height - 1):
        pending.append(y * width)
        pending.append(y * width + width - 1)

    while pending:
        index = pending.popleft()
        if connected[index] or not candidate[index]:
            continue
        connected[index] = 1
        x = index % width
        y = index // width
        if x:
            pending.append(index - 1)
        if x + 1 < width:
            pending.append(index + 1)
        if y:
            pending.append(index - width)
        if y + 1 < height:
            pending.append(index + width)

    mask_bytes = bytes(0 if connected[index] else 255 for index in range(count))
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes("L", (width, height), mask_bytes).save(destination, format="PNG")
    return destination


def _mask_metrics(mask_path: Path, expected_size: tuple[int, int]) -> dict[str, float | int]:
    mask = decode_mask(mask_path)
    try:
        if mask.size != expected_size:
            raise ValueError("turntable mask does not match its image")
        width, height = mask.size
        values = mask.convert("L").tobytes()
    finally:
        mask.close()
    foreground = [index for index, value in enumerate(values) if value > 127]
    if not foreground:
        return {
            "foreground_pixels": 0,
            "coverage": 0.0,
            "centroid_offset_x": 1.0,
            "centroid_offset_y": 1.0,
            "border_contact_ratio": 1.0,
        }
    foreground_count = len(foreground)
    sum_x = sum(index % width for index in foreground)
    sum_y = sum(index // width for index in foreground)
    centroid_x = sum_x / foreground_count
    centroid_y = sum_y / foreground_count
    border = sum(
        1
        for index in foreground
        if index % width in (0, width - 1) or index // width in (0, height - 1)
    )
    return {
        "foreground_pixels": foreground_count,
        "coverage": foreground_count / (width * height),
        "centroid_offset_x": abs((centroid_x + 0.5) / width - 0.5),
        "centroid_offset_y": abs((centroid_y + 0.5) / height - 0.5),
        "border_contact_ratio": border / foreground_count,
    }


def audit_turntable(candidates: Sequence[TurntableCandidate]) -> dict[str, Any]:
    """Apply the non-biometric structural gates for one eight-view set."""
    records = list(candidates)
    failed = [item.yaw for item in records] != list(TURNTABLE_YAWS)
    metrics: list[dict[str, float | int]] = []
    decoded_digests: list[str] = []
    for index, candidate in enumerate(records):
        expected_origin = "observed" if index == 0 else "generated"
        failed |= candidate.origin != expected_origin
        image = decode_still(candidate.image_path)
        try:
            failed |= image.size != (1024, 1024)
            decoded = image.convert("RGBA").tobytes()
        finally:
            image.close()
        decoded_digests.append(sha256_bytes(decoded))
        try:
            metrics.append(_mask_metrics(candidate.mask_path, (1024, 1024)))
        except ValueError:
            failed = True
            metrics.append(
                {
                    "foreground_pixels": 0,
                    "coverage": 0.0,
                    "centroid_offset_x": 1.0,
                    "centroid_offset_y": 1.0,
                    "border_contact_ratio": 1.0,
                }
            )

    failed |= len(decoded_digests) != len(set(decoded_digests))
    coverages = [float(item["coverage"]) for item in metrics]
    if coverages:
        failed |= min(coverages) < 0.20 or max(coverages) > 0.75
    centroid_offsets = [
        max(float(item["centroid_offset_x"]), float(item["centroid_offset_y"])) for item in metrics
    ]
    border_ratios = [float(item["border_contact_ratio"]) for item in metrics]
    if centroid_offsets:
        failed |= max(centroid_offsets) > 0.10
    if border_ratios:
        failed |= max(border_ratios) >= 0.01

    adjacent_ratios: list[float] = []
    if len(coverages) == 8 and all(value > 0 for value in coverages):
        adjacent_ratios = [
            coverages[(index + 1) % len(coverages)] / coverages[index]
            for index in range(len(coverages))
        ]
        failed |= min(adjacent_ratios) < 0.65 or max(adjacent_ratios) > 1.35
    else:
        failed = True

    def bounded(values: list[float], reducer, default: float = 0.0) -> float:
        return round(float(reducer(values)), 9) if values else default

    return {
        "status": "failed" if failed else "passed",
        "diagnostics": [VIEWSET_INCONSISTENT] if failed else [],
        "identity_consistency": "unmeasured",
        "metrics": {
            "minimum_foreground_coverage": bounded(coverages, min),
            "maximum_foreground_coverage": bounded(coverages, max),
            "maximum_centroid_offset": bounded(centroid_offsets, max),
            "maximum_border_contact_ratio": bounded(border_ratios, max),
            "minimum_adjacent_area_ratio": bounded(adjacent_ratios, min),
            "maximum_adjacent_area_ratio": bounded(adjacent_ratios, max),
        },
    }


def write_contact_sheet(candidates: Sequence[TurntableCandidate], destination: Path) -> Path:
    """Write a private four-by-two yaw review sheet without source names."""
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    records = list(candidates)
    if [item.yaw for item in records] != list(TURNTABLE_YAWS):
        raise ValueError("contact sheet requires the complete ordered yaw schedule")
    tile_size = 256
    sheet = Image.new("RGB", (tile_size * 4, tile_size * 2), (24, 24, 24))
    for index, candidate in enumerate(records):
        image = decode_still(candidate.image_path)
        try:
            tile = image.convert("RGB").resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        finally:
            image.close()
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, 70, 22), fill=(0, 0, 0))
        draw.text((5, 5), f"yaw {candidate.yaw}", fill=(255, 255, 255))
        sheet.paste(tile, ((index % 4) * tile_size, (index // 4) * tile_size))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG")
    return destination


def publish_turntable_viewset(
    *,
    plan_sha256: str,
    candidates: Sequence[TurntableCandidate],
    audit: Mapping[str, Any],
    actual_cost: str | None,
) -> dict[str, Any]:
    """Describe audited local bytes without publishing their paths or content."""
    if audit.get("status") != "passed":
        raise ValueError("a turntable viewset requires a passed audit")
    records = []
    aggregate_usage: dict[str, int | float] = {}
    for index, candidate in enumerate(candidates, start=1):
        usage = dict(candidate.reported_usage)
        for key, value in usage.items():
            aggregate_usage[key] = aggregate_usage.get(key, 0) + value
        records.append(
            {
                "label": f"view-{index}",
                "target_yaw": candidate.yaw,
                "pitch": 0,
                "roll": 0,
                "origin": candidate.origin,
                "image_sha256": sha256_file(candidate.image_path),
                "mask_sha256": sha256_file(candidate.mask_path),
                "byte_size": candidate.image_path.stat().st_size,
                "media_type": "image/png",
                "width": 1024,
                "height": 1024,
                "provider_request_id": candidate.provider_request_id,
                "reported_usage": usage,
            }
        )
    return build_turntable_viewset(
        plan_sha256=plan_sha256,
        views=records,
        audit=audit,
        reported_usage=aggregate_usage,
        actual_cost=actual_cost,
    )


__all__ = [
    "SOURCE_CUTOUT_NAME",
    "VIEWSET_INCONSISTENT",
    "TurntableCandidate",
    "audit_turntable",
    "derive_white_background_mask",
    "prepare_turntable_source",
    "publish_turntable_viewset",
    "write_contact_sheet",
]
