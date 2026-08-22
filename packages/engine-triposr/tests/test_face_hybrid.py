"""Face-anchor visual-hull geometry from private canonical turntable views."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from asset_mania_contracts import TURNTABLE_YAWS
from asset_mania_engine_triposr.face_hybrid import (
    CanonicalView,
    FaceHybridSettings,
    build_visual_hull,
    canonicalize_views,
    project_points,
)
from PIL import Image, ImageDraw


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_view(
    directory: Path,
    *,
    yaw: int,
    shift_x: int = 0,
    shift_y: int = 0,
    empty: bool = False,
) -> CanonicalView:
    image_path = directory / f"source-{yaw:03d}.png"
    mask_path = directory / f"mask-{yaw:03d}.png"
    image = Image.new("RGB", (1024, 1024), (212, 214, 218))
    mask = Image.new("L", (1024, 1024), 0)
    if not empty:
        box = (
            250 + shift_x,
            150 + shift_y,
            774 + shift_x,
            874 + shift_y,
        )
        ImageDraw.Draw(image).ellipse(box, fill=(70 + yaw // 8, 90, 120))
        ImageDraw.Draw(mask).ellipse(box, fill=255)
    image.save(image_path)
    mask.save(mask_path)
    return CanonicalView(yaw=yaw, image_path=image_path, mask_path=mask_path)


def _views(directory: Path) -> list[CanonicalView]:
    directory.mkdir(parents=True, exist_ok=True)
    return [
        _write_view(
            directory,
            yaw=yaw,
            shift_x=(index % 3 - 1) * 24,
            shift_y=(index % 2) * 18 - 9,
        )
        for index, yaw in enumerate(TURNTABLE_YAWS)
    ]


def _mask_geometry(path: Path) -> tuple[float, tuple[float, float]]:
    with Image.open(path) as opened:
        mask = np.asarray(opened.convert("L")) >= 128
    ys, xs = np.nonzero(mask)
    longest = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    return longest / 1024, (float(xs.mean()), float(ys.mean()))


def test_projection_axes_follow_triposr_yaw_convention() -> None:
    points = np.array([[1.0, 0.0, 0.30], [0.0, 1.0, -0.30]])

    yaw0 = project_points(points, yaw=0, resolution=101)
    yaw90 = project_points(points, yaw=90, resolution=101)

    assert yaw0[0] == pytest.approx([50.0, 25.0])
    assert yaw90[1] == pytest.approx([50.0, 75.0])


def test_canonicalization_centres_and_scales_without_changing_inputs(tmp_path: Path) -> None:
    views = _views(tmp_path / "inputs")
    before = {path: _sha256(path) for view in views for path in (view.image_path, view.mask_path)}

    result = canonicalize_views(views, tmp_path / "canonical")

    assert [view.yaw for view in result] == list(TURNTABLE_YAWS)
    for view in result:
        ratio, centroid = _mask_geometry(view.mask_path)
        assert ratio == pytest.approx(0.82, abs=0.01)
        assert centroid == pytest.approx((511.5, 511.5), abs=1.0)
        assert view.image_path.name == f"yaw-{view.yaw:03d}.png"
        assert view.mask_path.name == f"yaw-{view.yaw:03d}-mask.png"
    assert before == {path: _sha256(path) for path in before}


@pytest.mark.parametrize("mutation", ["wrong_order", "empty", "wrong_size", "existing"])
def test_canonicalization_fails_closed(tmp_path: Path, mutation: str) -> None:
    views = _views(tmp_path / "inputs")
    destination = tmp_path / "canonical"
    if mutation == "wrong_order":
        views[1], views[2] = views[2], views[1]
    elif mutation == "empty":
        views[-1] = _write_view(tmp_path / "inputs", yaw=315, empty=True)
    elif mutation == "wrong_size":
        Image.new("L", (512, 512), 255).save(views[-1].mask_path)
    else:
        destination.mkdir()

    with pytest.raises((TypeError, ValueError, FileExistsError)):
        canonicalize_views(views, destination)


def test_canonicalization_rejects_duplicate_decoded_images(tmp_path: Path) -> None:
    views = _views(tmp_path / "inputs")
    views[2].image_path.write_bytes(views[1].image_path.read_bytes())

    with pytest.raises(ValueError, match="duplicate"):
        canonicalize_views(views, tmp_path / "canonical")


def _analytic_views(
    directory: Path,
    *,
    smaller_yaws: set[int] | None = None,
    smaller_factor: float = 0.86,
) -> list[CanonicalView]:
    directory.mkdir(parents=True, exist_ok=True)
    smaller = smaller_yaws or set()
    centre = 511.5
    pixels_per_unit = 1023 / 1.2
    result = []
    for index, yaw in enumerate(TURNTABLE_YAWS):
        radians = np.deg2rad(yaw)
        # One 14% scale outlier still overlaps about 74% by area and is tolerated; two
        # independent outliers force the seven-vote hull to shrink and fail the mean-IoU gate.
        factor = smaller_factor if yaw in smaller else 1.0
        base_x, base_y, base_z = 0.40 * factor, 0.32 * factor, 0.50 * factor
        radius_u = np.sqrt((base_x * np.sin(radians)) ** 2 + (base_y * np.cos(radians)) ** 2)
        mask = Image.new("L", (1024, 1024), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse(
            (
                centre - radius_u * pixels_per_unit,
                centre - base_z * pixels_per_unit,
                centre + radius_u * pixels_per_unit,
                centre + base_z * pixels_per_unit,
            ),
            fill=255,
        )
        if yaw not in smaller:
            bump_x, bump_y, bump_z = 0.11, 0.09, 0.12
            bump_u = -np.sin(radians) * 0.39
            bump_radius_u = np.sqrt(
                (bump_x * np.sin(radians)) ** 2 + (bump_y * np.cos(radians)) ** 2
            )
            draw.ellipse(
                (
                    centre + (bump_u - bump_radius_u) * pixels_per_unit,
                    centre - bump_z * pixels_per_unit,
                    centre + (bump_u + bump_radius_u) * pixels_per_unit,
                    centre + bump_z * pixels_per_unit,
                ),
                fill=255,
            )
        image_path = directory / f"yaw-{yaw:03d}.png"
        mask_path = directory / f"yaw-{yaw:03d}-mask.png"
        Image.new("RGB", (1024, 1024), (40 + index * 20, 80, 120)).save(image_path)
        mask.save(mask_path)
        result.append(CanonicalView(yaw, image_path, mask_path))
    return result


def test_visual_hull_reconstructs_one_supported_subject_volume(tmp_path: Path) -> None:
    views = _analytic_views(tmp_path / "views")

    occupancy, metrics = build_visual_hull(views, FaceHybridSettings(48, 7, 0.08))

    assert occupancy.dtype == bool
    assert occupancy.shape == (48, 48, 48)
    assert occupancy.any()
    assert metrics["minimum_reprojection_iou"] >= 0.72
    assert metrics["mean_reprojection_iou"] >= 0.82


def test_visual_hull_tolerates_one_inconsistent_generated_silhouette(tmp_path: Path) -> None:
    views = _analytic_views(tmp_path / "views", smaller_yaws={315})

    occupancy, metrics = build_visual_hull(views, FaceHybridSettings(48, 7, 0.08))

    assert occupancy.any()
    assert metrics["minimum_reprojection_iou"] >= 0.72


def test_visual_hull_rejects_two_inconsistent_silhouettes(tmp_path: Path) -> None:
    views = _analytic_views(
        tmp_path / "views",
        smaller_yaws={90, 180},
        smaller_factor=0.60,
    )

    with pytest.raises(ValueError, match="visual hull reprojection gate failed"):
        build_visual_hull(views, FaceHybridSettings(48, 7, 0.08))
