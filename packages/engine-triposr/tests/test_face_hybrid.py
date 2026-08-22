"""Face-anchor visual-hull geometry from private canonical turntable views."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from asset_mania_contracts import TURNTABLE_YAWS
from asset_mania_engine_triposr.face_hybrid import (
    CanonicalView,
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
