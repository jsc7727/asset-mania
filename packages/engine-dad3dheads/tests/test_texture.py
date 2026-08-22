from pathlib import Path

import numpy as np
from asset_mania_engine_dad3dheads.texture import (
    DAD_TEXTURE_YAWS,
    DADTextureView,
    ViewVisibility,
    compute_view_visibility,
    select_triangle_views,
)
from PIL import Image


def _write_view(tmp_path: Path, yaw: int, camera_vertices: np.ndarray) -> DADTextureView:
    image = tmp_path / f"yaw-{yaw:03d}.png"
    mask = tmp_path / f"yaw-{yaw:03d}-mask.png"
    projection = tmp_path / f"yaw-{yaw:03d}.npz"
    Image.new("RGB", (1024, 1024), (yaw % 255, 100, 150)).save(image)
    Image.new("L", (1024, 1024), 255).save(mask)
    projected = np.array(
        [
            [160.0, 160.0],
            [800.0, 160.0],
            [160.0, 800.0],
            [160.0, 160.0],
            [800.0, 160.0],
            [160.0, 800.0],
        ]
    )
    np.savez_compressed(
        projection,
        projected_vertices=projected,
        camera_vertices=camera_vertices,
        image_shape=np.array([1024, 1024]),
    )
    return DADTextureView(
        yaw=yaw,
        origin="observed" if yaw == 0 else "generated",
        image_path=image,
        mask_path=mask,
        projection_path=projection,
    )


def test_visibility_keeps_front_triangle_and_occludes_rear(tmp_path: Path) -> None:
    camera = np.array(
        [
            [-1.0, -1.0, -0.4],
            [1.0, -1.0, -0.4],
            [-1.0, 1.0, -0.4],
            [-1.0, -1.0, 0.2],
            [1.0, -1.0, 0.2],
            [-1.0, 1.0, 0.2],
        ]
    )
    faces = np.array([[0, 2, 1], [3, 5, 4]], dtype=np.int64)
    view = _write_view(tmp_path, 0, camera)

    visibility = compute_view_visibility(view, faces, resolution=64, minimum_visible_pixels=4)

    assert visibility.eligible.tolist() == [True, False]
    assert visibility.visible_pixels[0] > 4
    assert visibility.visible_pixels[1] == 0


def test_back_facing_triangle_is_ineligible(tmp_path: Path) -> None:
    camera = np.array(
        [
            [-1.0, -1.0, -0.4],
            [1.0, -1.0, -0.4],
            [-1.0, 1.0, -0.4],
            [-1.0, -1.0, 0.2],
            [1.0, -1.0, 0.2],
            [-1.0, 1.0, 0.2],
        ]
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    view = _write_view(tmp_path, 0, camera)

    result = compute_view_visibility(view, faces, resolution=64)

    assert result.eligible.tolist() == [False]


def test_observed_face_priority_applies_only_after_eligibility() -> None:
    faces = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int64)
    face_indices = np.array([0, 1, 2], dtype=np.int64)
    records = []
    for yaw in DAD_TEXTURE_YAWS:
        eligible = np.array([yaw in (0, 45), yaw == 45])
        score = np.array([0.30 if yaw == 0 else 0.80 if yaw == 45 else 0.0, 0.7])
        visible = np.array([20 if eligible[0] else 0, 10 if eligible[1] else 0])
        records.append(ViewVisibility(yaw, eligible, score, visible))

    assignments = select_triangle_views(records, faces, face_indices)

    assert assignments.tolist() == [0, 45]


def test_unseen_triangle_uses_neutral_tile() -> None:
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    records = [
        ViewVisibility(yaw, np.array([False]), np.array([0.0]), np.array([0]))
        for yaw in DAD_TEXTURE_YAWS
    ]

    assert select_triangle_views(records, faces, np.array([], dtype=np.int64)).tolist() == [-1]
