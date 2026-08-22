from pathlib import Path

import numpy as np
from asset_mania_engine_dad3dheads.texture import (
    ATLAS_SIZE,
    DAD_TEXTURE_YAWS,
    DADTextureView,
    ViewVisibility,
    _remap_texture_seams,
    build_texture_atlas,
    build_textured_dad_glb,
    compute_view_visibility,
    select_triangle_views,
)
from asset_mania_pipeline import validate_glb
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


def test_atlas_uses_fixed_tiles_and_neutral_fallback(tmp_path: Path) -> None:
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
    views = [_write_view(tmp_path, yaw, camera) for yaw in DAD_TEXTURE_YAWS]
    output = tmp_path / "atlas.png"

    atlas = build_texture_atlas(views, output)

    assert atlas.size == (ATLAS_SIZE, ATLAS_SIZE)
    pixels = np.asarray(atlas)
    assert pixels[256, 256].tolist() == [0, 100, 150]
    assert pixels[1280, 1280].tolist() == [160, 145, 140]
    assert output.is_file()


def test_seam_remap_duplicates_only_cross_tile_vertices() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    normals = np.repeat(np.array([[0.0, 0.0, 1.0]]), 4, axis=0)
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    assignments = np.array([0, 45])
    projections = {
        0: np.array([[10, 10], [100, 10], [10, 100], [100, 100]], dtype=float),
        45: np.array([[20, 20], [200, 20], [20, 200], [200, 200]], dtype=float),
    }

    out_vertices, out_normals, out_faces, out_uv = _remap_texture_seams(
        vertices, normals, faces, assignments, projections, (1024, 1024)
    )

    assert len(out_vertices) == 6
    assert len(out_normals) == 6
    assert out_faces.shape == faces.shape
    assert not np.array_equal(out_faces[0, 1:], out_faces[1, [0, 2]])
    assert np.allclose(np.linalg.norm(out_normals, axis=1), 1.0)
    assert np.logical_and(out_uv >= 0, out_uv <= 1).all()


def test_textured_glb_embeds_image_texture_and_material(tmp_path: Path) -> None:
    obj = tmp_path / "head.obj"
    obj.write_text(
        "v -1 -1 -0.4\nv 1 -1 -0.4\nv -1 1 -0.4\nf 1 3 2\n",
        encoding="utf-8",
    )
    camera = np.array([[-1.0, -1.0, -0.4], [1.0, -1.0, -0.4], [-1.0, 1.0, -0.4]])
    views = []
    for yaw in DAD_TEXTURE_YAWS:
        image = tmp_path / f"single-{yaw}.png"
        mask = tmp_path / f"single-{yaw}-mask.png"
        projection = tmp_path / f"single-{yaw}.npz"
        Image.new("RGB", (1024, 1024), (yaw % 255, 120, 180)).save(image)
        Image.new("L", (1024, 1024), 255).save(mask)
        np.savez_compressed(
            projection,
            projected_vertices=np.array([[160, 160], [800, 160], [160, 800]], dtype=float),
            camera_vertices=camera,
            image_shape=np.array([1024, 1024]),
        )
        views.append(
            DADTextureView(
                yaw,
                "observed" if yaw == 0 else "generated",
                image,
                mask,
                projection,
            )
        )
    atlas = tmp_path / "atlas.png"
    glb = tmp_path / "textured.glb"

    result = build_textured_dad_glb(
        geometry_obj=obj,
        views=views,
        face_indices=np.array([0, 1, 2]),
        atlas_path=atlas,
        output_path=glb,
        visibility_resolution=64,
        enforce_gates=False,
    )

    container = validate_glb(glb)
    document = container.json_chunk
    assert result.triangle_count == 1
    assert len(document["images"]) == 1
    assert "uri" not in document["images"][0]
    assert len(document["textures"]) == 1
    assert document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0
