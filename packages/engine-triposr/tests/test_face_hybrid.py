"""Face-anchor visual-hull geometry from private canonical turntable views."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from asset_mania_contracts import TURNTABLE_YAWS
from asset_mania_engine_triposr.face_hybrid import (
    CanonicalView,
    FaceHybridSettings,
    _align_anchor,
    _blend_face_anchor,
    _project_vertex_colors,
    build_visual_hull,
    canonicalize_views,
    fuse_face_anchor,
    project_points,
)
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

pytestmark = [
    pytest.mark.filterwarnings("ignore:'pkgutil.find_loader' is deprecated:DeprecationWarning"),
    pytest.mark.filterwarnings("ignore:__array_wrap__ must accept context:DeprecationWarning"),
]


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


def _anchor_mesh():
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    vertices = np.asarray(mesh.vertices, dtype=float)
    vertices[:, 0] *= 0.40
    vertices[:, 1] *= 0.32
    vertices[:, 2] *= 0.50
    face_region = np.exp(-((vertices[:, 1] / 0.10) ** 2 + (vertices[:, 2] / 0.14) ** 2))
    vertices[:, 0] += np.where(vertices[:, 0] > 0, 0.10 * face_region, 0.0)
    mesh.vertices = vertices
    return mesh


def _normalised_vertices(mesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    return (vertices - (lower + upper) * 0.5) / np.ptp(vertices, axis=0).max()


def _anchor_silhouette(
    mesh,
    *,
    resolution: int,
    scale: float,
    translate_y: float,
    translate_z: float,
) -> np.ndarray:
    vertices = _normalised_vertices(mesh) * scale
    vertices[:, 1] += translate_y
    vertices[:, 2] += translate_z
    pixel_x = (vertices[:, 1] + 0.6) * ((resolution - 1) / 1.2)
    pixel_y = (0.6 - vertices[:, 2]) * ((resolution - 1) / 1.2)
    points = np.column_stack((pixel_x, pixel_y))
    hull = ConvexHull(points)
    image = Image.new("L", (resolution, resolution), 0)
    ImageDraw.Draw(image).polygon([tuple(point) for point in points[hull.vertices]], fill=255)
    return np.asarray(image) >= 128


def test_anchor_alignment_recovers_bounded_scale_and_translation() -> None:
    mesh = _anchor_mesh()
    target = _anchor_silhouette(
        mesh,
        resolution=64,
        scale=1.08,
        translate_y=0.04,
        translate_z=-0.03,
    )

    _aligned, metrics = _align_anchor(mesh, target, resolution=64)

    assert metrics["scale"] == pytest.approx(1.08, abs=0.011)
    assert metrics["translate_y"] == pytest.approx(0.04, abs=0.006)
    assert metrics["translate_z"] == pytest.approx(-0.03, abs=0.006)
    assert metrics["projection_iou"] >= 0.90


def test_anchor_alignment_rejects_a_target_outside_the_bounded_search() -> None:
    mesh = _anchor_mesh()
    target = _anchor_silhouette(
        mesh,
        resolution=64,
        scale=0.60,
        translate_y=0.30,
        translate_z=0.25,
    )

    with pytest.raises(ValueError, match="anchor alignment gate failed"):
        _align_anchor(mesh, target, resolution=64)


def _hybrid_grids(resolution: int = 48) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-0.6, 0.6, resolution)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    hull = (x / 0.45) ** 2 + (y / 0.34) ** 2 + (z / 0.50) ** 2 <= 1.0
    anchor = hull.copy()
    anchor[x < -0.04] = False
    return axis, anchor, hull


def test_face_anchor_blend_keeps_front_and_completes_rear() -> None:
    axis, anchor, hull = _hybrid_grids()

    hybrid, retention = _blend_face_anchor(anchor, hull, FaceHybridSettings(48, 7, 0.08))

    front = (
        int(np.argmin(abs(axis - 0.35))),
        int(np.argmin(abs(axis))),
        int(np.argmin(abs(axis))),
    )
    rear = (
        int(np.argmin(abs(axis + 0.35))),
        int(np.argmin(abs(axis))),
        int(np.argmin(abs(axis))),
    )
    assert retention >= 0.85
    assert hybrid[front]
    assert hybrid[rear]


def test_face_anchor_blend_rejects_excessive_front_clipping() -> None:
    axis, anchor, hull = _hybrid_grids()
    hull[np.meshgrid(axis, axis, axis, indexing="ij")[0] > 0.15] = False

    with pytest.raises(ValueError, match="front anchor retention gate failed"):
        _blend_face_anchor(anchor, hull, FaceHybridSettings(48, 7, 0.08))


def _colored_views(directory: Path, *, hide_yaw0: bool = False) -> list[CanonicalView]:
    directory.mkdir(parents=True, exist_ok=True)
    cardinal = {
        0: (240, 10, 10),
        90: (10, 240, 10),
        180: (10, 10, 240),
        270: (220, 220, 10),
    }
    result = []
    for yaw in TURNTABLE_YAWS:
        image_path = directory / f"color-{yaw:03d}.png"
        mask_path = directory / f"color-{yaw:03d}-mask.png"
        Image.new("RGB", (64, 64), cardinal.get(yaw, (32, 32, 32))).save(image_path)
        mask_value = 0 if hide_yaw0 and yaw == 0 else 255
        Image.new("L", (64, 64), mask_value).save(mask_path)
        result.append(CanonicalView(yaw, image_path, mask_path))
    return result


def test_vertex_colors_follow_the_best_facing_cameras(tmp_path: Path) -> None:
    views = _colored_views(tmp_path / "views")
    vertices = np.array([[0.30, 0.0, 0.0], [0.0, 0.30, 0.0], [-0.30, 0.0, 0.0]])
    normals = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

    colors, coverage = _project_vertex_colors(vertices, normals, views)

    assert colors[0, 0] > colors[0, 1] * 3
    assert colors[1, 1] > colors[1, 0] * 3
    assert colors[2, 2] > colors[2, 0] * 3
    assert coverage == 1.0
    assert (colors[:, 3] == 255).all()


def test_observed_front_color_is_not_used_when_its_mask_is_invalid(tmp_path: Path) -> None:
    views = _colored_views(tmp_path / "views", hide_yaw0=True)

    colors, coverage = _project_vertex_colors(
        np.array([[0.30, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0]]),
        views,
    )

    assert colors[0, 0] < 100
    assert coverage == 1.0


@pytest.mark.skipif(
    importlib.util.find_spec("torchmcubes") is None,
    reason="optional torchmcubes runtime is not installed in the workspace",
)
def test_face_hybrid_exports_one_colored_watertight_glb(tmp_path: Path) -> None:
    import trimesh

    views = _analytic_views(tmp_path / "views")
    anchor = tmp_path / "anchor.glb"
    _anchor_mesh().export(anchor, file_type="glb")
    output = tmp_path / "hybrid.glb"

    result = fuse_face_anchor(
        anchor_mesh=anchor,
        views=views,
        output_path=output,
        settings=FaceHybridSettings(48, 7, 0.08),
    )

    mesh = trimesh.load(str(output), process=False, force="mesh")
    assert result.manifold == "closed"
    assert result.component_count == 1
    assert result.signed_volume > 0
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert len(mesh.visual.vertex_colors) == len(mesh.vertices)
    assert result.color_coverage > 0.80


@pytest.mark.skipif(
    importlib.util.find_spec("torchmcubes") is None,
    reason="optional torchmcubes runtime is not installed in the workspace",
)
def test_face_hybrid_refuses_to_replace_an_existing_glb(tmp_path: Path) -> None:
    views = _analytic_views(tmp_path / "views")
    anchor = tmp_path / "anchor.glb"
    _anchor_mesh().export(anchor, file_type="glb")
    output = tmp_path / "hybrid.glb"
    output.write_bytes(b"existing")

    with pytest.raises(ValueError, match="overwrite"):
        fuse_face_anchor(
            anchor_mesh=anchor,
            views=views,
            output_path=output,
            settings=FaceHybridSettings(48, 7, 0.08),
        )
