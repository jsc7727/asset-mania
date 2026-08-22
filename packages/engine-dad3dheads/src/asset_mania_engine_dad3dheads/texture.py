"""Visibility-aware multi-view texture selection for fixed-topology DAD heads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
from asset_mania_pipeline import validate_glb
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from .mesh import (
    _DAD_TO_BLENDER,
    _load_mesh,
    _trimesh_numpy_two_compatibility,
    _vertex_normals,
    inspect_dad_mesh,
)

DAD_TEXTURE_YAWS = (0, 45, 90, 135, 180, 225, 270, 315)
ATLAS_TILE_SIZE = 512
ATLAS_SIZE = 1536
NEUTRAL_TILE = 8
_NEUTRAL_RGB = (160, 145, 140)
_TILE_BY_YAW = {yaw: index for index, yaw in enumerate(DAD_TEXTURE_YAWS)}


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


@dataclass(frozen=True, slots=True)
class DADTextureMeasurements:
    triangle_count: int
    vertex_count: int
    textured_triangle_fraction: float
    textured_surface_area_fraction: float
    observed_face_area_fraction: float
    generated_surface_area_fraction: float
    neutral_surface_area_fraction: float
    yaw_triangle_counts: dict[int, int]
    back_projection_violation_count: int
    non_manifold_edge_count: int
    winding_consistent: bool


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


def build_texture_atlas(views: list[DADTextureView], output_path: Path) -> Image.Image:
    if [item.yaw for item in views] != list(DAD_TEXTURE_YAWS):
        raise ValueError(f"texture views must use ordered yaws {list(DAD_TEXTURE_YAWS)}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    atlas = Image.new("RGB", (ATLAS_SIZE, ATLAS_SIZE), _NEUTRAL_RGB)
    for tile, view in enumerate(views):
        try:
            with Image.open(view.image_path) as opened:
                opened.load()
                if opened.size != (1024, 1024):
                    raise ValueError(f"yaw {view.yaw} texture image must be 1024 square")
                resized = opened.convert("RGB").resize(
                    (ATLAS_TILE_SIZE, ATLAS_TILE_SIZE), Image.Resampling.LANCZOS
                )
        except OSError as error:
            raise ValueError(f"yaw {view.yaw} texture image is unreadable") from error
        row, column = divmod(tile, 3)
        atlas.paste(resized, (column * ATLAS_TILE_SIZE, row * ATLAS_TILE_SIZE))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(output_path, format="PNG", compress_level=9)
    return atlas


def _tile_uv(projected: np.ndarray, yaw: int, image_shape: tuple[int, int]) -> np.ndarray:
    if yaw == -1:
        row, column = divmod(NEUTRAL_TILE, 3)
        return np.array(
            [
                (column * ATLAS_TILE_SIZE + ATLAS_TILE_SIZE * 0.5) / ATLAS_SIZE,
                1.0 - (row * ATLAS_TILE_SIZE + ATLAS_TILE_SIZE * 0.5) / ATLAS_SIZE,
            ]
        )
    tile = _TILE_BY_YAW[yaw]
    row, column = divmod(tile, 3)
    height, width = image_shape
    local_x = np.clip(projected[0] * ATLAS_TILE_SIZE / width, 2.0, ATLAS_TILE_SIZE - 3.0)
    local_y = np.clip(projected[1] * ATLAS_TILE_SIZE / height, 2.0, ATLAS_TILE_SIZE - 3.0)
    return np.array(
        [
            (column * ATLAS_TILE_SIZE + local_x + 0.5) / ATLAS_SIZE,
            1.0 - (row * ATLAS_TILE_SIZE + local_y + 0.5) / ATLAS_SIZE,
        ]
    )


def _remap_texture_seams(
    vertices: np.ndarray,
    normals: np.ndarray,
    faces: np.ndarray,
    assignments: np.ndarray,
    projections: dict[int, np.ndarray],
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions: list[np.ndarray] = []
    smooth_normals: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    remapped_faces = np.empty_like(faces)
    indices: dict[tuple[int, int], int] = {}
    for face_index, triangle in enumerate(faces):
        yaw = int(assignments[face_index])
        for corner, original_index_value in enumerate(triangle):
            original_index = int(original_index_value)
            key = (original_index, yaw)
            output_index = indices.get(key)
            if output_index is None:
                output_index = len(positions)
                indices[key] = output_index
                positions.append(vertices[original_index])
                smooth_normals.append(normals[original_index])
                projected = (
                    np.array([image_shape[1] * 0.5, image_shape[0] * 0.5])
                    if yaw == -1
                    else projections[yaw][original_index]
                )
                uvs.append(_tile_uv(projected, yaw, image_shape))
            remapped_faces[face_index, corner] = output_index
    return (
        np.asarray(positions, dtype=np.float64),
        np.asarray(smooth_normals, dtype=np.float64),
        remapped_faces,
        np.asarray(uvs, dtype=np.float64),
    )


def _validate_embedded_texture_glb(path: Path) -> None:
    document = validate_glb(path).json_chunk
    images = document.get("images", [])
    textures = document.get("textures", [])
    materials = document.get("materials", [])
    if len(images) != 1 or "uri" in images[0] or "bufferView" not in images[0]:
        raise ValueError("textured DAD GLB must embed exactly one image")
    if len(textures) != 1 or len(materials) != 1:
        raise ValueError("textured DAD GLB must contain one texture and material")
    reference = materials[0].get("pbrMetallicRoughness", {}).get("baseColorTexture", {})
    if reference.get("index") != 0:
        raise ValueError("textured DAD material does not reference the embedded texture")
    primitives = [
        primitive for mesh in document.get("meshes", []) for primitive in mesh.get("primitives", [])
    ]
    if not primitives or any("TEXCOORD_0" not in item.get("attributes", {}) for item in primitives):
        raise ValueError("textured DAD GLB has no TEXCOORD_0")


def build_textured_dad_glb(
    *,
    geometry_obj: Path,
    views: list[DADTextureView],
    face_indices: np.ndarray,
    atlas_path: Path,
    output_path: Path,
    visibility_resolution: int = 512,
    enforce_gates: bool = True,
) -> DADTextureMeasurements:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if [item.yaw for item in views] != list(DAD_TEXTURE_YAWS):
        raise ValueError(f"texture views must use ordered yaws {list(DAD_TEXTURE_YAWS)}")
    source_mesh = _load_mesh(geometry_obj)
    source_vertices = np.asarray(source_mesh.vertices, dtype=np.float64)
    faces = np.asarray(source_mesh.faces, dtype=np.int64)
    view_arrays = {view.yaw: _load_view_arrays(view) for view in views}
    if any(len(arrays[0]) != len(source_vertices) for arrays in view_arrays.values()):
        raise ValueError("DAD texture view topology differs from observed geometry")
    visibilities = [
        compute_view_visibility(view, faces, resolution=visibility_resolution) for view in views
    ]
    assignments = select_triangle_views(visibilities, faces, face_indices)
    atlas = build_texture_atlas(views, atlas_path)
    center = (source_vertices.min(axis=0) + source_vertices.max(axis=0)) * 0.5
    centered = source_vertices - center
    extent = float(np.ptp(centered, axis=0).max())
    if extent <= 0:
        raise ValueError("DAD texture geometry extent is invalid")
    transformed = (centered / extent) @ _DAD_TO_BLENDER.T
    source_normals = _vertex_normals(source_vertices, faces) @ _DAD_TO_BLENDER.T
    normal_lengths = np.linalg.norm(source_normals, axis=1)
    source_normals[normal_lengths > 0] /= normal_lengths[normal_lengths > 0, None]
    projections = {yaw: arrays[0] for yaw, arrays in view_arrays.items()}
    vertices, normals, remapped_faces, uvs = _remap_texture_seams(
        transformed,
        source_normals,
        faces,
        assignments,
        projections,
        (1024, 1024),
    )
    material = PBRMaterial(
        name="DAD multi-view texture",
        baseColorTexture=atlas,
        metallicFactor=0.0,
        roughnessFactor=0.65,
        alphaMode="OPAQUE",
    )
    visual = TextureVisuals(uv=uvs, material=material)
    textured = trimesh.Trimesh(
        vertices=vertices,
        faces=remapped_faces,
        visual=visual,
        process=False,
    )
    textured._cache["vertex_normals"] = normals
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _trimesh_numpy_two_compatibility():
        textured.export(str(output_path), file_type="glb")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise ValueError("textured DAD GLB export failed")
    _validate_embedded_texture_glb(output_path)

    area_vectors_a = transformed[faces[:, 1]] - transformed[faces[:, 0]]
    area_vectors_b = transformed[faces[:, 2]] - transformed[faces[:, 0]]
    areas = np.linalg.norm(np.cross(area_vectors_a, area_vectors_b), axis=1) * 0.5
    total_area = float(areas.sum())
    textured_mask = assignments != -1
    generated_mask = (assignments != -1) & (assignments != 0)
    neutral_area = float(areas[~textured_mask].sum() / total_area)
    face_set = {int(index) for index in np.asarray(face_indices, dtype=np.int64)}
    indexed_face = np.asarray(
        [sum(int(vertex) in face_set for vertex in triangle) >= 2 for triangle in faces], dtype=bool
    )
    yaw0_eligible = visibilities[0].eligible & indexed_face
    eligible_face_area = float(areas[yaw0_eligible].sum())
    observed_face_area = (
        float(areas[yaw0_eligible & (assignments == 0)].sum() / eligible_face_area)
        if eligible_face_area > 0
        else 0.0
    )
    visibility_by_yaw = {item.yaw: item for item in visibilities}
    violations = sum(
        1
        for index, yaw in enumerate(assignments)
        if yaw != -1 and not bool(visibility_by_yaw[int(yaw)].eligible[index])
    )
    counts = {yaw: int(np.count_nonzero(assignments == yaw)) for yaw in DAD_TEXTURE_YAWS}
    source_measurements = inspect_dad_mesh(geometry_obj)
    result = DADTextureMeasurements(
        triangle_count=len(faces),
        vertex_count=len(vertices),
        textured_triangle_fraction=float(np.count_nonzero(textured_mask) / len(faces)),
        textured_surface_area_fraction=1.0 - neutral_area,
        observed_face_area_fraction=observed_face_area,
        generated_surface_area_fraction=float(areas[generated_mask].sum() / total_area),
        neutral_surface_area_fraction=neutral_area,
        yaw_triangle_counts=counts,
        back_projection_violation_count=violations,
        non_manifold_edge_count=source_measurements.non_manifold_edge_count,
        winding_consistent=source_measurements.winding_consistent,
    )
    if enforce_gates:
        if result.textured_triangle_fraction < 0.80:
            raise ValueError("textured triangle coverage gate failed")
        if result.textured_surface_area_fraction < 0.85:
            raise ValueError("textured surface area gate failed")
        if result.observed_face_area_fraction < 0.75:
            raise ValueError("observed face area gate failed")
        if result.neutral_surface_area_fraction > 0.15:
            raise ValueError("neutral surface area gate failed")
        if result.back_projection_violation_count:
            raise ValueError("back projection gate failed")
        if any(counts[yaw] == 0 for yaw in DAD_TEXTURE_YAWS):
            raise ValueError("every DAD texture yaw must own at least one triangle")
        if result.vertex_count > len(source_vertices) * 8:
            raise ValueError("DAD texture seam vertex limit exceeded")
        if result.non_manifold_edge_count or not result.winding_consistent:
            raise ValueError("DAD texture source topology gate failed")
    return result
