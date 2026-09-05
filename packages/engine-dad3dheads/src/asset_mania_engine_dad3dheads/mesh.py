"""Deterministic validation and GLB conversion for DAD full-head OBJ output."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

_AREA_EPSILON = 1e-12
_NEUTRAL_RGBA = np.array([160, 145, 140, 255], dtype=np.uint8)
# Neutral FLAME is +Y up and +Z toward the viewer. Preserve X and Y while
# reflecting Z to glTF's canonical front; callers reverse triangle winding to
# compensate for this handedness change.
_DAD_TO_BLENDER = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
# Explicit compatibility path for pre-neutral-worker OBJ files, where -Y was
# treated as image-up. This proper rotation does not require a winding change.
_POSED_DAD_TO_GLTF = np.array(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
)


@dataclass(frozen=True, slots=True)
class DADMeshMeasurements:
    vertex_count: int
    triangle_count: int
    component_count: int
    boundary_edge_count: int
    boundary_loop_count: int
    non_manifold_edge_count: int
    winding_consistent: bool
    signed_volume: float | None
    observed_color_coverage: float


@contextmanager
def _trimesh_numpy_two_compatibility():
    original_allclose = trimesh.util.allclose

    def numpy_two_allclose(left, right, atol=1e-8):
        return float(np.ptp(np.asarray(left) - np.asarray(right))) < atol

    trimesh.util.allclose = numpy_two_allclose
    try:
        yield
    finally:
        trimesh.util.allclose = original_allclose


def _load_mesh(path: Path) -> trimesh.Trimesh:
    with _trimesh_numpy_two_compatibility():
        loaded = trimesh.load(str(path), process=False, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError("DAD mesh must contain exactly one geometry")
    return loaded


def _edge_counts(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.sort(
        np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0), axis=1
    )
    return np.unique(edges, axis=0, return_counts=True)


def _component_face_counts(vertex_count: int, faces: np.ndarray) -> list[int]:
    parent = list(range(vertex_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    used: set[int] = set()
    for face in faces:
        a, b, c = (int(value) for value in face)
        used.update((a, b, c))
        union(a, b)
        union(b, c)
    counts: dict[int, int] = {}
    for face in faces:
        root = find(int(face[0]))
        counts[root] = counts.get(root, 0) + 1
    return sorted(counts.values(), reverse=True)


def _boundary_loop_count(boundary_edges: np.ndarray) -> int:
    if len(boundary_edges) == 0:
        return 0
    adjacency: dict[int, set[int]] = {}
    for left, right in boundary_edges:
        adjacency.setdefault(int(left), set()).add(int(right))
        adjacency.setdefault(int(right), set()).add(int(left))
    remaining = set(adjacency)
    components = 0
    while remaining:
        components += 1
        pending = [remaining.pop()]
        while pending:
            current = pending.pop()
            for neighbour in adjacency[current]:
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    pending.append(neighbour)
    return components


def _measure(mesh: trimesh.Trimesh, *, observed_color_coverage: float) -> DADMeshMeasurements:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("DAD mesh vertices are invalid")
    if not np.isfinite(vertices).all():
        raise ValueError("DAD mesh vertices must be finite")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("DAD mesh triangles are invalid")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("DAD mesh face index is out of range")
    vectors_a = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    vectors_b = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    areas = np.linalg.norm(np.cross(vectors_a, vectors_b), axis=1) * 0.5
    if np.count_nonzero(areas > _AREA_EPSILON) != len(areas):
        raise ValueError("DAD mesh contains zero-area triangles")
    component_faces = _component_face_counts(len(vertices), faces)
    component_count = len(component_faces)
    fixed_eye_shells = (
        component_count == 3
        and component_faces[1] == component_faces[2]
        and component_faces[0] >= 2 * component_faces[1]
    )
    if component_count != 1 and not fixed_eye_shells:
        raise ValueError("DAD mesh has an unexpected fixed component topology")
    edges, counts = _edge_counts(faces)
    boundary_edges = edges[counts == 1]
    non_manifold = int(np.count_nonzero(counts > 2))
    if non_manifold:
        raise ValueError("DAD mesh contains non-manifold edges")
    signed_volume = float(mesh.volume) if len(boundary_edges) == 0 else None
    return DADMeshMeasurements(
        vertex_count=len(vertices),
        triangle_count=len(faces),
        component_count=component_count,
        boundary_edge_count=len(boundary_edges),
        boundary_loop_count=_boundary_loop_count(boundary_edges),
        non_manifold_edge_count=non_manifold,
        winding_consistent=bool(mesh.is_winding_consistent),
        signed_volume=signed_volume,
        observed_color_coverage=observed_color_coverage,
    )


def inspect_dad_mesh(path: Path) -> DADMeshMeasurements:
    return _measure(_load_mesh(path), observed_color_coverage=0.0)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > _AREA_EPSILON
    normals[valid] /= lengths[valid, None]
    return normals


def _load_projection(
    path: Path, vertex_count: int
) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            projected = np.asarray(archive["projected_vertices"], dtype=np.float64)
            camera_vertices = (
                np.asarray(archive["camera_vertices"], dtype=np.float64)
                if "camera_vertices" in archive.files
                else None
            )
            image_shape = np.asarray(archive["image_shape"], dtype=np.int64)
    except (OSError, KeyError, ValueError) as error:
        raise ValueError("DAD projection data is invalid") from error
    if projected.shape != (vertex_count, 2) or not np.isfinite(projected).all():
        raise ValueError("DAD projected vertices are invalid")
    if camera_vertices is not None and (
        camera_vertices.shape != (vertex_count, 3) or not np.isfinite(camera_vertices).all()
    ):
        raise ValueError("DAD camera vertices are invalid")
    if image_shape.shape != (2,) or np.any(image_shape <= 0):
        raise ValueError("DAD projection image shape is invalid")
    return projected, camera_vertices, (int(image_shape[0]), int(image_shape[1]))


def _export_create_only(mesh: trimesh.Trimesh, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with _trimesh_numpy_two_compatibility():
        mesh.export(str(path), file_type="glb")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("DAD GLB export failed")
    _measure(_load_mesh(path), observed_color_coverage=0.0)


def convert_dad_mesh(
    *,
    obj_path: Path,
    projection_path: Path,
    source_image: Path,
    plain_glb: Path,
    colored_glb: Path,
    geometry_pose: str = "neutral",
) -> DADMeshMeasurements:
    if geometry_pose not in {"neutral", "posed"}:
        raise ValueError("DAD geometry pose must be neutral or posed")
    if plain_glb.exists():
        raise FileExistsError(f"refusing to overwrite {plain_glb}")
    if colored_glb.exists():
        raise FileExistsError(f"refusing to overwrite {colored_glb}")
    source_mesh = _load_mesh(obj_path)
    measurements = _measure(source_mesh, observed_color_coverage=0.0)
    original_vertices = np.asarray(source_mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(source_mesh.faces, dtype=np.int64).copy()
    projected, camera_vertices, image_shape = _load_projection(
        projection_path, len(original_vertices)
    )
    with Image.open(source_image) as opened:
        image = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    if image.shape[:2] != image_shape:
        raise ValueError("source image differs from DAD projection dimensions")

    center = (original_vertices.min(axis=0) + original_vertices.max(axis=0)) * 0.5
    centered = original_vertices - center
    extent = np.max(np.max(centered, axis=0) - np.min(centered, axis=0))
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("DAD mesh extent is invalid")
    # Unit longest extent is a relative-shape convention for consistent framing,
    # not metric scale and not direct size comparability with MICA or DECA.
    transform = _DAD_TO_BLENDER if geometry_pose == "neutral" else _POSED_DAD_TO_GLTF
    transformed = (centered / extent) @ transform.T
    transformed_faces = faces[:, [0, 2, 1]] if geometry_pose == "neutral" else faces
    plain_mesh = trimesh.Trimesh(vertices=transformed, faces=transformed_faces, process=False)
    _export_create_only(plain_mesh, plain_glb)

    visibility_vertices = camera_vertices if camera_vertices is not None else original_vertices
    normals = _vertex_normals(visibility_vertices, faces)
    rounded = np.rint(projected).astype(np.int64)
    height, width = image_shape
    valid = (
        (normals[:, 2] > 0)
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    colors = np.repeat(_NEUTRAL_RGBA[None, :], len(original_vertices), axis=0)
    colors[valid, :3] = image[rounded[valid, 1], rounded[valid, 0]]
    colored_mesh = trimesh.Trimesh(vertices=transformed, faces=transformed_faces, process=False)
    colored_mesh.visual.vertex_colors = colors
    _export_create_only(colored_mesh, colored_glb)
    coverage = float(np.count_nonzero(valid) / len(valid))
    return replace(measurements, observed_color_coverage=coverage)
