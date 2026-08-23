"""Numeric validation and bounded fusion for local FLAME face geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_GEOMETRY_FIELDS = frozenset({"vertices", "faces", "source_projection", "detail_displacement"})


@dataclass(frozen=True, slots=True)
class FaceGeometryData:
    vertices: np.ndarray
    faces: np.ndarray
    source_projection: np.ndarray
    detail_displacement: np.ndarray


@dataclass(frozen=True, slots=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        return (
            np.asarray(points, dtype=np.float64) @ self.rotation.T * self.scale + self.translation
        )


@dataclass(frozen=True, slots=True)
class FaceGeometryMeasurements:
    vertex_count: int
    triangle_count: int
    non_manifold_edge_count: int
    winding_consistent: bool
    longest_extent_metres: float
    maximum_displacement_metres: float
    rms_displacement_metres: float
    face_displacement_coverage: float
    outside_face_displacement_count: int


def load_face_geometry(path: Path, *, expected_topology: np.ndarray) -> FaceGeometryData:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != _GEOMETRY_FIELDS:
                raise ValueError("face geometry archive inventory is invalid")
            vertices = np.asarray(archive["vertices"], dtype=np.float64).copy()
            faces = np.asarray(archive["faces"], dtype=np.int64).copy()
            projection = np.asarray(archive["source_projection"], dtype=np.float64).copy()
            displacement = np.asarray(archive["detail_displacement"], dtype=np.float64).copy()
    except (OSError, KeyError, ValueError) as error:
        if isinstance(error, ValueError) and "inventory" in str(error):
            raise
        raise ValueError("face geometry archive is unreadable") from error
    topology = np.asarray(expected_topology, dtype=np.int64)
    if vertices.shape != (5023, 3):
        raise ValueError("face geometry vertices must have shape (5023, 3)")
    if faces.shape != (9976, 3) or topology.shape != (9976, 3):
        raise ValueError("face geometry faces must have shape (9976, 3)")
    if not np.array_equal(faces, topology):
        raise ValueError("face geometry topology differs from the sealed topology")
    if projection.shape != (5023, 2):
        raise ValueError("face geometry source projection must have shape (5023, 2)")
    if displacement.shape != (5023,):
        raise ValueError("face geometry displacement must have shape (5023,)")
    if not all(np.isfinite(item).all() for item in (vertices, projection, displacement)):
        raise ValueError("face geometry contains non-finite values")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("face geometry topology index is out of range")
    if (
        np.any(faces[:, 0] == faces[:, 1])
        or np.any(faces[:, 1] == faces[:, 2])
        or np.any(faces[:, 2] == faces[:, 0])
    ):
        raise ValueError("face geometry contains a degenerate triangle")
    return FaceGeometryData(vertices, faces, projection, displacement)


def fit_similarity_transform(source: np.ndarray, target: np.ndarray) -> SimilarityTransform:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("similarity fit requires matching 3D point arrays")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("similarity fit contains non-finite values")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    variance = float(np.mean(np.sum(centered_source**2, axis=1)))
    if variance <= 1e-15:
        raise ValueError("similarity fit source has zero variance")
    covariance = centered_target.T @ centered_source / len(source)
    left, singular, right_transpose = np.linalg.svd(covariance)
    signs = np.ones(3, dtype=np.float64)
    if np.linalg.det(left @ right_transpose) < 0:
        signs[-1] = -1.0
    rotation = left @ np.diag(signs) @ right_transpose
    scale = float(np.sum(singular * signs) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("similarity fit scale is invalid")
    translation = target_mean - scale * (rotation @ source_mean)
    return SimilarityTransform(scale, rotation, translation)


def _adjacency(vertex_count: int, faces: np.ndarray) -> list[set[int]]:
    result = [set() for _ in range(vertex_count)]
    for triangle in np.asarray(faces, dtype=np.int64):
        a, b, c = map(int, triangle)
        result[a].update((b, c))
        result[b].update((a, c))
        result[c].update((a, b))
    return result


def build_face_taper(
    *, vertex_count: int, faces: np.ndarray, face_indices: np.ndarray
) -> np.ndarray:
    indices = {int(value) for value in np.asarray(face_indices, dtype=np.int64)}
    if not indices or min(indices) < 0 or max(indices) >= vertex_count:
        raise ValueError("face indices are invalid")
    adjacency = _adjacency(vertex_count, faces)
    taper = np.zeros(vertex_count, dtype=np.float64)
    taper[list(indices)] = 1.0
    first = {neighbor for index in indices for neighbor in adjacency[index]} - indices
    taper[list(first)] = 0.75
    second = {neighbor for index in first for neighbor in adjacency[index]} - indices - first
    taper[list(second)] = 0.25
    return taper


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-15
    normals[valid] /= lengths[valid, None]
    return normals


def _topology_measurements(faces: np.ndarray) -> tuple[int, bool]:
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    undirected = np.sort(directed, axis=1)
    unique, inverse, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    non_manifold = int(np.count_nonzero(counts > 2))
    winding = True
    for edge_index in range(len(unique)):
        occurrences = directed[inverse == edge_index]
        if len(occurrences) == 2 and np.array_equal(occurrences[0], occurrences[1]):
            winding = False
            break
    return non_manifold, winding


def fuse_mica_deca_geometry(
    *,
    mica: FaceGeometryData,
    deca: FaceGeometryData,
    face_indices: np.ndarray,
    inner_face_indices: np.ndarray,
) -> tuple[FaceGeometryData, FaceGeometryMeasurements]:
    if not np.array_equal(mica.faces, deca.faces):
        raise ValueError("MICA and DECA topology differs")
    inner = np.asarray(inner_face_indices, dtype=np.int64)
    if inner.ndim != 1 or len(inner) < 3 or inner.min() < 0 or inner.max() >= len(mica.vertices):
        raise ValueError("inner face indices are invalid")
    transform = fit_similarity_transform(deca.vertices[inner], mica.vertices[inner])
    aligned_displacement = np.asarray(deca.detail_displacement, dtype=np.float64) * transform.scale
    maximum = float(np.max(np.abs(aligned_displacement)))
    rms = float(np.sqrt(np.mean(aligned_displacement**2)))
    if maximum > 0.003:
        raise ValueError("maximum displacement exceeds 0.003 metres")
    if rms > 0.0015:
        raise ValueError("RMS displacement exceeds 0.0015 metres")
    taper = build_face_taper(
        vertex_count=len(mica.vertices), faces=mica.faces, face_indices=face_indices
    )
    tapered_region = taper > 0
    coverage = float(
        np.count_nonzero(np.isfinite(aligned_displacement[tapered_region]))
        / np.count_nonzero(tapered_region)
    )
    if coverage < 0.90:
        raise ValueError("face displacement coverage is below 0.90")
    applied = aligned_displacement * taper
    outside_count = int(np.count_nonzero(applied[~tapered_region]))
    if outside_count:
        raise ValueError("detail displacement escaped the tapered face region")
    normals = _vertex_normals(mica.vertices, mica.faces)
    vertices = mica.vertices + normals * applied[:, None]
    result = FaceGeometryData(
        vertices=vertices,
        faces=mica.faces.copy(),
        source_projection=mica.source_projection.copy(),
        detail_displacement=applied,
    )
    non_manifold, winding = _topology_measurements(result.faces)
    measurements = FaceGeometryMeasurements(
        vertex_count=len(vertices),
        triangle_count=len(result.faces),
        non_manifold_edge_count=non_manifold,
        winding_consistent=winding,
        longest_extent_metres=float(np.ptp(vertices, axis=0).max()),
        maximum_displacement_metres=maximum,
        rms_displacement_metres=rms,
        face_displacement_coverage=coverage,
        outside_face_displacement_count=outside_count,
    )
    return result, measurements
