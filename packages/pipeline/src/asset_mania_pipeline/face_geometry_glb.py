"""Neutral, texture-free glTF export for validated face geometry."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from .containers import validate_glb
from .face_geometry import FaceGeometryData, FaceGeometryMeasurements, _topology_measurements


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


def export_clay_glb(data: FaceGeometryData, output_path: Path) -> FaceGeometryMeasurements:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    vertices = np.asarray(data.vertices, dtype=np.float64)
    faces = np.asarray(data.faces, dtype=np.int64)
    displacement = np.asarray(data.detail_displacement, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("clay geometry must contain triangle positions")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("clay geometry must be non-empty")
    if not np.isfinite(vertices).all() or not np.isfinite(displacement).all():
        raise ValueError("clay geometry contains non-finite values")
    if displacement.shape != (len(vertices),):
        raise ValueError("clay displacement must match the vertex count")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("clay face index is out of range")
    material = PBRMaterial(
        name="Asset Mania neutral clay",
        baseColorFactor=[0.62, 0.62, 0.64, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.55,
        alphaMode="OPAQUE",
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=TextureVisuals(material=material),
        process=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _trimesh_numpy_two_compatibility():
        mesh.export(str(output_path), file_type="glb")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise ValueError("clay GLB export failed")
    document = validate_glb(output_path).json_chunk
    if document.get("images") or document.get("textures"):
        raise ValueError("clay GLB must not contain a texture")
    if any("uri" in buffer for buffer in document.get("buffers", [])):
        raise ValueError("clay GLB must not contain an external buffer")
    non_manifold, winding = _topology_measurements(faces)
    maximum = float(np.max(np.abs(displacement)))
    rms = float(np.sqrt(np.mean(displacement**2)))
    return FaceGeometryMeasurements(
        vertex_count=len(vertices),
        triangle_count=len(faces),
        non_manifold_edge_count=non_manifold,
        winding_consistent=winding,
        longest_extent_metres=float(np.ptp(vertices, axis=0).max()),
        maximum_displacement_metres=maximum,
        rms_displacement_metres=rms,
        face_displacement_coverage=float(
            np.count_nonzero(np.isfinite(displacement)) / len(vertices)
        ),
        outside_face_displacement_count=0,
    )
