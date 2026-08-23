from pathlib import Path

import numpy as np
import pytest
from asset_mania_pipeline import validate_glb
from asset_mania_pipeline.face_geometry import FaceGeometryData
from asset_mania_pipeline.face_geometry_glb import export_clay_glb


def tetrahedron() -> FaceGeometryData:
    vertices = np.array(
        [
            [-0.05, -0.05, 0.02],
            [0.05, -0.05, 0.02],
            [0.00, 0.06, 0.02],
            [0.00, 0.00, -0.08],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
    return FaceGeometryData(
        vertices=vertices,
        faces=faces,
        source_projection=np.zeros((4, 2), dtype=np.float64),
        detail_displacement=np.zeros(4, dtype=np.float64),
    )


def test_clay_glb_has_neutral_material_and_no_texture(tmp_path: Path) -> None:
    output = tmp_path / "clay.glb"

    measured = export_clay_glb(tetrahedron(), output)

    document = validate_glb(output).json_chunk
    assert measured.vertex_count == 4
    assert measured.triangle_count == 4
    assert document["materials"][0]["name"] == "Asset Mania neutral clay"
    assert "images" not in document
    assert "textures" not in document
    assert "baseColorTexture" not in document["materials"][0]["pbrMetallicRoughness"]
    assert all("uri" not in buffer for buffer in document["buffers"])


def test_clay_export_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "clay.glb"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_clay_glb(tetrahedron(), output)


def test_clay_export_rejects_nonfinite_geometry(tmp_path: Path) -> None:
    data = tetrahedron()
    data.vertices[0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        export_clay_glb(data, tmp_path / "clay.glb")
