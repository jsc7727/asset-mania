from pathlib import Path

import numpy as np
import pytest
from asset_mania_pipeline.face_geometry import (
    FaceGeometryData,
    build_face_taper,
    fit_similarity_transform,
    fuse_mica_deca_geometry,
    load_face_geometry,
)


def flame_topology() -> np.ndarray:
    # A consistently wound triangulated 68-gon with 4,955 interior insertions.
    # Inserting a vertex into one triangle replaces it with three triangles,
    # preserving a manifold disk while adding exactly two faces each time.
    faces = [[0, index, index + 1] for index in range(1, 67)]
    for vertex in range(68, 5023):
        a, b, c = faces.pop()
        faces.extend(((a, b, vertex), (b, c, vertex), (c, a, vertex)))
    return np.asarray(faces, dtype=np.int64)


def write_geometry(
    path: Path,
    *,
    faces: np.ndarray | None = None,
    extra: dict[str, np.ndarray] | None = None,
    longest_extent_metres: float | None = None,
) -> Path:
    x = np.linspace(-0.08, 0.08, 5023)
    vertices = np.stack([x, np.sin(x * 20) * 0.1, np.cos(x * 20) * 0.08], axis=1)
    if longest_extent_metres is not None:
        vertices *= longest_extent_metres / float(np.ptp(vertices, axis=0).max())
    arrays = {
        "vertices": vertices.astype(np.float32),
        "faces": flame_topology() if faces is None else faces,
        "source_projection": np.zeros((5023, 2), dtype=np.float32),
        "detail_displacement": np.zeros(5023, dtype=np.float32),
    }
    arrays.update(extra or {})
    np.savez_compressed(path, **arrays)
    return path


def test_numeric_geometry_requires_exact_flame_topology(tmp_path: Path) -> None:
    topology = flame_topology()
    path = write_geometry(tmp_path / "geometry.npz")

    data = load_face_geometry(path, expected_topology=topology)

    assert data.vertices.shape == (5023, 3)
    assert data.faces.shape == (9976, 3)
    assert data.source_projection.shape == (5023, 2)


@pytest.mark.parametrize("longest_extent_metres", [0.309499189, 0.32])
def test_geometry_archive_accepts_full_head_extent_boundaries(
    tmp_path: Path, longest_extent_metres: float
) -> None:
    path = write_geometry(
        tmp_path / "geometry.npz", longest_extent_metres=longest_extent_metres
    )

    data = load_face_geometry(path, expected_topology=flame_topology())

    assert np.isclose(np.ptp(data.vertices, axis=0).max(), longest_extent_metres)


@pytest.mark.parametrize("longest_extent_metres", [0.149999, 0.320001])
def test_geometry_archive_rejects_extent_outside_full_head_boundaries(
    tmp_path: Path, longest_extent_metres: float
) -> None:
    path = write_geometry(
        tmp_path / "geometry.npz", longest_extent_metres=longest_extent_metres
    )

    with pytest.raises(ValueError, match="between 0.15 and 0.32 metres"):
        load_face_geometry(path, expected_topology=flame_topology())


@pytest.mark.parametrize("bad_key", ["embedding", "landmarks", "crop", "shape_parameters"])
def test_geometry_archive_rejects_private_feature_fields(tmp_path: Path, bad_key: str) -> None:
    path = write_geometry(tmp_path / "geometry.npz", extra={bad_key: np.zeros(1)})

    with pytest.raises(ValueError, match="inventory"):
        load_face_geometry(path, expected_topology=flame_topology())


def test_geometry_archive_rejects_changed_topology_and_nonfinite_values(tmp_path: Path) -> None:
    changed = flame_topology()
    changed[0] = [2, 1, 0]
    path = write_geometry(tmp_path / "changed.npz", faces=changed)
    with pytest.raises(ValueError, match="topology differs"):
        load_face_geometry(path, expected_topology=flame_topology())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("faces", flame_topology().astype(np.float64)),
        ("vertices", np.zeros((5023, 3), dtype=np.int16)),
        ("source_projection", np.zeros((5023, 2), dtype=np.complex64)),
        ("detail_displacement", np.zeros(5023, dtype=np.uint8)),
    ],
)
def test_geometry_archive_rejects_unsupported_raw_dtypes(
    tmp_path: Path, field: str, value: np.ndarray
) -> None:
    path = write_geometry(tmp_path / f"bad-{field}.npz", extra={field: value})

    with pytest.raises(ValueError, match="dtype is unsupported"):
        load_face_geometry(path, expected_topology=flame_topology())


def test_geometry_archive_rejects_zero_area_triangle(tmp_path: Path) -> None:
    vertices = np.zeros((5023, 3), dtype=np.float32)
    vertices[:, 0] = np.linspace(-0.08, 0.08, 5023)
    path = write_geometry(tmp_path / "collinear.npz", extra={"vertices": vertices})

    with pytest.raises(ValueError, match="zero-area triangle"):
        load_face_geometry(path, expected_topology=flame_topology())

    path = write_geometry(
        tmp_path / "nonfinite.npz",
        extra={"detail_displacement": np.full(5023, np.nan, dtype=np.float32)},
    )
    with pytest.raises(ValueError, match="non-finite"):
        load_face_geometry(path, expected_topology=flame_topology())


def test_similarity_fit_recovers_known_metric_transform() -> None:
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    angle = np.deg2rad(25.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]]
    )
    target = source @ rotation.T * 1.07 + np.array([0.2, -0.4, 0.1])

    transform = fit_similarity_transform(source, target)

    assert np.max(np.abs(transform.apply(source) - target)) < 1e-9


def test_face_taper_uses_two_adjacency_rings() -> None:
    faces = np.array([[0, 1, 2], [2, 1, 3], [3, 1, 4], [4, 1, 5]], dtype=np.int64)

    taper = build_face_taper(vertex_count=7, faces=faces, face_indices=np.array([0, 2]))

    assert taper.tolist() == [1.0, 0.75, 1.0, 0.75, 0.25, 0.25, 0.0]


def geometry_fixture(vertices: np.ndarray, displacement: np.ndarray) -> FaceGeometryData:
    faces = np.array([[0, 1, 2], [2, 1, 3], [3, 1, 4], [4, 1, 5]], dtype=np.int64)
    return FaceGeometryData(
        vertices=vertices,
        faces=faces,
        source_projection=np.zeros((len(vertices), 2), dtype=np.float64),
        detail_displacement=displacement,
    )


def test_fusion_keeps_mica_positions_outside_taper_and_bounds_detail() -> None:
    mica_vertices = np.array(
        [
            [-0.08, 0.02, -0.01],
            [-0.03, 0.08, -0.02],
            [0.02, 0.09, -0.03],
            [0.07, 0.03, -0.02],
            [0.08, -0.04, 0.00],
            [0.02, -0.11, 0.01],
            [0.00, -0.14, 0.02],
        ]
    )
    deca_vertices = mica_vertices * 2.0 + np.array([0.5, -0.2, 0.1])
    mica = geometry_fixture(mica_vertices, np.zeros(7))
    deca = geometry_fixture(deca_vertices, np.full(7, 0.002))

    result, measured = fuse_mica_deca_geometry(
        mica=mica,
        deca=deca,
        face_indices=np.array([0, 2]),
        inner_face_indices=np.array([0, 1, 2, 3]),
    )

    assert np.array_equal(result.faces, mica.faces)
    assert np.allclose(result.vertices[6], mica.vertices[6])
    assert measured.maximum_displacement_metres <= 0.003
    assert measured.rms_displacement_metres <= 0.0015
    assert measured.outside_face_displacement_count == 0


@pytest.mark.parametrize(
    ("displacement", "message"),
    [
        (np.array([0.0031] * 6), "maximum displacement"),
        (np.array([0.0016] * 6), "RMS displacement"),
    ],
)
def test_fusion_rejects_out_of_bounds_detail(displacement: np.ndarray, message: str) -> None:
    vertices = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.1],
                [0.1, 0.1, 0.0],
                [0.1, 0.0, 0.1],
            ]
        )
        * 1.6
    )
    mica = geometry_fixture(vertices, np.zeros(6))
    deca = geometry_fixture(vertices, displacement)

    with pytest.raises(ValueError, match=message):
        fuse_mica_deca_geometry(
            mica=mica,
            deca=deca,
            face_indices=np.arange(6),
            inner_face_indices=np.array([0, 1, 2, 3]),
        )
