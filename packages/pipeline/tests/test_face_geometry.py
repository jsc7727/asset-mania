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
    path = write_geometry(tmp_path / "geometry.npz", longest_extent_metres=longest_extent_metres)

    data = load_face_geometry(path, expected_topology=flame_topology())

    assert np.isclose(np.ptp(data.vertices, axis=0).max(), longest_extent_metres)


@pytest.mark.parametrize("longest_extent_metres", [0.149999, 0.320001, 0.324885711])
def test_geometry_archive_rejects_extent_outside_full_head_boundaries(
    tmp_path: Path, longest_extent_metres: float
) -> None:
    path = write_geometry(tmp_path / "geometry.npz", longest_extent_metres=longest_extent_metres)

    with pytest.raises(ValueError, match="between 0.15 and 0.32 metres"):
        load_face_geometry(path, expected_topology=flame_topology())


def test_geometry_archive_explicit_extent_opt_out_accepts_raw_deca_unchanged(
    tmp_path: Path,
) -> None:
    path = write_geometry(tmp_path / "deca.npz", longest_extent_metres=0.324885711)

    data = load_face_geometry(
        path,
        expected_topology=flame_topology(),
        validate_extent=False,
    )

    with np.load(path, allow_pickle=False) as archive:
        assert np.array_equal(data.vertices, archive["vertices"])


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_geometry_archive_extent_opt_out_still_rejects_zero_axis(tmp_path: Path, axis: int) -> None:
    path = write_geometry(tmp_path / f"zero-{axis}.npz")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {field: archive[field].copy() for field in archive.files}
    arrays["vertices"][:, axis] = 0.0
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="positive extent on every axis"):
        load_face_geometry(
            path,
            expected_topology=flame_topology(),
            validate_extent=False,
        )


def test_geometry_archive_extent_opt_out_still_rejects_nonfinite_vertices(tmp_path: Path) -> None:
    vertices = np.full((5023, 3), np.nan, dtype=np.float32)
    path = write_geometry(tmp_path / "nonfinite-deca.npz", extra={"vertices": vertices})

    with pytest.raises(ValueError, match="non-finite"):
        load_face_geometry(
            path,
            expected_topology=flame_topology(),
            validate_extent=False,
        )


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
    source = write_geometry(tmp_path / "source.npz")
    with np.load(source, allow_pickle=False) as archive:
        invalid_vertices = archive["vertices"].copy()
    first_triangle = flame_topology()[0]
    invalid_vertices[first_triangle[0]] = invalid_vertices[first_triangle[1]]
    path = write_geometry(tmp_path / "collinear.npz", extra={"vertices": invalid_vertices})

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


def grid_faces(size: int) -> np.ndarray:
    faces: list[tuple[int, int, int]] = []
    for row in range(size - 1):
        for column in range(size - 1):
            top_left = row * size + column
            top_right = top_left + 1
            bottom_left = top_left + size
            bottom_right = bottom_left + 1
            faces.extend(
                ((top_left, bottom_left, top_right), (top_right, bottom_left, bottom_right))
            )
    return np.asarray(faces, dtype=np.int64)


def test_face_taper_feathers_two_rings_inward_without_expanding_sealed_mask() -> None:
    size = 7
    faces = grid_faces(size)
    face_indices = np.array(
        [row * size + column for row in range(1, 6) for column in range(1, 6)],
        dtype=np.int64,
    )

    taper = build_face_taper(
        vertex_count=size * size,
        faces=faces,
        face_indices=face_indices,
    )

    expected = np.zeros((size, size), dtype=np.float64)
    expected[1:6, 1:6] = 0.25
    expected[2:5, 2:5] = 0.75
    expected[3, 3] = 1.0
    assert np.array_equal(taper.reshape(size, size), expected)


@pytest.mark.parametrize(
    "face_indices",
    [np.array([0.0, 1.0, 2.0]), np.array([0, 1, 1], dtype=np.int64)],
)
def test_face_taper_rejects_noninteger_or_duplicate_face_indices(
    face_indices: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="face indices are invalid"):
        build_face_taper(vertex_count=4, faces=np.array([[0, 1, 2]]), face_indices=face_indices)


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
        face_indices=np.arange(6),
        inner_face_indices=np.array([0, 1, 2, 3]),
    )

    assert np.array_equal(result.faces, mica.faces)
    assert np.allclose(result.vertices[6], mica.vertices[6])
    assert measured.maximum_displacement_metres <= 0.003
    assert measured.rms_displacement_metres <= 0.0015
    assert measured.outside_face_displacement_count == 0


def test_fusion_ignores_malicious_detail_outside_original_face_mask() -> None:
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
    deca = geometry_fixture(
        deca_vertices,
        np.array([0.002, 0.002, 0.002, 0.002, 0.002, 1000.0, np.nan]),
    )

    result, measured = fuse_mica_deca_geometry(
        mica=mica,
        deca=deca,
        face_indices=np.arange(5),
        inner_face_indices=np.array([0, 1, 2, 3]),
    )

    assert np.array_equal(result.vertices[5:], mica.vertices[5:])
    assert np.array_equal(result.detail_displacement[5:], np.zeros(2))
    assert measured.maximum_displacement_metres == pytest.approx(0.001)
    assert measured.rms_displacement_metres == pytest.approx(0.001)
    assert measured.face_displacement_coverage == 1.0
    assert measured.outside_face_displacement_count == 0


@pytest.mark.parametrize(
    ("face_indices", "inner_face_indices"),
    [
        (np.arange(6, dtype=np.float64), np.array([0, 1, 2])),
        (np.array([0, 1, 2, 3, 4, 5]), np.array([0.0, 1.0, 2.0])),
        (np.array([0, 1, 2, 3, 4, 5]), np.array([0, 1, 1])),
        (np.array([0, 1, 2, 3, 4, 5]), np.array([0, 1, 6])),
        (np.array([0, 1, 2]), np.array([0, 1, 2])),
    ],
)
def test_fusion_rejects_invalid_or_nonproper_face_fit_masks(
    face_indices: np.ndarray,
    inner_face_indices: np.ndarray,
) -> None:
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
    deca = geometry_fixture(vertices, np.zeros(6))

    with pytest.raises(ValueError, match="indices are invalid"):
        fuse_mica_deca_geometry(
            mica=mica,
            deca=deca,
            face_indices=face_indices,
            inner_face_indices=inner_face_indices,
        )


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
