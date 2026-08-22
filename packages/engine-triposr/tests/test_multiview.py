"""Yaw-aware normalization and voxel consensus for multi-view TripoSR meshes."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from asset_mania_engine_triposr import (
    FusionSettings,
    YawMesh,
    fuse_turntable_meshes,
    normalize_and_rotate,
    vote_occupancy,
)


def _rotate_about_z(vertices: np.ndarray, yaw: int) -> np.ndarray:
    radians = np.deg2rad(yaw)
    rotation = np.array(
        [
            [np.cos(radians), -np.sin(radians), 0.0],
            [np.sin(radians), np.cos(radians), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return vertices @ rotation.T


def _asymmetric_vertices() -> np.ndarray:
    return np.array(
        [
            [-2.0, -1.0, -0.5],
            [2.0, -1.0, -0.5],
            [2.0, 1.0, -0.5],
            [-2.0, 1.0, -0.5],
            [-1.5, -0.8, 1.5],
            [1.5, -0.8, 1.5],
            [1.5, 0.8, 1.5],
            [-1.5, 0.8, 1.5],
        ]
    )


def test_known_yaw_is_removed_before_consensus() -> None:
    vertices = _asymmetric_vertices()
    rotated = _rotate_about_z(vertices, 90)

    restored = normalize_and_rotate(rotated, yaw=90)
    reference = normalize_and_rotate(vertices, yaw=0)

    assert np.allclose(restored, reference, atol=1e-6)
    assert np.max(np.ptp(restored, axis=0)) == pytest.approx(1.0)
    assert np.mean(restored, axis=0) == pytest.approx([0.0, 0.0, 0.0])


def test_four_of_eight_votes_survive_one_outlier() -> None:
    base = np.zeros((8, 8, 8), dtype=bool)
    base[2:6, 2:6, 2:6] = True
    outlier = np.zeros_like(base)
    outlier[0:2, 0:2, 0:2] = True

    fused = vote_occupancy([base.copy() for _ in range(7)] + [outlier])

    assert np.array_equal(fused, base)


def test_six_inputs_default_to_three_votes() -> None:
    occupied = np.zeros((4, 4, 4), dtype=bool)
    occupied[1, 1, 1] = True
    empty = np.zeros_like(occupied)

    fused = vote_occupancy([occupied, occupied, occupied, empty, empty, empty])

    assert fused[1, 1, 1]


def test_fewer_than_six_or_mismatched_grids_are_refused() -> None:
    grid = np.zeros((4, 4, 4), dtype=bool)
    with pytest.raises(ValueError, match="six"):
        vote_occupancy([grid] * 5)
    with pytest.raises(ValueError, match="shape"):
        vote_occupancy([grid] * 5 + [np.zeros((5, 5, 5), dtype=bool)])


def _inputs(directory: Path) -> list[YawMesh]:
    result = []
    for index, yaw in enumerate((0, 45, 90, 135, 180, 225, 270, 315), start=1):
        path = directory / f"mesh-{index}.ply"
        path.write_bytes(f"mesh-{index}".encode())
        result.append(
            YawMesh(
                yaw=yaw,
                path=path,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    return result


def test_duplicate_yaw_or_existing_output_is_refused_before_mesh_load(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs[-1] = YawMesh(yaw=270, path=inputs[-1].path, sha256=inputs[-1].sha256)
    with pytest.raises(ValueError, match="yaws"):
        fuse_turntable_meshes(inputs, tmp_path / "fused.glb", FusionSettings(48))

    output = tmp_path / "exists.glb"
    output.write_bytes(b"existing")
    with pytest.raises(ValueError, match="overwrite"):
        fuse_turntable_meshes(_inputs(tmp_path), output, FusionSettings(48))


def test_fewer_than_six_closed_meshes_is_refused(tmp_path: Path, monkeypatch) -> None:
    import asset_mania_engine_triposr.multiview as module

    class FakeMesh:
        def __init__(self, closed: bool) -> None:
            self.is_watertight = closed
            self.is_winding_consistent = True
            self.vertices = _asymmetric_vertices()
            self.faces = np.array([[0, 1, 2]], dtype=int)

    states = iter([False, False, False, True, True, True, True, True])
    monkeypatch.setattr(module, "_load_mesh", lambda path: FakeMesh(next(states)))

    with pytest.raises(ValueError, match="six closed"):
        fuse_turntable_meshes(_inputs(tmp_path), tmp_path / "fused.glb", FusionSettings(48))


@pytest.mark.skipif(
    importlib.util.find_spec("torchmcubes") is None,
    reason="optional torchmcubes runtime is not installed in the workspace",
)
def test_eight_yaw_ellipsoids_fuse_to_a_closed_glb(tmp_path: Path) -> None:
    import trimesh

    inputs = []
    for index, yaw in enumerate((0, 45, 90, 135, 180, 225, 270, 315), start=1):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.45)
        mesh.vertices[:, 0] *= 0.82
        mesh.vertices[:, 2] *= 1.18
        radians = np.deg2rad(yaw)
        rotation = np.array(
            [
                [np.cos(radians), -np.sin(radians), 0.0],
                [np.sin(radians), np.cos(radians), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        mesh.vertices = mesh.vertices @ rotation.T
        path = tmp_path / f"ellipsoid-{index}.ply"
        mesh.export(path, file_type="ply")
        inputs.append(
            YawMesh(
                yaw=yaw,
                path=path,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )

    output = tmp_path / "fused.glb"
    result = fuse_turntable_meshes(inputs, output, FusionSettings(48))

    assert output.is_file()
    assert result.manifold == "closed"
    assert result.signed_volume > 0
    assert result.eligible_mesh_count == 8
    assert result.minimum_votes == 4
