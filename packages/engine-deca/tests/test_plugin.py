import json
from pathlib import Path

import numpy as np
import pytest
from asset_mania_engine_deca.plugin import (
    DecaPluginSettings,
    DecaPrediction,
    execute_deca_request,
    sample_uv_displacement,
    validate_deca_runtime,
)
from asset_mania_pipeline import (
    build_face_geometry_plugin_request,
    write_face_geometry_plugin_request,
)

REVISION = "a" * 40
CHECKPOINT = "b" * 64
FLAME = "c" * 64
RIGHTS = "d" * 64


def topology() -> np.ndarray:
    indices = np.arange(9976, dtype=np.int64)
    return np.stack([indices % 5023, (indices + 1) % 5023, (indices + 2) % 5023], axis=1)


def settings(tmp_path: Path) -> DecaPluginSettings:
    source_root = tmp_path / "deca-source"
    source_root.mkdir()
    (source_root / "decalib").mkdir()
    (source_root / "decalib" / "deca.py").write_text("official source marker", encoding="utf-8")
    checkpoint = tmp_path / "deca_model.tar"
    checkpoint.write_bytes(b"checkpoint")
    flame = tmp_path / "generic_model.pkl"
    flame.write_bytes(b"flame")
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    return DecaPluginSettings(
        source_root=source_root,
        isolated_home=isolated,
        checkpoint_path=checkpoint,
        flame_path=flame,
        revision=REVISION,
        checkpoint_sha256=CHECKPOINT,
        flame_sha256=FLAME,
    )


def request_files(tmp_path: Path):
    source = tmp_path / "private-source.png"
    source.write_bytes(b"authorized portrait")
    request = build_face_geometry_plugin_request(
        plugin="deca-local",
        profile="detail-displacement-v1",
        plugin_revision=REVISION,
        source_image=source,
        output_directory=tmp_path / "output",
        device="cuda",
        checkpoint_sha256=CHECKPOINT,
        topology="flame-2020-5023",
        face_rights_receipt_sha256=RIGHTS,
    )
    request_path = tmp_path / "request.json"
    write_face_geometry_plugin_request(request, request_path)
    return request, request_path, tmp_path / "result.json"


def fake_backend(_source: Path, _settings: DecaPluginSettings) -> DecaPrediction:
    vertices = np.zeros((5023, 3), dtype=np.float32)
    vertices[:, 0] = np.linspace(-0.08, 0.08, 5023)
    return DecaPrediction(
        vertices=vertices,
        faces=topology(),
        source_projection=np.zeros((5023, 2), dtype=np.float32),
        detail_displacement=np.full(5023, 0.0005, dtype=np.float32),
    )


def test_uv_displacement_sampling_has_fixed_orientation() -> None:
    displacement = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    uv = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]], dtype=np.float64)

    sampled = sample_uv_displacement(displacement, uv)

    assert np.allclose(sampled, [0.0, 3.0, 1.5])


def test_runtime_requires_exact_revision_and_digests(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    checkpoint = validate_deca_runtime(
        configured,
        revision_reader=lambda _path: REVISION,
        digest_reader=lambda path: CHECKPOINT if path == configured.checkpoint_path else FLAME,
    )
    assert checkpoint == configured.checkpoint_path
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        validate_deca_runtime(
            configured,
            revision_reader=lambda _path: REVISION,
            digest_reader=lambda _path: "e" * 64,
        )


def test_worker_writes_only_numeric_detail_geometry(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    request, request_path, result_path = request_files(tmp_path)

    exit_code = execute_deca_request(
        request_path,
        result_path,
        configured,
        backend=fake_backend,
        revision_reader=lambda _path: REVISION,
        digest_reader=lambda path: CHECKPOINT if path == configured.checkpoint_path else FLAME,
    )

    assert exit_code == 0
    with np.load(request.output_directory / "geometry.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {
            "vertices",
            "faces",
            "source_projection",
            "detail_displacement",
        }
        assert np.allclose(archive["detail_displacement"], 0.0005)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["ephemeral_identity_feature_used"] is False
    assert result["persisted_identity_feature_count"] == 0
    forbidden = ("*albedo*", "*texture*", "*landmark*", "*vis*", "*.obj", "*.mat")
    assert not [path for pattern in forbidden for path in tmp_path.rglob(pattern)]


def test_worker_rejects_nonfinite_displacement_before_output(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    request, request_path, result_path = request_files(tmp_path)

    def invalid_backend(_source: Path, _settings: DecaPluginSettings) -> DecaPrediction:
        prediction = fake_backend(_source, _settings)
        prediction.detail_displacement[0] = np.nan
        return prediction

    with pytest.raises(ValueError, match="non-finite"):
        execute_deca_request(
            request_path,
            result_path,
            configured,
            backend=invalid_backend,
            revision_reader=lambda _path: REVISION,
            digest_reader=lambda path: CHECKPOINT if path == configured.checkpoint_path else FLAME,
        )
    assert not request.output_directory.exists()
