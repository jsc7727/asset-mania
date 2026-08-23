import json
from pathlib import Path

import numpy as np
import pytest
from asset_mania_engine_mica.plugin import (
    MicaPluginSettings,
    MicaPrediction,
    execute_mica_request,
    validate_mica_runtime,
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


def settings(tmp_path: Path) -> MicaPluginSettings:
    source_root = tmp_path / "mica-source"
    source_root.mkdir()
    (source_root / "demo.py").write_text("official source marker", encoding="utf-8")
    checkpoint = tmp_path / "mica.tar"
    checkpoint.write_bytes(b"checkpoint")
    flame = tmp_path / "flame.pkl"
    flame.write_bytes(b"flame")
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    return MicaPluginSettings(
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
        plugin="mica-local",
        profile="identity-neutral-v1",
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


def fake_backend(_source: Path, _settings: MicaPluginSettings) -> MicaPrediction:
    vertices = np.zeros((5023, 3), dtype=np.float32)
    vertices[:, 0] = np.linspace(-0.08, 0.08, 5023)
    return MicaPrediction(
        vertices=vertices,
        faces=topology(),
        source_projection=np.zeros((5023, 2), dtype=np.float32),
    )


def test_runtime_requires_exact_revision_and_digests(tmp_path: Path) -> None:
    configured = settings(tmp_path)

    checkpoint = validate_mica_runtime(
        configured,
        revision_reader=lambda _path: REVISION,
        digest_reader=lambda path: CHECKPOINT if path == configured.checkpoint_path else FLAME,
    )

    assert checkpoint == configured.checkpoint_path
    with pytest.raises(ValueError, match="revision mismatch"):
        validate_mica_runtime(
            configured,
            revision_reader=lambda _path: "e" * 40,
            digest_reader=lambda path: CHECKPOINT if path == configured.checkpoint_path else FLAME,
        )


def test_worker_persists_geometry_but_no_identity_feature(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    request, request_path, result_path = request_files(tmp_path)

    exit_code = execute_mica_request(
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
        assert np.count_nonzero(archive["detail_displacement"]) == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["ephemeral_identity_feature_used"] is True
    assert result["persisted_identity_feature_count"] == 0
    assert not list(tmp_path.rglob("identity*.npy"))
    assert not list(tmp_path.rglob("*crop*.png"))


def test_worker_rejects_wrong_plugin_and_existing_output(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    _request, request_path, result_path = request_files(tmp_path)
    document = json.loads(request_path.read_text(encoding="utf-8"))
    document["plugin"] = "deca-local"
    document["profile"] = "detail-displacement-v1"
    request_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="request differs from MICA"):
        execute_mica_request(
            request_path,
            result_path,
            configured,
            backend=fake_backend,
            revision_reader=lambda _path: REVISION,
            digest_reader=lambda path: CHECKPOINT if path == configured.checkpoint_path else FLAME,
        )


def test_worker_rejects_invalid_prediction_without_writing_output(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    request, request_path, result_path = request_files(tmp_path)

    def invalid_backend(_source: Path, _settings: MicaPluginSettings) -> MicaPrediction:
        return MicaPrediction(
            vertices=np.zeros((2, 3)),
            faces=np.array([[0, 1, 1]]),
            source_projection=np.zeros((2, 2)),
        )

    with pytest.raises(ValueError, match="5,023 vertices"):
        execute_mica_request(
            request_path,
            result_path,
            configured,
            backend=invalid_backend,
            revision_reader=lambda _path: REVISION,
            digest_reader=lambda path: CHECKPOINT if path == configured.checkpoint_path else FLAME,
        )

    assert not request.output_directory.exists()
    assert not result_path.exists()
