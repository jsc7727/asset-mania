import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from asset_mania_engine_mica.plugin import (
    MicaPluginSettings,
    MicaPrediction,
    _directory_sha256,
    _official_backend,
    _require_checkpoint_keys,
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


def test_checkpoint_keys_fail_closed() -> None:
    with pytest.raises(ValueError, match="missing required model keys"):
        _require_checkpoint_keys({"arcface": {}})


def test_official_bridge_binds_sealed_flame_without_full_mica_renderer() -> None:
    source = inspect.getsource(_official_backend)
    assert "cfg.model.flame_model_path = str(settings.flame_path)" in source
    assert "Arcface" in source
    assert "Generator" in source
    assert "find_model_using_name" not in source


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
    detector = isolated / ".insightface" / "models" / "antelopev2"
    detector.mkdir(parents=True)
    for name in ("scrfd_10g_bnkps.onnx", "2d106det.onnx"):
        (detector / name).write_bytes(name.encode("utf-8"))
    return MicaPluginSettings(
        source_root=source_root,
        isolated_home=isolated,
        checkpoint_path=checkpoint,
        flame_path=flame,
        detector_path=detector,
        revision=REVISION,
        checkpoint_sha256=CHECKPOINT,
        flame_sha256=FLAME,
        detector_sha256=_directory_sha256(detector),
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
        coordinate_unit="metres",
    )


def file_digest_reader(configured: MicaPluginSettings):
    return {
        configured.checkpoint_path: CHECKPOINT,
        configured.flame_path: FLAME,
    }.__getitem__


def test_runtime_requires_exact_revision_and_digests(tmp_path: Path) -> None:
    configured = settings(tmp_path)

    checkpoint = validate_mica_runtime(
        configured,
        revision_reader=lambda _path: REVISION,
        digest_reader=file_digest_reader(configured),
        detector_digest_reader=_directory_sha256,
        clean_reader=lambda _path: True,
    )

    assert checkpoint == configured.checkpoint_path
    with pytest.raises(ValueError, match="revision mismatch"):
        validate_mica_runtime(
            configured,
            revision_reader=lambda _path: "e" * 40,
            digest_reader=file_digest_reader(configured),
            detector_digest_reader=_directory_sha256,
            clean_reader=lambda _path: True,
        )


def test_runtime_rejects_dirty_source_and_unsealed_detector(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    digests = {
        configured.checkpoint_path: CHECKPOINT,
        configured.flame_path: FLAME,
    }
    with pytest.raises(ValueError, match="source tree is not clean"):
        validate_mica_runtime(
            configured,
            revision_reader=lambda _path: REVISION,
            digest_reader=digests.__getitem__,
            detector_digest_reader=_directory_sha256,
            clean_reader=lambda _path: False,
        )
    configured.detector_path.joinpath("2d106det.onnx").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="detector asset digest mismatch"):
        validate_mica_runtime(
            configured,
            revision_reader=lambda _path: REVISION,
            digest_reader=digests.__getitem__,
            detector_digest_reader=_directory_sha256,
            clean_reader=lambda _path: True,
        )


def test_worker_rejects_non_metre_or_implausible_geometry(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    _request, request_path, result_path = request_files(tmp_path)
    prediction = fake_backend(Path(), configured)
    prediction = MicaPrediction(
        prediction.vertices * 10,
        prediction.faces,
        prediction.source_projection,
        "millimetres",
    )
    with pytest.raises(ValueError, match="explicitly use metres"):
        execute_mica_request(
            request_path,
            result_path,
            configured,
            backend=lambda *_args: prediction,
            revision_reader=lambda _path: REVISION,
            digest_reader=file_digest_reader(configured),
            detector_digest_reader=_directory_sha256,
            clean_reader=lambda _path: True,
        )


def test_worker_sanitizes_credentials_before_external_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = settings(tmp_path)
    _request, request_path, result_path = request_files(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-cross-boundary")
    monkeypatch.setenv("ASSET_MANIA_SAFE_FLAG", "present")

    def inspecting_backend(source: Path, runtime: MicaPluginSettings) -> MicaPrediction:
        assert "OPENAI_API_KEY" not in __import__("os").environ
        assert __import__("os").environ["ASSET_MANIA_SAFE_FLAG"] == "present"
        return fake_backend(source, runtime)

    execute_mica_request(
        request_path,
        result_path,
        configured,
        backend=inspecting_backend,
        revision_reader=lambda _path: REVISION,
        digest_reader=file_digest_reader(configured),
        detector_digest_reader=_directory_sha256,
        clean_reader=lambda _path: True,
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
        digest_reader=file_digest_reader(configured),
        detector_digest_reader=_directory_sha256,
        clean_reader=lambda _path: True,
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
            digest_reader=file_digest_reader(configured),
            detector_digest_reader=_directory_sha256,
            clean_reader=lambda _path: True,
        )


def test_worker_rejects_invalid_prediction_without_writing_output(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    request, request_path, result_path = request_files(tmp_path)

    def invalid_backend(_source: Path, _settings: MicaPluginSettings) -> MicaPrediction:
        return MicaPrediction(
            vertices=np.zeros((2, 3)),
            faces=np.array([[0, 1, 1]]),
            source_projection=np.zeros((2, 2)),
            coordinate_unit="metres",
        )

    with pytest.raises(ValueError, match="5,023 vertices"):
        execute_mica_request(
            request_path,
            result_path,
            configured,
            backend=invalid_backend,
            revision_reader=lambda _path: REVISION,
            digest_reader=file_digest_reader(configured),
            detector_digest_reader=_directory_sha256,
            clean_reader=lambda _path: True,
        )

    assert not request.output_directory.exists()
    assert not result_path.exists()
