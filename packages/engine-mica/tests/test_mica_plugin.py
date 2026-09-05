import ast
import json
import shutil
import socket
import subprocess
import sys
import types
from pathlib import Path

import asset_mania_engine_mica.plugin as mica_plugin
import numpy as np
import pytest
from asset_mania_engine_mica.plugin import (
    MicaPluginSettings,
    MicaPrediction,
    _bind_mica_model_assets,
    _directory_sha256,
    _require_checkpoint_keys,
    _restore_chumpy_numpy_aliases,
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


def test_official_backend_rejects_cpu_before_reading_source(monkeypatch, tmp_path: Path) -> None:
    cv2 = types.SimpleNamespace(
        IMREAD_COLOR=1,
        imread=lambda *_args: pytest.fail("source must not be read before CUDA validation"),
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    with pytest.raises(ValueError, match="CUDA unavailable"):
        mica_plugin._official_backend(
            tmp_path / "source.png",
            settings(tmp_path),
            cuda_validator=lambda _torch: (_ for _ in ()).throw(ValueError("CUDA unavailable")),
        )


def test_official_backend_rejects_cpu_detector_before_reading_source(
    monkeypatch, tmp_path: Path
) -> None:
    configured = settings(tmp_path)

    class RetinaFace:
        session = types.SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])

    cv2 = types.SimpleNamespace(
        IMREAD_COLOR=1,
        imread=lambda *_args: pytest.fail("source must not be read before detector validation"),
    )
    insightface = types.ModuleType("insightface")
    insightface.model_zoo = types.SimpleNamespace(get_model=lambda *_args, **_kwargs: RetinaFace())
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "insightface", insightface)

    with pytest.raises(ValueError, match="CUDA execution provider"):
        mica_plugin._official_backend(
            tmp_path / "source.png", configured, cuda_validator=lambda _torch: None
        )


@pytest.mark.parametrize("alias", ["bool", "int", "float", "complex", "object", "unicode", "str"])
def test_chumpy_numpy_compatibility_preserves_each_existing_legacy_alias(alias: str) -> None:
    sentinel = object()
    numpy_module = types.SimpleNamespace(**{alias: sentinel})

    _restore_chumpy_numpy_aliases(numpy_module)

    assert getattr(numpy_module, alias) is sentinel


def test_chumpy_numpy_compatibility_is_idempotent() -> None:
    numpy_module = types.SimpleNamespace()
    _restore_chumpy_numpy_aliases(numpy_module)
    restored = dict(vars(numpy_module))

    _restore_chumpy_numpy_aliases(numpy_module)

    assert vars(numpy_module) == restored


def test_mica_model_assets_bind_exact_clean_revision_paths(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    flame_data = configured.source_root / "data" / "FLAME2020"
    flame_data.mkdir(parents=True)
    landmark_embedding = flame_data / "landmark_embedding.npy"
    head_template = flame_data / "head_template.obj"
    landmark_embedding.write_bytes(b"tracked landmarks")
    head_template.write_bytes(b"tracked topology")
    cfg = types.SimpleNamespace(model=types.SimpleNamespace())

    configured = MicaPluginSettings(
        source_root=configured.source_root / ".." / configured.source_root.name,
        isolated_home=configured.isolated_home,
        checkpoint_path=configured.checkpoint_path,
        flame_path=configured.flame_path.parent / "." / configured.flame_path.name,
        detector_path=configured.detector_path,
        revision=configured.revision,
        checkpoint_sha256=configured.checkpoint_sha256,
        flame_sha256=configured.flame_sha256,
        detector_sha256=configured.detector_sha256,
    )

    _bind_mica_model_assets(cfg, configured)

    assert cfg.model.flame_model_path == str(configured.flame_path.resolve(strict=True))
    assert cfg.model.flame_lmk_embedding_path == str(landmark_embedding.resolve(strict=True))
    assert cfg.model.topology_path == str(head_template.resolve(strict=True))


def test_mica_model_assets_reject_missing_sealed_flame(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    configured.flame_path.unlink()
    flame_data = configured.source_root / "data" / "FLAME2020"
    flame_data.mkdir(parents=True)
    (flame_data / "landmark_embedding.npy").write_bytes(b"tracked landmarks")
    (flame_data / "head_template.obj").write_bytes(b"tracked topology")
    cfg = types.SimpleNamespace(model=types.SimpleNamespace())

    with pytest.raises(ValueError, match="user-supplied FLAME asset is unavailable"):
        _bind_mica_model_assets(cfg, configured)


@pytest.mark.parametrize("missing_name", ["landmark_embedding.npy", "head_template.obj"])
def test_mica_model_assets_reject_missing_tracked_assets(tmp_path: Path, missing_name: str) -> None:
    configured = settings(tmp_path)
    flame_data = configured.source_root / "data" / "FLAME2020"
    flame_data.mkdir(parents=True)
    for name in ("landmark_embedding.npy", "head_template.obj"):
        if name != missing_name:
            (flame_data / name).write_bytes(b"tracked asset")
    cfg = types.SimpleNamespace(model=types.SimpleNamespace())

    with pytest.raises(ValueError, match="tracked MICA FLAME asset is unavailable"):
        _bind_mica_model_assets(cfg, configured)


def test_network_denial_preserves_socket_imports_and_refuses_connections(monkeypatch) -> None:
    original_socket = (
        socket.socket.__base__
        if getattr(socket.socket, "_asset_mania_network_denied", False)
        else socket.socket
    )
    original_create_connection = socket.create_connection
    requests = types.ModuleType("requests")

    class Session:
        def request(self, *_args, **_kwargs):
            pytest.fail("request must be denied")

        def get(self, url: str):
            return self.request("GET", url)

    requests.sessions = types.SimpleNamespace(Session=Session)
    requests.Session = Session
    monkeypatch.setattr(socket, "socket", original_socket)
    monkeypatch.setattr(socket, "create_connection", original_create_connection)
    monkeypatch.setitem(sys.modules, "requests", requests)

    mica_plugin._deny_network()
    denied_socket = socket.socket
    socket.create_connection = lambda *_args, **_kwargs: pytest.fail(
        "create_connection denial must be reasserted"
    )
    requests.sessions.Session.request = lambda *_args, **_kwargs: pytest.fail(
        "requests denial must be reasserted"
    )
    mica_plugin._deny_network()

    class ImportStyleSocket(socket.socket):
        pass

    assert socket.socket is denied_socket
    assert issubclass(ImportStyleSocket, original_socket)
    with socket.socket() as connection:
        with pytest.raises(RuntimeError, match="network denied during MICA inference"):
            connection.connect(("127.0.0.1", 9))
        with pytest.raises(RuntimeError, match="network denied during MICA inference"):
            connection.connect_ex(("127.0.0.1", 9))
        for method in ("send", "sendall", "sendto", "sendmsg"):
            if hasattr(connection, method):
                with pytest.raises(RuntimeError, match="network denied during MICA inference"):
                    getattr(connection, method)(b"blocked")
    with pytest.raises(RuntimeError, match="network denied during MICA inference"):
        socket.create_connection(("127.0.0.1", 9))
    with pytest.raises(RuntimeError, match="network denied during MICA inference"):
        requests.Session().get("https://example.invalid")


def test_network_denial_allows_fresh_ssl_import_without_requests() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import builtins; "
                "from asset_mania_engine_mica.plugin import _deny_network; "
                "real_import = builtins.__import__; "
                "builtins.__import__ = lambda name, *args, **kwargs: "
                "(_ for _ in ()).throw(ImportError('requests unavailable')) "
                "if name == 'requests' else real_import(name, *args, **kwargs); "
                "_deny_network(); import socket, ssl; "
                "assert issubclass(ssl.SSLSocket, socket.socket); "
                "connection = socket.socket(); "
                'exec("def _raises_runtime_error(check):\\n'
                "    try:\\n        check()\\n"
                "    except RuntimeError:\\n        return True\\n"
                '    return False"); '
                "checks = (lambda: connection.connect(('127.0.0.1', 9)), "
                "lambda: connection.connect_ex(('127.0.0.1', 9)), "
                "lambda: socket.create_connection(('127.0.0.1', 9))); "
                "assert all(_raises_runtime_error(check) for check in checks)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_scrfd_bridge_uses_only_sealed_model_and_selects_centered_face(tmp_path: Path) -> None:
    detector_directory = tmp_path / "antelopev2"
    detector_directory.mkdir()
    sealed_model = detector_directory / "scrfd_10g_bnkps.onnx"
    sealed_model.write_bytes(b"sealed")
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    bboxes = np.array([[1, 2, 30, 40, 0.8], [50, 60, 150, 180, 0.95]])
    keypoints = np.arange(20, dtype=np.float32).reshape(2, 5, 2)
    calls: list[tuple[object, ...]] = []

    class FakeDetector:
        session = types.SimpleNamespace(
            get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        def prepare(self, *, ctx_id: int, input_size: tuple[int, int]) -> None:
            calls.append(("prepare", ctx_id, input_size))

        def detect(self, value: np.ndarray, *, max_num: int, metric: str):
            calls.append(("detect", value, max_num, metric))
            return bboxes, keypoints

    def detector_factory(path: str, *, providers: list[str]) -> FakeDetector:
        calls.append(("factory", path, providers))
        return FakeDetector()

    bbox, kps, score = mica_plugin._detect_face_with_scrfd(
        image,
        detector_directory,
        detector_factory=detector_factory,
        center_selector=lambda values, source: 1,
    )

    assert calls[0] == ("factory", str(sealed_model), ["CUDAExecutionProvider"])
    assert calls[1] == ("prepare", 0, (224, 224))
    assert calls[2][0] == "detect"
    assert calls[2][1] is image
    assert calls[2][2:] == (0, "default")
    np.testing.assert_array_equal(bbox, bboxes[1, :4])
    np.testing.assert_array_equal(kps, keypoints[1])
    assert score == pytest.approx(0.95)


def test_scrfd_bridge_fails_closed_without_cuda_session_provider(tmp_path: Path) -> None:
    model = tmp_path / "scrfd_10g_bnkps.onnx"
    model.write_bytes(b"sealed")
    detector = types.SimpleNamespace(
        session=types.SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
    )
    with pytest.raises(ValueError, match="CUDA execution provider"):
        mica_plugin._detect_face_with_scrfd(
            np.zeros((10, 10, 3), dtype=np.uint8),
            tmp_path,
            center_selector=lambda *_args: 0,
            detector_factory=lambda *_args, **_kwargs: detector,
        )


def test_scrfd_bridge_rejects_missing_sealed_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sealed SCRFD detector is unavailable"):
        mica_plugin._detect_face_with_scrfd(
            np.zeros((224, 224, 3), dtype=np.uint8),
            tmp_path,
            detector_factory=lambda *_args, **_kwargs: pytest.fail("factory must not run"),
            center_selector=lambda _values, _source: 0,
        )


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


def test_sealed_worker_runs_help_from_an_isolated_copy(tmp_path: Path) -> None:
    worker = tmp_path / "plugin.py"
    shutil.copyfile(Path(sys.modules[MicaPluginSettings.__module__].__file__), worker)

    completed = subprocess.run(
        [sys.executable, "-I", str(worker), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--request" in completed.stdout


def test_sealed_worker_parses_as_cpython_39() -> None:
    worker = Path(sys.modules[MicaPluginSettings.__module__].__file__)

    ast.parse(worker.read_text(encoding="utf-8"), filename=str(worker), feature_version=(3, 9))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema": "asset-mania.face-geometry-plugin-request.v2"}, "schema"),
        ({"plugin": "unknown-local"}, "unsupported face geometry plugin"),
        ({"plugin": "deca-local"}, "profile does not belong"),
        ({"profile": "detail-displacement-v1"}, "profile does not belong"),
        ({"plugin_revision": "A" * 40}, "revision"),
        ({"checkpoint_sha256": "B" * 64}, "checkpoint digest"),
        ({"face_rights_receipt_sha256": "D" * 64}, "rights digest"),
        ({"source_image": "relative.png"}, "source image must be absolute"),
        ({"output_directory": "relative-output"}, "output directory must be absolute"),
        ({"device": "cpu"}, "device must be cuda"),
        ({"topology": "other"}, "topology"),
        ({"network": "allowed"}, "network must be denied"),
    ],
)
def test_private_request_validation_matches_public_v1(
    tmp_path: Path, changes: dict[str, str], message: str
) -> None:
    _request, request_path, _result_path = request_files(tmp_path)
    document = json.loads(request_path.read_text(encoding="utf-8"))
    document.update(changes)
    request_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        __import__("asset_mania_engine_mica.plugin", fromlist=["_load_request"])._load_request(
            request_path
        )


def test_private_request_rejects_source_inside_output_directory(tmp_path: Path) -> None:
    _request, request_path, _result_path = request_files(tmp_path)
    document = json.loads(request_path.read_text(encoding="utf-8"))
    document["source_image"] = str(Path(document["output_directory"]) / "source.png")
    request_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="source image must not be contained"):
        __import__("asset_mania_engine_mica.plugin", fromlist=["_load_request"])._load_request(
            request_path
        )


def fake_backend(_source: Path, _settings: MicaPluginSettings) -> MicaPrediction:
    vertices = np.zeros((5023, 3), dtype=np.float32)
    vertices[:, 0] = np.linspace(-0.08, 0.08, 5023)
    return MicaPrediction(
        vertices=vertices,
        faces=topology(),
        source_projection=np.zeros((5023, 2), dtype=np.float32),
        coordinate_unit="metres",
    )


@pytest.mark.parametrize("extent", [0.309499189, 0.32])
def test_prediction_validator_accepts_full_head_extent_boundaries(extent: float) -> None:
    prediction = fake_backend(Path(), object())
    prediction.vertices[:, 0] = np.linspace(0.0, extent, 5023)

    vertices, _faces, _projection = mica_plugin._validate_prediction(prediction)

    assert np.isclose(np.ptp(vertices, axis=0).max(), extent)


@pytest.mark.parametrize("extent", [0.149999, 0.320001])
def test_prediction_validator_rejects_extent_outside_full_head_boundaries(extent: float) -> None:
    prediction = fake_backend(Path(), object())
    prediction.vertices[:, 0] = np.linspace(0.0, extent, 5023)

    with pytest.raises(ValueError, match="between 0.15 and 0.32 metres"):
        mica_plugin._validate_prediction(prediction)


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
    monkeypatch.setenv("ASSET_MANIA_SAFE_FLAG", "must-not-cross-boundary")
    monkeypatch.setenv("PRIVATE_KEY", "must-not-cross-boundary")
    monkeypatch.setenv("SESSION_COOKIE", "must-not-cross-boundary")

    def inspecting_backend(source: Path, runtime: MicaPluginSettings) -> MicaPrediction:
        assert "OPENAI_API_KEY" not in __import__("os").environ
        assert "ASSET_MANIA_SAFE_FLAG" not in __import__("os").environ
        assert "PRIVATE_KEY" not in __import__("os").environ
        assert "SESSION_COOKIE" not in __import__("os").environ
        assert __import__("os").environ["HOME"] == str(configured.isolated_home)
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
