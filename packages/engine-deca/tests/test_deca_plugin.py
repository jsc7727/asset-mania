import ast
import inspect
import json
import shutil
import socket
import subprocess
import sys
import types
from pathlib import Path

import asset_mania_engine_deca.plugin as deca_plugin
import numpy as np
import pytest
from asset_mania_engine_deca.plugin import (
    DecaPluginSettings,
    DecaPrediction,
    _decompose_code,
    _directory_sha256,
    _require_checkpoint_keys,
    execute_deca_request,
    sample_position_uv_displacement,
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
DETECTOR = "e" * 64
RIGHTS = "d" * 64


def topology() -> np.ndarray:
    indices = np.arange(9976, dtype=np.int64)
    return np.stack([indices % 5023, (indices + 1) % 5023, (indices + 2) % 5023], axis=1)


def test_checkpoint_keys_fail_closed() -> None:
    with pytest.raises(ValueError, match="missing required model keys"):
        _require_checkpoint_keys({"E_flame": {}, "E_detail": {}})


def test_official_backend_rejects_cpu_before_reading_source(monkeypatch, tmp_path: Path) -> None:
    cv2 = types.SimpleNamespace(
        IMREAD_COLOR=1,
        imread=lambda *_args: pytest.fail("source must not be read before CUDA validation"),
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    with pytest.raises(ValueError, match="CUDA unavailable"):
        deca_plugin._official_backend(
            tmp_path / "source.png",
            settings(tmp_path),
            cuda_validator=lambda _torch: (_ for _ in ()).throw(ValueError("CUDA unavailable")),
        )


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

    deca_plugin._deny_network()
    denied_socket = socket.socket
    socket.create_connection = lambda *_args, **_kwargs: pytest.fail(
        "create_connection denial must be reasserted"
    )
    requests.sessions.Session.request = lambda *_args, **_kwargs: pytest.fail(
        "requests denial must be reasserted"
    )
    deca_plugin._deny_network()

    class ImportStyleSocket(socket.socket):
        pass

    assert socket.socket is denied_socket
    assert issubclass(ImportStyleSocket, original_socket)
    with socket.socket() as connection:
        with pytest.raises(RuntimeError, match="network denied during DECA inference"):
            connection.connect(("127.0.0.1", 9))
        with pytest.raises(RuntimeError, match="network denied during DECA inference"):
            connection.connect_ex(("127.0.0.1", 9))
        for method in ("send", "sendall", "sendto", "sendmsg"):
            if hasattr(connection, method):
                with pytest.raises(RuntimeError, match="network denied during DECA inference"):
                    getattr(connection, method)(b"blocked")
    with pytest.raises(RuntimeError, match="network denied during DECA inference"):
        socket.create_connection(("127.0.0.1", 9))
    with pytest.raises(RuntimeError, match="network denied during DECA inference"):
        requests.Session().get("https://example.invalid")


def test_network_denial_allows_fresh_ssl_import_without_requests() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import builtins; "
                "from asset_mania_engine_deca.plugin import _deny_network; "
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


def test_decompose_code_preserves_sealed_numeric_parameter_slices() -> None:
    class ModelConfig:
        param_list = ("shape", "exp", "pose", "cam", "light")
        n_shape = 2
        n_exp = 1
        n_pose = 3
        n_cam = 3
        n_light = 27

    parameters = np.arange(36, dtype=np.float32)[None, :]

    result = _decompose_code(parameters, ModelConfig())

    assert np.array_equal(result["shape"], [[0.0, 1.0]])
    assert np.array_equal(result["pose"], [[3.0, 4.0, 5.0]])
    assert result["light"].shape == (1, 9, 3)
    assert np.array_equal(result["light"][0, 0], [9.0, 10.0, 11.0])


def test_position_uv_sampling_flips_v_and_samples_seam_corners_before_mean() -> None:
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    uv_faces = np.array([[0, 1, 2], [4, 2, 3]], dtype=np.int64)
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    displacement = np.array([[0.0, 0.0], [0.0, 4.0]], dtype=np.float64)

    result = sample_position_uv_displacement(displacement, 4, faces, uv, uv_faces)

    assert np.allclose(result, [0.5, 4.0, 0.0, 0.0])


def test_position_uv_sampling_rejects_unmapped_and_out_of_range_topology() -> None:
    displacement = np.zeros((2, 2), dtype=np.float64)
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    uv_faces = np.array([[0, 1, 2]], dtype=np.int64)

    with pytest.raises(ValueError, match="leaves a position unmapped"):
        sample_position_uv_displacement(displacement, 4, faces, uv, uv_faces)
    with pytest.raises(ValueError, match="UV topology is out of range"):
        sample_position_uv_displacement(displacement, 3, faces, uv, [[0, 1, 3]])


@pytest.mark.parametrize(
    ("faces", "uv_faces", "message"),
    [
        ([[0.5, 1, 2]], [[0, 1, 2]], "position topology indices must be integers"),
        ([[0, 1, 2]], [[0, 1.5, 2]], "UV topology indices must be integers"),
    ],
)
def test_position_uv_sampling_rejects_fractional_topology_indices(
    faces: list[list[float]], uv_faces: list[list[float]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        sample_position_uv_displacement(
            np.zeros((2, 2), dtype=np.float64),
            3,
            faces,
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            uv_faces,
        )


@pytest.mark.parametrize("unused_uv", [[np.nan, 0.0], [1.1, 0.0]])
def test_position_uv_sampling_validates_unused_uv_coordinates(unused_uv: list[float]) -> None:
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], unused_uv])

    with pytest.raises(ValueError, match="UV"):
        sample_position_uv_displacement(
            np.zeros((2, 2), dtype=np.float64),
            3,
            [[0, 1, 2]],
            uv,
            [[0, 1, 2]],
        )


def settings(tmp_path: Path) -> DecaPluginSettings:
    source_root = tmp_path / "deca-source"
    source_root.mkdir()
    (source_root / "decalib").mkdir()
    (source_root / "decalib" / "deca.py").write_text("official source marker", encoding="utf-8")
    data = source_root / "data"
    data.mkdir()
    (data / "head_template.obj").write_text("tracked topology", encoding="utf-8")
    (data / "fixed_displacement_256.npy").write_bytes(b"tracked displacement")
    checkpoint = tmp_path / "deca_model.tar"
    checkpoint.write_bytes(b"checkpoint")
    flame = tmp_path / "generic_model.pkl"
    flame.write_bytes(b"flame")
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    detector = isolated / ".insightface" / "models" / "antelopev2"
    detector.mkdir(parents=True)
    (detector / "scrfd_10g_bnkps.onnx").write_bytes(b"detector")
    return DecaPluginSettings(
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


def test_center_face_selection_uses_bbox_center_nearest_image_center() -> None:
    boxes = np.array([[0, 0, 20, 20, 0.9], [40, 40, 70, 70, 0.8], [80, 80, 99, 99, 0.99]])
    assert deca_plugin._select_center_face(boxes, (100, 100)) == 1


def test_deca_similarity_crop_and_inverse_projection_are_numerically_sealed() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[20:80, 60:140] = 255
    cv2 = types.SimpleNamespace(
        warpAffine=lambda _image, _transform, size: np.zeros((size[1], size[0], 3), dtype=np.uint8)
    )
    crop, inverse = deca_plugin._deca_face_crop(
        image, np.array([60.0, 20.0, 140.0, 80.0]), output_size=224, cv2_module=cv2
    )
    original = deca_plugin._project_crop_to_source(
        np.array([[0.0, 0.0], [223.0, 223.0], [111.5, 111.5]]), inverse
    )
    assert crop.shape == (224, 224, 3)
    np.testing.assert_allclose(original[2], [100.0, 50.0], atol=0.6)
    np.testing.assert_allclose(original[0], [56.25, 6.25], atol=0.6)


def test_chumpy_compatibility_and_deca_assets_are_bound_to_tracked_paths(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    numpy_module = types.SimpleNamespace()
    deca_plugin._restore_chumpy_numpy_aliases(numpy_module)
    assert numpy_module.bool is bool
    cfg = types.SimpleNamespace()
    deca_plugin._bind_deca_model_assets(cfg, configured)
    assert cfg.flame_model_path == str(configured.flame_path.resolve(strict=True))
    assert cfg.topology_path == str(
        (configured.source_root / "data/head_template.obj").resolve(strict=True)
    )


def test_scrfd_detector_requires_cuda_provider_and_returns_center_bbox(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    calls = []

    class Detector:
        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        def prepare(self, **_kwargs):
            pass

        def detect(self, _image, **_kwargs):
            return np.array([[0, 0, 10, 10, 0.9], [40, 40, 60, 60, 0.8]]), np.zeros((2, 5, 2))

    def factory(path, *, providers):
        calls.append((path, providers))
        return Detector()

    bbox = deca_plugin._detect_face_with_scrfd(
        np.zeros((100, 100, 3), dtype=np.uint8),
        configured.detector_path,
        detector_factory=factory,
    )
    assert calls == [
        (
            str(configured.detector_path / "scrfd_10g_bnkps.onnx"),
            ["CUDAExecutionProvider"],
        )
    ]
    np.testing.assert_array_equal(bbox, [40, 40, 60, 60])


def test_scrfd_detector_fails_closed_without_cuda_provider(tmp_path: Path) -> None:
    configured = settings(tmp_path)

    class Detector:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    with pytest.raises(ValueError, match="CUDA execution provider"):
        deca_plugin._detect_face_with_scrfd(
            np.zeros((10, 10, 3), dtype=np.uint8),
            configured.detector_path,
            detector_factory=lambda *_args, **_kwargs: Detector(),
        )


def digest_reader(configured: DecaPluginSettings):
    return {
        configured.checkpoint_path: CHECKPOINT,
        configured.flame_path: FLAME,
    }.__getitem__


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


def test_sealed_worker_runs_help_from_an_isolated_copy(tmp_path: Path) -> None:
    worker = tmp_path / "plugin.py"
    shutil.copyfile(Path(inspect.getfile(DecaPluginSettings)), worker)

    completed = subprocess.run(
        [sys.executable, "-I", str(worker), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--request" in completed.stdout


def test_sealed_worker_parses_as_cpython_39() -> None:
    worker = Path(inspect.getfile(DecaPluginSettings))

    ast.parse(worker.read_text(encoding="utf-8"), filename=str(worker), feature_version=(3, 9))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema": "asset-mania.face-geometry-plugin-request.v2"}, "schema"),
        ({"plugin": "unknown-local"}, "unsupported face geometry plugin"),
        ({"plugin": "mica-local"}, "profile does not belong"),
        ({"profile": "identity-neutral-v1"}, "profile does not belong"),
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
        __import__("asset_mania_engine_deca.plugin", fromlist=["_load_request"])._load_request(
            request_path
        )


def test_private_request_rejects_source_inside_output_directory(tmp_path: Path) -> None:
    _request, request_path, _result_path = request_files(tmp_path)
    document = json.loads(request_path.read_text(encoding="utf-8"))
    document["source_image"] = str(Path(document["output_directory"]) / "source.png")
    request_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="source image must not be contained"):
        __import__("asset_mania_engine_deca.plugin", fromlist=["_load_request"])._load_request(
            request_path
        )


def fake_backend(_source: Path, _settings: DecaPluginSettings) -> DecaPrediction:
    vertices = np.zeros((5023, 3), dtype=np.float32)
    vertices[:, 0] = np.linspace(-0.08, 0.08, 5023)
    vertices[:, 1] = np.linspace(-0.06, 0.06, 5023)
    vertices[:, 2] = np.linspace(-0.04, 0.04, 5023)
    return DecaPrediction(
        vertices=vertices,
        faces=topology(),
        source_projection=np.zeros((5023, 2), dtype=np.float32),
        detail_displacement=np.full(5023, 0.0005, dtype=np.float32),
        coordinate_unit="metres",
    )


def test_prediction_validator_defers_raw_absolute_extent_until_alignment() -> None:
    prediction = fake_backend(Path(), object())
    prediction.vertices[:, 0] = np.linspace(0.0, 0.324885711, 5023)

    vertices, _faces, _projection, _displacement = deca_plugin._validate_prediction(prediction)

    assert np.isclose(np.ptp(vertices, axis=0).max(), 0.324885711)
    assert np.array_equal(vertices, prediction.vertices)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_prediction_validator_rejects_zero_extent_on_every_axis(axis: int) -> None:
    prediction = fake_backend(Path(), object())
    prediction.vertices[:, axis] = 0.0

    with pytest.raises(ValueError, match="positive finite extent on every axis"):
        deca_plugin._validate_prediction(prediction)


def test_prediction_validator_rejects_nonfinite_vertices() -> None:
    prediction = fake_backend(Path(), object())
    prediction.vertices[0, 1] = np.inf

    with pytest.raises(ValueError, match="non-finite"):
        deca_plugin._validate_prediction(prediction)


def test_prediction_validator_rejects_invalid_topology() -> None:
    prediction = fake_backend(Path(), object())
    prediction.faces[0, 0] = 5023

    with pytest.raises(ValueError, match="out-of-range face index"):
        deca_plugin._validate_prediction(prediction)


def test_uv_displacement_sampling_has_fixed_orientation() -> None:
    rows, columns = np.mgrid[0:4, 0:4]
    displacement = rows * 10.0 + columns
    uv = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]], dtype=np.float64)

    sampled = sample_uv_displacement(displacement, uv)

    assert np.allclose(sampled, [30.0, 3.0, 16.5])


def test_runtime_requires_exact_revision_and_digests(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    checkpoint = validate_deca_runtime(
        configured,
        revision_reader=lambda _path: REVISION,
        digest_reader=digest_reader(configured),
        clean_reader=lambda _path: True,
    )
    assert checkpoint == configured.checkpoint_path
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        validate_deca_runtime(
            configured,
            revision_reader=lambda _path: REVISION,
            digest_reader=lambda _path: "e" * 64,
            clean_reader=lambda _path: True,
        )


def test_runtime_rejects_dirty_source(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    digests = {
        configured.checkpoint_path: CHECKPOINT,
        configured.flame_path: FLAME,
    }
    with pytest.raises(ValueError, match="source tree is not clean"):
        validate_deca_runtime(
            configured,
            revision_reader=lambda _path: REVISION,
            digest_reader=digests.__getitem__,
            clean_reader=lambda _path: False,
        )
    configured.detector_path.joinpath("scrfd_10g_bnkps.onnx").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="detector asset digest mismatch"):
        validate_deca_runtime(
            configured,
            revision_reader=lambda _path: REVISION,
            digest_reader=digests.__getitem__,
            clean_reader=lambda _path: True,
        )


def test_worker_rejects_non_metre_or_implausible_geometry(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    _request, request_path, result_path = request_files(tmp_path)
    prediction = fake_backend(Path(), configured)
    prediction = DecaPrediction(
        prediction.vertices * 10,
        prediction.faces,
        prediction.source_projection,
        prediction.detail_displacement,
        "millimetres",
    )
    with pytest.raises(ValueError, match="explicitly use metres"):
        execute_deca_request(
            request_path,
            result_path,
            configured,
            backend=lambda *_args: prediction,
            revision_reader=lambda _path: REVISION,
            digest_reader=digest_reader(configured),
            clean_reader=lambda _path: True,
        )


def test_worker_uses_exact_environment_allowlist_before_external_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = settings(tmp_path)
    _request, request_path, result_path = request_files(tmp_path)
    monkeypatch.setenv("ASSET_MANIA_SAFE_FLAG", "drop")
    monkeypatch.setenv("PRIVATE_KEY", "drop")
    monkeypatch.setenv("SESSION_COOKIE", "drop")
    monkeypatch.setenv("ASSET_MANIA_DECA_DETECTOR_PATH", str(configured.detector_path))

    def inspecting_backend(source: Path, runtime: DecaPluginSettings) -> DecaPrediction:
        environment = __import__("os").environ
        assert "ASSET_MANIA_SAFE_FLAG" not in environment
        assert "PRIVATE_KEY" not in environment
        assert "SESSION_COOKIE" not in environment
        assert environment["ASSET_MANIA_DECA_DETECTOR_PATH"] == str(configured.detector_path)
        assert environment["HOME"] == str(configured.isolated_home)
        return fake_backend(source, runtime)

    execute_deca_request(
        request_path,
        result_path,
        configured,
        backend=inspecting_backend,
        revision_reader=lambda _path: REVISION,
        digest_reader=digest_reader(configured),
        clean_reader=lambda _path: True,
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
        digest_reader=digest_reader(configured),
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
        assert np.allclose(archive["detail_displacement"], 0.0005)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["ephemeral_identity_feature_used"] is False
    assert result["persisted_identity_feature_count"] == 0
    forbidden = ("*albedo*", "*texture*", "*landmark*", "*vis*", "*.obj", "*.mat")
    assert not [path for pattern in forbidden for path in request.output_directory.rglob(pattern)]


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
            digest_reader=digest_reader(configured),
            clean_reader=lambda _path: True,
        )
    assert not request.output_directory.exists()
