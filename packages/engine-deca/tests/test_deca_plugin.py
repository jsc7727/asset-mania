import ast
import inspect
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from asset_mania_engine_deca.plugin import (
    DecaPluginSettings,
    DecaPrediction,
    _decompose_code,
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
RIGHTS = "d" * 64


def topology() -> np.ndarray:
    indices = np.arange(9976, dtype=np.int64)
    return np.stack([indices % 5023, (indices + 1) % 5023, (indices + 2) % 5023], axis=1)


def test_checkpoint_keys_fail_closed() -> None:
    with pytest.raises(ValueError, match="missing required model keys"):
        _require_checkpoint_keys({"E_flame": {}, "E_detail": {}})


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
        ["py", "-3.9", "-I", str(worker), "--help"],
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
    return DecaPrediction(
        vertices=vertices,
        faces=topology(),
        source_projection=np.zeros((5023, 2), dtype=np.float32),
        detail_displacement=np.full(5023, 0.0005, dtype=np.float32),
        coordinate_unit="metres",
    )


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
            digest_reader=digest_reader(configured),
            clean_reader=lambda _path: True,
        )
    assert not request.output_directory.exists()
