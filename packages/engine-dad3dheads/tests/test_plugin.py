import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from asset_mania_engine_dad3dheads.plugin import (
    DADPluginSettings,
    _install_python312_compatibility,
    _neutral_mesh_vertices,
    _write_projection,
    run_face_plugin,
    validate_dad_runtime,
)
from asset_mania_pipeline import (
    DAD_PLUGIN,
    build_face_plugin_request,
    write_face_plugin_request,
)

REVISION = "68cc9b51974e2628f7a8f8ed2dadc5f73b3f8aa7"
DIGEST = "a" * 64


def test_neutral_mesh_vertices_remove_nonzero_global_pose() -> None:
    neutral = np.array(
        [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.05]], dtype=np.float64
    )
    angle = np.deg2rad(35.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]]
    )
    posed = neutral @ rotation.T
    parameters = object()

    class HeadMesh:
        def vertices_3d(self, supplied, *, zero_rotation=False):
            assert supplied is parameters
            assert zero_rotation is True
            return neutral[None, ...]

    class Predictor:
        head_mesh = HeadMesh()

    predictions = {"3d_vertices": posed, "3dmm_params": parameters}

    result = _neutral_mesh_vertices(Predictor(), predictions)

    assert np.allclose(result, neutral)
    assert not np.allclose(result, predictions["3d_vertices"])


def test_python312_compatibility_restores_getargspec(monkeypatch) -> None:
    monkeypatch.delattr(inspect, "getargspec", raising=False)
    for name in ("bool", "int", "float", "complex", "object", "unicode", "str"):
        monkeypatch.delattr(np, name, raising=False)

    _install_python312_compatibility()

    assert inspect.getargspec is inspect.getfullargspec
    assert np.bool is np.bool_
    assert np.int is int
    assert np.float is float
    assert np.complex is complex
    assert np.object is object
    assert np.unicode is str
    assert np.str is str


def test_projection_payload_preserves_camera_vertices(tmp_path: Path) -> None:
    path = tmp_path / "projection.npz"
    projected = np.array([[1.0, 2.0], [3.0, 4.0]])
    camera = np.array([[0.1, 0.2, -0.3], [0.4, 0.5, 0.6]])

    _write_projection(path, projected, camera, (64, 32))

    with np.load(path, allow_pickle=False) as archive:
        assert np.array_equal(archive["projected_vertices"], projected)
        assert np.array_equal(archive["camera_vertices"], camera)
        assert archive["image_shape"].tolist() == [64, 32]


def _request(tmp_path: Path):
    source = tmp_path / "source.png"
    source.write_bytes(b"synthetic")
    return build_face_plugin_request(
        plugin=DAD_PLUGIN,
        plugin_revision=REVISION,
        source_image=source,
        output_directory=tmp_path / "output",
        device="cuda",
        checkpoint_sha256=DIGEST,
    )


def test_runtime_requires_pinned_source_and_preplaced_checkpoint(tmp_path: Path) -> None:
    settings = DADPluginSettings(
        source_root=tmp_path / "source",
        isolated_home=tmp_path / "home",
        revision=REVISION,
        checkpoint_sha256=DIGEST,
    )
    with pytest.raises(ValueError, match="pinned source revision is unavailable"):
        validate_dad_runtime(settings)


def test_runtime_rejects_revision_and_digest_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "predictor.py").write_text("# synthetic", encoding="utf-8")
    static = source / "model_training/model/static"
    static.mkdir(parents=True)
    (static / "flame.pkl").write_bytes(b"flame")
    home = tmp_path / "home"
    checkpoint = home / ".dad_checkpoints/dad_3dheads.trcd"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    settings = DADPluginSettings(source, home, REVISION, DIGEST)

    with pytest.raises(ValueError, match="source revision mismatch"):
        validate_dad_runtime(
            settings,
            revision_reader=lambda _root: "b" * 40,
            digest_reader=lambda _path: DIGEST,
        )
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        validate_dad_runtime(
            settings,
            revision_reader=lambda _root: REVISION,
            digest_reader=lambda _path: "b" * 64,
        )


def test_fake_plugin_round_trip_is_create_only_and_closed(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    write_face_plugin_request(request, request_path)
    script = tmp_path / "fake_plugin.py"
    script.write_text(
        """
import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--request', required=True)
p.add_argument('--result', required=True)
a = p.parse_args()
request = json.loads(Path(a.request).read_text(encoding='utf-8'))
out = Path(request['output_directory'])
out.mkdir()
(out / 'head.obj').write_text('v 0 0 0\\nv 1 0 0\\nv 0 1 0\\nf 1 2 3\\n', encoding='utf-8')
(out / 'projection.npz').write_bytes(b'synthetic')
result = {
  'schema': 'asset-mania.face-plugin-result.v0',
  'plugin': request['plugin'], 'status': 'succeeded',
  'raw_mesh': str((out / 'head.obj').resolve()),
  'projection_data': str((out / 'projection.npz').resolve()),
  'vertex_count': 3, 'triangle_count': 1, 'elapsed_seconds': 0.01,
  'device': 'cuda', 'checkpoint_sha256': request['checkpoint_sha256'],
}
Path(a.result).write_text(json.dumps(result), encoding='utf-8')
""".lstrip(),
        encoding="utf-8",
    )

    result = run_face_plugin(
        [sys.executable, str(script)],
        request,
        request_path,
        result_path,
        timeout_seconds=30,
    )

    assert result.status == "succeeded"
    assert result.vertex_count == 3
    assert result.checkpoint_sha256 == DIGEST


def test_nonzero_plugin_exit_has_no_fallback(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    write_face_plugin_request(request, request_path)

    with pytest.raises(ValueError, match="face plugin exited with 7"):
        run_face_plugin(
            [sys.executable, "-c", "raise SystemExit(7)"],
            request,
            request_path,
            result_path,
            timeout_seconds=30,
        )
    assert not result_path.exists()


def test_result_file_cannot_preexist(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    write_face_plugin_request(request, request_path)
    result_path.write_text(json.dumps({}), encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_face_plugin(
            [sys.executable, "-c", "pass"],
            request,
            request_path,
            result_path,
            timeout_seconds=30,
        )
