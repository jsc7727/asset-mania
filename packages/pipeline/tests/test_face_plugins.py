import json
from dataclasses import replace
from pathlib import Path

import pytest
from asset_mania_pipeline.face_plugins import (
    DAD_PLUGIN,
    FacePluginRequest,
    build_face_plugin_request,
    load_face_plugin_result,
    write_face_plugin_request,
)

REVISION = "68cc9b51974e2628f7a8f8ed2dadc5f73b3f8aa7"
DIGEST = "a" * 64


def _request(tmp_path: Path) -> FacePluginRequest:
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


def test_request_is_closed_and_requires_cuda(tmp_path: Path) -> None:
    request = _request(tmp_path)

    assert request.schema == "asset-mania.face-plugin-request.v0"
    assert request.network == "denied-during-inference"
    assert request.source_image.is_absolute()
    with pytest.raises(ValueError, match="device must be cuda"):
        replace(request, device="cpu")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("plugin", "triposr-local", "unsupported face plugin"),
        ("plugin_revision", "main", "revision must be a SHA-1"),
        ("checkpoint_sha256", "short", "checkpoint digest must be SHA-256"),
    ],
)
def test_request_rejects_unpinned_fields(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    request = _request(tmp_path)
    with pytest.raises(ValueError, match=message):
        replace(request, **{field: value})


def test_request_rejects_relative_or_existing_output(tmp_path: Path) -> None:
    request = _request(tmp_path)
    with pytest.raises(ValueError, match="source image must be absolute"):
        replace(request, source_image=Path("source.png"))
    request.output_directory.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_face_plugin_request(request, tmp_path / "request.json")


def test_result_rejects_extra_fields_and_mismatch(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request.output_directory.mkdir()
    mesh = request.output_directory / "head.obj"
    projection = request.output_directory / "projection.npz"
    mesh.write_text("v 0 0 0\n", encoding="utf-8")
    projection.write_bytes(b"x")
    payload = {
        "schema": "asset-mania.face-plugin-result.v0",
        "plugin": DAD_PLUGIN,
        "status": "succeeded",
        "raw_mesh": str(mesh.resolve()),
        "projection_data": str(projection.resolve()),
        "vertex_count": 1,
        "triangle_count": 0,
        "elapsed_seconds": 0.1,
        "device": "cuda",
        "checkpoint_sha256": DIGEST,
        "surprise": True,
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fields outside the v0 allowlist"):
        load_face_plugin_result(result_path, request)

    del payload["surprise"]
    payload["checkpoint_sha256"] = "b" * 64
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        load_face_plugin_result(result_path, request)
