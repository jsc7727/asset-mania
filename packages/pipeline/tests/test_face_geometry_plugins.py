import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import asset_mania_pipeline.face_geometry_plugins as protocol
import pytest
from asset_mania_pipeline.face_geometry_plugins import (
    DECA_PLUGIN,
    MICA_PLUGIN,
    build_face_geometry_plugin_request,
    load_face_geometry_plugin_result,
    run_face_geometry_plugin,
    write_face_geometry_plugin_request,
)

REVISION = "a" * 40
CHECKPOINT = "b" * 64
RIGHTS = "c" * 64


def geometry_request(tmp_path: Path, *, plugin: str = MICA_PLUGIN):
    source = tmp_path / "source.png"
    source.write_bytes(b"synthetic")
    return build_face_geometry_plugin_request(
        plugin=plugin,
        profile="identity-neutral-v1" if plugin == MICA_PLUGIN else "detail-displacement-v1",
        plugin_revision=REVISION,
        source_image=source,
        output_directory=tmp_path / "output",
        device="cuda",
        checkpoint_sha256=CHECKPOINT,
        topology="flame-2020-5023",
        face_rights_receipt_sha256=RIGHTS,
    )


def successful_result(request, result_path: Path) -> None:
    request.output_directory.mkdir()
    geometry = request.output_directory / "geometry.npz"
    geometry.write_bytes(b"numeric geometry")
    result_path.write_text(
        json.dumps(
            {
                "schema": "asset-mania.face-geometry-plugin-result.v1",
                "plugin": request.plugin,
                "profile": request.profile,
                "status": "succeeded",
                "geometry": str(geometry),
                "vertex_count": 5023,
                "triangle_count": 9976,
                "elapsed_seconds": 0.1,
                "device": "cuda",
                "checkpoint_sha256": CHECKPOINT,
                "topology": "flame-2020-5023",
                "ephemeral_identity_feature_used": request.plugin == MICA_PLUGIN,
                "persisted_identity_feature_count": 0,
            }
        ),
        encoding="utf-8",
    )


def test_geometry_request_binds_rights_topology_and_network(tmp_path: Path) -> None:
    request = geometry_request(tmp_path)

    assert request.network == "denied-during-inference"
    assert request.topology == "flame-2020-5023"
    assert request.face_rights_receipt_sha256 == RIGHTS
    assert request.source_image.is_absolute()
    assert request.output_directory.is_absolute()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("plugin_revision", "main", "revision must be a SHA-1"),
        ("checkpoint_sha256", "short", "checkpoint digest must be SHA-256"),
        ("face_rights_receipt_sha256", "short", "rights digest must be SHA-256"),
        ("device", "cpu", "device must be cuda"),
        ("topology", "unknown", "topology must be flame-2020-5023"),
    ],
)
def test_geometry_request_rejects_unpinned_or_unsupported_fields(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    request = geometry_request(tmp_path)

    with pytest.raises(ValueError, match=message):
        replace(request, **{field: value})


def test_plugin_and_profile_pair_is_closed(tmp_path: Path) -> None:
    request = geometry_request(tmp_path)

    with pytest.raises(ValueError, match="profile does not belong to plugin"):
        replace(request, profile="detail-displacement-v1")
    with pytest.raises(ValueError, match="unsupported face geometry plugin"):
        replace(request, plugin="dad3dheads-local")


def test_request_write_is_create_only(tmp_path: Path) -> None:
    request = geometry_request(tmp_path)
    request_path = tmp_path / "request.json"

    write_face_geometry_plugin_request(request, request_path)

    assert request_path.is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_face_geometry_plugin_request(request, request_path)


def test_success_inventory_is_exactly_numeric_geometry(tmp_path: Path) -> None:
    request = geometry_request(tmp_path)
    result_path = tmp_path / "result.json"
    successful_result(request, result_path)

    result = load_face_geometry_plugin_result(result_path, request)

    assert result.geometry == request.output_directory / "geometry.npz"
    assert result.persisted_identity_feature_count == 0
    (request.output_directory / "crop.png").write_bytes(b"private")
    with pytest.raises(ValueError, match="inventory is unexpected"):
        load_face_geometry_plugin_result(result_path, request)


def test_result_rejects_extra_fields_and_checkpoint_mismatch(tmp_path: Path) -> None:
    request = geometry_request(tmp_path)
    result_path = tmp_path / "result.json"
    successful_result(request, result_path)
    document = json.loads(result_path.read_text(encoding="utf-8"))
    document["embedding"] = [1.0]
    result_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="fields outside the v1 allowlist"):
        load_face_geometry_plugin_result(result_path, request)

    del document["embedding"]
    document["checkpoint_sha256"] = "d" * 64
    result_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        load_face_geometry_plugin_result(result_path, request)


def test_failed_result_exposes_no_geometry(tmp_path: Path) -> None:
    request = geometry_request(tmp_path, plugin=DECA_PLUGIN)
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema": "asset-mania.face-geometry-plugin-result.v1",
                "plugin": request.plugin,
                "profile": request.profile,
                "status": "execution_failed",
                "geometry": str(request.output_directory / "geometry.npz"),
                "vertex_count": 0,
                "triangle_count": 0,
                "elapsed_seconds": 0.1,
                "device": "cuda",
                "checkpoint_sha256": CHECKPOINT,
                "topology": "flame-2020-5023",
                "ephemeral_identity_feature_used": False,
                "persisted_identity_feature_count": 0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="failed result must not expose geometry"):
        load_face_geometry_plugin_result(result_path, request)


def test_success_requires_exact_topology_counts_and_identity_feature_flag(tmp_path: Path) -> None:
    request = geometry_request(tmp_path)
    result_path = tmp_path / "result.json"
    successful_result(request, result_path)
    document = json.loads(result_path.read_text(encoding="utf-8"))
    document["vertex_count"] = 5022
    result_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="exact FLAME topology counts"):
        load_face_geometry_plugin_result(result_path, request)

    document["vertex_count"] = 5023
    document["ephemeral_identity_feature_used"] = False
    result_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="identity feature flag"):
        load_face_geometry_plugin_result(result_path, request)


def test_launcher_uses_fixed_environment_allowlist_by_default(tmp_path: Path, monkeypatch) -> None:
    request = geometry_request(tmp_path)
    result_path = tmp_path / "result.json"
    seen = {}

    def recording_run(_arguments, **kwargs):
        seen.update(kwargs["env"])
        successful_result(request, result_path)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(protocol.subprocess, "run", recording_run)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-plugin")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-reach-plugin")
    monkeypatch.setenv("ASSET_MANIA_MICA_SOURCE_ROOT", "private-runtime")
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))

    run_face_geometry_plugin(
        ["tool"],
        request,
        tmp_path / "request.json",
        result_path,
        timeout_seconds=10,
    )

    assert "OPENAI_API_KEY" not in seen
    assert "AWS_SECRET_ACCESS_KEY" not in seen
    assert seen["ASSET_MANIA_MICA_SOURCE_ROOT"] == "private-runtime"
    assert "PATH" in seen
