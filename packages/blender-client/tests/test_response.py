"""Only a bounded, closed, resealed, contained response is believed."""

import copy
import json
from pathlib import Path

import pytest
from asset_mania_blender_client import MAX_RESPONSE_BYTES, ResponseInvalid, load_response
from asset_mania_contracts import canonical_digest, canonical_json

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "tests" / "fixtures" / "v2"
OPERATIONS = ("preflight", "condition", "bake", "export", "validate")


def _example(operation: str) -> dict:
    return json.loads(
        (EXAMPLES / f"blender-response-v1-{operation}.json").read_text(encoding="utf-8")
    )


def _reseal(response: dict) -> dict:
    preimage = {key: value for key, value in response.items() if key != "response_sha256"}
    return {**preimage, "response_sha256": canonical_digest(preimage)}


def _write(staging: Path, response: dict) -> Path:
    path = staging / "response.json"
    path.write_text(canonical_json(response), encoding="utf-8")
    return path


def _load(staging: Path, response: dict, *, operation: str, **kwargs):
    return load_response(
        _write(staging, response),
        request_id=response["request_id"],
        operation=operation,
        staging_root=staging,
        **kwargs,
    )


def _stage_outputs(staging: Path, response: dict) -> None:
    for output in response["outputs"]:
        path = staging / output["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


@pytest.mark.parametrize("operation", OPERATIONS)
def test_each_normative_response_is_accepted(staging: Path, operation: str) -> None:
    response = _example(operation)
    _stage_outputs(staging, response)
    assert _load(staging, response, operation=operation) == response


@pytest.mark.parametrize("operation", OPERATIONS)
def test_a_response_for_another_operation_is_refused(staging: Path, operation: str) -> None:
    response = _example(operation)
    other = next(name for name in OPERATIONS if name != operation)
    with pytest.raises(ResponseInvalid, match="another operation"):
        _load(staging, response, operation=other)


def test_a_response_for_another_request_is_refused(staging: Path) -> None:
    response = _example("preflight")
    with pytest.raises(ResponseInvalid, match="another request"):
        load_response(
            _write(staging, response),
            request_id="request-preflight-2",
            operation="preflight",
            staging_root=staging,
        )


def test_a_missing_response_is_refused(staging: Path) -> None:
    with pytest.raises(ResponseInvalid, match="no readable response"):
        load_response(
            staging / "absent.json",
            request_id="request-preflight-1",
            operation="preflight",
            staging_root=staging,
        )


def test_an_oversized_response_is_refused(staging: Path) -> None:
    response = _example("preflight")
    with pytest.raises(ResponseInvalid, match="exceeds"):
        _load(staging, response, operation="preflight", max_bytes=16)


def test_the_default_size_limit_is_one_mebibyte() -> None:
    assert MAX_RESPONSE_BYTES == 1048576


def test_a_non_json_response_is_refused(staging: Path) -> None:
    path = staging / "response.json"
    path.write_bytes(b"\xff\xfe not json")
    with pytest.raises(ResponseInvalid, match="not UTF-8 JSON"):
        load_response(
            path,
            request_id="request-preflight-1",
            operation="preflight",
            staging_root=staging,
        )


def test_a_json_array_response_is_refused(staging: Path) -> None:
    path = staging / "response.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ResponseInvalid, match="not an object"):
        load_response(
            path,
            request_id="request-preflight-1",
            operation="preflight",
            staging_root=staging,
        )


def test_an_unknown_field_is_refused(staging: Path) -> None:
    response = _reseal({**_example("preflight"), "stdout": "Blender 5.2.0"})
    with pytest.raises(ResponseInvalid, match="unknown fields"):
        _load(staging, response, operation="preflight")


@pytest.mark.parametrize(
    "field",
    ["request_id", "operation", "status", "diagnostics", "portable_labels", "outputs", "metrics"],
)
def test_a_missing_field_is_refused(staging: Path, field: str) -> None:
    response = _example("preflight")
    del response[field]
    with pytest.raises(ResponseInvalid, match="missing fields"):
        load_response(
            _write(staging, response),
            request_id="request-preflight-1",
            operation="preflight",
            staging_root=staging,
        )


def test_an_edited_response_fails_the_reseal(staging: Path) -> None:
    response = _example("preflight")
    response["metrics"]["object_count"] = 99
    with pytest.raises(ResponseInvalid, match="response_sha256"):
        _load(staging, response, operation="preflight")


@pytest.mark.parametrize("operation", OPERATIONS)
def test_metrics_from_another_operation_are_refused(staging: Path, operation: str) -> None:
    response = _example(operation)
    donor = next(name for name in OPERATIONS if name != operation)
    response = _reseal({**response, "metrics": _example(donor)["metrics"]})
    with pytest.raises(ResponseInvalid, match="metrics"):
        _load(staging, response, operation=operation)


def test_an_unknown_metric_is_refused(staging: Path) -> None:
    response = _example("bake")
    response["metrics"]["gpu_seconds"] = 12
    response = _reseal(response)
    with pytest.raises(ResponseInvalid, match="metrics keys"):
        _load(staging, response, operation="bake")


def test_unsorted_diagnostics_are_refused(staging: Path) -> None:
    response = _reseal(
        {**_example("preflight"), "diagnostics": ["PASS_INVALID", "CAMERA_NOT_FOUND"]}
    )
    with pytest.raises(ResponseInvalid, match="diagnostics"):
        _load(staging, response, operation="preflight")


def test_unsorted_portable_labels_are_refused(staging: Path) -> None:
    response = _reseal({**_example("preflight"), "portable_labels": ["mesh-1", "camera-1"]})
    with pytest.raises(ResponseInvalid, match="portable labels"):
        _load(staging, response, operation="preflight")


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "../escape.json",
        "passes/../../escape.exr",
        "passes//double.exr",
        "passes\\windows.exr",
    ],
)
def test_a_malicious_output_path_is_refused(staging: Path, path: str) -> None:
    response = copy.deepcopy(_example("bake"))
    response["outputs"][0]["path"] = path
    response = _reseal(response)
    with pytest.raises(ResponseInvalid, match="leaves the staging root"):
        _load(staging, response, operation="bake")


def test_an_output_symlink_out_of_staging_is_refused(staging: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (staging / "linked").symlink_to(outside, target_is_directory=True)

    response = copy.deepcopy(_example("bake"))
    response["outputs"][0]["path"] = "linked/baked-texture.png"
    response = _reseal(response)
    with pytest.raises(ResponseInvalid, match="leaves the staging root"):
        _load(staging, response, operation="bake")


def test_unordered_outputs_are_refused(staging: Path) -> None:
    response = copy.deepcopy(_example("export"))
    response["outputs"].reverse()
    response = _reseal(response)
    _stage_outputs(staging, response)
    with pytest.raises(ResponseInvalid, match="ordered by relative path"):
        _load(staging, response, operation="export")


def test_a_repeated_output_path_is_refused(staging: Path) -> None:
    response = copy.deepcopy(_example("export"))
    response["outputs"][1]["path"] = response["outputs"][0]["path"]
    response = _reseal(response)
    _stage_outputs(staging, response)
    with pytest.raises(ResponseInvalid, match="repeat a path"):
        _load(staging, response, operation="export")


def test_a_failed_response_may_not_claim_a_valid_output(staging: Path) -> None:
    response = copy.deepcopy(_example("bake"))
    response["status"] = "failed"
    response["diagnostics"] = ["BAKE_CONTEXT_INVALID"]
    response = _reseal(response)
    _stage_outputs(staging, response)
    with pytest.raises(ResponseInvalid, match="valid output"):
        _load(staging, response, operation="bake")


def test_a_failed_response_with_incomplete_outputs_is_accepted(staging: Path) -> None:
    response = copy.deepcopy(_example("bake"))
    response["status"] = "failed"
    response["diagnostics"] = ["BAKE_CONTEXT_INVALID"]
    response["outputs"][0]["validation"]["status"] = "incomplete"
    response = _reseal(response)
    _stage_outputs(staging, response)
    assert _load(staging, response, operation="bake")["status"] == "failed"


def test_an_unknown_status_is_refused(staging: Path) -> None:
    response = _reseal({**_example("preflight"), "status": "partly"})
    with pytest.raises(ResponseInvalid, match="status"):
        _load(staging, response, operation="preflight")


def test_a_wrong_schema_identifier_is_refused(staging: Path) -> None:
    response = _reseal({**_example("preflight"), "schema_id": "asset-mania/run-manifest"})
    with pytest.raises(ResponseInvalid, match="schema identifier"):
        _load(staging, response, operation="preflight")


def test_a_wrong_schema_version_is_refused(staging: Path) -> None:
    response = _reseal({**_example("preflight"), "schema_version": "2.0"})
    with pytest.raises(ResponseInvalid, match="schema version"):
        _load(staging, response, operation="preflight")


def test_an_unknown_operation_is_refused(staging: Path) -> None:
    """The operation must be one the schema declares, even when both sides agree on it."""
    response = _reseal({**_example("preflight"), "operation": "teleport"})
    with pytest.raises(ResponseInvalid, match="not a worker operation"):
        _load(staging, response, operation="teleport")
