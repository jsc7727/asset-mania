import json
from copy import deepcopy
from pathlib import Path

import pytest
from asset_mania_contracts import (
    DiagnosticCode,
    ResultStatus,
    build_manifest,
    canonical_json,
    load_manifest_schema,
)
from jsonschema import ValidationError, validate

SUCCESS_MANIFEST = {
    "schema_version": "1.0",
    "run_id": "run-1",
    "command": "inspect",
    "tool_version": "0.1.0",
    "created_at": "2026-08-19T00:00:00Z",
    "inputs": [
        {
            "label": "input-1",
            "sha256": "a" * 64,
            "byte_size": 12,
            "media_type": "image/png",
        }
    ],
    "environment": {},
    "parameters": {"workflow": "image-to-3d", "kind": "object"},
    "capabilities": {},
    "artifacts": [],
    "result": {
        "status": "succeeded",
        "diagnostics": ["WORKFLOW_NOT_IMPLEMENTED"],
    },
    "warnings": [],
}

FAILURE_MANIFEST = {
    "schema_version": "1.0",
    "run_id": "run-2",
    "command": "inspect",
    "tool_version": "0.1.0",
    "created_at": "2026-08-19T00:00:00Z",
    "inputs": [],
    "environment": {},
    "parameters": {"workflow": "image-to-3d", "kind": "object"},
    "capabilities": {},
    "artifacts": [],
    "result": {"status": "failed", "diagnostics": ["INPUT_NOT_FOUND"]},
    "warnings": [],
}

SUCCESS_FIXTURE_PATH = Path(__file__).parents[3] / "tests" / "fixtures" / "manifest-v1-success.json"


def test_build_manifest_uses_portable_labels_and_canonical_json():
    manifest = build_manifest(
        run_id="run-1",
        created_at="2026-08-19T00:00:00Z",
        tool_version="0.1.0",
        input_sha256="a" * 64,
        byte_size=12,
        media_type="image/png",
        parameters={"workflow": "image-to-3d", "kind": "object"},
        result_status=ResultStatus.SUCCEEDED,
        diagnostics=[DiagnosticCode.WORKFLOW_NOT_IMPLEMENTED],
    )

    assert manifest["inputs"] == [
        {
            "label": "input-1",
            "sha256": "a" * 64,
            "byte_size": 12,
            "media_type": "image/png",
        }
    ]
    assert "Users/" not in canonical_json(manifest)


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "workflow": "image-to-3d",
            "kind": "object",
            "output_directory": "/Users/example/runs",
        },
        {
            "workflow": "image-to-3d",
            "kind": "object",
            "api_token": "secret-token",
        },
    ],
)
def test_build_manifest_rejects_parameters_outside_the_v1_allowlist(
    parameters: dict[str, object],
):
    with pytest.raises(ValueError, match="parameters"):
        build_manifest(
            run_id="run-1",
            created_at="2026-08-19T00:00:00Z",
            tool_version="0.1.0",
            input_sha256="a" * 64,
            byte_size=12,
            media_type="image/png",
            parameters=parameters,
            result_status=ResultStatus.SUCCEEDED,
            diagnostics=[],
        )


@pytest.mark.parametrize("manifest", [SUCCESS_MANIFEST, FAILURE_MANIFEST])
def test_literal_manifests_validate_against_v1_schema(manifest: dict[str, object]):
    validate(instance=manifest, schema=load_manifest_schema())


def test_schema_rejects_an_absolute_source_path_field():
    manifest = deepcopy(SUCCESS_MANIFEST)
    manifest["inputs"][0]["source_path"] = "/Users/example/private.png"

    with pytest.raises(ValidationError):
        validate(instance=manifest, schema=load_manifest_schema())


def test_committed_success_fixture_is_readable_portable_and_schema_valid():
    payload = SUCCESS_FIXTURE_PATH.read_text(encoding="utf-8")
    manifest = json.loads(payload)

    validate(instance=manifest, schema=load_manifest_schema())
    assert manifest["inputs"][0]["label"] == "input-1"
    assert "source_path" not in payload
    assert "/Users/" not in payload
    assert "\\Users\\" not in payload


def test_internal_error_is_a_closed_v1_manifest_diagnostic():
    manifest = deepcopy(FAILURE_MANIFEST)
    manifest["result"]["diagnostics"] = [DiagnosticCode.INTERNAL_ERROR.value]

    validate(instance=manifest, schema=load_manifest_schema())


def test_output_storage_diagnostic_is_centrally_owned_but_never_manifested():
    assert DiagnosticCode.OUTPUT_STORAGE_UNAVAILABLE.value == "OUTPUT_STORAGE_UNAVAILABLE"
    manifest = deepcopy(FAILURE_MANIFEST)
    manifest["result"]["diagnostics"] = [DiagnosticCode.OUTPUT_STORAGE_UNAVAILABLE.value]

    with pytest.raises(ValidationError):
        validate(instance=manifest, schema=load_manifest_schema())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("environment", {"home_directory": "/Users/example"}),
        (
            "parameters",
            {
                "workflow": "image-to-3d",
                "kind": "object",
                "private_prompt": "a private prompt",
            },
        ),
        ("capabilities", {"provider_token": "secret-token"}),
        ("artifacts", [{"path": "/Users/example/private.glb"}]),
        ("warnings", ["/Users/example/private.png"]),
    ],
)
def test_schema_rejects_unversioned_or_nonportable_field_contents(
    field: str,
    value: object,
):
    manifest = deepcopy(SUCCESS_MANIFEST)
    manifest[field] = value

    with pytest.raises(ValidationError):
        validate(instance=manifest, schema=load_manifest_schema())
