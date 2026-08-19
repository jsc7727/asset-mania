from copy import deepcopy

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


@pytest.mark.parametrize("manifest", [SUCCESS_MANIFEST, FAILURE_MANIFEST])
def test_literal_manifests_validate_against_v1_schema(manifest: dict[str, object]):
    validate(instance=manifest, schema=load_manifest_schema())


def test_schema_rejects_an_absolute_source_path_field():
    manifest = deepcopy(SUCCESS_MANIFEST)
    manifest["inputs"][0]["source_path"] = "/Users/example/private.png"

    with pytest.raises(ValidationError):
        validate(instance=manifest, schema=load_manifest_schema())
