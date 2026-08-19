import json
from importlib.resources import files

from .diagnostics import DiagnosticCode, ResultStatus

_V1_WORKFLOWS = {"image-to-3d", "scene-to-image"}
_V1_IMAGE_KINDS = {"object", "character", "face-head"}


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"


def _validate_v1_parameters(parameters: dict[str, object]) -> dict[str, object]:
    if set(parameters) - {"workflow", "kind"}:
        raise ValueError("parameters contain fields outside the v1 allowlist")

    workflow = parameters.get("workflow")
    if workflow not in _V1_WORKFLOWS:
        raise ValueError("parameters.workflow must be a supported v1 workflow")

    if workflow == "image-to-3d":
        kind = parameters.get("kind")
        if kind not in _V1_IMAGE_KINDS:
            raise ValueError("parameters.kind must be a supported image-to-3d kind")
    elif "kind" in parameters:
        raise ValueError("parameters.kind is only valid for image-to-3d")

    return dict(parameters)


def build_manifest(
    *,
    run_id: str,
    created_at: str,
    tool_version: str,
    input_sha256: str,
    byte_size: int,
    media_type: str,
    parameters: dict[str, object],
    result_status: ResultStatus,
    diagnostics: list[DiagnosticCode],
) -> dict[str, object]:
    safe_parameters = _validate_v1_parameters(parameters)

    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "command": "inspect",
        "tool_version": tool_version,
        "created_at": created_at,
        "inputs": [
            {
                "label": "input-1",
                "sha256": input_sha256,
                "byte_size": byte_size,
                "media_type": media_type,
            }
        ],
        "environment": {},
        "parameters": safe_parameters,
        "capabilities": {},
        "artifacts": [],
        "result": {
            "status": result_status.value,
            "diagnostics": sorted(diagnostic.value for diagnostic in diagnostics),
        },
        "warnings": [],
    }


def load_manifest_schema() -> dict[str, object]:
    schema_path = files("asset_mania_contracts").joinpath("schema/manifest-v1.schema.json")
    return json.loads(schema_path.read_text(encoding="utf-8"))
