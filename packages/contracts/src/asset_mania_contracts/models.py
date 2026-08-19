import json
from importlib.resources import files

from .diagnostics import DiagnosticCode, ResultStatus


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"


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
        "parameters": parameters,
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
